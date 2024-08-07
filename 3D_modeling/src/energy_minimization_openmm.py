"""
OpenMM Protein Minimization Module

This module provides functionality for performing energy minimization on protein structures
using the OpenMM molecular dynamics simulation toolkit. It applies a series of restraints
to gradually relax the protein structure while maintaining the positions of metal ions.

The main function, perform_minimization, carries out a multi-step minimization process:
1. Apply restraints to all atoms except solvent
2. Reduce restraints to only backbone atoms and metal ions
3. Further reduce restraints to only metal ions

This approach allows for careful relaxation of the protein structure while preserving
the overall fold and metal ion positions.

Author: Xin Dai
Date: 07-2024
"""

import openmm
import os
from pathlib import Path
from openmm import (
    app,
    unit,
    LangevinMiddleIntegrator,
    CustomExternalForce,
    MonteCarloBarostat,
)
from openmm.app import PDBxFile
from pdb_utils import remove_water_from_model
from constants import standard_residues, backbone_atoms


def perform_minimization(
    prmtop_file,
    inpcrd_file,
    output_file,
    restraint_force_constant=41840,
    use_gpu=True,
    gpu_index="0",
    metal_ion_name="MN",
    verbose=False,
):
    """
    Perform energy minimization on a protein structure with gradual relaxation of restraints.

    This function applies a series of energy minimizations with decreasing restraints,
    starting from all non-solvent atoms and ending with only metal ions restrained.

    Args:
        prmtop_file (str): Path to the AMBER topology file.
        inpcrd_file (str): Path to the AMBER coordinate file.
        output_file (str): Path to save the minimized structure.
        restraint_force_constant (float): Initial force constant for restraints in kJ/mol/nm^2.
        use_gpu (bool): Whether to use GPU acceleration.
        gpu_index (str): GPU device index to use if use_gpu is True.
        metal_ion_name (str): Name of the metal ion in the structure.
        verbose (bool): Whether to print detailed progress information.

    Returns:
        str: Path to the saved minimized structure file.
    """

    def safe_execute(action, simulation, last_state, *args, **kwargs):
        """
        Safely execute a simulation step and handle potential errors.

        Args:
            action (callable): The simulation action to perform.
            simulation (openmm.app.Simulation): The current simulation object.
            last_state (openmm.State): The last successful state.
            *args, **kwargs: Additional arguments for the action.

        Returns:
            tuple: (success (bool), new_state (openmm.State))
        """
        try:
            action(*args, **kwargs)
            new_state = simulation.context.getState(getEnergy=True, getPositions=True)
            return True, new_state
        except Exception as e:
            print(f"An error occurred: {e}")
            if last_state:
                with open(output_file, "w") as f:
                    PDBxFile.writeFile(
                        simulation.topology, last_state.getPositions(), f
                    )
                remove_water_from_model(output_file, output_file)
                print(f"Recovered state saved to {output_file}")
            return False, last_state

    def apply_restraints(
        system,
        topology,
        restrain_atoms=standard_residues,
        restraint_force_constant=1000,
    ):
        """
        Apply harmonic restraints to specified atoms in the system.

        Args:
            system (openmm.System): The OpenMM system object.
            topology (openmm.app.Topology): The topology of the system.
            restrain_atoms (set): Set of atom names to restrain.
            restraint_force_constant (float): Force constant for the restraints.

        Returns:
            openmm.System: The system with added restraints.
        """
        restraint_force = CustomExternalForce(
            "k*periodicdistance(x, y, z, x0, y0, z0)^2"
        )
        restraint_force.addGlobalParameter(
            "k", restraint_force_constant * unit.kilojoules_per_mole / unit.nanometer
        )
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")

        for atom in topology.atoms():
            if atom.residue.name in restrain_atoms:
                position = inpcrd.positions[atom.index]
                restraint_force.addParticle(
                    atom.index, position.value_in_unit(unit.nanometers)
                )

        system.addForce(restraint_force)
        return system

    def build_system(
        apply_restraints_flag=True,
        restrain_atoms=None,
        restraint_force_constant=None,
        constant_pressure=False,
    ):
        """
        Build an OpenMM system with optional restraints and barostat.

        Args:
            apply_restraints_flag (bool): Whether to apply restraints.
            restrain_atoms (set): Set of atom names to restrain.
            restraint_force_constant (float): Force constant for restraints.
            constant_pressure (bool): Whether to use constant pressure conditions.

        Returns:
            openmm.System: The constructed system.
        """
        system = prmtop.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1 * unit.nanometer,
            constraints=app.HBonds,
        )

        if constant_pressure:
            barostat = MonteCarloBarostat(1.0 * unit.atmospheres, 298 * unit.kelvin, 25)
            system.addForce(barostat)

        if (
            apply_restraints_flag
            and restrain_atoms
            and restraint_force_constant is not None
        ):
            system = apply_restraints(
                system,
                prmtop.topology,
                restrain_atoms,
                restraint_force_constant,
            )

        return system

    def create_simulation(system, current_state=None):
        """
        Create an OpenMM simulation object with an integrator and optional state.

        Args:
            system (openmm.System): The OpenMM system object.
            current_state (openmm.State): Optional state to set in the simulation context.

        Returns:
            openmm.app.Simulation: The created simulation object.
        """
        integrator = LangevinMiddleIntegrator(temperature, friction, timestep)
        simulation = app.Simulation(
            prmtop.topology, system, integrator, platform, properties
        )
        if inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
        if current_state is not None:
            simulation.context.setState(current_state)

        return simulation

    # Load input files
    prmtop = app.AmberPrmtopFile(prmtop_file)
    inpcrd = app.AmberInpcrdFile(inpcrd_file)

    # Define simulation constants
    temperature = 298 * unit.kelvin
    friction = 1 / unit.picosecond
    timestep = 1 * unit.femtoseconds

    # Setup platform
    platform_name = "CUDA" if use_gpu else "CPU"
    platform = openmm.Platform.getPlatformByName(platform_name)
    properties = {"CudaPrecision": "mixed", "DeviceIndex": gpu_index} if use_gpu else {}

    # First minimization: Restraint on all atoms except for solvent
    system = build_system(
        apply_restraints_flag=True,
        restrain_atoms=standard_residues.union({metal_ion_name}),
        restraint_force_constant=restraint_force_constant,
        constant_pressure=False,
    )
    simulation = create_simulation(system)
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    last_successful_state = simulation.context.getState(
        getEnergy=True, getPositions=True, getVelocities=True
    )
    print(f"Initial energy: {last_successful_state.getPotentialEnergy()}")
    if verbose:
        print(f"Start minimization with restraint {restraint_force_constant:.1f}...")
    success, last_successful_state = safe_execute(
        simulation.minimizeEnergy, simulation, last_successful_state, maxIterations=0
    )
    if not success:
        return output_file

    # Second minimization: Reduce restraint to backbone atoms and metal ions
    restraint_force_constant *= 0.5
    system = build_system(
        apply_restraints_flag=True,
        restrain_atoms=backbone_atoms.union({metal_ion_name}),
        restraint_force_constant=restraint_force_constant,
        constant_pressure=True,
    )
    simulation = create_simulation(system, last_successful_state)
    if verbose:
        print(
            f"Minimizing energy on new system with restraints on backbone and ions only..."
        )
    success, last_successful_state = safe_execute(
        simulation.minimizeEnergy, simulation, last_successful_state, maxIterations=0
    )
    if verbose:
        print(f"Current energy: {last_successful_state.getPotentialEnergy()}")
    if not success:
        return output_file

    # Third minimization: Further reduce restraint to only metal ions
    restraint_force_constant *= 0.5
    system = build_system(
        apply_restraints_flag=True,
        restrain_atoms={metal_ion_name},
        restraint_force_constant=restraint_force_constant,
        constant_pressure=True,
    )
    simulation = create_simulation(system, last_successful_state)
    if verbose:
        print(f"Minimizing energy with restraint on metal ions only...")
    success, last_successful_state = safe_execute(
        simulation.minimizeEnergy, simulation, last_successful_state, maxIterations=0
    )
    if not success:
        return output_file

    print(
        f"Final energy after minimization: {last_successful_state.getPotentialEnergy()}"
    )

    # Save the minimized structure
    output_dir = Path(output_file).parent
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        PDBxFile.writeFile(simulation.topology, last_successful_state.getPositions(), f)

    remove_water_from_model(output_file, output_file, verbose=verbose)

    return output_file


if __name__ == "__main__":
    # Example usage
    prmtop_file = "amber.prmtop"
    inpcrd_file = "amber.inpcrd"
    output_file = "output.cif"
    perform_minimization(
        prmtop_file,
        inpcrd_file,
        output_file,
        verbose=True,
        restraint_force_constant=41840,
    )
