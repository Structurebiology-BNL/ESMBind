"""
Metal Ion Placement in Protein Structures

This module provides functionality to add metal ions to protein structures based on 
predicted binding residues. It includes functions for manipulating PDB structures, 
identifying potential binding sites, and placing metal ions in optimal positions.

The main function, process_pdb_with_ions, takes a PDB structure and predicted binding 
residues as input, adds the specified metal ions to the structure, and optionally 
runs tleap and parmed for further processing.
Author: Xin Dai
"""

import os
import numpy as np
from Bio.PDB import (
    MMCIFParser,
    Atom,
    Residue,
    Structure,
    Model,
    NeighborSearch,
)
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from sklearn.cluster import AgglomerativeClustering
from numba import jit, prange
from constants import binding_atoms, ion_binding_distances, ion_coordination_data
from pdb_utils import preprocess_structure, save_structure, initialize_chain
from utils import execute_tleap_and_parmed
import warnings

# Suppress specific PDBConstructionWarnings
warnings.simplefilter("ignore", PDBConstructionWarning)


# Structure Manipulation Functions
def add_metal_ion(chain, new_residue_id, ion_coord, element="ZN"):
    """
    Add a metal ion to the given chain at the specified coordinates.

    Args:
        chain (Bio.PDB.Chain.Chain): The chain to add the ion to.
        new_residue_id (int): The residue ID for the new ion.
        ion_coord (numpy.ndarray): The 3D coordinates of the ion.
        element (str): The element symbol of the ion (default: "ZN" for zinc).

    Returns:
        None
    """
    element_name = element.rjust(4)
    new_residue = Residue.Residue((element, new_residue_id, " "), element, " ")
    ion_atom = Atom.Atom(
        name=element_name,
        coord=ion_coord,
        bfactor=0.00,
        occupancy=1.0,
        altloc=" ",
        fullname=element_name,
        serial_number=new_residue_id,
        element=element,
    )
    new_residue.add(ion_atom)
    chain.add(new_residue)


def finalize_structure(
    new_chain, terminal_residue, num_added_ions, num_existing_residues
):
    """
    Create a new structure with the modified chain and add the terminal residue if present.

    Args:
        new_chain (Bio.PDB.Chain.Chain): The modified chain.
        terminal_residue (Bio.PDB.Residue.Residue): The terminal residue to add, if any.
        num_added_ions (int): The number of ions added to the structure.
        num_existing_residues (int): The number of residues in the original structure.

    Returns:
        Bio.PDB.Structure.Structure: The finalized structure.
    """
    if terminal_residue:
        new_terminal_residue = terminal_residue.copy()
        new_terminal_residue.id = (" ", num_existing_residues + num_added_ions + 1, " ")
        new_chain.add(new_terminal_residue)
    new_structure = Structure.Structure("new_protein")
    new_model = Model.Model(0)
    new_model.add(new_chain)
    new_structure.add(new_model)
    return new_structure


def copy_chain_without_terminal_residue(chain, new_chain):
    """
    Copy a chain to a new chain object, excluding the terminal residue.

    Args:
        chain (Bio.PDB.Chain.Chain): The original chain to copy.
        new_chain (Bio.PDB.Chain.Chain): The new chain to copy into.

    Returns:
        tuple: (new_chain, terminal_residue, new_residue_id_counter)
            - new_chain: The copied chain without the terminal residue.
            - terminal_residue: The terminal residue, if found.
            - new_residue_id_counter: The next available residue ID.
    """
    terminal_residue = None
    new_residue_id_counter = 1

    for residue in chain:
        is_terminal = any(atom.name in ["OXT", "HXT"] for atom in residue)
        if is_terminal:
            terminal_residue = residue
        else:
            new_residue = residue.copy()
            new_residue.id = (" ", new_residue_id_counter, " ")
            new_chain.add(new_residue)
            new_residue_id_counter += 1

    return new_chain, terminal_residue, new_residue_id_counter


# Ion Placement Functions
def get_initial_ion_placements(chain, predicted_residues, ion):
    """
    Get initial placements for ions based on predicted binding residues using center of mass of binding atoms.
    Args:
        chain (Bio.PDB.Chain.Chain): The protein chain.
        predicted_residues (list): List of residue IDs predicted to bind the ion.
        ion (str): The ion type (e.g., "ZN" for zinc).

    Returns:
        list: A list of tuples (position, [residues], atom_count) for each potential binding site.
    """
    placements = []
    for residue in chain:
        if residue.id[1] in predicted_residues:
            binding_atoms = get_binding_atoms(residue, ion)
            if binding_atoms:
                position = calculate_center_of_mass(binding_atoms)
                if position is not None:
                    placements.append((position, [residue], len(binding_atoms)))
    print(f"Initial placements for {ion}: {len(placements)}")
    return placements


def calculate_center_of_mass(atoms):
    """
    Calculate the center of mass of a group of atoms.

    Args:
        atoms (list): A list of Bio.PDB.Atom.Atom objects.

    Returns:
        numpy.ndarray or None: The 3D coordinates of the center of mass, or None if the list is empty.
    """
    if not atoms:
        return None

    total_mass = 0
    weighted_coords = np.zeros(3)

    for atom in atoms:
        mass = atom.mass
        total_mass += mass
        weighted_coords += atom.coord * mass

    return weighted_coords / total_mass if total_mass > 0 else None


def get_binding_atoms(residue, ion):
    """
    Get all potential binding atoms for a residue and a specific ion.

    Args:
        residue (Bio.PDB.Residue.Residue): The residue to check for binding atoms.
        ion (str): The ion type (e.g., "ZN" for zinc).

    Returns:
        list: A list of atoms that are potential binding sites for the ion.
    """
    binding_atom_list = []
    for atom in residue:
        if atom.name in binding_atoms[ion].get(residue.resname, {}):
            binding_atom_list.append(atom)
        elif atom.name == "O" and "O" in binding_atoms[ion]:
            binding_atom_list.append(atom)
    return binding_atom_list


@jit(nopython=True)
def is_atom_in_binding_atoms(atom, binding_atoms):
    for binding_atom in binding_atoms:
        if (
            np.all(atom["coord"] == binding_atom["coord"])
            and atom["element"] == binding_atom["element"]
        ):
            return True
    return False


@jit(nopython=True)
def score_position(pos, binding_atoms, all_atoms, ion_distances, default_distance):
    binding_score = 0.0
    clash_score = 0.0

    for i in range(len(binding_atoms)):
        atom = binding_atoms[i]
        distance = np.linalg.norm(pos - atom["coord"])
        ideal_distance = default_distance
        for j in range(len(ion_distances)):
            if ion_distances[j]["element"] == atom["element"]:
                ideal_distance = ion_distances[j]["distance"]
                break

        if 0.8 * ideal_distance <= distance <= 1.2 * ideal_distance:
            binding_score += 1 - abs(distance - ideal_distance) / (0.2 * ideal_distance)
        elif distance < 0.8 * ideal_distance:
            return float("-inf")

    for i in range(len(all_atoms)):
        atom = all_atoms[i]
        if not is_atom_in_binding_atoms(atom, binding_atoms):
            distance = np.linalg.norm(pos - atom["coord"])
            if distance < 2.0:
                return float("-inf")
            elif distance < 2.5:
                clash_score += 1 - (distance / 2.5)

    return binding_score - clash_score


@jit(nopython=True, parallel=True)
def score_positions(
    positions, binding_atoms, all_atoms, ion_distances, default_distance
):
    scores = np.empty(len(positions))
    for i in prange(len(positions)):
        scores[i] = score_position(
            positions[i], binding_atoms, all_atoms, ion_distances, default_distance
        )
    return scores


def create_grid(center, radius, resolution):
    x, y, z = np.mgrid[
        center[0] - radius : center[0] + radius : resolution,
        center[1] - radius : center[1] + radius : resolution,
        center[2] - radius : center[2] + radius : resolution,
    ]
    return np.vstack([x.ravel(), y.ravel(), z.ravel()]).T


def find_optimal_ion_position(
    binding_atoms,
    all_atoms,
    initial_position,
    ion,
    search_radius=2.5,
    grid_resolution=0.05,
):
    """
    Find the optimal ion position by searching for empty spaces near binding atoms.
    Args:
    binding_atoms (list): List of binding atoms.
    all_atoms (list): List of all atoms in the structure.
    initial_position (numpy.ndarray): Initial position for the ion.
    ion (str): The ion type (e.g., "ZN" for zinc).
    search_radius (float): Radius to search around the initial position.
    grid_resolution (float): Resolution of the search grid.
    Returns:
    numpy.ndarray: Optimal position for the ion.
    """
    # Convert binding_atoms and all_atoms to NumPy arrays for Numba compatibility
    binding_atoms_array = np.array(
        [(atom.coord, atom.element) for atom in binding_atoms],
        dtype=[("coord", float, (3,)), ("element", "U2")],
    )
    all_atoms_array = np.array(
        [(atom.coord, atom.element) for atom in all_atoms],
        dtype=[("coord", float, (3,)), ("element", "U2")],
    )

    # Convert ion_binding_distances to a Numba-compatible format
    ion_distances = np.array(
        [
            (element, distance)
            for element, distance in ion_binding_distances[ion].items()
        ],
        dtype=[("element", "U2"), ("distance", float)],
    )
    default_distance = 2.2  # Default distance if element is not found

    # Create grid positions around the initial position
    positions = create_grid(initial_position, search_radius, grid_resolution)

    # Score positions using the optimized function
    scores = score_positions(
        positions, binding_atoms_array, all_atoms_array, ion_distances, default_distance
    )

    # If no valid positions found, gradually increase the search radius
    if np.all(scores == float("-inf")):
        for _ in range(5):
            search_radius *= 1.2
            print(f"Expanding search radius to {search_radius:.2f} A")
            positions = create_grid(initial_position, search_radius, grid_resolution)
            scores = score_positions(
                positions,
                binding_atoms_array,
                all_atoms_array,
                ion_distances,
                default_distance,
            )
            if np.any(scores != float("-inf")):
                break

    # If still no valid positions, return the center of binding atoms
    if np.all(scores == float("-inf")):
        print(
            "Warning: No valid position found. Returning center of mass of binding atoms."
        )
        return calculate_center_of_mass(binding_atoms)

    optimal_position = positions[np.argmax(scores)]
    return optimal_position


def cluster_ion_placements(placements, structure, distance_threshold=7.0, ion="ZN"):
    """
    Cluster ion placements and find optimal positions for ions, ensuring at least one cluster is returned.
    Optimizes position even for a single placement.

    Args:
        placements (list): List of initial ion placements.
        structure (Bio.PDB.Structure.Structure): The protein structure.
        distance_threshold (float): Maximum distance between placements to be clustered.
        ion (str): The ion type (e.g., "ZN" for zinc).

    Returns:
        list: A list of optimized ion placements, with at least one placement.
    """
    if not placements:
        return []

    all_atoms = list(structure.get_atoms())
    ns = NeighborSearch(all_atoms)

    if len(placements) == 1:
        return optimize_single_placement(placements[0], ns, ion)

    positions = np.array([p[0] for p in placements])

    # Start with the original distance threshold
    current_threshold = distance_threshold
    min_threshold = 0.5 * distance_threshold
    while current_threshold >= min_threshold:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=current_threshold
        ).fit(positions)

        num_clusters = len(set(clustering.labels_))

        if num_clusters >= 1:
            break  # We found at least one cluster

        current_threshold *= 0.8  # Decrease threshold by 20% if no clusters found

    # If no clusters were found even at minimum threshold, treat each placement as its own cluster
    if current_threshold < min_threshold:
        clustering.labels_ = np.arange(len(placements))

    optimized_placements = []
    for label in set(clustering.labels_):
        cluster_indices = np.where(clustering.labels_ == label)[0]
        cluster_placements = [placements[j] for j in cluster_indices]

        all_binding_atoms = [
            atom
            for p in cluster_placements
            for residue in p[1]
            for atom in get_binding_atoms(residue, ion)
        ]
        initial_position = calculate_center_of_mass(all_binding_atoms)

        if initial_position is not None:
            nearby_atoms = ns.search(initial_position, 8.0)  # 8.0 Angstrom radius

            optimal_position = find_optimal_ion_position(
                all_binding_atoms, nearby_atoms, initial_position, ion
            )

            merged_residues = list(set([r for p in cluster_placements for r in p[1]]))
            merged_atom_count = sum(p[2] for p in cluster_placements)

            optimized_placements.append(
                (optimal_position, merged_residues, merged_atom_count)
            )

    return optimized_placements


def optimize_single_placement(placement, ns, ion):
    """
    Optimize a single ion placement.

    Args:
        placement (tuple): A single ion placement (position, [residues], atom_count).
        ns (Bio.PDB.NeighborSearch): NeighborSearch object for the structure.
        ion (str): The ion type (e.g., "ZN" for zinc).

    Returns:
        list: A list containing the optimized placement.
    """
    position, residues, _ = placement
    all_binding_atoms = [
        atom for residue in residues for atom in get_binding_atoms(residue, ion)
    ]

    nearby_atoms = ns.search(position, 8.0)  # 8.0 Angstrom radius
    optimal_position = find_optimal_ion_position(
        all_binding_atoms, nearby_atoms, position, ion
    )

    return [(optimal_position, residues, len(all_binding_atoms))]


def ion_placement(chain, predicted_residues, ion):
    """
    Determine the final ion positions based on predicted binding residues.

    Args:
        chain (Bio.PDB.Chain.Chain): The protein chain.
        predicted_residues (list): List of residue IDs predicted to bind the ion.
        ion (str): The ion type (e.g., "ZN" for zinc).

    Returns:
        tuple: (ion_positions, valid_groups)
            - ion_positions: List of 3D coordinates for placed ions.
            - valid_groups: List of residue groups associated with each ion position.
    """
    initial_placements = get_initial_ion_placements(chain, predicted_residues, ion)
    clustered_placements = cluster_ion_placements(initial_placements, chain, ion=ion)

    ion_positions = []
    valid_groups = []
    min_binding_atoms = ion_coordination_data[ion]["min_coordination"]
    for position, residues, atom_count in clustered_placements:
        if atom_count >= min_binding_atoms:
            print(f"Adding ion placement with {atom_count} binding atoms")
            ion_positions.append(position)
            valid_groups.append(residues)
        else:
            print(f"Ignoring ion placement with {atom_count} binding atoms")

    print(f"Final # of ions for {ion}: {len(ion_positions)}")
    return ion_positions, valid_groups


# Main Processing Function
def process_pdb_with_ions(
    pdb_id,
    chain_id,
    predicted_residues,
    ion,
    pdb_directory="./",
    output_file=None,
    output_file_before_minimization=None,
    temp_dir=None,
    run_tleap=True,
    input_format=None,
):
    """
    Process a PDB or CIF structure by adding metal ions based on predicted binding residues.
    Args:
        pdb_id (str): The PDB ID of the structure.
        chain_id (str): The chain ID to process.
        predicted_residues (list): List of residue IDs predicted to bind the ion.
        ion (str): The ion type to add (e.g., "ZN" for zinc).
        pdb_directory (str): Directory containing the input files (default: "./").
        output_file (str): Path to save the output PDB file (optional).
        output_file_before_minimization (str): Path to save the output PDB file before minimization (optional).
        temp_dir (str): Directory for temporary files (optional).
        run_tleap (bool): Whether to run tleap and parmed after processing (default: True).
        input_format (str): Format of the input file ('pdb' or 'cif'). If None, it will be auto-detected (default: None).
    Returns:
        dict: A dictionary mapping added ion residue IDs to their associated binding residues.
    Raises:
        FileNotFoundError: If the input file is not found.
        ValueError: If the input file format is not supported or cannot be determined.
        Exception: If there's an error during tleap and parmed execution.
    Note:
        The intermediate cleaned file is always in CIF format, regardless of the input format.
    """
    # Determine input file format and path
    if input_format is None:
        if os.path.exists(os.path.join(pdb_directory, f"{pdb_id}.cif")):
            input_format = "cif"
        elif os.path.exists(os.path.join(pdb_directory, f"{pdb_id}.pdb")):
            input_format = "pdb"
        else:
            raise FileNotFoundError(
                f"No .cif or .pdb file found for {pdb_id} in {pdb_directory}"
            )

    input_file = os.path.join(pdb_directory, f"{pdb_id}.{input_format}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    # Remove all heteroatoms and water molecules
    cleaned_tmp_model = os.path.join(temp_dir, f"cleaned_{pdb_id}.cif")
    preprocess_structure(input_file, cleaned_tmp_model, output_format="cif")

    # Parse the cleaned structure (always in CIF format)
    parser = MMCIFParser(QUIET=True)
    original_structure = parser.get_structure("protein", cleaned_tmp_model)

    # Initialize and copy the chain
    chain, new_chain = initialize_chain(original_structure, chain_id)
    new_chain, terminal_residue, new_residue_id_counter = (
        copy_chain_without_terminal_residue(chain, new_chain)
    )
    num_existing_residues = len(list(new_chain.get_residues()))

    # Place ions in the structure
    ion_positions, valid_groups = ion_placement(new_chain, predicted_residues, ion)

    # Add ions to the structure
    ion_to_group = {}
    for position, group in zip(ion_positions, valid_groups):
        new_residue_id_counter += 1
        add_metal_ion(new_chain, new_residue_id_counter, position, element=ion)
        ion_to_group[new_residue_id_counter] = [residue.id[1] for residue in group]

    # Finalize the new structure
    new_structure = finalize_structure(
        new_chain, terminal_residue, len(ion_to_group), num_existing_residues
    )

    # Save the new structure
    new_pdb_file = os.path.join(temp_dir, f"{pdb_id}_with_{ion}.pdb")
    save_structure(new_structure, new_pdb_file, original_chain_id=chain_id)
    if output_file_before_minimization:
        save_structure(
            new_structure, output_file_before_minimization, original_chain_id=chain_id
        )

    # Run tleap and parmed if requested
    if run_tleap:
        try:
            execute_tleap_and_parmed(new_pdb_file, ion, temp_dir=temp_dir)
        except Exception as e:
            print(f"Error executing tleap and parmed: {e}")
            if output_file:  # fall back to save the structure without minimization
                save_structure(new_structure, output_file, original_chain_id=chain_id)
            raise

    return ion_to_group
