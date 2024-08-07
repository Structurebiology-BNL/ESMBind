import os
import requests
from openmm import app
from Bio.PDB import (
    PDBIO,
    Select,
    PDBParser,
    MMCIFParser,
    MMCIFIO,
    Chain,
)
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from constants import substitutions
import warnings

# Suppress specific PDBConstructionWarnings
warnings.simplefilter("ignore", PDBConstructionWarning)


# Custom Select class for modifying residues
class ResidueSelect(Select):
    def __init__(self, substitutions):
        self.substitutions = substitutions

    def accept_residue(self, residue):
        if residue.resname in self.substitutions:
            residue.resname = self.substitutions[residue.resname]
        return 1


class NonWaterSelect(Select):
    """Only accepts residues that are NOT water."""

    def accept_residue(self, residue):
        return residue.get_resname() != "HOH"


class NonHeteroAtomSelect(Select):
    """Selects only non-hetero (i.e., standard protein and nucleic acid) atoms, excluding water."""

    def accept_residue(self, residue):
        return not (residue.id[0] != " " or residue.get_resname() == "HOH")


class ChainAndNonWaterSelect(Select):
    """Selects a specific chain and non-water residues."""

    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        return residue.get_resname() != "HOH"


def replace_nonstandard_residues(input_file, substitutions):
    """
    Replace non-standard residues in a PDB or CIF file with standard residues.

    Parameters:
    input_file : str
        Path to the input PDB or CIF file.
    substitutions : dict
        Dictionary mapping non-standard residue names to standard ones.
    """
    file_format = "pdb" if input_file.lower().endswith(".pdb") else "cif"
    parser = PDBParser() if file_format == "pdb" else MMCIFParser()

    structure = parser.get_structure("structure", input_file)

    io = PDBIO() if file_format == "pdb" else MMCIFIO()
    io.set_structure(structure)
    select = ResidueSelect(substitutions)

    io.save(input_file, select=select)


def parse_structure(file_path):
    if file_path.endswith(".cif"):
        parser = MMCIFParser()
    else:
        parser = PDBParser()
    return parser.get_structure(file_path.split("/")[-1], file_path)


def sanitize_chain_id(chain_id):
    """Convert chain ID to a single character if it's not already."""
    if len(chain_id) == 1 and chain_id.isalpha():
        return chain_id
    return chain_id[0] if chain_id[0].isalpha() else "A"


## File Saving Functions
def save_structure(structure, output_path, original_chain_id=None):
    """Save the structure to a file, handling both PDB and mmCIF formats."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)

        if output_path.endswith(".pdb"):
            io = PDBIO()
            io.set_structure(structure)
        elif output_path.endswith(".cif"):
            io = MMCIFIO()
            io.set_structure(structure)
        else:
            raise ValueError("Unsupported file format. Please use '.pdb' or '.cif'.")

        if original_chain_id:
            sanitized_id = sanitize_chain_id(original_chain_id)
            io.save(output_path, ChainAndNonWaterSelect(sanitized_id))
        else:
            io.save(output_path, NonWaterSelect())


def preprocess_structure(input_file, output_file, output_format="cif"):
    """
    Remove all heteroatoms and water molecules from the input structure file,
    then save in the specified format (PDB or mmCIF).

    Args:
    input_file (str): Path to the input structure file (PDB or mmCIF format).
    output_file (str): Path to save the processed structure file.
    output_format (str): Format of the output file ('pdb' or 'cif', default: 'cif').

    Raises:
    ValueError: If the input file format is not supported or the output format is invalid.
    """
    # Detect input file format
    if input_file.lower().endswith((".cif", ".mmcif")):
        parser = MMCIFParser()
    elif input_file.lower().endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Unsupported input file format. Please use PDB or mmCIF.")

    # Parse the structure
    structure = parser.get_structure("temp_structure", input_file)

    # Choose the appropriate writer based on the output format
    if output_format.lower() == "cif":
        io = MMCIFIO()
    elif output_format.lower() == "pdb":
        io = PDBIO()
    else:
        raise ValueError("Invalid output format. Please use 'pdb' or 'cif'.")

    # Set the structure and save
    io.set_structure(structure)
    io.save(output_file, select=NonHeteroAtomSelect())


def fix_pdb(file_path):
    """
    Replace non-standard residues in a PDB or mmCIF file with standard residues.
    If the file does not exist, check for a .cif file, or download it using the download_pdb function.
    """
    if not os.path.exists(file_path):
        # Check if a .cif file exists with the same name
        cif_path = os.path.splitext(file_path)[0] + ".cif"
        if os.path.exists(cif_path):
            file_path = cif_path
        else:
            # Download the file if it doesn't exist
            pdb_id = os.path.splitext(os.path.basename(file_path))[0]
            download_folder = os.path.dirname(file_path)
            file_path = download_pdb(pdb_id, download_folder)

    replace_nonstandard_residues(file_path, substitutions)

    return file_path


def download_pdb(pdb_id, download_folder):
    pdb_id = pdb_id.upper()  # Ensure PDB ID is in uppercase
    url_cif = f"https://files.rcsb.org/download/{pdb_id}.cif"
    url_pdb = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        response = requests.get(url_cif)
        response.raise_for_status()  # Raise an error for bad status codes

        file_path = os.path.join(download_folder, f"{pdb_id}.cif")
        with open(file_path, "wb") as file:
            file.write(response.content)

        return file_path
    except requests.exceptions.HTTPError:
        try:
            response = requests.get(url_pdb)
            response.raise_for_status()  # Raise an error for bad status codes

            file_path = os.path.join(download_folder, f"{pdb_id}.pdb")
            with open(file_path, "wb") as file:
                file.write(response.content)

            return file_path
        except requests.exceptions.HTTPError:
            raise Exception(
                f"Failed to download {pdb_id} from both {url_cif} and {url_pdb}"
            )


def initialize_chain(structure, chain_id):
    chain = structure[0][chain_id]
    sanitized_id = sanitize_chain_id(chain_id)
    new_chain = Chain.Chain(sanitized_id)
    return chain, new_chain


def remove_water_from_model(input_cif_path, output_cif_path, verbose=False):
    """
    Reads a mmCIF file, removes water molecules, and saves the result to a new file.

    Parameters:
    input_cif_path (str): Path to the input cif file.
    output_cif_path (str): Path to the output cif file with water removed.
    """
    parser = MMCIFParser(QUIET=True)
    io = MMCIFIO()
    structure = parser.get_structure("Structure", input_cif_path)
    io.set_structure(structure)
    io.save(output_cif_path, NonWaterSelect())
    if verbose:
        print(f"Saved {output_cif_path}, with water molecules removed.")


def write_without_solvent_mmcif(topology, positions, output_file_name):
    # Filter out water and create a new topology and positions list
    new_topology = app.Topology()
    new_chain = new_topology.addChain()
    new_positions = []

    for chain in topology.chains():
        for residue in chain.residues():
            if residue.name == "HOH":
                continue  # Skip water residues

            new_residue = new_topology.addResidue(residue.name, new_chain)
            for atom in residue.atoms():
                new_topology.addAtom(atom.name, atom.element, new_residue)
                new_positions.append(positions[atom.index])

    # Save as mmCIF file
    with open(output_file_name, "w") as outfile:
        app.PDBxFile.writeFile(new_topology, new_positions, outfile)
