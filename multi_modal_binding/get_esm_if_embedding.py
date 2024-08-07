import argparse
import esm.inverse_folding
import torch
from Bio import SeqIO
import numpy as np
import os
import logging


def get_structure(pdb_path, chain_id):
    structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
    coords, seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    return coords, seq


def embedding(fasta_file, output_dir, pdb_folder, chain_id="A"):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load FASTA file and get IDs
    with open(fasta_file) as handle:
        recs = list(SeqIO.parse(handle, "fasta"))
    ids = [rec.id.split("|")[1] for rec in recs]
    seqs = {rec.id.split("|")[1]: str(rec.seq) for rec in recs}

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)

    failed_ids = []
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for id in ids:
            output_file = os.path.join(output_dir, f"{id}.npy")
            if os.path.exists(output_file):
                logging.info(f"Embedding for {id} already exists. Skipping.")
                continue

            pdb_path = os.path.join(pdb_folder, f"{id}.pdb")
            if not os.path.exists(pdb_path):
                logging.warning(f"Failed to find PDB file for {id}")
                failed_ids.append(id)
                continue

            try:
                coords, seq = get_structure(pdb_path, chain_id)
                if seq != seqs[id]:
                    logging.warning(f"Sequence mismatch for {id}")
                    failed_ids.append(id)
                    continue

                rep = esm.inverse_folding.util.get_encoder_output(
                    model, alphabet, coords
                )
                np.save(output_file, rep.detach().cpu().numpy())
                logging.info(f"Successfully generated embedding for {id}")
            except Exception as e:
                logging.error(f"Error processing {id}: {str(e)}")
                failed_ids.append(id)

    if failed_ids:
        with open(os.path.join(output_dir, "failed_ids.txt"), "w") as f:
            for id in failed_ids:
                f.write(f"{id}\n")
        logging.warning(f"Failed IDs: {failed_ids}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings from FASTA and PDB files"
    )
    parser.add_argument("fasta_file", help="Path to the input FASTA file")
    parser.add_argument(
        "output_dir", help="Path to the output directory for embeddings"
    )
    parser.add_argument("pdb_folder", help="Path to the folder containing PDB files")
    parser.add_argument("--chain_id", default="A", help="Chain ID to use (default: A)")

    args = parser.parse_args()

    embedding(args.fasta_file, args.output_dir, args.pdb_folder, args.chain_id)


if __name__ == "__main__":
    main()
