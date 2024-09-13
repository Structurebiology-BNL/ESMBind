import argparse
import torch
import os
from Bio import SeqIO
from tqdm import tqdm
import numpy as np
from esm import pretrained
import logging


def esm_embedding(ID_list, seq_list, output_path, batch_size, device):
    protein_features = {}
    os.makedirs(output_path, exist_ok=True)

    # Load the fixed model esm2_t33_650M_UR50D
    model, alphabet = pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    data = list(zip(ID_list, seq_list))
    with torch.no_grad():
        for j in tqdm(range(0, len(data), batch_size)):
            partial_data = data[j : j + batch_size]
            batch_labels, _, batch_tokens = batch_converter(partial_data)
            result = model(
                batch_tokens.to(device),
                repr_layers=[len(model.layers)],
                return_contacts=False,
            )
            embedding = (
                result["representations"][len(model.layers)].detach().cpu().numpy()
            )
            batch_lens=(batch_tokens != alphabet.padding_idx).sum(1)

            for seq_num in range(len(embedding)):
                seq_len = batch_lens[seq_num]
                # get rid of cls and eos token
                seq_emd = embedding[seq_num][1 : (seq_len - 1)]
                np.save(os.path.join(output_path, batch_labels[seq_num]), seq_emd)
                protein_features[batch_labels[seq_num]] = seq_emd

    return protein_features


def process_fasta(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    seq_list, ID_list = [], []
    for rec in records:
        seq_list.append(str(rec.seq))
        ID_list.append(rec.id.split("|")[1])

    assert len(ID_list) == len(seq_list), "Broken FASTA input"
    assert len(seq_list) == len(set(seq_list)), "Duplicate entries found"
    return ID_list, seq_list


def main():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings using ESM2 (esm2_t33_650M_UR50D)"
    )
    parser.add_argument("fasta_file", help="Path to the input FASTA file")
    parser.add_argument(
        "output_dir", help="Path to the output directory for embeddings"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device number if using CUDA"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Set device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Process FASTA file
    logging.info(f"Processing FASTA file: {args.fasta_file}")
    ID_list, seq_list = process_fasta(args.fasta_file)
    logging.info(f"Processed {len(ID_list)} sequences")

    # Generate embeddings
    logging.info("Generating embeddings using ESM2 (esm2_t33_650M_UR50D)")
    esm_embedding(
        ID_list,
        seq_list,
        output_path=args.output_dir,
        batch_size=args.batch_size,
        device=device,
    )

    logging.info("Embedding generation completed")


if __name__ == "__main__":
    main()
