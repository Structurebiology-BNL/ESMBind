# ESMBind

ESMBind ([manuscript](https://www.sciencedirect.com/science/article/pii/S0022283625000282)) is a deep learning and physics-based workflow for predicting metal-binding proteins and modeling their 3D structures with bound metal ions. It combines evolutionary scale modeling (ESM) for residue-level metal binding prediction with physics-based protein-metal modeling to generate detailed 3D structures of protein-metal complexes.

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- Numba 0.60.0
- OpenMM 8.1.1
- AmberTools 23.3

We recommend using mamba to manage dependencies.

## Usage

ESMBind consists of two main components:

1. Deep learning model for residue-level metal binding prediction
2. 3D modeling of protein-metal complexes

### 1. Deep Learning Model

#### Inference

To use the pre-trained model for inference:

```bash
python multi_modal_binding/inference.py --config multi_modal_binding/configs/inference.json
```

We have provided pretrained model weights in the `multi_modal_binding/model/trained_weights` directory. The inference script will automatically load these weights for prediction.

After running the inference script, the predictions will be saved in a pickle file in the `multi_modal_binding/results/` directory, which will be used in the 3D modeling step.

#### Training

To train the model from scratch:

```bash
python multi_modal_binding/train.py --config multi_modal_binding/configs/training.json
```

### Example Data

Both the inference and training scripts assume that embeddings from ESM and ESM-IF are precomputed. The `multi_modal_binding/datasets` directory contains a sample dataset that can be used for both training and inference. Labels are not required for inference; any standard FASTA file will work.

We provide two example scripts, `get_esm_embedding.py` and `get_esm_if_embedding.py`, to generate embeddings for the input sequences and structures. Please refer to the official [ESM repository](https://github.com/facebookresearch/esm) for more details.

### 2. 3D Modeling

After obtaining residue-level predictions from the deep learning model, follow these steps to generate 3D structures with placed metal ions:

1. Convert probability predictions to binding residues:
   ```bash
   python 3D_modeling/src/parse_dl_results.py path/to/predictions_file.pkl ION_TYPE --lower_factor 0.5
   ```
   Replace `path/to/predictions_file.pkl` with the path to your predictions file, `ION_TYPE` with the type of ion you're analyzing (e.g., CA, ZN, MG, etc.), and adjust the `lower_factor` as needed.

2. Generate 3D structures:
   ```bash
   cd 3D_modeling
   bash run-3d-modeling.sh
   ```

   Please revise the `run-3d-modeling.sh` script to specify the input and output directories before running. This script processes the parsed deep learning predictions, places metal ions, and performs energy minimization to produce the final 3D structures.

## Data

This project uses data from:
- [Protein Data Bank](https://www.rcsb.org/)
- [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)
- [BioLiP database](https://zhanggroup.org/BioLiP/index.cgi)

## License

This source code is licensed under the CSI approved 3-clause BSD license found in the LICENSE file in the root directory of this source tree.

## Citation
If you find our work useful, please cite our work as:
```
@article{ESMBind2025,
  title={Predicting metal-binding proteins and structures through integration of evolutionary-scale and physics-based modeling},
  author={Dai, Xin and Henderson, Max and Yoo, Shinjae and Liu, Qun},
  journal={Journal of Molecular Biology},
  pages={168962},
  year={2025},
  publisher={Elsevier}
}

```

## Contact

For questions or issues, please open an issue on GitHub or contact Xin Dai (xdai@bnl.gov).
