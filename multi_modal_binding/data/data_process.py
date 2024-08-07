import logging
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import KFold
from .datasets import MultiModalDataset, MultiModalTestDataset


def calculate_pos_weight(label_list):
    pos_num, neg_num = [], []
    for label in label_list:
        pos_ = sum([int(i) for i in label])
        pos_num.append(pos_)
        neg_num.append(len(label) - pos_)

    pos_weight = sum(pos_num) / sum(neg_num)
    return pos_weight


def process_fasta_file(fasta_file):
    with open(fasta_file, "r") as file:
        lines = file.readlines()

    ID_list, seq_list, label_list = [], [], []
    for i in range(0, len(lines), 3):
        id = lines[i].strip().replace(">", "")
        ID_list.append(id)
        seq = lines[i + 1].strip()
        label = list(map(int, list(lines[i + 2].strip())))
        assert len(seq) == len(label), "seq and label length mismatch"
        seq_list.append(seq)
        label_list.append(label)
    assert len(ID_list) == len(seq_list), "broken fasta input"
    assert len(ID_list) == len(label_list), "broken fasta input"
    assert len(seq_list) == len(set(seq_list)), "duplicate entries found"

    return ID_list, seq_list, label_list


def feature_extraction(ID_list, conf, ligand=None, inference=False):
    if not inference:
        assert ligand is not None, "ligand must be provided for training"
        precomputed_feature_path = conf.data.precomputed_feature
    else:
        precomputed_feature_path = conf.inference.precomputed_feature

    protein_features = multimodal_embedding(
        ID_list,
        precomputed_feature_path=precomputed_feature_path,
        normalize=conf.data.normalize,
        ligand=ligand,
        inference=inference,
    )

    return protein_features


def get_train_val_ids(ids, fold_to_use_as_val=1, random_seed=1):
    # Initialize the KFold class with 5 splits
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    # Split the IDs into 5 folds
    folds = {}
    for i, (_, val_index) in enumerate(kf.split(ids)):
        fold_ids = [ids[idx] for idx in val_index]
        folds[i + 1] = fold_ids
    # Get the validation IDs from the selected fold
    val_ids = folds[fold_to_use_as_val]

    # Get the training IDs from the remaining folds
    train_ids = []
    for fold, ids in folds.items():
        if fold != fold_to_use_as_val:
            train_ids.extend(ids)

    return train_ids, val_ids


def prep_train_dataset(
    conf,
    random_seed=0,
    ligand="MN",
):
    """
    Prepare the training dataset for protein-ligand binding prediction.

    Args:
        conf (object): Configuration object containing data and training settings.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 0.
        ligand (str, optional): Ligand identifier. Defaults to "MN".
        training (bool, optional): Flag indicating whether to prepare the training dataset. Defaults to True.

    Returns:
        tuple: A tuple containing the training dataset, validation dataset (if training=True), and positive weight.
               The training dataset and validation dataset are instances of MultiModalDataset,
               depending on the data type specified in the configuration object.
    """
    ID_list, _, label_list = process_fasta_file(conf.data.fasta_path)
    pos_weight = calculate_pos_weight(label_list)
    label_data = dict(zip(ID_list, label_list))

    protein_features = feature_extraction(
        ID_list,
        conf,
        ligand=ligand,
    )
    fold_to_use_as_val = getattr(conf.training, "fold_to_use_as_val", 1)
    train_id, val_id = get_train_val_ids(
        ID_list, fold_to_use_as_val, random_seed=random_seed
    )
    train_dataset = MultiModalDataset(train_id, label_data, protein_features)
    val_dataset = MultiModalDataset(val_id, label_data, protein_features)

    logging.info(
        "Train and val samples for {}: {}, {}".format(
            ligand, len(train_dataset), len(val_dataset)
        )
    )
    return train_dataset, val_dataset, pos_weight


def prep_test_dataset(conf, ligand=None):
    """
    Return test dataset for inference (without label information)
    """
    with open(conf.inference.fasta_path) as handle:
        recs = list(SeqIO.parse(handle, "fasta"))

    ID_list = [rec.id for rec in recs]

    protein_features = feature_extraction(
        ID_list,
        conf,
        ligand=ligand,
        inference=True,
    )
    test_dataset = MultiModalTestDataset(ID_list, protein_features)

    logging.info("Total # of test samples for is {}".format(len(test_dataset)))

    return test_dataset


def multimodal_embedding(
    ID_list,
    precomputed_feature_path=None,
    normalize=True,
    ligand="CU",
    inference=False,
):
    protein_embeddings = {}
    max_repr_esm = np.load(
        "multi_modal_binding/data/normalization_constants/esm_repr_max.npy"
    )
    min_repr_esm = np.load(
        "multi_modal_binding/data/normalization_constants/esm_repr_min.npy"
    )
    max_repr_esm_if = np.load(
        "multi_modal_binding/data/normalization_constants/esm_if_repr_max.npy"
    )
    min_repr_esm_if = np.load(
        "multi_modal_binding/data/normalization_constants/esm_if_repr_min.npy"
    )
    if inference:
        feature_path_1 = precomputed_feature_path + "/esm"
        feature_path_2 = precomputed_feature_path + "/esm_if"
    else:
        feature_path_1 = precomputed_feature_path + "/{}/esm".format(ligand)
        feature_path_2 = precomputed_feature_path + "/{}/esm_if".format(ligand)

    for id in ID_list:
        embeddding_esm = np.load(feature_path_1 + "/{}.npy".format(id))
        embedding_esm_if = np.load(feature_path_2 + "/{}.npy".format(id))
        if normalize:
            embeddding_esm = (embeddding_esm - min_repr_esm) / (
                max_repr_esm - min_repr_esm
            )
            embedding_esm_if = (embedding_esm_if - min_repr_esm_if) / (
                max_repr_esm_if - min_repr_esm_if
            )

        protein_embeddings[id] = [embeddding_esm, embedding_esm_if]

    return protein_embeddings
