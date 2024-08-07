import torch
import numpy as np
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    """
    The training dataset is per-residue basis, i.e., concatenating residues from different sequences
    """

    def __init__(self, ID_list, label_data, protein_features):
        feat_1 = []
        feat_2 = []
        label_list = []
        for id in ID_list:
            feat_1.append(protein_features[id][0])
            feat_2.append(protein_features[id][1])
            label_list.append(label_data[id])

        self.targets = np.concatenate(label_list, dtype=np.float32)
        self.feat_1 = np.concatenate(feat_1, dtype=np.float32)
        self.feat_2 = np.concatenate(feat_2, dtype=np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feat_1 = self.feat_1[idx]
        feat_2 = self.feat_2[idx]
        target = self.targets[idx]
        return feat_1, feat_2, target


class MultiModalTestDataset(Dataset):
    """
    The test dataset is per-sequence, so it needs padding and masking
    """

    def __init__(self, ID_list, protein_features):
        self.ID_list = ID_list
        self.protein_features = protein_features
        self.feature_dim_1 = protein_features[ID_list[0]][0].shape[1]
        self.feature_dim_2 = protein_features[ID_list[0]][1].shape[1]

    def __len__(self):
        return len(self.ID_list)

    def __getitem__(self, idx):
        protein_id = self.ID_list[idx]
        protein_feat = self.protein_features[protein_id]
        feat_1, feat_2 = protein_feat[0], protein_feat[1]

        return protein_id, feat_1, feat_2

    def padding(self, batch, maxlen):
        batch_feat_1 = []
        batch_feat_2 = []
        batch_protein_mask = []
        batch_id = []
        for id, feat_1, feat_2 in batch:
            batch_id.append(id)
            padded_feat_1 = np.zeros((maxlen, self.feature_dim_1))
            padded_feat_1[: feat_1.shape[0]] = feat_1
            padded_feat_1 = torch.tensor(padded_feat_1, dtype=torch.float)
            batch_feat_1.append(padded_feat_1)

            padded_feat_2 = np.zeros((maxlen, self.feature_dim_2))
            padded_feat_2[: feat_2.shape[0]] = feat_2
            padded_feat_2 = torch.tensor(padded_feat_2, dtype=torch.float)
            batch_feat_2.append(padded_feat_2)

            protein_mask = np.zeros(maxlen)
            protein_mask[: feat_1.shape[0]] = 1
            protein_mask = torch.tensor(protein_mask, dtype=torch.long)
            batch_protein_mask.append(protein_mask)

        return (
            batch_id,
            torch.stack(batch_feat_1),
            torch.stack(batch_feat_2),
            torch.stack(batch_protein_mask),
        )

    def collate_fn(self, batch):
        maxlen = max([feat_1.shape[0] for _, feat_1, _ in batch])
        protein_id, feat_1, feat_2, protein_mask = self.padding(batch, maxlen)
        return protein_id, feat_1, feat_2, protein_mask
