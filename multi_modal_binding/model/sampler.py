from typing import Callable
import pandas as pd
import torch
import torch.utils.data

"""
adapted from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
"""


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        pos_ratio: float = 0.5,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        # label_to_count = df["label"].value_counts()
        # label_to_count = pd.Series({"0.0": 1, "1.0": self.pos_ratio}, name="count")
        # self.pos_numbers = label_to_count[1]
        # weights = 1.0 / label_to_count[df["label"]]

        # self.weights = torch.DoubleTensor(weights.to_list())
        self.num_neg = (df["label"] == 0).sum()
        self.num_pos = (df["label"] == 1).sum()

        # Create a probability tensor
        self.weights = torch.tensor(
            [
                (
                    (1 - pos_ratio) / self.num_neg
                    if label == 0
                    else pos_ratio / self.num_pos
                )
                for label in df["label"]
            ]
        )

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.targets
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.targets
        else:
            raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples