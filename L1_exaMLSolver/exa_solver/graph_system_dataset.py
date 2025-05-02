from typing import Sequence
from pathlib import Path
from collections import namedtuple

import numpy as np
import scipy.sparse
import torch
import torch_geometric.data
import zipfile
from torch.utils.data import Dataset

from .augmentations import FeatureAugmentation

System = namedtuple("System", ["A_indices", "A_values", "b", "x"])


class GraphSystemDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_matrices: int,
        feature_augmentations: Sequence[FeatureAugmentation] = tuple(),
    ):
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.num_matrices = num_matrices
        self.feature_augmentations = tuple(feature_augmentations)

        self._paths = tuple(sorted(self.dataset_dir.glob("*.npz")))
        if len(self._paths) != self.num_matrices:
            raise ValueError("The dataset size differs from the expected one")

    def __len__(self) -> int:
        return self.num_matrices

    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        system = None
        while system is None:
            filepath = self._paths[idx]
            with np.load(str(filepath)) as npz_file:
                # prevent crash from random failures with unknown cause
                try:
                    system = System(**npz_file)
                except zipfile.BadZipFile as e:
                    print(
                        f"Skip file after {type(e).__name__}:", e.with_traceback(None)
                    )
                    idx += 1
                    continue
        A_values = system.A_values.astype(np.float32)
        b = system.b.astype(np.float32)
        m = scipy.sparse.coo_matrix(
            (A_values, list(system.A_indices)),
            shape=(system.b.size, system.b.size),
            dtype=np.float32,
        )
        features = [b[:, np.newaxis], m.diagonal()[:, np.newaxis]]
        for augmentation in self.feature_augmentations:
            augmentation_features = augmentation(m, b)
            features.append(augmentation_features)
        features = np.column_stack(features)

        data = torch_geometric.data.Data(
            torch.from_numpy(features),
            edge_index=torch.from_numpy(system.A_indices.astype(np.int64)),
            edge_attr=torch.from_numpy(system.A_values),
            y=torch.from_numpy(system.x),
            b=torch.from_numpy(system.b),
        )
        return data

    @property
    def feature_dim(self) -> int:
        return 2 + sum(
            augmentation.feature_dim for augmentation in self.feature_augmentations
        )
