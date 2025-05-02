from pathlib import Path
from typing import Any, Union, Tuple, Dict

import yaml
import torch
import torch.nn
import torch_geometric.data
import torch_geometric.loader
from torch import Tensor
from torch.utils.data import DataLoader

from . import gnn
from . import preprocessors
from . import augmentations
from .graph_system_dataset import GraphSystemDataset


class UnpackingCollater(torch_geometric.loader.dataloader.Collater):
    def __init__(self, dataset):
        super().__init__(dataset=dataset, follow_batch=tuple(), exclude_keys=tuple())

    def __call__(
        self, batch: torch_geometric.data.Data
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        collated_batch = super().__call__(batch)
        if isinstance(collated_batch, torch_geometric.data.Batch):
            return (
                collated_batch.x,
                collated_batch.edge_index,
                collated_batch.edge_attr,
                collated_batch.batch,
                collated_batch.y,
                collated_batch.b,
            )
        return collated_batch


class Config:
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path).expanduser()
        with self.config_path.open("r") as f:
            self.config = yaml.safe_load(f)

    def _get_dataset(self, train: bool) -> GraphSystemDataset:
        dataset_config = self.config["DATASET"]
        split_dataset_config = dataset_config["TRAIN" if train else "TEST"]
        dataset_augmentations = []
        if "AUGMENTATIONS" in dataset_config:
            augmentations_config = dataset_config["AUGMENTATIONS"]
            for augmentation_config in augmentations_config:
                augmentation_class = getattr(augmentations, augmentation_config["NAME"])
                augmentation_kwargs = {
                    k.lower(): v for k, v in augmentation_config.items() if k != "NAME"
                }
                augmentation = augmentation_class(**augmentation_kwargs)
                dataset_augmentations.append(augmentation)
        return GraphSystemDataset(
            dataset_dir=split_dataset_config["DIRECTORY"],
            num_matrices=split_dataset_config["NUM_MATRICES"],
            feature_augmentations=dataset_augmentations,
        )

    def get_preprocessors(self) -> Tuple[preprocessors.Preprocessor, ...]:
        if "AUGMENTATIONS" not in self.config["DATASET"]:
            return tuple()
        augmentations_config = self.config["DATASET"]["AUGMENTATIONS"]
        processors = []
        for augmentation_config in augmentations_config:
            preprocessor_class = getattr(
                preprocessors,
                augmentation_config["NAME"].replace("Augmentation", "Preprocessor"),
            )
            preprocessor_kwargs = {
                k.lower(): v for k, v in augmentation_config.items() if k != "NAME"
            }
            processor = preprocessor_class(**preprocessor_kwargs)
            processors.append(processor)
        return tuple(processors)

    def get_train_dataset(self) -> GraphSystemDataset:
        return self._get_dataset(train=True)

    def get_test_dataset(self) -> GraphSystemDataset:
        return self._get_dataset(train=False)

    def _get_loader(self, train: bool) -> DataLoader:
        dataset = self._get_dataset(train)
        if train:
            batch_size = self.config["OPTIMIZER"]["BATCH_SIZE"]
        else:
            batch_size = self.config["TEST"]["BATCH_SIZE"]
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train,
            collate_fn=UnpackingCollater(dataset=dataset), 
            num_workers=4,  
            pin_memory=True,
        )
        return loader

    def get_train_loader(self) -> DataLoader:
        return self._get_loader(train=True)

    def get_test_loader(self) -> DataLoader:
        return self._get_loader(train=False)

    def get_module_params(self) -> Dict[str, Any]:
        optimizer_config = self.config["OPTIMIZER"]
        params = {
            k.lower(): v
            for k, v in optimizer_config.items()
            if k not in ("NAME", "BATCH_SIZE", "EPOCHS")
        }
        params["optimizer"] = getattr(torch.optim, optimizer_config["NAME"])

        if "SCHEDULER" in self.config:
            scheduler_config = self.config["SCHEDULER"]
            params["lr_scheduler"] = getattr(
                torch.optim.lr_scheduler, scheduler_config["NAME"]
            )

            # Adjust casing for specific keys like T_MAX
            scheduler_kwargs = {}
            for k, v in scheduler_config.items():
                if k == "T_MAX":  # Handle case-sensitive parameters explicitly
                    scheduler_kwargs["T_max"] = v
                elif k != "NAME":
                    scheduler_kwargs[k.lower()] = v  # Lowercase for other parameters

            params.update(scheduler_kwargs)

        return params

    def get_epochs(self) -> int:
        return self.config["OPTIMIZER"]["EPOCHS"]

    def get_model(self, n_features: int) -> torch.nn.Module:
        architecture_config = self.config["ARCHITECTURE"]
        model_class = getattr(gnn, architecture_config["NAME"])
        model_kwargs = {
            k.lower(): v for k, v in architecture_config.items() if k != "NAME"
        }
        model_kwargs["n_features"] = n_features
        model = model_class(**model_kwargs)
        return model

    def set_learning_rate(self, lr: float) -> None:
        self.config["OPTIMIZER"]["LEARNING_RATE"] = lr

    def save(self, output_dir: Union[str, Path]) -> None:
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.config_path.name
        with output_path.open("w") as f:
            yaml.safe_dump(
                self.config, f, default_flow_style=False, allow_unicode=True, indent=4
            )

    def get_convergence_params(self):
        convergence_config = self.config.get("CONVERGENCE", {})
        return {
            "rtol": convergence_config.get("RTOL", 1e-5),
            "atol": convergence_config.get("ATOL", 1e-50),
            "dtol": convergence_config.get("DTOL", 1e5),
            "patience": convergence_config.get("PATIENCE", 3),
            "max_epochs": convergence_config.get("MAX_EPOCHS", 50)
        }