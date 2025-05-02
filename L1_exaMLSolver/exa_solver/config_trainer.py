from typing import Union
from pathlib import Path

import torch.nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


from .config import Config
from .neural_solver import NeuralSolver


class ConfigTrainer:
    def __init__(
        self, config: Config, output_dir: Union[None, Path, str], **trainer_kwargs
    ):
        self.config = config
        self.output_dir = output_dir
        self.trainer_kwargs = trainer_kwargs

        # Initialize trainer and module as None
        self.trainer = None
        self.module = None

        # Initialize logger and checkpointing
        if output_dir is None:
            self.logger = False
            self.enable_checkpointing = False
        else:
            self.logger = TensorBoardLogger(
                str(output_dir),
                name="",
                default_hp_metric=False,
            )
            self.enable_checkpointing = True

        # Set data loaders and input dimensions
        self.train_loader = self.config.get_train_loader()
        self.val_loader = self.config.get_test_loader()
        self.input_dim = self.train_loader.dataset.feature_dim

    def set_hyperparams(self, params: dict = None):
        """Set hyperparameters for the trainer."""
        if params is None:
            params = self.config.get_module_params()  # Default for traditional training

        # Ensure logger is properly set and obtain the correct directory
        checkpoint_dir = Path(self.logger.log_dir) / "checkpoints" if self.logger else Path(self.output_dir) / "checkpoints"

        # Define ModelCheckpoint for saving the best model
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="residual/val_rmse_normalized_by_b",  # Metric to track
            filename="best-checkpoint-epoch={epoch:02d}-step={step}-residual={residual/val_rmse_normalized_by_b:.4f}",
            save_top_k=1,  # Keep only the best model
            mode="min",  
            verbose=True,
            auto_insert_metric_name=False,  
        )

        # Define ModelCheckpoint for always saving the latest checkpoint
        latest_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="latest-checkpoint-epoch={epoch:02d}-step={step}-residual={residual/val_rmse_normalized_by_b:.4f}",
            save_last=True,  # Save only the latest checkpoint
            verbose=True,
            auto_insert_metric_name=False,  
        )

        # Add both callbacks to the trainer
        callbacks = [
            LearningRateMonitor(),
            TQDMProgressBar(refresh_rate=100),
            EarlyStopping(
                monitor="residual/val_rmse_normalized_by_b",  
                patience=10,  
                mode="min",  
                verbose=True,  
            ),
            best_checkpoint_callback,  # Saves best checkpoint
            latest_checkpoint_callback,  # Always saves latest checkpoint
        ]

        # Initialize the module and trainer
        self.module = NeuralSolver(**params)
        self.trainer = pl.Trainer(
            logger=self.logger,
            enable_checkpointing=self.enable_checkpointing,
            callbacks=callbacks,  
            max_epochs=self.config.get_epochs(),
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            benchmark=True,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="value",
            strategy="fsdp",
            **self.trainer_kwargs,
        )

    def fit(self, model: torch.nn.Module) -> None:
        if self.trainer is None or self.module is None:
            raise ValueError(
                "The trainer or module has not been initialized. Call `set_hyperparams` first."
            )
        self.module.set_model(model)
        return self.trainer.fit(self.module, self.train_loader, self.val_loader)


