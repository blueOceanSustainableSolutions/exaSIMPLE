#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
import random

from .config import Config
from .config_trainer import ConfigTrainer
from .neural_solver import NeuralSolver
from .single_inference import SingleInference

from .optuna_objective import run_optuna_study

torch.set_float32_matmul_precision("high")

def set_seed(seed: int):
    """
    Sets seeds for reproducibility in PyTorch, NumPy, and Python's random module.
    """
    pl.seed_everything(seed)
    random.seed(seed)  # Python's random
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch for CUDA
    torch.cuda.manual_seed_all(seed)  # PyTorch for multi-GPU setups



class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Command line interface for exaSIMPLE Solver, built on top of Neural Sparse Linear Solvers",
            usage=(
                "python3 -m exas <command> [<args>]\n"
                "\n"
                "train       Train the model\n"
                "eval        Evaluate the model\n"
                "export      Export a trained model\n"
                "optuna      Run Optuna hyperparameter tuning\n"
            ),
        )
        parser.add_argument(
            "command",
            type=str,
            help="Sub-command to run",
            choices=(
                "train",
                "eval",
                "export",
                "optuna",
            ),
        )


        args = parser.parse_args(sys.argv[1:2])
        command = args.command.replace("-", "_")
        if not hasattr(self, command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        getattr(self, command)()

    @staticmethod
    def train() -> None:

        set_seed(42)

        warnings.filterwarnings(
            "ignore",
            ".*Trying to infer the `batch_size` from an ambiguous collection.*",
        )
        parser = argparse.ArgumentParser(
            description="Train the model",
            usage="python3 -m exas train config-path [--output-dir OUTPUT-DIR] [--checkpoint CHECKPOINT]",
        )
        parser.add_argument(
            "config_path",
            metavar="config-path",
            help="Path to configuration file",
        )
        parser.add_argument("--output-dir", help="Output directory", default="./runs")
        parser.add_argument(
            "--checkpoint", help="Path to checkpoint file to resume training", default=None
        )
        args = parser.parse_args(sys.argv[2:])

        NeuralSolver.log_gpu_usage(tag="GPU USAGE - Starting Point")

        config = Config(args.config_path)

        config_trainer = ConfigTrainer(
            config,
            Path(args.output_dir).expanduser(),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=4 if torch.cuda.is_available() else 1, 
        )


        """CHECK GPUS"""
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            print(f"GPUs available: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        else:
            print("No GPUs detected. Falling back to CPU.")
        """"""

        config_trainer.set_hyperparams()

        model = config.get_model(config_trainer.input_dim)

        if args.checkpoint:
            print(f"Attempting to load checkpoint: {args.checkpoint}")
            checkpoint = torch.load(
                args.checkpoint,
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            print("Checkpoint successfully loaded. Adjusting state_dict keys...")

            # Adjust model state_dict keys to match the model keys
            adjusted_state_dict = {
                key.replace("model.", ""): value for key, value in checkpoint["state_dict"].items()
            }
            model.load_state_dict(adjusted_state_dict)

            # Resume epoch progress
            if "epoch" in checkpoint:
                config_trainer.trainer.fit_loop.epoch_progress.current.completed = checkpoint["epoch"]
                print(f"Resuming training at epoch {checkpoint['epoch']}")

            print(f"Resumed training from checkpoint: {args.checkpoint}")

        config.save(config_trainer.trainer.logger.log_dir)
        config_trainer.fit(model)

    @staticmethod
    def eval() -> None:
        parser = argparse.ArgumentParser(
            description="Evaluate the model",
            usage="python3 -m exas eval config-path --checkpoint CHECKPOINT",
        )
        parser.add_argument(
            "config_path",
            metavar="config-path",
            help="Path to configuration file",
        )
        parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
        args = parser.parse_args(sys.argv[2:])

        config = Config(args.config_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        test_loader = config.get_test_loader()
        model = config.get_model(test_loader.dataset.feature_dim)
        module = NeuralSolver(**checkpoint["hyper_parameters"])
        module.set_model(model)
        module.load_state_dict(checkpoint["state_dict"])
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            num_nodes=1
        )

        results = trainer.test(module, dataloaders=test_loader)
        print(results)

    @staticmethod
    def export() -> None:
        parser = argparse.ArgumentParser(
            description="Export a trained model",
            usage="python3 -m exas export config-path --checkpoint CHECKPOINT",
        )
        parser.add_argument(
            "config_path",
            metavar="config-path",
            help="Path to configuration file",
        )
        parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
        parser.add_argument(
            "--output-path", help="Output directory", default="model.pt"
        )
        parser.add_argument("--gpu", help="Export model for GPU", action="store_true")
        args = parser.parse_args(sys.argv[2:])

        config = Config(args.config_path)
        device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        test_dataset = config.get_test_dataset()

        model = config.get_model(test_dataset.feature_dim)
        processors = config.get_preprocessors()
        module = SingleInference(model, processors)
        module.load_state_dict(checkpoint["state_dict"])
        module = module.to(device).eval().requires_grad_(False)
        test_sample = test_dataset[0]
        test_inputs = (
            test_sample.b.to(device),
            test_sample.edge_index.to(device),
            test_sample.edge_attr.to(device),
        )
        traced_module = torch.jit.trace(
            module,
            test_inputs,
        )
        traced_module = torch.jit.freeze(traced_module)
        traced_module.save(args.output_path)


if __name__ == "__main__":
    set_seed(42)
    if sys.argv[1] == "optuna":
        run_optuna_study()
    else:
        CLI()

