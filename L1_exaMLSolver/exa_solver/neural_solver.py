from typing import Any, Type, Optional, Tuple, List, Dict

import torch
import torch_scatter
import pytorch_lightning as pl
from torch import Tensor
from datetime import datetime


from .metrics import L1Distance, L2Distance, L2Ratio, VectorAngle, RMSE
from .losses import CosineDistanceLoss, ResidualLoss

import os
import numpy as np
import csv

import subprocess




class NeuralSolver(pl.LightningModule):

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        optimizer: Type[torch.optim.Optimizer],
        lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        **scheduler_kwargs: Dict[str, Any],
    ):
        super().__init__()
        # Define the CSV path for end-of-epoch metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"<csv_path>"

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scheduler_kwargs = scheduler_kwargs

        self.model = None
        # self.criterion = CosineDistanceLoss()
        self.criterion = ResidualLoss()
        self.elementwise_metric = torch.nn.L1Loss()
        self.systemwise_metrics = torch.nn.ModuleDict(
            {
                "l2_ratio": L2Ratio(),
                "l2_distance": L2Distance(),
                "l1_distance": L1Distance(),
                "angle": VectorAngle(),
                "rmse": RMSE()
            }
        )

    # ensure all metrics are set to GPU device
    def set_device_for_metrics(self, device):
        for metric in self.systemwise_metrics.values():
            metric.to(device)

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch_map: Optional[Tensor] = None
    ) -> Tensor:
        """
        Ensures that if batch_map is None, it defaults to a tensor of zeros,
        which effectively means all nodes belong to the same graph.
        """
        if batch_map is None:
            batch_map = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            print("Warning: batch_map is None. Assuming a single graph.")

        b = x[:, 0]
        b_max = torch_scatter.scatter(b.abs(), batch_map, reduce="max")
        edge_batch_map = batch_map[edge_index[0]]
        matrix_max = torch_scatter.scatter(
            edge_weight.abs(), edge_batch_map, reduce="max"
        )
        x[:, 0] /= b_max[batch_map]
        x[:, 1] /= matrix_max[batch_map]
        scaled_weights = edge_weight / matrix_max[edge_batch_map]
        y_direction = self.model(x, edge_index, scaled_weights, batch_map)
        return y_direction

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        x, edge_index, edge_weight, batch_map, y, b = batch
        n_systems = batch_map.max().item() + 1

        # check if conversion is needed else avoid overhead
        edge_weight = edge_weight if edge_weight.dtype == torch.float32 else edge_weight.to(torch.float32)
        b = b if b.dtype == torch.float32 else b.to(torch.float32)
        y = y if y.dtype == torch.float32 else y.to(torch.float32)

        y_direction = self(x, edge_index, edge_weight, batch_map)

        # computationaly heavy
        matrix = torch.sparse_coo_tensor(
            edge_index, edge_weight, (b.size(0), b.size(0)), dtype=torch.float32
        )

        # computationaly heavy
        b_direction = torch.mv(matrix, y_direction)

        # revert back if required
        # y_loss = self.criterion(y_direction, y, batch_map)
        # b_loss = self.criterion(b_direction, b, batch_map)
        # loss = y_loss + b_loss

        # DEBUGGING
        # print(f"y_direction shape: {y_direction.shape}")
        # print(f"b shape: {b.shape}")
        # print(f"matrix shape: {matrix.shape}")
        # print(f"batch_map shape: {batch_map.shape}")

        # directly minimising the residual
        # Only this for ResidualLoss
        loss = self.criterion(y_direction, b, matrix, batch_map)

        # Only this for CosineDistanceLoss
        # loss = self.criterion(y_direction, y, batch_map)  

        # for debugging
        # Track gradient norms to monitor for exploding/vanishing gradients
        # grad_norm = 0.0
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         grad_norm += param.grad.data.norm(2).item()

        # Log gradient norm
        # self.log("grad_norm", grad_norm, on_step=True, on_epoch=True, prog_bar=True, batch_size=n_systems)

        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=True, prog_bar=True, batch_size=n_systems)

        # Log training loss
        # self.log("loss/train_solution", y_loss, batch_size=n_systems, sync_dist=True)
        # self.log("loss/train_residual", b_loss, batch_size=n_systems, sync_dist=True)
        # Only this for ResidualLoss and solo Cosine
        self.log("loss/train", loss, batch_size=n_systems, sync_dist=True)

        # log norm RMSE
        residual_error = torch.norm(b - b_direction, p=2) / torch.norm(b, p=2)
        self.log("train/residual_error_normed", residual_error, batch_size=n_systems)

        # Early stopping check
        if hasattr(self.trainer, "early_stopping"):
            stop, reason = self.trainer.early_stopping.check(residual_error.item(), torch.norm(b, p=2).item())
            if stop:
                self.trainer.should_stop = True
                self.log("early_stopping/reason", reason)

        # log MAE
        y_mae = torch.mean(torch.abs(y_direction - y))
        b_mae = torch.mean(torch.abs(b_direction - b))
        self.log("train/y_mae", y_mae, batch_size=n_systems)
        self.log("train/b_mae", b_mae, batch_size=n_systems)

        # log R-squared (R²) for Model Fit Quality
        ss_total = torch.sum((y - torch.mean(y)) ** 2)
        ss_residual = torch.sum((y - y_direction) ** 2)
        r_squared = 1 - ss_residual / ss_total
        self.log("train/r_squared", r_squared, batch_size=n_systems)

        # # log standard dev of gradience
        # grads = [param.grad.data.norm(2) for param in self.model.parameters() if param.grad is not None]
        # if grads:  # Only compute variance if there are gradients
        #     grad_variance = torch.std(torch.stack(grads))
        #     self.log("grad_variance", grad_variance, on_step=True, on_epoch=True, prog_bar=True, batch_size=n_systems)
        # else:
        #     grad_variance = 0.0  # or handle differently if needed
        #     self.log("grad_variance", grad_variance, on_step=True, on_epoch=True, prog_bar=True, batch_size=n_systems)

        return loss

    def _evaluation_step(
        self,
        phase_name: str,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        x, edge_index, edge_weight, batch_map, y, b = batch
        n_systems = batch_map.max().item() + 1

        """
        DEBUGGING STATEMENTS
        """
        # verify contents
        # print(f"Batch {batch_idx} contents:")
        # print(f"x (Input features): shape = {x.shape}, values = {x[:5]}")
        # print(f"edge_index (Graph edges): shape = {edge_index.shape}, values = {edge_index[:, :5]}")
        # print(f"edge_weight (Edge weights): shape = {edge_weight.shape}, values = {edge_weight[:5]}")
        # print(f"batch_map (Batch indices): shape = {batch_map.shape}, values = {batch_map[:5]}")
        # print(f"y (Real solution x): shape = {y.shape}, values = {y[:5]}")
        # print(f"b (Target vector): shape = {b.shape}, values = {b[:5]}")

        # set device for metric from GPU
        device = y.device
        self.set_device_for_metrics(device)

        matrix = torch.sparse_coo_tensor(
            edge_index, edge_weight, (b.size(0), b.size(0)), dtype=torch.float64
        )
        y_direction = self(x, edge_index, edge_weight.to(torch.float32), batch_map)
        y_direction = y_direction.to(torch.float64)

        """
        commented out but can use for debugging / tracking
        """
        # store y_direction (save generated x vectors)
        # self.log(f"{phase_name}_y_direction", y_direction, batch_size=n_systems)

        save_dir = f"./predictions/{phase_name}/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"batch_{batch_idx}.npz")

        # store original x vector for comparison
        real_x = batch[4].cpu().numpy()

        np.savez(
            save_path,
            A_indices=edge_index.cpu().numpy(),
            A_values=edge_weight.cpu().numpy(),
            x_model=y_direction.detach().cpu().numpy(),  # save model-generated x
            x_real=real_x, # save real x
            b=b.cpu().numpy(),
        )

        p_direction = torch.mv(matrix, y_direction)
        p_squared_norm = torch_scatter.scatter_sum(p_direction.square(), batch_map)
        bp_dot_product = torch_scatter.scatter_sum(p_direction * b, batch_map)
        scaler = torch.clamp_min(bp_dot_product / p_squared_norm, 1e-16)
        y_hat = y_direction * scaler[batch_map]
        b_hat = p_direction * scaler[batch_map]
        
        # works for CosineDistanceLoss but not for ResidualLoss
        # y_loss = self.criterion(y_hat, y, batch_map)
        # b_loss = self.criterion(b_hat, b, batch_map)
        # loss = y_loss + b_loss
        loss = self.criterion(y_direction, b, matrix, batch_map)
        # self.log(f"loss/{phase_name}_solution", y_loss, batch_size=n_systems, sync_dist=True)
        # self.log(f"loss/{phase_name}_residual", b_loss, batch_size=n_systems, sync_dist=True)
        self.log(f"loss/{phase_name}", loss, batch_size=n_systems, sync_dist=True)

        # **Compute element-wise residual relative to b**
        eps = 1e-8  # Avoid division by zero
        residual_elementwise = torch.abs(b_hat - b) / (torch.abs(b) + eps)
        mean_elementwise_residual = torch.mean(residual_elementwise)

        # **Log element-wise residual stat**
        self.log(f"residual/{phase_name}_elementwise_mean", mean_elementwise_residual, batch_size=n_systems, sync_dist=True)

        for metric_name, metric in self.systemwise_metrics.items():
            self.log(
                f"metrics/{phase_name}_{metric_name}",
                metric(y_hat, y, batch_map),
                batch_size=n_systems,
            )
            # log residual metrics related to b_hat (A * predicted x)
            residual_metric_value = metric(b_hat, b, batch_map)
            self.log(
                f"residual/{phase_name}_{metric_name}",
                residual_metric_value,
                batch_size=n_systems,
            )

            # normalise residuals by b norm
            residual_norm_b = residual_metric_value / torch.norm(b, p=2).item()
            self.log(f"residual/{phase_name}_{metric_name}_normalized_by_b", residual_norm_b, batch_size=n_systems, sync_dist=True)

        self.log(
            f"metrics/{phase_name}_absolute_error",
            self.elementwise_metric(y_hat, y),
            batch_size=y.size(0),
        )


    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self._evaluation_step("val", batch, batch_idx)

    def test_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self._evaluation_step("test", batch, batch_idx)

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizers = []
        schedulers = []
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        optimizers.append(optimizer)
        if self.lr_scheduler is not None:
            schedulers.append(
                {
                    "scheduler": self.lr_scheduler(optimizer, **self.scheduler_kwargs),
                    "interval": "epoch",
                    "name": "lr",
                }
            )
        return optimizers, schedulers

    @staticmethod
    def log_gpu_usage(tag=""):
        """Logs the GPU memory and utilization using nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                print(f"--- GPU Usage ({tag}) ---")
                for i, line in enumerate(result.stdout.strip().split("\n")):
                    gpu_util, mem_used, mem_total = map(int, line.split(", "))
                    print(f"GPU {i} | Utilization: {gpu_util}% | Memory: {mem_used} MB / {mem_total} MB")
                print("---------------------------")
            else:
                print(f"Error while running nvidia-smi: {result.stderr}")
        except FileNotFoundError:
            print("nvidia-smi command not found. Ensure NVIDIA drivers and CUDA are properly installed.")

    def on_train_epoch_end(self):
        try:
            self.log_gpu_usage(tag="GPU USAGE - End of Epoch")
        except Exception as e:
            print(f"An error occurred while logging GPU usage: {e}")


    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "optimizer_states" in checkpoint:
            print("Restoring optimizer states...")
            self.trainer.optimizers[0].load_state_dict(checkpoint["optimizer_states"])