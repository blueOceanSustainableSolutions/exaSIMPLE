import torch

from torch import Tensor
from torch_scatter import scatter_sum


class ResidualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-12  # Avoid division by zero

    def forward(
        self,
        preds: Tensor,  # Predicted solution x
        target_b: Tensor,  # True right-hand side b
        matrix: Tensor,  # Sparse matrix A (torch.sparse_coo_tensor)
        batch_map: Tensor,  # Batch map
    ) -> Tensor:
        # Compute residual: r = b - Ax_pred
        Ax_pred = torch.sparse.mm(matrix, preds.unsqueeze(-1)).squeeze(-1)
        residual = target_b - Ax_pred

        # Compute L2 residual norm
        res_l2 = torch.norm(residual, p=2) / (torch.norm(target_b, p=2) + self.eps)

        return res_l2.mean()

    
class CosineDistanceLoss(torch.nn.Module):
    def forward(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> Tensor:
        eps = 1e-8
        preds_norm = scatter_sum(preds.square(), batch_map).sqrt_() + eps
        target_norm = scatter_sum(target.square(), batch_map).sqrt_() + eps
        dot_product = scatter_sum(preds * target, batch_map)
        cosine = torch.clamp(dot_product / (preds_norm * target_norm), -1.0, 1.0)
        cosine_distance = 1 - cosine

        # Fallback: Assign high loss for problematic batches
        if torch.isnan(cosine_distance).any() or torch.isinf(cosine_distance).any():
            print("Problematic batch detected in CosineDistanceLoss")
            return torch.tensor(1e4, requires_grad=True).to(preds.device)  # Assign a high penalty

        return cosine_distance.mean()



class L1DistanceLoss(torch.nn.Module):
    def forward(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> Tensor:
        absolute_difference = torch.abs(preds - target)
        l1_distance = scatter_sum(absolute_difference, batch_map)

        return l1_distance.mean()


class L2DistanceLoss(torch.nn.Module):
    def forward(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> Tensor:
        eps = 1e-8  # Ensure stability for sqrt operation
        squared_difference = torch.square(preds - target)
        l2_distance = scatter_sum(squared_difference, batch_map).sqrt_() + eps

        return l2_distance.mean()
