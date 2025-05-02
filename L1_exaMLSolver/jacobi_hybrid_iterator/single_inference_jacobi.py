from typing import Tuple
import torch
from torch import Tensor

class SingleInference(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessors: Tuple[torch.nn.Module, ...] = tuple(),
    ):
        super().__init__()
        self.model = model
        self.preprocessors = preprocessors

    def forward(self, b: Tensor, m_indices: Tensor, m_values: Tensor, x: Tensor = None) -> Tensor:
        """Performs a single inference pass with iterative Jacobi updates replacing previous augmentations."""

        m = torch.sparse_coo_tensor(m_indices, m_values, dtype=torch.float32)

        # Extract diagonal elements
        diagonal_mask = m_indices[0] == m_indices[1]
        diagonal_values = m_values[diagonal_mask].to(dtype=torch.float32)
        diagonal = torch.zeros_like(b, dtype=torch.float32)
        diagonal[m_indices[0][diagonal_mask]] = diagonal_values

        if x is None:
            x = torch.zeros_like(b, dtype=torch.float32)
        x_features = torch.cat([x.unsqueeze(-1), b.to(torch.float32).unsqueeze(-1), diagonal.unsqueeze(-1)], dim=-1)

        print(f"x_features[:, -1] sample values: {x_features[:, -1][:5]}")

        # Collect augmentations
        new_aug_features = []
        for preprocessor in self.preprocessors:
            aug_feature = preprocessor(m, b.to(dtype=torch.float32), diagonal)
            print(f"Preprocessor: {preprocessor.__class__.__name__} | Output Shape: {aug_feature.shape}")
            new_aug_features.append(aug_feature)
            
        # Ensure augmentation is properly stacked
        if new_aug_features:
            jacobi_features = torch.cat(new_aug_features, dim=-1)
            print(f"Final stacked augmentation shape: {jacobi_features.shape}")  # Debug
            assert jacobi_features.shape[1] == 21, f"Jacobi augmentation features should have 20 columns but got {jacobi_features.shape[1]}"
            x_features = torch.cat([x_features[:, :2], jacobi_features], dim=-1)  # Replace old features

        # Normalize input system
        b_max = torch.linalg.vector_norm(b, ord=torch.inf)
        m_max = torch.linalg.vector_norm(m_values, ord=torch.inf)
        b_max = torch.clamp_min(b_max, 1e-16)
        m_max = torch.clamp_min(m_max, 1e-16)

        x_features[:, 0] /= b_max
        x_features[:, 1] /= m_max
        scaled_m_values = m_values / m_max

        # Run inference model
        y_direction = self.model(x_features.to(torch.float32), m_indices, scaled_m_values.to(torch.float32))
        y_direction = y_direction.to(torch.float32)

        # Compute projection direction
        p_direction = torch.mv(m, y_direction)
        p_squared_norm = p_direction.square().sum()
        bp_dot_product = p_direction.dot(b.to(torch.float32))

        # Avoid zero division in scaling
        scaler = torch.clamp_min(bp_dot_product / torch.clamp_min(p_squared_norm, 1e-16), 1e-16)

        # Scale the prediction
        y_hat = y_direction * scaler
        return y_hat
