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

    def forward(self, b: Tensor, m_indices: Tensor, m_values: Tensor) -> Tensor:
        # Create the sparse matrix with consistent dtype
        m = torch.sparse_coo_tensor(m_indices, m_values, dtype=torch.float32)

        # Get the diagonal of the sparse matrix
        diagonal_mask = m_indices[0] == m_indices[1]
        diagonal_values = m_values[diagonal_mask].to(dtype=torch.float32)
        diagonal = torch.zeros_like(b, dtype=torch.float32)
        diagonal[m_indices[0][diagonal_mask]] = diagonal_values

        # Preprocess the input features
        x = torch.stack([b.to(torch.float32), diagonal], dim=-1)
        features = [x]
        for preprocessor in self.preprocessors:
            features.append(preprocessor(m, b.to(dtype=torch.float32), diagonal))
        x = torch.cat(features, dim=-1)

        # Rescale input system
        b_max = torch.linalg.vector_norm(b, ord=torch.inf)
        m_max = torch.linalg.vector_norm(m_values, ord=torch.inf)
        
        # Avoid division by zero by clamping to a minimum value
        b_max = torch.clamp_min(b_max, 1e-16)
        m_max = torch.clamp_min(m_max, 1e-16)

        x[:, 0] /= b_max
        x[:, 1] /= m_max
        scaled_m_values = m_values / m_max

        # Run the model
        y_direction = self.model(
            x.to(torch.float32), m_indices, scaled_m_values.to(torch.float32)
        )
        
        # Ensure output dtype consistency
        y_direction = y_direction.to(torch.float32)

        # Compute p_direction and scaling factor
        p_direction = torch.mv(m, y_direction)
        p_squared_norm = p_direction.square().sum()
        bp_dot_product = p_direction.dot(b.to(torch.float32))
        
        # Avoid division by zero in the scaler
        scaler = torch.clamp_min(bp_dot_product / torch.clamp_min(p_squared_norm, 1e-16), 1e-16)

        # Scale the predicted direction
        y_hat = y_direction * scaler
        return y_hat
