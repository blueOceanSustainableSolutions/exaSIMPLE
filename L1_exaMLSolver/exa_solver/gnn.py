from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn import GraphNorm, GATConv, SAGEConv, EdgeConv, GINConv, GCNConv
import torch.utils.checkpoint as checkpoint
from torch_geometric.nn import GATConv, GCNConv, GraphNorm
import torch.nn as nn
import torch
torch.set_float32_matmul_precision("high")


class GraphBlock(nn.Module):
    def __init__(self, width: int, num_gat_layers: int = 2):
        """
        Parameters:
        - width: Number of hidden units.
        - num_gat_layers: Number of sequential GATConv layers.
        """
        super().__init__()
        self.width = width
        self.num_gat_layers = num_gat_layers

        # Normalization Layers
        self.layer_norm = nn.LayerNorm(width)

        # GATConv Layers
        self.gat_convs = nn.ModuleList(
            [GATConv(width, width, edge_dim=1, add_self_loops=True) for _ in range(num_gat_layers)]
        )

        # Activation Function
        self.activation = nn.GELU()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        batch_map: Optional[Tensor] = None,
    ) -> Tensor:

        # Pass through multiple GATConv layers
        xx = x
        for gat_layer in self.gat_convs:
            xx = gat_layer(xx, edge_index, edge_weight)
            xx = self.activation(xx)

        xx = x + xx  # Residual connection
        xx = self.layer_norm(xx)  # Apply LayerNorm AFTER residual
        return xx


class GNNSolver(nn.Module):
    def __init__(
        self,
        n_features: int,
        depth: int,
        width: int,
    ):
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        self.width = width

        self.node_embedder = nn.Linear(n_features, width)
        self.blocks = nn.ModuleList([GraphBlock(width) for _ in range(depth)])
        self.regressor = nn.Sequential(
            nn.Linear(width, width),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch_map: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.node_embedder(x)
        for block in self.blocks:
            x = block(x, edge_index, edge_weight, batch_map)
        solution = self.regressor(x)
        return solution.squeeze(dim=-1)