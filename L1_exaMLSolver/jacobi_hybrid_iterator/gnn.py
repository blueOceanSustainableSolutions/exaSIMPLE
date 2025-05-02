from typing import Optional
from torch import Tensor
from torch_sparse import spmm
import torch.nn as nn
from torch_geometric.nn import GraphNorm, GATConv, SAGEConv, EdgeConv, GINConv, GCNConv
from torch_geometric import utils
import torch.utils.checkpoint as checkpoint
from torch_geometric.nn import GATConv, GCNConv, GraphNorm
import torch.nn as nn
import torch
torch.set_float32_matmul_precision("high")

# OPTIMISE: JACOBI REFINEMENT!!!!
class GraphBlock(nn.Module):
    def __init__(self, width: int, num_gat_layers: int = 2):
        super().__init__()
        self.width = width
        self.num_gat_layers = num_gat_layers

        self.layer_norm = nn.LayerNorm(width)
        self.gat_convs = nn.ModuleList(
            [GATConv(width, width, edge_dim=1, add_self_loops=True) 
             for _ in range(num_gat_layers)]
        )
        self.activation = nn.GELU()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        b: Optional[Tensor] = None,  # True RHS vector (not batch indices)
    ) -> Tensor:
        # GAT layer processing
        xx = x
        for gat_layer in self.gat_convs:
            xx = gat_layer(xx, edge_index, edge_weight)
            xx = self.activation(xx)
        
        xx = x + xx  # Residual connection
        xx = self.layer_norm(xx)

        # Jacobi refinement
        if b is not None:
            # confirm (b should not be none)
            # print("b is not None - Jacobi Refinement triggered")
            # Compute diagonal of A (degree matrix)
            # edge_index[0] == row indices of the edges between the nodes
            # x.size(0) total number of nodes 
            A_diag = utils.degree(
                edge_index[0], x.size(0), dtype=x.dtype
            ) + 1e-10  # Avoid division by zero
            
            # Compute Ax = A @ xx using sparse matmul
            Ax = spmm(
                edge_index, # sparse structure of A
                edge_weight,   # value of nonzero entries in A
                m=x.size(0),  # Number of nodes
                n=x.size(0),  # Same as m (square matrix)
                matrix=xx     # current node feature (current solution x)
            )
            
            # Residual: r = b - Ax
            r = b.unsqueeze(-1) - Ax  
            
            # Jacobi step: x = x + D^{-1} @ r
            xx = xx + (r / A_diag.unsqueeze(-1))

        return xx

class GNNSolver(nn.Module):
    def __init__(self, n_features: int, depth: int, width: int, num_jacobi_iters: int = 3):
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        self.width = width
        self.num_jacobi_iters = num_jacobi_iters

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
        b: Tensor  # True RHS vector (e.g., shape [num_nodes])
    ) -> Tensor:
        x = self.node_embedder(x)
        for block in self.blocks:
            for _ in range(self.num_jacobi_iters):
                x = block(x, edge_index, edge_weight, b)  # Pass RHS `b` to blocks
        solution = self.regressor(x)
        return solution.squeeze(dim=-1)

