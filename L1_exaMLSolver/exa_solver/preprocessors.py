from abc import ABCMeta, abstractmethod

from torch import Tensor
import torch
import torch.nn.functional as F


class Preprocessor(torch.nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        raise NotImplementedError

    def forward(self, m: Tensor, b: Tensor, d: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.degree})"


class ArnoldiPreprocessor(Preprocessor):
    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        return self.degree

    def forward(self, m: Tensor, b: Tensor, d: Tensor) -> Tensor:
        features = []
        v = F.normalize(b, dim=0)
        for _ in range(self.degree):
            v = F.normalize(m.mv(v), p=torch.inf, dim=0)
            features.append(v)
        features = torch.stack(features, dim=-1)
        return features


class JacobiPreprocessor(Preprocessor):
    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        return self.degree + 1

    def forward(self, m: Tensor, b: Tensor, d: Tensor) -> Tensor:
        bias = torch.unsqueeze(b / d, dim=1)
        features = [bias]

        indices = m._indices()
        h_matrix = torch.sparse_coo_tensor(indices, m._values() / d[indices[0]])
        diagonal_mask = indices[0] == indices[1]
        h_matrix._values().masked_fill_(diagonal_mask, 0.0)

        v = bias
        for _ in range(self.degree):
            v = torch.sparse.addmm(bias, h_matrix, v)
            features.append(v)
        features = torch.cat(features, dim=-1)
        features = F.normalize(features, p=torch.inf, dim=0)
        return features


class ConjugateGradientPreprocessor(Preprocessor):
    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        return self.degree

    def forward(self, m: Tensor, b: Tensor, d: Tensor) -> Tensor:
        v = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        r_squared_norm = r.square().sum()
        features = []
        for _ in range(self.degree):
            eps = 1e-8
            Ap = m.mv(p)
            alpha = r_squared_norm / torch.clamp_min((p * Ap).sum(), eps)

            v = v + alpha * p
            r = r - alpha * Ap
            r1_squared_norm = r.square().sum()
            beta = r1_squared_norm / r_squared_norm

            p = r + beta * p
            r_squared_norm = r1_squared_norm
            features.append(v)
        features = torch.stack(features, dim=-1)
        features = F.normalize(features, p=torch.inf, dim=0)
        return features
