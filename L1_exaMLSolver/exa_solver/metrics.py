import torch
from torch import Tensor
from torchmetrics import Metric
from torch_scatter import scatter_sum



class L1Distance(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "distance",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        self.distance += torch.abs(preds - target).sum()
        self.total += batch_map.max() + 1

    def compute(self) -> Tensor:
        return self.distance / self.total


class L2Distance(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "distance",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        squared_difference = torch.square(preds - target)
        distance = torch.sqrt(scatter_sum(squared_difference, batch_map))
        self.distance += distance.sum()
        self.total += distance.size(0)

    def compute(self) -> Tensor:
        return self.distance / self.total


class L1Ratio(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "ratio", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        target_norm = scatter_sum(torch.abs(target), batch_map)
        distance = scatter_sum(torch.abs(preds - target), batch_map)
        ratio = distance / target_norm
        self.ratio += ratio.sum()
        self.total += ratio.size(0)

    def compute(self) -> Tensor:
        return self.ratio / self.total


class L2Ratio(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "ratio", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        target_norm = scatter_sum(torch.square(target), batch_map)
        distance = scatter_sum(torch.square(preds - target), batch_map)
        ratio = torch.sqrt(distance / target_norm)
        self.ratio += ratio.sum()
        self.total += ratio.size(0)

    def compute(self) -> Tensor:
        return self.ratio / self.total


class RMSE(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_squared_error", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        squared_error = torch.square(preds - target)
        self.sum_squared_error += torch.sum(squared_error)
        self.total += batch_map.max() + 1

    def compute(self) -> Tensor:
        return torch.sqrt(self.sum_squared_error / self.total)



class VectorAngle(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state(
            "angle",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        eps = 1e-12
        preds_norm = scatter_sum(preds.square(), batch_map).sqrt_()
        target_norm = scatter_sum(target.square(), batch_map).sqrt_()
        dot_product = scatter_sum(preds * target, batch_map)
        cosine = dot_product / torch.clamp_min(preds_norm * target_norm, eps)
        self.angle += torch.arccos(cosine).sum()
        self.total += cosine.size(0)

    def compute(self) -> Tensor:
        return self.angle / self.total
