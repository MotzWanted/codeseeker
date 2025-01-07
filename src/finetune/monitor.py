import abc
import math
import typing as typ

import torch
import torch.distributed as dist


class Aggregator(abc.ABC, torch.nn.Module):
    """Aggregates merics."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the metric stats."""

    @abc.abstractmethod
    def update(self, values: torch.Tensor) -> None:
        """Update the metrics stats."""

    @abc.abstractmethod
    def get(self) -> torch.Tensor:
        """Return the metric value averaged over all updates."""

    @abc.abstractmethod
    def all_reduce(self) -> None:
        """Synchronize all processes by summing stats."""


class MeanAggregator(Aggregator):
    """Computes the mean."""

    _total: torch.Tensor
    _count: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self._total = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        self._count = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        self.reset()

    def reset(self) -> None:
        """Reset the metric stats."""
        self._total.data.fill_(0.0)
        self._count.data.fill_(0.0)

    def update(self, values: torch.Tensor) -> None:
        """Update the metrics stats."""
        values = values.detach()
        values_no_nan = values[~torch.isnan(values)]
        if values_no_nan.numel() == 0:
            return
        self._total += values_no_nan.sum()
        self._count += values_no_nan.numel()

    def get(self) -> torch.Tensor:
        """Return the metric value averaged over all updates."""
        return (self._total / self._count).mean()

    def all_reduce(self) -> None:
        dist.all_reduce(self._total.data, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._count.data, op=dist.ReduceOp.SUM)


class MaxAggregator(Aggregator):
    """Computes the max."""

    _max: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self._max = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        self.reset()

    def reset(self) -> None:
        """Reset the metric stats."""
        self._max.data.fill_(-math.inf)

    def update(self, values: torch.Tensor) -> None:
        """Update the metrics stats."""
        values = values.detach()
        values_no_nan = values[~torch.isnan(values)]
        if values_no_nan.numel() == 0:
            return
        self._max.data = torch.maximum(self._max.data, torch.max(values_no_nan))

    def get(self) -> torch.Tensor:
        """Return the metric value averaged over all updates."""
        return self._max.data.clone()

    def all_reduce(self) -> None:
        """Synchronize all processes by summing stats."""
        dist.all_reduce(self._max.data, op=dist.ReduceOp.MAX)


class SumAggregator(Aggregator):
    """Computes the sum."""

    _sum: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self._sum = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        self.reset()

    def reset(self) -> None:
        """Reset the metric stats."""
        self._sum.data.fill_(0.0)

    def update(self, values: torch.Tensor) -> None:
        """Update the metrics stats."""
        values = values.detach()
        values_no_nan = values[~torch.isnan(values)]
        if values_no_nan.numel() == 0:
            return
        self._sum += values_no_nan.sum()

    def get(self) -> torch.Tensor:
        """Return the metric value averaged over all updates."""
        return self._sum.data.clone()

    def all_reduce(self) -> None:
        """Synchronize all processes by summing stats."""
        dist.all_reduce(self._sum.data, op=dist.ReduceOp.SUM)


class ClassAggregator(Aggregator):
    """Aggregates metrics for individual classes."""

    _sum: torch.Tensor

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.register_buffer("_sum", torch.zeros(num_classes))

    def reset(self) -> None:
        """Reset the metric stats for all classes."""
        self._sum.zero_()

    def update(self, values: torch.Tensor) -> None:
        """Update the metrics stats for each class."""
        values = values.detach()
        values_no_nan = values[~torch.isnan(values)]
        if values_no_nan.numel() == 0:
            return
        if self._sum.device != values.device:
            values = values.to(self._sum.device)
        self._sum += values

    def get(self) -> torch.Tensor:
        """Return the metric value averaged over all updates for each class."""
        return self._sum.data.clone()

    def all_reduce(self) -> None:
        """Synchronize all processes by summing stats."""
        if dist.is_initialized():
            dist.all_reduce(self._sum, op=dist.ReduceOp.SUM)


class Monitor(abc.ABC, torch.nn.Module):
    """Monitor retrieval performances."""

    aggregators: torch.nn.ModuleDict

    @abc.abstractmethod
    def update(self, **kws: typ.Any) -> None:
        """Compute metrics and update the aggregators."""
        ...

    def synchronize(self) -> None:
        """Synchronize aggregators between process."""
        if dist.is_initialized():
            dist.barrier()
            for agg in self.aggregators.values():
                agg.all_reduce()

    def reset(self) -> None:
        """Reset aggregators."""
        for agg in self.aggregators.values():
            agg.reset()

    def get(self) -> dict[str, torch.Tensor]:
        """Get values from all aggregators."""
        return {name: agg.get() for name, agg in self.aggregators.items()}

    def compute(self, synchronize: bool = True) -> dict[str, torch.Tensor]:
        """Sync, get values and reset."""
        if synchronize:
            self.synchronize()
        outputs = self.get()
        self.reset()
        return outputs


class MeanMonitor(Monitor):
    """Tracks the average of values."""

    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self.keys = keys
        self.aggregators = torch.nn.ModuleDict(
            {
                **{k: MeanAggregator() for k in keys},
            }
        )  # type: ignore

    def update(self, **kws: typ.Any) -> None:
        """Compute metrics and update the aggregators."""
        for key in self.keys:
            self.aggregators[key].update(kws[key])
