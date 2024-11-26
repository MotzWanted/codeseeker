import os
import typing as typ

import rich
import torch
import transformers

from tools.pprint import pprint_batch


class Callback:
    """Base class for callbacks."""

    def on_fit_start(self, *, step: int, model: transformers.PreTrainedModel) -> None:
        """Called at the beginning of the training loop."""

    def on_fit_end(self, *, step: int, status: str, model: transformers.PreTrainedModel) -> None:
        """Called at the end of the training loop."""

    def on_epoch_start(self, *, step: int) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, *, step: int) -> None:
        """Called at the end of each epoch."""

    def on_train_batch_start(self, *, batch: dict[str, torch.Tensor], step: int) -> None:
        """Called at the beginning of each training batch."""

    def on_train_batch_end(self, *, batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor], step: int) -> None:
        """Called at the end of each training batch."""

    def on_validation_start(self, *, step: int) -> None:
        """Called at the beginning of the validation loop."""

    def on_validation_end(self, *, step: int, metrics: dict[str, typ.Any], model: transformers.PreTrainedModel) -> None:
        """Called at the end of the validation loop."""

    def on_validation_batch_start(self, *, batch: dict[str, torch.Tensor], step: int) -> None:
        """Called at the beginning of a validation step."""

    def on_validation_batch_end(
        self, batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor], step: int
    ) -> None:
        """Called at the end of a validation step."""


class PrintCallback(Callback):
    """A callback that prints the batches and outputs."""

    def on_fit_start(self, *, step: int, model: transformers.PreTrainedModel) -> None:
        """Called at the beginning of the training loop."""
        rich.print(f"[bold yellow]on_fit_start[/bold yellow] (step={step}, rank={_get_node_rank()})")

    def on_fit_end(self, *, step: int, status: str, model: transformers.PreTrainedModel) -> None:
        """Called at the end of the training loop."""
        rich.print(f"[bold yellow]on_fit_end[/bold yellow] (step={step}, status={status}, rank={_get_node_rank()})")

    def on_epoch_start(self, *, step: int) -> None:
        """Called at the beginning of each epoch."""
        rich.print(f"[bold yellow]on_epoch_start[/bold yellow] (step={step}, rank={_get_node_rank()})")

    def on_epoch_end(self, *, step: int) -> None:
        """Called at the end of each epoch."""
        rich.print(f"[bold yellow]on_epoch_end[/bold yellow] (step={step}, rank={_get_node_rank()})")

    def on_train_batch_start(self, *, batch: dict[str, torch.Tensor], step: int) -> None:
        """Called at the beginning of each training batch."""
        pprint_batch(batch, header=f"on_train_batch_start (step={step}, rank={_get_node_rank()}) - batch")

    def on_train_batch_end(self, *, batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor], step: int) -> None:
        """Called at the end of each training batch."""
        pprint_batch(batch, header=f"on_train_batch_start (step={step}, rank={_get_node_rank()}) - batch")
        pprint_batch(output, header=f"on_train_batch_start (step={step}, rank={_get_node_rank()}) - model output")

    def on_validation_start(self, *, step: int) -> None:
        """Called at the beginning of the validation loop."""
        rich.print(f"[bold yellow]on_validation_start[/bold yellow] (step={step}, rank={_get_node_rank()})")

    def on_validation_end(self, *, step: int, metrics: dict[str, typ.Any], model: transformers.PreTrainedModel) -> None:
        """Called at the end of the validation loop."""
        rich.print(f"[bold yellow]on_validation_end[/bold yellow] (step={step}, rank={_get_node_rank()})")
        rich.print(metrics)

    def on_validation_batch_start(self, *, batch: dict[str, torch.Tensor], step: int) -> None:
        """Called at the beginning of the validation loop."""
        pprint_batch(batch, header=f"on_validation_batch_start (step={step}, rank={_get_node_rank()}) - batch")

    def on_validation_batch_end(
        self, batch: dict[str, torch.Tensor], output: dict[str, torch.Tensor], step: int
    ) -> None:
        """Called at the end of the validation loop."""
        pprint_batch(batch, header=f"on_validation_batch_end (step={step}, rank={_get_node_rank()}) - batch")
        pprint_batch(output, header=f"on_validation_batch_end (step={step}, rank={_get_node_rank()}) - model output")


def _get_node_rank() -> str:
    """Get the node rank."""
    node_rank = os.environ.get("NODE_RANK", "0")
    local_rank = os.environ.get("LOCAL_RANK", "0")
    return f"{local_rank}:{node_rank}"
