import argparse
import json
import pathlib
import shutil
import typing as typ

from lightning.fabric.loggers.logger import Logger
from lightning_utilities.core.rank_zero import rank_zero_only


class JsonLogger(Logger):
    """Lighnint logger as a jsonl file."""

    def __init__(self, log_dir: str, remove_existing: bool = False) -> None:
        self._log_dir = pathlib.Path(log_dir)
        if remove_existing and self.log_dir.exists():
            shutil.rmtree(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self) -> pathlib.Path:  # type: ignore
        """Return the log directory."""
        return self._log_dir

    def get_log_file(self, name: str) -> str:
        """Return the log file name."""
        return (self.log_dir / name).as_posix()

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: None | int = None) -> None:
        """Records metrics. This method logs metrics as soon as it received them."""
        if step is not None:
            metrics["step"] = step
        with open(self.get_log_file("metrics.jsonl"), "a") as f:
            json.dump(metrics, f)
            f.write("\n")

    @rank_zero_only
    def log_hyperparams(
        self, params: dict[str, typ.Any] | argparse.Namespace, *args: typ.Any, **kwargs: typ.Any
    ) -> None:
        """Record hyperparameters."""
        with open(self.get_log_file("params.json"), "w") as f:
            json.dump(params, f)

    @property
    def name(self) -> None | str:
        """Return the experiment name."""
        pass

    @property
    def version(self) -> None | int | str:
        """Return the experiment version."""
        pass
