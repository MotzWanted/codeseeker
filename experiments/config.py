from collections import OrderedDict
import hashlib
import pathlib
import typing as typ
import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

from finetune.monitor import ClassAggregator, MeanAggregator, Monitor

DUMP_FOLDER = pathlib.Path("~/research/codeseeker/experiments").expanduser()


class BaseArguments(BaseSettings):
    """Args for the script."""

    experiment_id: str
    experiment_name: str

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    api_base: str
    deployment: str
    pretrained_model_path: str | None = None
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"
    use_cache: bool = True  # whether to cache on request level

    dataset: str
    seed: str

    prompt_name: str

    num_workers: int
    batch_size: int

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)

    @pydantic.computed_field
    def _deployment_name(self) -> str:
        """Get the model name."""
        return self.deployment.split("/")[-1]

    @pydantic.computed_field
    def _hash(self) -> str:
        """Create unique identifier for the arguments"""
        model_dict = self.model_dump(
            exclude={
                "_hash",
                "_deployment_name",
                "experiment_folder",
                "experiment_name",
                "seed",
            }
        )
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    @pydantic.computed_field
    def experiment_folder(self) -> str:
        """Get the experiment name."""
        return f"{self.experiment_id}/{self.experiment_name}/{self._deployment_name}"

    @pydantic.computed_field
    def _seeds(self) -> list[int]:
        """Get the seeds."""
        return (
            [int(x) for x in self.seed.split(":")]
            if ":" in self.seed
            else [int(self.seed)]
        )

    @pydantic.computed_field
    def _datasets(self) -> list[str]:
        """Get the datasets."""
        return (
            [x for x in self.dataset.split(":")]
            if ":" in self.dataset
            else [self.dataset]
        )

    @pydantic.computed_field
    def _prompts(self) -> list[str]:
        """Get the prompts."""
        return (
            [x for x in self.prompt_name.split(":")]
            if ":" in self.prompt_name
            else [self.prompt_name]
        )


def list2tensor_vectorized(
    dim_x: int, dim_y: int, indices: list[set[int | float]]
) -> torch.Tensor:
    """Convert a list of indices to a sparse tensor."""
    row_indices = []
    col_indices = []
    values = []

    for i, preds in enumerate(indices):
        preds = torch.tensor(
            list(preds), dtype=torch.float32
        )  # Convert the set to a PyTorch tensor
        pred_signs = torch.where(preds < 0, -1, 1)  # Determine the sign
        pred_indices = torch.abs(preds) - 1  # Get absolute indices (0-based)

        # Filter valid indices (within bounds)
        valid_mask = (pred_indices >= 0) & (pred_indices < dim_y)
        valid_count = int(valid_mask.sum().item())  # Explicitly convert to Python int
        row_indices.extend([i] * valid_count)  # Repeat row index for valid preds
        col_indices.extend(
            pred_indices[valid_mask].to(torch.int).tolist()
        )  # Valid column indices
        values.extend(pred_signs[valid_mask].tolist())  # Valid signs

    # Convert row_indices, col_indices, and values to PyTorch tensors
    row_indices = torch.tensor(row_indices, dtype=torch.long)
    col_indices = torch.tensor(col_indices, dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float32)

    # Create the sparse tensor
    sparse_tensor = torch.zeros((dim_x, dim_y), dtype=torch.float32)
    sparse_tensor[row_indices, col_indices] = values

    return sparse_tensor


class TrieClassificationMonitor(Monitor):
    """Monitor for classification tasks."""

    def __init__(
        self,
        trie: OrderedDict[str, int],
        keys: list[str] = [],
    ) -> None:
        super().__init__()
        self.keys = keys
        self.num_classes = len(trie)
        self.trie = trie
        self.class_aggregators = torch.nn.ModuleDict(
            {
                **{
                    k: ClassAggregator(self.num_classes)
                    for k in ["tp", "fp", "fn", "tn"]
                },
            }
        )
        self.aggregators = torch.nn.ModuleDict(
            {
                **{
                    k: ClassAggregator(self.num_classes)
                    for k in ["tp", "fp", "fn", "tn"]
                },
                **{k: MeanAggregator() for k in [*self.keys, "_hit", "pos_ratio"]},
            }
        )

    def get(self) -> dict[str, torch.Tensor]:
        """Get values from all aggregators."""
        output = {key: self.aggregators[key].get() for key in self.keys}
        tp = self.aggregators["tp"].get()
        fp = self.aggregators["fp"].get()
        fn = self.aggregators["fn"].get()
        tn = self.aggregators["tn"].get()
        # Micro F1
        micro_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
        micro_recall = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
        output["f1_micro"] = (
            2
            * (micro_precision * micro_recall)
            / (micro_precision + micro_recall + 1e-10)
        )

        # Macro F1 (ignoring TN-only classes)
        precision_per_class = tp / (tp + fp + 1e-10)
        recall_per_class = tp / (tp + fn + 1e-10)
        f1_per_class = (
            2
            * (precision_per_class * recall_per_class)
            / (precision_per_class + recall_per_class + 1e-10)
        )

        # Exclude classes where TP + FP + FN = 0 (TN-only classes)
        valid_classes = (tp + fp + fn) > 0
        if valid_classes.any():  # Ensure there are valid classes
            macro_f1 = f1_per_class[valid_classes].mean()
        else:
            macro_f1 = torch.tensor(
                0.0
            )  # Handle edge case where no valid classes exist

        output["f1_macro"] = macro_f1

        # Classification Metrics
        output["recall"] = micro_recall
        output["precision"] = micro_precision
        output["specificity"] = tn.sum() / (tn.sum() + fp.sum() + 1e-10)
        output["accuracy"] = self.aggregators["_hit"].get()
        output["prediction-bias-ratio"] = self.aggregators["pos_ratio"].get()
        # output["table"] = self._make_table_date(f1_per_class, tp, fp, fn, self.aggregators["tn"].get())
        return output

    def update(
        self,
        *,
        target_inputs: list[list[str]],
        pred_inputs: list[list[str]],
        **kws: typ.Any,
    ) -> None:
        """Update the metrics."""
        for key in self.keys:
            self.aggregators[key].update(kws[key])

        targets = [set(t) for t in target_inputs]
        predictions = [set(p) for p in pred_inputs]

        prediction_ids = []
        target_ids = []
        for targets, predictions in zip(targets, predictions):
            prediction_ids.append(
                [
                    self.trie[code_name]
                    for code_name in predictions
                    if code_name in self.trie
                ]
            )
            target_ids.append([self.trie[code_name] for code_name in targets])

        prediction_matrix = list2tensor_vectorized(
            len(prediction_ids), self.num_classes, prediction_ids
        )
        target_matrix = list2tensor_vectorized(
            len(target_ids), self.num_classes, target_ids
        )

        # Compute the true/false negatives/positives
        conf_matrix = self._make_conf_matrix(target_matrix, prediction_matrix)
        for k, v in conf_matrix.items():
            self.aggregators[k].update(v)

    @staticmethod
    def _make_conf_matrix(
        targets: torch.Tensor, preds: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute the true/false positives."""
        _targets = targets.bool()
        _preds = preds.bool()

        return {
            "tp": (_targets & _preds).sum(dim=0),
            "fp": (_preds & ~_targets).sum(dim=0),
            "fn": (_targets & ~_preds).sum(dim=0),
            "tn": (~_targets & ~_preds).sum(dim=0),
            "_hit": (~(_targets ^ _preds))
            .all(dim=1)
            .float(),  # counting row wise exact matches
            "pos_ratio": preds.sum() / targets.sum(),
        }
