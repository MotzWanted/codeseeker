from collections import OrderedDict, defaultdict
from functools import partial
import hashlib
import json
import pathlib
import typing

import datasets
from loguru import logger
import pydantic
import rich
from rich.progress import track
import torch

from finetune.monitor import ClassAggregator, MeanAggregator, Monitor
from trie.base import Trie

from trie.icd import ICD10Trie
from throughster.factory import create_interface

from trie.models import Term


def get_detailed_instruct(
    task_description: str, query: str, prompt_template: str
) -> str:
    return prompt_template.format(task=task_description, query=query)


def format_query_prompt(row: dict, task: str, prompt_template: str) -> dict:
    """Format the query prompt for a given row."""
    return {"query_prompt": get_detailed_instruct(task, row["note"], prompt_template)}


def format_dataset(
    dataset: datasets.Dataset | datasets.DatasetDict,
    trie: Trie,
    debug: bool = False,
) -> datasets.Dataset:
    """Format the dataset."""

    if isinstance(dataset, datasets.DatasetDict):
        dataset = datasets.concatenate_datasets(list(dataset.values()))

    trie_lookup_set = set(trie.lookup.keys())  # Ensure fast lookup

    all_codes = set()
    filtered_out = set()

    def filter_targets(batch):
        nonlocal all_codes, filtered_out
        filtered_batch = []
        for codes in batch["targets"]:
            all_codes.update(codes)
            filtered = [code for code in codes if code in trie_lookup_set]
            filtered_batch.append(filtered)
            filtered_out.update(set(codes) - set(filtered))
        return {"targets": filtered_batch}

    dataset = dataset.map(filter_targets, batched=True)

    if filtered_out:
        logger.warning(
            f"Number of filtered codes ({len(filtered_out)}): `{filtered_out}`"
        )

    if debug:
        return dataset.select(range(10))
    return dataset


def build_icd_trie(year: int = 2022) -> ICD10Trie:
    trie = ICD10Trie.from_cms(year=year)
    trie.parse()
    return trie


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


def analyse_agent_metrics(
    eval_data: list[dict[str, typing.Any]],
    xml_trie,
    ranks: list[int],
    strict: bool = False,
) -> dict[str, float]:
    """Evaluate retrieval metrics (micro recall and precision) at different rank cutoffs."""
    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_seen = defaultdict(int)

    for row in track(eval_data, total=len(eval_data), description="Evaluating metrics"):
        row_dict = dict(row)
        target_codes = set(row_dict["targets"])

        accumulated_codes = set()
        tp_at_k = {}
        seen_at_k = {}

        for i, term_id in enumerate(row_dict["terms"], start=1):
            if strict:
                accumulated_codes.update(
                    xml_trie.get_term_codes(term_id, subterms=False)
                )
            else:
                accumulated_codes.update(
                    xml_trie.get_term_codes(term_id, subterms=True)
                )

            if i in ranks:
                tp_at_k[i] = len(accumulated_codes & target_codes)
                seen_at_k[i] = len(accumulated_codes)

        for k in sorted(ranks):
            tp = tp_at_k.get(k, 0)
            seen = seen_at_k.get(k, 0)
            fn = len(target_codes) - tp

            total_tp[k] += tp
            total_fp[k] += fn
            total_seen[k] += seen

    metrics_results = {}
    for k in sorted(ranks):
        recall = (
            total_tp[k] / (total_tp[k] + total_fp[k])
            if (total_tp[k] + total_fp[k]) > 0
            else 0.0
        )
        precision = total_tp[k] / total_seen[k] if total_seen[k] > 0 else 0.0
        metrics_results[f"recall@{k}"] = recall
        metrics_results[f"precision@{k}"] = precision

        rich.print(f"\nRetrieval metrics at rank {k}:")
        rich.print(f"  Micro Recall: {recall:.4f}")
        rich.print(f"  Micro Precision: {precision:.4f}")

    return metrics_results


def _init_client_fn(
    provider: str,
    api_base: str,
    endpoint: str,
    deployment: str,
    use_cache: bool,
    api_version: str | None = None,
    **kwargs,
) -> typing.Callable:
    return partial(
        create_interface,
        provider=provider,
        api_base=api_base,
        api_version=api_version,
        endpoint=endpoint,
        model_name=deployment,
        use_cache=use_cache,
        cache_dir=str(pathlib.Path(f"~/.cache/throughster/{deployment}").expanduser()),
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
        **kws: typing.Any,
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


def evaluate_and_dump_metrics(
    eval_data: datasets.Dataset,
    trie: OrderedDict[str, int],
    dump_path: pathlib.Path,
    file_prefix: str,
):
    """Compute, print, and optionally dump evaluation metrics."""
    overall_monitor = TrieClassificationMonitor(trie=trie)
    contextual_monitor = TrieClassificationMonitor(trie=trie)

    overall_monitor.update(
        target_inputs=eval_data["targets"], pred_inputs=eval_data["output"]
    )
    contextual_monitor.update(
        target_inputs=eval_data["subset_targets"], pred_inputs=eval_data["output"]
    )

    overall_metrics = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in overall_monitor.get().items()
    }
    contextual_metrics = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in contextual_monitor.get().items()
    }

    rich.print(f"[blue][{file_prefix}]Evaluation Metrics.[/blue]")
    rich.print(f"[blue][Overall] {overall_metrics}[/blue]")
    rich.print(f"[blue][Contextual] {contextual_metrics}[/blue]")

    for metrics in [overall_metrics, contextual_metrics]:
        with open(dump_path / f"{file_prefix}_metrics.json", "w") as f:
            json.dump(metrics, f)

    with open(dump_path / "responses.json", "w") as f:
        cols_to_remove = set(_get_dataset(eval_data).column_names) - set(
            ["aid", "note_type", "output", "targets", "response"]
        )
        dump_data = eval_data.remove_columns(list(cols_to_remove))
        json.dump(dump_data.to_list(), f)

    return metrics


def make_group_id(term: Term, cross_references: bool = False) -> str:
    if not term.code:
        return term.id if cross_references else ""
    key_tuple = (
        term.code,
        term.manifestation_code if term.manifestation_code else "",
        hashlib.md5(term.see.encode()).hexdigest() if term.see else "",
        hashlib.md5(term.see_also.encode()).hexdigest() if term.see_also else "",
    )
    key_str = "-".join(key_tuple)
    return key_str


class CandidateCode(pydantic.BaseModel):
    """Model for candidate codes."""

    id: str
    code: str
    description: str
    path: str
