from collections import defaultdict
from functools import partial
import pathlib
import typing

import datasets
from loguru import logger
import rich
from rich.progress import track

from trie.base import Trie
from intervaltree import IntervalTree
from rouge_score import rouge_scorer

from trie.icd import ICD10Trie
from throughster.factory import create_interface


def _build_ground_truth(evidence_spans: list[dict], note: str) -> IntervalTree:
    ground_truth = IntervalTree()
    for d in evidence_spans:
        code = d["code"]
        for start, end in d["locations"]:
            span = note[start : end + 1]
            ground_truth[start:end] = (code, span)
    return ground_truth


def _extract_predictions(
    predictions_data: list[dict], note: str, ground_truth: IntervalTree
) -> tuple[list[str], list[str], defaultdict[str, list[str]]]:
    predictions, references = [], []
    lead_terms: defaultdict[str, list[str]] = defaultdict(list)

    for pred in predictions_data:
        for span in pred.get("spans", []):
            start = note.find(span)
            if start == -1:
                continue

            end = start + len(span)
            overlaps = ground_truth[start:end]
            for overlap in overlaps:
                _, evidence = overlap.data
                references.append(evidence)
                predictions.append(span)

        lead_term = pred.get("lead_term")
        if lead_term:
            lead_terms[lead_term].extend(pred.get("modifiers", []))

    return predictions, references, lead_terms


def _get_predicted_codes(
    lead_terms: dict[str, list[str]], trie: Trie, n_terms: int
) -> set[str]:
    index_terms = trie.find_terms(
        list(lead_terms.keys()), main_terms=True, limit=n_terms
    )
    unique_codes = set()
    for term in index_terms:
        unique_codes.update(trie.get_all_term_codes(term.id))
    return unique_codes


def _compute_rouge(predictions: list[str], references: list[str]) -> float:
    if not predictions or not references:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    f1_scores = []

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        f1_scores.append(score["rougeL"].fmeasure)

    return sum(f1_scores) / len(f1_scores)


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
    **kwargs,
) -> typing.Callable:
    return partial(
        create_interface,
        provider=provider,
        api_base=api_base,
        endpoint=endpoint,
        model_name=deployment,
        use_cache=use_cache,
        cache_dir=str(pathlib.Path(f"~/.cache/throughster/{deployment}").expanduser()),
    )
