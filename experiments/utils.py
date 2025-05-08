from collections import defaultdict
import typing

import datasets
from loguru import logger
import rich
from rich.progress import track

from trie.base import Trie
from intervaltree import IntervalTree
from rouge_score import rouge_scorer

from trie.icd import ICD10Trie


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

    unique_codes: set[str] = set()
    for codes in dataset["targets"]:
        unique_codes.update(codes)
    dataset = dataset.map(
        lambda row: {
            **row,
            "targets": [code for code in row["targets"] if code in trie.lookup],
        }
    )

    filtered_codes: set[str] = set()
    for codes in dataset["targets"]:
        filtered_codes.update(codes)

    # print difference between unique_codes and filtered_codes
    filtered_codes = unique_codes - filtered_codes
    if filtered_codes:
        logger.warning(
            f"Number of filtered codes ({len(filtered_codes)}): `{filtered_codes}`"
        )
    if debug:
        return dataset.select(range(10))
    return dataset


def build_icd_trie(year: int = 2022) -> ICD10Trie:
    trie = ICD10Trie.from_cms(year=year)
    trie.parse()
    return trie


def analyse_agent_metrics(
    eval_data: list[dict[str, typing.Any]],
    xml_trie: Trie,
    ranks: list[int],
    strict: bool = False,
) -> dict[str, float]:
    """Evaluate retrieval metrics (recall and precision) at different rank cutoffs."""
    sum_recall = defaultdict(float)
    sum_precision = defaultdict(float)
    n_examples = len(eval_data)

    for row in track(eval_data, total=len(eval_data), description="Evaluating metrics"):
        row_dict = dict(row)
        term_ids = [t["id"] for t in row_dict["retrieved_terms"]]
        target_codes = set(row_dict["targets"])

        accumulated_codes = set()
        accumulated_lead_terms = set()
        tp_at_k = {}
        seen_at_k = {}
        terms_per_lead_at_k = {}

        # iterate over retrieved term IDs and update the true‑positive tally
        for i, term_id in enumerate(term_ids, start=1):
            accumulated_lead_terms.add(term_id.split(".")[0])
            if strict:
                term = xml_trie.index[term_id]
                if term.code:
                    accumulated_codes.add(term.code)
                if term.manifestation_code:
                    accumulated_codes.add(term.manifestation_code)
            else:
                accumulated_codes.update(xml_trie.get_all_term_codes(term_id))

            if i in ranks:
                tp_at_k[i] = len(accumulated_codes & target_codes)
                seen_at_k[i] = len(accumulated_codes)
                terms_per_lead_at_k[i] = i / len(accumulated_lead_terms)

        # fill missing ranks with the last computed value
        last_tp, last_seen = 0, 1
        for k in sorted(ranks):
            tp_here = tp_at_k.get(k, last_tp)
            seen_here = seen_at_k.get(k, last_seen)
            negatives_per_positives = seen_here - tp_here
            sum_recall[f"recall@{k}"] += tp_here / len(target_codes)
            sum_precision[f"precision@{k}"] += tp_here / seen_here
            sum_precision[f"neg_per_pos@{k}"] += negatives_per_positives
            sum_precision[f"terms_per_lead@{k}"] += terms_per_lead_at_k.get(k, 0.0)
            last_tp, last_seen = tp_here, seen_here

    # convert sums → means
    metrics_results = {
        **{k: v / n_examples for k, v in sum_recall.items()},
        **{k: v / n_examples for k, v in sum_precision.items()},
    }

    # print metrics
    for k in sorted(ranks):
        rich.print(f"\nRetrieval metrics at rank {k}:")
        rich.print(f"  Recall: {metrics_results[f'recall@{k}']:.4f}")
        rich.print(f"  Precision: {metrics_results[f'precision@{k}']:.4f}")
        rich.print(
            f"  Negatives per positives: {metrics_results[f'neg_per_pos@{k}']:.4f}"
        )
        rich.print(f"  Terms per lead: {metrics_results[f'terms_per_lead@{k}']:.4f}")

    return metrics_results
