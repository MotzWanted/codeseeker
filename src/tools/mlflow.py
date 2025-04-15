import mlflow
import typing as typ
from mlflow.metrics import MetricValue, make_metric
import pandas as pd

from agents.metrics import AlignmentMetrics


def predict(
    client: mlflow.tracking.MlflowClient,
    corpus: list[str],
    queries: list[str],
    predictions: list[list[int]],
    seed: int,
    num_shots: int,
    prompt_name: str,
) -> dict:
    """Predict the alignment."""
    if seed != 1:
        return {"predictions": predictions}
    root_span = client.start_trace(
        name="Generate Alignment",
        inputs={"corpus": corpus, "queries": queries},
        tags={"num_shots": num_shots, "prompt_name": prompt_name},
    )
    request_id = root_span.request_id
    for i, (query, prediction) in enumerate(zip(queries, predictions), 1):
        child_span = client.start_span(
            name=f"Predicting Alignment ({i}/{len(queries)})",
            # Specify the request ID to which the child span belongs.
            request_id=request_id,
            # Also specify the ID of the parent span to build the span hierarchy.
            # You can access the span ID via `span_id` property of the span object.
            parent_id=root_span.span_id,
            # Each span has its own inputs.
            inputs={"corpus": corpus, "query": query},
        )
        client.end_span(
            request_id=request_id,
            span_id=child_span.span_id,
            # Set the output(s) of the span.
            outputs=prediction,
            # Set the completion status, such as "OK" (default), "ERROR", etc.
            status="OK",
        )
    client.end_trace(
        request_id=request_id,
        # Set the output(s) of the span.
        outputs={"predictions": predictions},
    )
    return {"predictions": predictions}


def model_fn(*args: tuple, client: mlflow.tracking.MlflowClient, seed: int, num_shots: int, prompt_name: str) -> dict:
    df = typ.cast(pd.DataFrame, args[0])
    for i, row in df.iterrows():
        _ = predict(client, row["corpus"], row["queries"], row["predictions"], seed, num_shots, prompt_name)
    return pd.Series(df["predictions"], index=df.index)


def f1_score(predictions, targets) -> MetricValue:
    """Evaluate the alignment metrics using F1 score."""
    precision_scores = [AlignmentMetrics.recall_precision(t, p)["precision"] for t, p in zip(targets, predictions)]
    recall_scores = [AlignmentMetrics.recall_precision(t, p)["recall"] for t, p in zip(targets, predictions)]

    f1_scores = [
        2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0 for prec, rec in zip(precision_scores, recall_scores)
    ]

    return MetricValue(scores=f1_scores, aggregate_results={"mean": sum(f1_scores) / len(f1_scores)})


def recall(predictions, targets) -> MetricValue:
    """Evaluate the alignment metrics."""
    scores = [AlignmentMetrics.recall_precision(t, p)["recall"] for t, p in zip(targets, predictions)]
    return MetricValue(scores=scores, aggregate_results={"mean": sum(scores) / len(scores)})


def precision(predictions, targets) -> MetricValue:
    """Evaluate the alignment metrics."""
    scores = [AlignmentMetrics.recall_precision(t, p)["precision"] for t, p in zip(targets, predictions)]
    return MetricValue(scores=scores, aggregate_results={"mean": sum(scores) / len(scores)})


def exact_match(predictions, targets) -> MetricValue:
    """Evaluate the alignment metrics."""
    scores = [AlignmentMetrics.exact_match(t, p)["exact_match"] for t, p in zip(targets, predictions)]
    return MetricValue(scores=scores, aggregate_results={"mean": sum(scores) / len(scores)})


def positive_ratio(predictions, targets) -> MetricValue:
    """Evaluate the alignment metrics."""
    scores = [AlignmentMetrics.positive_ratio(t, p)["positive_ratio"] for t, p in zip(targets, predictions)]
    return MetricValue(scores=scores, aggregate_results={"mean": sum(scores) / len(scores)})


def negative_ratio(predictions, targets) -> MetricValue:
    """Evaluate the alignment metrics."""
    scores = [AlignmentMetrics.negative_ratio(t, p)["negative_ratio"] for t, p in zip(targets, predictions)]
    return MetricValue(scores=scores, aggregate_results={"mean": sum(scores) / len(scores)})


DEFAULT_METRICS = [
    make_metric(
        eval_fn=recall,
        greater_is_better=True,
    ),
    make_metric(
        eval_fn=precision,
        greater_is_better=True,
    ),
    make_metric(
        eval_fn=f1_score,
        greater_is_better=True,
    ),
    make_metric(
        eval_fn=exact_match,
        greater_is_better=True,
    ),
    make_metric(
        eval_fn=positive_ratio,
        greater_is_better=False,
    ),
    make_metric(
        eval_fn=negative_ratio,
        greater_is_better=False,
    ),
]
