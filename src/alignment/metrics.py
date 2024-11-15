from collections import defaultdict
import typing as typ

DEFAULT_METRICS = ["recall", "precision", "exact_match", "positive_ratio", "negative_ratio"]
AvailableMetrics = typ.Literal["recall", "precision", "exact_match", "positive_ratio", "negative_ratio"]


class AlignmentMetrics:
    def __init__(
        self,
        labels_key: str,
        predictions_key: str,
        metrics: list[AvailableMetrics] = DEFAULT_METRICS,  # type: ignore
    ) -> None:
        self.labels_key = labels_key
        self.predictions_key = predictions_key
        self.metrics = list(metrics)

    def __call__(self, batch: dict[str, list[typ.Any]], idx: list[int] | None = None) -> dict[str, list[typ.Any]]:
        """Compute the metrics."""
        self._validate_input(batch)

        labels = batch[self.labels_key]
        predictions = batch[self.predictions_key]

        results = self.generate_metrics(labels, predictions)
        outputs = defaultdict(list)
        for metric, value in results.items():
            outputs[metric].append(value)

        return {**outputs}

    def generate_metrics(self, labels: list[list[int]], predictions: list[list[int]]) -> dict[AvailableMetrics, float]:
        """Generate the metrics."""
        results = {}
        for metric in self.metrics:
            if metric == "recall" or metric == "precision":
                recall_precision_results = self.recall_precision(labels, predictions)
                results.update(recall_precision_results)
            else:
                results[metric] = getattr(self, metric)(labels, predictions)[metric]
        return results

    @staticmethod
    def recall_precision(labels: list[list[int]], predictions: list[list[int]]) -> dict[AvailableMetrics, float]:
        """Compute the recall and precision, ignoring [0] lists for true positives and false positives."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        for y, y_hat in zip(labels, predictions):
            if y == y_hat == [0]:
                true_negatives += 1
            elif y != [0] and y_hat == [0]:
                false_negatives += 1
            elif y == y_hat != [0]:
                true_positives += len(y)
            else:
                true_positives += len(set(y) & set(y_hat))
                false_positives += len(set(y_hat) - set(y))

        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
        return {
            "recall": round(recall, 2),
            "precision": round(precision, 2),
        }

    @staticmethod
    def exact_match(labels: list[list[int]], predictions: list[list[int]]) -> dict[AvailableMetrics, float]:
        """Compute the exact match."""
        exact_matches = sum(1 for y, y_hat in zip(labels, predictions) if set(y) == set(y_hat))
        return {"exact_match": round(exact_matches / len(labels), 2)}

    @staticmethod
    def negative_ratio(labels: list[list[int]], predictions: list[list[int]]) -> dict[AvailableMetrics, float]:
        """Compute the specific true negative ratio, meaning when both y and y_hat are [0].
        Thus, a ratio of 1.0 means that predictions contain the same amount of [0] elements as the labels.
        A ratio of 0.5 means that predictions contain half the amount of [0] elements as the labels.
        A ratio of 2.0 means that predictions contain twice the amount of [0] elements as the"""
        specific_true_negatives = sum(1 for y, y_hat in zip(labels, predictions) if y == [0] and y_hat == [0])
        relevant_cases = sum(1 for y in labels if y == [0])
        specific_tnr = specific_true_negatives / relevant_cases if relevant_cases > 0 else 1.0
        return {"negative_ratio": round(specific_tnr, 2)}

    @staticmethod
    def positive_ratio(labels: list[list[int]], predictions: list[list[int]]) -> dict[AvailableMetrics, float]:
        """Compute the ratio of positive elements in the predictions compared to the labels.
        Thus, a ratio of 1.0 means that predictions contain the same amount of elements as the labels.
        A ratio of 0.5 means that predictions contain half the amount of elements as the labels.
        A ratio of 2.0 means that predictions contain twice the amount of elements as the labels."""
        total_label_length = sum(len(set(y) - {0}) for y in labels)
        total_prediction_length = sum(len(set(y_hat) - {0}) for y_hat in predictions)
        if total_label_length == 0:
            return {"positive_ratio": float("inf") if total_prediction_length > 0 else 1.0}
        return {"positive_ratio": round(total_prediction_length / total_label_length, 2)}

    def _validate_input(self, batch: dict[str, typ.Any]) -> None:
        if self.labels_key not in batch:
            raise ValueError(f"Missing key: {self.labels_key}")
        if self.predictions_key not in batch:
            raise ValueError(f"Missing key: {self.predictions_key}")
        if not isinstance(batch[self.labels_key], list):
            raise ValueError(f"Invalid type for key {self.labels_key}: {type(batch[self.labels_key])}")
        if not isinstance(batch[self.predictions_key], list):
            raise ValueError(f"Invalid type for key {self.predictions_key}: {type(batch[self.predictions_key])}")
