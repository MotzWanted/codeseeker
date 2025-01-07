import typing as typ


DEFAULT_METRICS = ["recall", "precision", "exact_match", "positive_ratio", "negative_ratio"]
AvailableMetrics = typ.Literal["recall", "precision", "exact_match", "positive_ratio", "negative_ratio"]


class ConfusionMatrix:
    def __init__(
        self,
        targets_key: str,
        predictions_key: str,
    ) -> None:
        self.targets_key = targets_key
        self.predictions_key = predictions_key

    def __call__(self, batch: dict[str, list[typ.Any]], idx: list[int] | None = None) -> dict[str, list[float]]:
        """Compute the metrics."""
        if self.targets_key not in batch:
            raise ValueError(f"Key `{self.targets_key}` not found in batch.")
        if self.predictions_key not in batch:
            raise ValueError(f"Key `{self.predictions_key}` not found in batch.")

        preds_input = self.parse_labels_fn(batch[self.predictions_key])
        target_input = self.parse_labels_fn(batch[self.targets_key])

        # Compute the true/false negatives/positives
        return self._make_conf_matrix(target_input, preds_input)

    @staticmethod
    def parse_labels_fn(labels: list[list[list[int]]]) -> list[set[int]]:
        """Parse the labels."""
        return [set(multi_label) for multi_label in labels]

    @staticmethod
    def _make_conf_matrix(targets: list[set[int]], preds: list[set[int]]) -> dict[str, list[int]]:
        """Compute the true/false positives."""

        def _tp(pred: set[int], target: set[int]) -> int:
            """True positives."""
            return len(pred & target)

        def _fp(pred: set[int], target: set[int]) -> int:
            """False positives."""
            return len(pred - target)

        def _fn(pred: set[int], target: set[int]) -> int:
            """False negatives."""
            return len(target - pred)

        def _hit(pred: set[int], target: set[int]) -> int:
            """Hit."""
            return int(pred == target)

        return {
            "tp": sum([_tp(p, t) for p, t in zip(preds, targets)]),
            "fp": sum([_fp(p, t) for p, t in zip(preds, targets)]),
            "fn": sum([_fn(p, t) for p, t in zip(preds, targets)]),
            "hits": sum([_hit(p, t) for p, t in zip(preds, targets)]),
        }
