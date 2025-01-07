import pytest
import torch
from finetune.helpers import ClassificationMonitor

CLASSES = 5


@pytest.fixture
def classification_monitor():
    def parse_labels_fn(label_str):
        return set(map(int, label_str.split(",")))

    monitor = ClassificationMonitor(classes=CLASSES)
    return monitor


@pytest.mark.parametrize(
    "targets_input, preds_input, expected_metrics",
    [
        (
            [{1, 2, 3}, {2}, {1, 3}],
            [{1, 2, 3}, {2}, {1, 3}],
            {
                "accuracy": torch.tensor(1.0),
                "f1_micro": torch.tensor(1.0),
                "f1_macro": torch.tensor(1.0),
                "positive_ratio": torch.tensor(1.0),
            },
        ),
        (
            [{1, 2, 3}, {1, 2}, {1, 3}],
            [{2, 3}, {2}, {3, 4}],
            {
                "accuracy": torch.tensor(0.0),
                "f1_micro": torch.tensor(0.6667),
                "f1_macro": torch.tensor(0.5),
                "positive_ratio": torch.tensor(0.714),
            },
        ),
    ],
)
def test_classification_monitor(
    targets_input: list[set[int]],
    preds_input: list[set[int]],
    expected_metrics: dict[str, torch.Tensor],
    classification_monitor: ClassificationMonitor,
):
    classification_monitor.update(
        target_input_ids=targets_input,
        preds_input_ids=preds_input,
        tp=torch.zeros(CLASSES),
        fp=torch.zeros(CLASSES),
        fn=torch.zeros(CLASSES),
        tn=torch.zeros(CLASSES),
        _hit=torch.zeros(0),
    )

    metrics = classification_monitor.get()

    # Verify the metrics computation
    for key, value in expected_metrics.items():
        assert key in metrics
        assert torch.allclose(metrics[key], value, atol=1e-3)
