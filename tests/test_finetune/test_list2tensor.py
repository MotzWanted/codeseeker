import pytest
import torch
from finetune.helpers import list2tensor_vectorized


@pytest.mark.parametrize(
    "dim_x, dim_y, indices, expected_tensor",
    [
        # Basic case
        (
            3,
            5,
            [{1, 2, 3}, {2}, {1, 3}],
            torch.tensor(
                [[1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
            ),
        ),
        # Negative case
        (
            3,
            5,
            [{1, -2, 3}, {2}, {-1, -3}],
            torch.tensor(
                [[1.0, -1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, -1.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
        ),
        # Out-of-bounds case
        (
            2,
            5,
            [{1, 2, 6}, {0}],
            torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
        # Large case
        (
            3,
            10,
            [{1, 2, 3}, {4}, {3, 5}],
            torch.tensor(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        ),
    ],
)
def test_list2tensor_vectorized(dim_x: int, dim_y: int, indices: list[set[int | float]], expected_tensor: torch.Tensor):
    result_tensor = list2tensor_vectorized(dim_x, dim_y, indices)
    assert result_tensor.shape == expected_tensor.shape, f"Test failed for indices={indices}"
    assert torch.equal(result_tensor, expected_tensor), f"Test failed for indices={indices}"
