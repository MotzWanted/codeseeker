from abc import ABC, abstractmethod
import json
from pathlib import Path
import re

import numpy as np

from agents.models import Alignment

PATH_TO_TEMPLATES = Path(__file__).parent / "templates"


def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    def sanitize_value(val):
        # Recursively sanitize strings within nested structures
        if isinstance(val, str):
            # Replace non-printable characters with a space
            return re.sub(r"[^\x20-\x7E]", " ", val)
        return val

    sanitized_value = sanitize_value(value)
    return json.dumps(sanitized_value, ensure_ascii=False)


def list2matrix(dim_x: int, dim_y: int, alignment_indices: list[list[int | float]]) -> np.ndarray:
    sparse_matrix = np.zeros((dim_x, dim_y), dtype=np.float32)
    for i, preds in enumerate(alignment_indices):
        for pred in preds:
            pred_sign = -1 if pred < 0 else 1
            pred_idx = abs(pred) - 1
            if 0 <= pred_idx < dim_y:
                sparse_matrix[i, pred_idx] = pred_sign
    return sparse_matrix


# @numba.jit(cache=True, nogil=True, fastmath=True)
def matrix2list(sparse_matrix: np.ndarray) -> list:
    alignment_indices = []
    for i in range(sparse_matrix.shape[0]):
        non_zero_indices = np.where(sparse_matrix[i] == 1)[0] + 1
        if len(non_zero_indices) > 0:
            row_indices = [non_zero_indices[j] for j in range(len(non_zero_indices))]
            alignment_indices.append(row_indices)
        else:
            alignment_indices.append([0])  # Append [0] for zero rows
    return alignment_indices


class Aligner(ABC):
    """An abstract class for an alignment model."""

    @abstractmethod
    async def predict(self, *args, **kwargs) -> Alignment: ...
