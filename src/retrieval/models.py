import abc
import math
import typing as typ
from numbers import Number

import numba
import numpy as np
import numpy.typing as npt
import torch
from numba.typed import List
from typing_extensions import Self, Type

ArrayIndexType: typ.TypeAlias = typ.Union[int, slice, typ.Iterable[int]]

_T = typ.TypeVar("_T")
_T_co = typ.TypeVar("_T_co", covariant=True)


@typ.runtime_checkable
class Sequence(typ.Protocol[_T_co]):
    """A sequence of data."""

    def __getitem__(self, __it: int) -> _T_co: ...

    def __len__(self) -> int: ...


@typ.runtime_checkable
class DictsSequence(typ.Protocol[_T]):
    """A sequence of dictionaries."""

    def __getitem__(self, __it: int) -> dict[str, _T]: ...

    def __len__(self) -> int: ...


class RetrievalData(abc.ABC):
    """Model search results."""

    __slots__ = ("scores", "indices", "labels", "allow_unsafe", "meta")
    _expected_dim: int
    _str_sep: str = ""
    _repr_sep: str = ""
    scores: np.ndarray
    indices: np.ndarray
    labels: None | np.ndarray
    meta: dict[str, typ.Any]

    def __init__(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
        labels: None | np.ndarray = None,
        meta: None | dict[str, typ.Any] = None,
        allow_unsafe: bool = False,
    ):
        dim = len(indices.shape)
        # note: only check shapes up to the number of dimensions of the indices. This allows
        # for the scores to have more dimensions than the indices, e.g. for the case of
        # merging two batches.
        if not allow_unsafe and scores.shape[:dim] != indices.shape[:dim]:
            raise ValueError(
                f"The shapes of `scores` and `indices` must match up to the dimension of `indices`, "
                f"but got {_array_repr(scores)} and {_array_repr(indices)}"
            )
        if labels is not None and (scores.shape[:dim] != labels.shape[:dim]):
            raise ValueError(
                "The shapes of `scores` and `labels` must match up to the dimension of `indices`, "
            )
        if len(scores.shape) != self._expected_dim:
            raise ValueError(
                f"Scores must be {self._expected_dim}D, "
                f"but got {_array_repr(scores)} and {_array_repr(indices)}"
            )

        self.allow_unsafe = allow_unsafe
        self.scores = scores
        self.indices = indices
        self.labels = labels
        self.meta = meta or {}

    @classmethod
    def cast(
        cls: Type[Self],
        scores: npt.ArrayLike,
        indices: npt.ArrayLike,
        labels: None | npt.ArrayLike = None,
        meta: None | dict[str, typ.Any] = None,
        allow_unsafe: bool = False,
    ) -> Self:
        """Cast the input to a `RetrievalData` object."""
        return cls(
            scores=_cast_to_numpy(scores),
            indices=_cast_to_numpy(indices),
            labels=_cast_to_numpy(labels) if labels is not None else None,
            meta=meta,
            allow_unsafe=allow_unsafe,
        )

    @abc.abstractmethod
    def __getitem__(self, item: ArrayIndexType) -> "RetrievalData":
        """Slice the data."""
        ...

    @abc.abstractmethod
    def __iter__(self) -> typ.Iterable["RetrievalData"]:
        """Iterate over the data."""
        ...

    def __len__(self) -> int:
        """Length of the data."""
        return len(self.scores)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data."""
        return self.scores.shape

    def _get_repr_parts(self) -> list[str]:
        return [
            f"{type(self).__name__}[{_type_repr(self.scores)}](",
            f"scores={repr(self.scores)}, ",
            f"indices={repr(self.indices)}, ",
            f"labels={repr(self.labels)}, ",
            f"meta={repr(self.meta)}",
            ")",
        ]

    def __repr__(self) -> str:
        """Representation of the object."""
        parts = self._get_repr_parts()
        return self._repr_sep.join(parts[:-1]) + parts[-1]

    def __str__(self) -> str:
        """String representation of the object."""
        parts = self._get_repr_parts()
        return self._str_sep.join(parts[:-1]) + parts[-1]

    def __eq__(self, other: object) -> bool:
        """Compare two `RetrievalData` objects."""
        if not isinstance(other, type(self)):
            raise NotImplementedError(f"Cannot compare {type(self)} with {type(other)}")
        op = {
            torch.Tensor: torch.all,
            np.ndarray: np.all,
        }[type(self.scores)]
        return op(self.scores == other.scores) and op(self.indices == other.indices)

    def to_dict(self) -> dict[str, None | list[Number]]:
        """Convert to a dictionary."""
        return {
            "scores": self.scores.tolist(),
            "indices": self.indices.tolist(),
            "labels": self.labels.tolist() if self.labels is not None else None,
        }


class RetrievalTuple(RetrievalData):
    """A single search result."""

    _expected_dim = 0

    def __getitem__(self, item: typ.Any) -> typ.Any:  # noqa: ANN401
        """Not implemented for single samples."""
        raise NotImplementedError("RetrievalTuple is not iterable")

    def __iter__(self) -> typ.Any:  # noqa: ANN401
        """Not implemented for single samples."""
        raise NotImplementedError("RetrievalTuple is not iterable")


class RetrievalSample(RetrievalData):
    """A single value of a search result."""

    _expected_dim = 1
    _str_sep: str = ""

    @property
    def shape(self) -> tuple[int]:
        """Shape of the data."""
        return self.scores.shape  # type: ignore

    def __getitem__(self, item: int) -> RetrievalTuple:
        """Get a single value from the sample."""
        return RetrievalTuple(
            scores=self.scores[item],
            indices=self.indices[item],
            labels=self.labels[item] if self.labels is not None else None,
        )

    def __iter__(self) -> typ.Iterable[RetrievalTuple]:
        """Iterate over the sample dimension."""
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other: Self) -> "RetrievalBatch":
        """Concatenate two samples along the sample dimension."""
        return _stack_samples([self, other])


class RetrievalBatch(RetrievalData):
    """A batch of search results."""

    _expected_dim = 2
    _str_sep: str = "\n"

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the data."""
        return self.scores.shape  # type: ignore

    def __getitem__(self, item: int) -> RetrievalSample:
        """Get a single sample from the batch."""
        return RetrievalSample(
            scores=self.scores[item],
            indices=self.indices[item],
            labels=self.labels[item] if self.labels is not None else None,
        )

    def __iter__(self) -> typ.Iterable[RetrievalSample]:
        """Iterate over the batch dimension."""
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other: "RetrievalBatch") -> "RetrievalBatch":
        """Concatenate two batches along the batch dimension."""
        return RetrievalBatch(
            scores=np.concatenate([self.scores, other.scores]),
            indices=np.concatenate([self.indices, other.indices]),
            labels=_merge_labels(self.labels, other.labels, np.concatenate),
        )

    @classmethod
    def stack_samples(cls: Type[Self], samples: typ.Iterable[RetrievalSample]) -> Self:
        """Stack a list of samples into a batch."""
        result = _stack_samples(samples)
        return cls(scores=result.scores, indices=result.indices, labels=result.labels)

    @classmethod
    def concatenate_batches(cls: Type[Self], batches: typ.Iterable[Self]) -> Self:
        """Concatenate a list of batches into a batch."""
        output = None
        for batch in batches:
            output = batch if output is None else output + batch

        if output is None:
            raise ValueError("Cannot concatenate an empty list of batches")
        return typ.cast(Self, output)


@numba.njit(cache=True)
def _write_array(arr: np.ndarray, writes: list[np.ndarray]) -> None:
    for j in numba.prange(len(writes)):
        y = writes[j]
        arr[j, : len(y)] = y


def _stack_np_1darrays(
    arrays: list[np.ndarray], fill_values: typ.Any
) -> np.ndarray:  # noqa: ANN401
    """Stack a list of 1D arrays into a 2D array."""
    if not isinstance(arrays, list):
        raise TypeError(f"Expected a list, but got {type(arrays)}")
    if not all(isinstance(arr, np.ndarray) for arr in arrays):
        raise TypeError(f"Expected a list of numpy arrays, but got {type(arrays)}")

    # Array size
    batch_size, max_len = max(len(arr) for arr in arrays), len(arrays)

    # Create a new array and fill it with the fill value
    output = np.full((max_len, batch_size), fill_values, dtype=arrays[0].dtype)
    _write_array(output, List(arrays))  # type: ignore

    return output


def _stack_samples(samples: typ.Iterable[RetrievalSample]) -> RetrievalBatch:
    """Stack a list of samples into a batch."""
    scores = [sample.scores for sample in samples]
    indices = [sample.indices for sample in samples]
    labels = [sample.labels for sample in samples]
    if any(lbl is None for lbl in labels):
        labels = None
    return RetrievalBatch(
        scores=_stack_np_1darrays(scores, fill_values=-math.inf),  # type: ignore
        indices=_stack_np_1darrays(indices, fill_values=-1),  # type: ignore
        labels=_stack_np_1darrays(labels, -1) if labels is not None else None,  # type: ignore
    )


def _type_repr(x: typ.Any) -> str:  # noqa: ANN401
    return f"{type(x).__name__}"


def _array_repr(x: np.ndarray | torch.Tensor) -> str:
    return f"{type(x).__name__}(shape={x.shape}, dtype={x.dtype}))"


def _merge_labels(
    a: None | np.ndarray,
    b: None | np.ndarray,
    op: typ.Callable[[list[np.ndarray]], np.ndarray],
) -> None | np.ndarray:
    if a is None and a is None:
        return None
    if a is None:
        a = np.full_like(b, fill_value=-1)
    if b is None:
        b = np.full_like(a, fill_value=-1)
    return op([a, b])


def _cast_to_numpy(x: npt.ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def _slice_sequence_of_arrays(arr: Sequence[np.ndarray], indices: slice) -> np.ndarray:
    indices_list = range(*indices.indices(len(arr)))
    return np.stack([arr[i] for i in indices_list])


class LazyArray(abc.ABC, Sequence[np.ndarray]):
    """A class that handles input array and provides lazy slicing into np.ndarray."""

    @typ.overload
    def __getitem__(self, __it: int) -> np.ndarray: ...

    @typ.overload
    def __getitem__(self, __it: slice) -> np.ndarray: ...

    def __getitem__(self, item: int | slice) -> np.ndarray:
        """Slice the vector and return the result."""
        if isinstance(item, int):
            return self._getitem_int(item)
        if isinstance(item, slice):
            return self._getitem_slice(item)
        raise TypeError(f"Unsupported index type: {type(item)}")

    @abc.abstractmethod
    def _getitem_int(self, item: int) -> np.ndarray:
        """Slice the vector and return the result."""
        raise NotImplementedError

    def _getitem_slice(self, item: slice) -> np.ndarray:
        return self._slice_arr(item)

    @abc.abstractmethod
    def _get_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def _slice_arr(self, indices: slice) -> np.ndarray:
        """Slice the vector and return the result."""
        return _slice_sequence_of_arrays(self, indices)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the vector."""
        return self._get_shape()

    def __len__(self) -> int:
        """Return the length of the vector."""
        return self.shape[0]

    def __repr__(self) -> str:
        """String representation of the vector handler."""
        return f"{type(self).__name__}(shape={self.shape})"


def slice_arrays_sequence(arr: Sequence[np.ndarray], indices: slice) -> np.ndarray:
    """Slice an array and return the result."""
    if isinstance(arr, np.ndarray):
        return arr[indices]
    if isinstance(arr, LazyArray):
        return arr._slice_arr(indices)

    return _slice_sequence_of_arrays(arr, indices)
