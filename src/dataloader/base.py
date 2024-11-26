import os
import pathlib
import typing as typ

import datasets


CACHE_DIR = str(pathlib.Path(os.environ.get("CACHE_DIR", "~/.cache/docgen")).expanduser())
DATASETS_CACHE_PATH = str(pathlib.Path(CACHE_DIR, "datasets"))


@typ.runtime_checkable
class DatasetLoader(typ.Protocol):
    """A dataset loader."""

    def __call__(
        self, subset: None | str = None, split: None | str = None, **kws: typ.Any
    ) -> datasets.DatasetDict | datasets.Dataset:
        """Load a dataset."""
        ...
