import json
import typing as typ

import numpy as np
import pydantic

from dataloader.adapt.base import Adapter, BaseModel
from dataloader.adapt.utils import (
    shuffle_classes_randomly,
    sort_classes_alphabetically,
)
from dataloader.base import DatasetOptions
from dataloader.constants import PROJECT_ROOT
from tools.code_trie import (
    XMLTrie,
    get_code_objects,
)

_MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/medical-coding-systems/icd"
_NEGATIVES_DIR = PROJECT_ROOT / "data/medical-coding-systems/negatives"


class MimicModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE® Step 2 Clinical Skills exam."""

    subject_id: int
    hadm_id: int
    note_id: str | None
    text: str
    codes: list[str]


class MimicAdapter(Adapter):
    """Adapter for the MedQA dataset."""

    input_model = MimicModel
    output_model = BaseModel

    _cm_trie: XMLTrie | None = None
    _pcs_trie: XMLTrie | None = None
    _negatives_data: dict[str, list] | None = None

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Load the XMLTrie and negatives JSON only if not already loaded."""
        if cls._cm_trie is None:
            cls._cm_trie = XMLTrie.from_xml_file(
                str(_MEDICAL_CODING_SYSTEMS_DIR / "icd10cm_tabular_2025.xml"), "icd10cm"
            )
        if cls._pcs_trie is None:
            cls._pcs_trie = XMLTrie.from_xml_file(
                str(_MEDICAL_CODING_SYSTEMS_DIR / "icd10pcs_tables_2025.xml"),
                "icd10pcs",
            )
        if cls._negatives_data is None:
            with (_NEGATIVES_DIR / "icd10_negatives.json").open() as f:
                cls._negatives_data = json.load(f)

    @property
    def cm_trie(self) -> XMLTrie:
        self._ensure_loaded()
        return type(self)._cm_trie  # type: ignore

    @property
    def pcs_trie(self) -> XMLTrie:
        self._ensure_loaded()
        return type(self)._pcs_trie  # type: ignore

    @property
    def negatives(self) -> dict[str, list]:
        self._ensure_loaded()
        return type(self)._negatives_data  # type: ignore


def sample_negatives(
    negatives: list[list[str]],
    positives: list[str],
    per_positive: int,
    seed: int | str = 42,
) -> list[str]:
    # Ensure the total number of samples does not exceed 50
    negatives_to_sample = len(positives) * per_positive
    if negatives_to_sample == 0:
        return []
    rng = np.random.RandomState(int(seed))
    # Step 1: Determine the initial fair share per sublist
    num_sublists = len(negatives)
    if num_sublists == 0:
        raise ValueError("No negatives to sample from")
    base_samples_per_sublist = negatives_to_sample // num_sublists
    remainder = negatives_to_sample % num_sublists  # Leftover samples

    selected_negatives = []
    remaining_negatives = negatives_to_sample

    # Step 2: Assign samples as evenly as possible
    for i, sublist in enumerate(sorted(negatives)):
        if remaining_negatives <= 0:
            break
        unique_sublist = [
            code
            for code in sublist
            if code not in positives and code not in selected_negatives
        ]

        if len(unique_sublist) == 0:
            continue

        indices = np.arange(len(unique_sublist))
        weights = np.exp(-0.5 * indices)  # Exponential decay
        weights /= weights.sum()  # Normalize to get probabilities

        num_to_sample = min(
            len(unique_sublist), base_samples_per_sublist + (1 if i < remainder else 0)
        )
        sampled = rng.choice(
            unique_sublist, size=num_to_sample, replace=False, p=weights
        ).tolist()

        selected_negatives.extend(sampled)
        remaining_negatives -= len(sampled)

    if selected_negatives != negatives_to_sample:
        raise ValueError(
            f"Sampled {len(selected_negatives)} negatives, but expected {negatives_to_sample}"
        )

    return selected_negatives


def string_to_seed(string: str) -> int:
    # Hash the string using SHA-256 (or another hash algorithm)
    return abs(hash(string)) % (2**32)


class MimicForTrainingAdapter(MimicAdapter):
    """Adapter for the MedQA dataset."""

    input_model = MimicModel
    output_model = BaseModel

    @classmethod
    def translate_row(
        cls, row: dict[str, typ.Any], options: DatasetOptions
    ) -> BaseModel:
        """Adapt a row."""

        def _format_row(
            row: dict[str, typ.Any], options: DatasetOptions
        ) -> dict[str, typ.Any]:
            cm_trie = cls().cm_trie
            pcs_trie = cls().pcs_trie
            negatives_data = cls().negatives
            struct_row = cls.input_model(**row)
            _id = f"{struct_row.subject_id}_{struct_row.hadm_id}_{struct_row.note_id}"
            seed = string_to_seed(_id)
            positives = get_code_objects(
                cm_trie,  # type: ignore
                pcs_trie,  # type: ignore
                struct_row.codes,
            )
            negatives: list[list[list[str]]] = [
                negatives_data[code] for code in positives if code in negatives_data
            ]
            sampled_negatives = sample_from_nested_list(
                negatives,
                positives=positives,
                negatives=options.negatives,
                total=options.total,
                seed=seed,
            )
            classes = {**positives, **sampled_negatives}
            order_fn = {
                "alphabetical": sort_classes_alphabetically,
                "random": shuffle_classes_randomly,
            }[options.order]
            ordered_classes = order_fn(classes, seed)
            if len(ordered_classes) < 50:
                raise ValueError(
                    f"Length of sorted classes is less than 50: {len(ordered_classes)}"
                )

            return {
                "aid": _id,
                "classes": json.dumps(ordered_classes),
                "note": struct_row.text,
                "targets": list(positives.keys()),
            }

        formatted_row = _format_row(row, options)
        try:
            output_model = cls.output_model(**formatted_row)
        except pydantic.ValidationError as e:
            print(f"Error in row: {row}")
            raise e
        return output_model


class MimicIdentifyAdapter(MimicAdapter):
    """Adapter for the MedQA dataset."""

    input_model = MimicModel
    output_model = MimicModel

    @classmethod
    def translate_row(
        cls, row: dict[str, typ.Any], options: DatasetOptions
    ) -> BaseModel:
        """Adapt a row."""
        return cls.output_model(**row)
