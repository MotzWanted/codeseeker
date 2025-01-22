import hashlib
import json
import random
import typing as typ

import pydantic

from dataloader.adapt.base import Adapter, BaseInferenceModel, BaseTrainingModel
from dataloader.adapt.utils import shuffle_classes_randomly, sort_classes_alphabetically
from dataloader.base import DatasetOptions
from dataloader.constants import PROJECT_ROOT
from tools.code_trie import XMLTrie, get_hard_negatives_for_list_of_codes, get_random_negatives_for_codes

_MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/medical-coding-systems/icd"


class MimicModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE® Step 2 Clinical Skills exam."""

    subject_id: int
    hadm_id: int
    note_id: str | None
    text: str
    codes: list[str]
    classes: dict[str, str]

    @pydantic.field_validator("classes", mode="before")
    @classmethod
    def validate_dict_string(cls, v: dict | str) -> dict[str, str]:
        """Ensure that the classes are always a dictionary."""
        if isinstance(v, dict):
            return {str(key): value for key, value in v.items()}
        _v: dict = json.loads(v)
        return {str(key): value for key, value in _v.items()}


class MimicAdapter(Adapter):
    """Adapter for the MedQA dataset."""

    input_model = MimicModel
    output_model = BaseInferenceModel

    @property
    def _cm_trie(self) -> XMLTrie:
        return XMLTrie.from_xml_file(_MEDICAL_CODING_SYSTEMS_DIR / "icd10cm_tabular_2025.xml", "icd10cm")

    @property
    def _pcs_trie(self) -> XMLTrie:
        return XMLTrie.from_xml_file(_MEDICAL_CODING_SYSTEMS_DIR / "icd10pcs_tables_2025.xml", "icd10pcs")


def _split_into_pcs_and_cm(cm_trie: XMLTrie, pcs_trie: XMLTrie, codes: list[str]) -> tuple[list[str], list[str]]:
    """Split the codes into PCS and CM codes."""
    pcs_codes = [], cm_codes = []
    for code in codes:
        if code in cm_trie.lookup:
            cm_codes.append(code)
        elif code in pcs_trie.lookup:
            pcs_codes.append(code)
        elif code[:4] in pcs_trie.lookup:
            pcs_codes.append(code)
        elif code[:3] in cm_trie.lookup:
            cm_codes.append(code)
        else:
            raise ValueError(f"Code {code} is not in the CM or PCS trie")
    return pcs_codes, cm_codes


def _sample_negatives_proportional_to_code_frequency(
    pcs_codes: list[str], cm_codes: list[str], num: int, seed: int
) -> list[str]:
    """Sample negatives proportional to the code frequency."""
    weight_cm = len(cm_codes) / (len(cm_codes) + len(pcs_codes))
    weight_pcs = len(pcs_codes) / (len(cm_codes) + len(pcs_codes))
    negatives = random.choices(
        cm_codes + pcs_codes, weights=[weight_cm] * len(cm_codes) + [weight_pcs] * len(pcs_codes), k=num, seed=seed
    )
    return negatives


def _sample_hard_negatives(
    cm_trie: XMLTrie, pcs_trie: XMLTrie, pcs_codes: list[str], cm_codes: list[str], num: int, seed: int
) -> list[str]:
    """Sample hard negatives."""
    cm_hard_negatives = get_hard_negatives_for_list_of_codes(cm_codes, cm_trie, num=num)
    pcs_hard_negatives = get_hard_negatives_for_list_of_codes(pcs_codes, pcs_trie, num=num)

    if len(cm_hard_negatives) + len(pcs_hard_negatives) < num:
        raise ValueError(f"Number of hard negatives is less than {num}")

    return _sample_negatives_proportional_to_code_frequency(pcs_hard_negatives, cm_hard_negatives, num, seed)


def _sample_random_negatives(
    cm_trie: XMLTrie, pcs_trie: XMLTrie, pcs_codes: list[str], cm_codes: list[str], num: int, seed: int
) -> list[str]:
    """Sample soft negatives."""
    cm_rand_negatives = get_random_negatives_for_codes(cm_codes, cm_trie, num=num)
    pcs_rand_negatives = get_random_negatives_for_codes(pcs_codes, pcs_trie, num=num)

    if len(cm_rand_negatives) + len(pcs_rand_negatives) < num:
        raise ValueError(f"Number of soft negatives is less than {num}")

    return _sample_negatives_proportional_to_code_frequency(cm_rand_negatives, pcs_rand_negatives, num, seed)


def sample_negatives(
    cm_trie: XMLTrie,
    pcs_trie: XMLTrie,
    classes: dict[str, str],
    codes: list[list[str]],
    num_negatives: int,
    seed: int,
    hard_ratio: float = 0.0,
) -> dict[str, str]:
    """Sample negative features."""
    if num_negatives < 0:
        # Return all features if negatives is less than zero
        return classes
    cm_codes, pcs_codes = _split_into_pcs_and_cm(cm_trie, pcs_trie, codes)

    random_negatives: set[str] = set().union(
        *_sample_random_negatives(cm_trie, pcs_trie, pcs_codes, cm_codes, int(num_negatives * (1 - hard_ratio)), seed)
    )

    positive_codes: set[str] = set().union(*codes)

    num_hard_negatives = int(num_negatives * hard_ratio)
    hard_negatives = _sample_hard_negatives(cm_trie, pcs_trie, pcs_codes, cm_codes, num_hard_negatives, seed)
    negatives = hard_negatives + random_negatives
    # Add selected negatives and postive codes to result
    result = classes.copy()
    for k in negatives + positive_codes:
        result[k] = classes[k]

    return result


def string_to_seed(string: str) -> int:
    # Hash the string using SHA-256 (or another hash algorithm)
    hash_object = hashlib.sha256(string.encode())
    # Convert the hash to an integer
    hash_int = int(hash_object.hexdigest(), 16)
    # Reduce the integer to a manageable size, if needed (e.g., for compatibility with random.seed)
    return hash_int % (2**32)


class MimicInferenceAdapter(Adapter):
    """Adapter for the MedQA dataset."""

    input_model = MimicModel
    output_model = BaseInferenceModel

    @classmethod
    def translate_row(cls, row: dict[str, typ.Any], options: DatasetOptions) -> BaseInferenceModel:
        """Adapt a row."""

        def _format_row(row: dict[str, typ.Any], options: DatasetOptions) -> dict[str, typ.Any]:
            struct_row = cls.input_model(**row)
            _id = f"{struct_row.subject_id}_{struct_row.hadm_id}_{struct_row.note_id}"
            seed = string_to_seed(_id)
            classes = sample_negatives(
                cm_trie=cls._cm_trie,
                pcs_trie=cls._pcs_trie,
                classes=struct_row.classes,
                codes=[struct_row.codes],
                seed=seed,
                hard_ratio=options.hard_negatives,
                num_negatives=options.negatives,
            )
            order_fn = {"alphabetical": sort_classes_alphabetically, "random": shuffle_classes_randomly}[options.order]
            ordered_classes = order_fn(classes, seed)
            return {
                "aid": _id,
                "classes": ordered_classes,
                "segments": [struct_row.text],
                "targets": [struct_row.codes],
            }

        formatted_row = _format_row(row, options)
        return cls.output_model(**formatted_row)


class MimicForTrainingAdapter(MimicAdapter):
    """Adapter for the MedQA dataset."""

    input_model = MimicModel
    output_model = BaseTrainingModel

    @classmethod
    def translate_row(cls, row: dict[str, typ.Any], options: DatasetOptions) -> BaseTrainingModel:
        """Adapt a row."""

        def _format_row(row: dict[str, typ.Any], options: DatasetOptions) -> dict[str, typ.Any]:
            struct_row = cls.input_model(**row)
            _id = f"{struct_row.subject_id}_{struct_row.hadm_id}_{struct_row.note_id}"
            seed = string_to_seed(_id)
            code2class = sample_negatives(
                cm_trie=cls._cm_trie,
                pcs_trie=cls._pcs_trie,
                classes=struct_row.classes,
                codes=[struct_row.codes],
                seed=seed,
                hard_ratio=options.hard_negatives,
                num_negatives=options.negatives,
            )
            order_fn = {"alphabetical": sort_classes_alphabetically, "random": shuffle_classes_randomly}[options.order]
            ordered_classes = order_fn(code2class, seed)

            return {
                "aid": _id,
                "classes": ordered_classes,
                "segments": struct_row.text,
                "targets": struct_row.codes,
            }

        formatted_row = _format_row(row, options)
        return cls.output_model(**formatted_row)
