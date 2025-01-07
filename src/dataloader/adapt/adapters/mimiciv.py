import json
import typing as typ
import pydantic


from dataloader.adapt.base import Adapter, BaseTrainingModel, BaseInferenceModel
from dataloader.adapt.utils import sample_negatives, shuffle_classes
from dataloader.base import DatasetOptions


class MimicIvModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE® Step 2 Clinical Skills exam."""

    subject_id: int
    hadm_id: int
    note_id: str
    text: str
    codes: list[str]
    negatives: list[str]
    classes: dict[str, str]

    @pydantic.field_validator("classes", mode="before")
    @classmethod
    def validate_dict_string(cls, v: dict | str) -> dict[str, str]:
        """Ensure that the classes are always a dictionary."""
        if isinstance(v, dict):
            return {str(key): value for key, value in v.items()}
        _v: dict = json.loads(v)
        return {str(key): value for key, value in _v.items()}


class MimicIvInferenceAdapter(Adapter):
    """Adapter for the MedQA dataset."""

    input_model = MimicIvModel
    output_model = BaseInferenceModel

    @classmethod
    def translate_row(cls, row: dict[str, typ.Any], options: DatasetOptions) -> BaseInferenceModel:
        """Adapt a row."""

        def _format_row(row: dict[str, typ.Any], options: DatasetOptions) -> dict[str, typ.Any]:
            struct_row = cls.input_model(**row)
            code2class = sample_negatives(
                struct_row.classes,
                [struct_row.codes],
                options.negatives,
                options.seed,
                options.hard_negatives,
                struct_row.negatives,
            )
            _, shuffled_classes, shuffled_targets = shuffle_classes(code2class, [struct_row.codes], options.seed)
            return {
                "aid": f"{struct_row.subject_id}_{struct_row.hadm_id}_{struct_row.note_id}",
                "classes": shuffled_classes,
                "segments": [struct_row.text],
                "targets": shuffled_targets,
                "index2code": {str(idx): code for idx, code in enumerate(code2class, start=1)},
            }

        formatted_row = _format_row(row, options)
        return cls.output_model(**formatted_row)


class MimicIvForTrainingAdapter(Adapter):
    """Adapter for the MedQA dataset."""

    input_model = MimicIvModel
    output_model = BaseTrainingModel

    @classmethod
    def translate_row(cls, row: dict[str, typ.Any], options: DatasetOptions) -> BaseTrainingModel:
        """Adapt a row."""

        def _format_row(row: dict[str, typ.Any], options: DatasetOptions) -> dict[str, typ.Any]:
            struct_row = cls.input_model(**row)
            code2class = sample_negatives(
                struct_row.classes,
                [struct_row.codes],
                options.negatives,
                options.seed,
                options.hard_negatives,
                struct_row.negatives,
            )
            shuffled_codes, shuffled_classes, shuffled_targets = shuffle_classes(
                code2class, [struct_row.codes], options.seed
            )
            return {
                "aid": f"{struct_row.subject_id}_{struct_row.hadm_id}_{struct_row.note_id}",
                "classes": shuffled_classes,
                "segments": struct_row.text,
                "targets": shuffled_targets[0],
                "codes": shuffled_codes,
            }

        formatted_row = _format_row(row, options)
        return cls.output_model(**formatted_row)
