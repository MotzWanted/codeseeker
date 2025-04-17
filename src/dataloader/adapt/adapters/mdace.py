import typing as typ
import pydantic


from dataloader.adapt.adapters.mimic import MimicAdapter, sample_negatives
from dataloader.adapt.base import BaseModel
from dataloader.base import DatasetOptions
from dataloader.constants import PROJECT_ROOT
from tools.code_trie import get_code_guidelines, get_code_objects

_MEDICAL_CODING_SYSTEMS_DIR = PROJECT_ROOT / "data/medical-coding-systems/icd"
_NEGATIVES_DIR = PROJECT_ROOT / "data/medical-coding-systems/negatives"

"""
ICD system: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2022/
"""


class MdaceAnnotationModel(pydantic.BaseModel):
    """Model for a clinical patient note annotation."""

    code: list[str]
    code_type: list[str]
    spans: list[list[list[int]]]

    @pydantic.computed_field
    def location(self) -> list[list[tuple[int, int]]]:
        """Get the location of the annotation."""
        return [[(s[0], s[-1])] for span_list in self.spans for s in span_list]


class MdaceDataModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE® Step 2 Clinical Skills exam."""

    hadm_id: int
    note_id: int
    note_type: str
    note_subtype: str
    text: str
    annotations: MdaceAnnotationModel


class MdaceAdapter(MimicAdapter):
    """Adapter for the MedQA dataset."""

    input_model: typ.Type[MdaceDataModel] = MdaceDataModel
    output_model: typ.Type[BaseModel] = BaseModel

    @classmethod
    def translate_row(
        cls, row: dict[str, typ.Any], options: DatasetOptions
    ) -> BaseModel:
        """Adapt a row."""

        def _format_row(
            row: dict[str, typ.Any], options: DatasetOptions
        ) -> dict[str, typ.Any]:
            struct_row = cls.input_model(**row)
            targets = []
            for code in struct_row.annotations.code:
                if code not in targets:
                    targets.append(code)
            return {
                "aid": f"{struct_row.hadm_id}_{struct_row.note_id}",
                "note": struct_row.text,
                "targets": targets,
            }

        formatted_row = _format_row(row, options)
        return cls.output_model(**formatted_row)


class MdaceLegacyAdapter(MdaceAdapter):
    """Adapter for the MedQA dataset."""

    input_model: typ.Type[MdaceDataModel] = MdaceDataModel
    output_model: typ.Type[BaseModel] = BaseModel

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
            positives = list(set(code for code in struct_row.annotations.code))
            negatives: list[list[str]] = [
                negatives_data[code] for code in positives if code in negatives_data
            ]
            sampled_negatives = sample_negatives(
                negatives,
                positives=positives,
                per_positive=options.negatives,
                seed=options.seed,
            )
            classes = get_code_objects(
                cm_trie,
                pcs_trie,
                positives + sampled_negatives,
            )
            guidelines = get_code_guidelines(
                cm_trie,
                pcs_trie,
                [code.name for code in classes],
            )
            return {
                "aid": f"{struct_row.hadm_id}_{struct_row.note_id}",
                "guidelines": guidelines,
                "classes": [code.model_dump() for code in classes],
                "note": struct_row.text,
                "targets": positives,
            }

        formatted_row = _format_row(row, options)
        return cls.output_model(**formatted_row)
