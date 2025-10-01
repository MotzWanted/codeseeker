import typing as typ
import pydantic


from dataloader.adapt.base import BaseModel, Adapter
from dataloader.base import DatasetOptions

"""
ICD system: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2022/
"""


class MdaceAnnotationModel(pydantic.BaseModel):
    code: list[str]
    location: list[list[tuple[int, int]]] = pydantic.Field(alias="spans")

    @pydantic.field_validator("location", mode="before")
    @classmethod
    def convert_from_span_format(
        cls, value: list[list[list[int]]]
    ) -> list[list[tuple[int, int]]]:
        """Convert list[list[list[int]]] -> list[list[tuple[int, int]]]"""
        if not isinstance(value, list):
            raise TypeError("Expected a list for location field")
        return [[(span[0], span[-1]) for span in span_list] for span_list in value]


class MdaceDataModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE® Step 2 Clinical Skills exam."""

    hadm_id: int
    note_id: int
    note_type: str
    note_subtype: str
    text: str
    annotations: MdaceAnnotationModel


class MdaceAdapter(Adapter):
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
            evidence_spans = [
                {"code": code, "locations": spans}
                for code, spans in zip(
                    struct_row.annotations.code,
                    struct_row.annotations.location,
                )
            ]
            return {
                "aid": f"{struct_row.hadm_id}_{struct_row.note_id}",
                "note": struct_row.text,
                "note_type": struct_row.note_type,
                "evidence_spans": evidence_spans,
                "targets": targets,
            }

        formatted_row = _format_row(row, options)
        return cls.output_model(**formatted_row)
