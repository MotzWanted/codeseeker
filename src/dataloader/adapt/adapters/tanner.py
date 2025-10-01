import typing as typ
import pydantic


from dataloader.adapt.base import BaseModel, Adapter
from dataloader.base import DatasetOptions

"""
ICD system: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2022/
"""


class TannerDataModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE® Step 2 Clinical Skills exam."""

    MRN: int
    HAR: int
    note: str
    diagnosis_codes: list[str]
    cpt_codes: list[str]
    claim_denial: str


class TannerAdapter(Adapter):
    """Adapter for the MedQA dataset."""

    input_model: typ.Type[TannerDataModel] = TannerDataModel
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
            return {
                "aid": f"{struct_row.MRN}_{struct_row.HAR}",
                "note": struct_row.note,
                "targets": struct_row.diagnosis_codes,
                "claim_denial": struct_row.claim_denial,
            }

        formatted_row = _format_row(row, options)
        return cls.output_model(**formatted_row)
