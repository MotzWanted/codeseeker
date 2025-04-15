import ast
import typing as typ
import pydantic


from dataloader.adapt.base import Adapter, BaseModel
from dataloader.adapt.utils import create_labels, flatten_fewshots
from dataloader.base import DatasetOptions
from segmenters import Segment


class MedDecAnnotationModel(pydantic.BaseModel):
    """Model for a clinical patient note annotation."""

    annotator_id: list[str]
    annotation_id: list[str]
    start_offset: list[int]
    end_offset: list[int]
    category: list[str]
    category_id: list[str]
    decision: list[str]

    @pydantic.computed_field
    def location(self) -> list[list[tuple[int, int]]]:
        """Get the location of the annotation."""
        return [[(start, end)] for start, end in zip(self.start_offset, self.end_offset)]


class MedDecDataModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE® Step 2 Clinical Skills exam."""

    subject_id: int
    hadm_id: int
    note_id: int
    text: str
    classes: dict[str, str]
    annotations: MedDecAnnotationModel

    @pydantic.field_validator("classes", mode="before")
    @classmethod
    def validate_dict_string(cls, v: dict | str) -> dict:
        """Ensure that the classes are always a dictionary."""
        if isinstance(v, dict):
            return v
        return ast.literal_eval(v)


class MedDecAdapter(Adapter):
    """Adapter for the MedQA dataset."""

    input_model: typ.Type[MedDecDataModel] = MedDecDataModel
    output_model: typ.Type[BaseModel] = BaseModel

    @classmethod
    def translate_row(cls, row: dict[str, typ.Any], options: DatasetOptions) -> BaseModel:
        """Adapt a row."""

        def _format_row(row: dict[str, typ.Any], options: DatasetOptions) -> dict[str, typ.Any]:
            struct_row = cls.input_model(**row)
            segments: list[Segment] = list(options.segmenter(struct_row.text))
            text_segments: list[str] = [chunk.text for chunk in segments]
            classes, targets = [], []
            if struct_row.annotations is not None:
                classes, targets = create_labels(
                    segments=segments,
                    targets=struct_row.annotations.category_id,
                    spans=struct_row.annotations.location,
                    classes=struct_row.classes,
                    negatives=options.negatives,
                    seed=options.seed,
                )
            return {
                "aid": f"{struct_row.hadm_id}_{struct_row.subject_id}_{struct_row.note_id}",
                "classes": classes,
                "segments": text_segments,
                "targets": targets,
            }

        formatted_row = _format_row(row, options)
        fewshots = None
        if "fewshots" in row and row["fewshots"]:
            formatted_fewshots = [_format_row(row, options) for row in row["fewshots"]]
            fewshots = flatten_fewshots(formatted_fewshots, options.seed)

        return cls.output_model(**formatted_row, fewshots=fewshots)
