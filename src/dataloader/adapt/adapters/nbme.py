import ast
import typing as typ
import pydantic


from dataloader.adapt.base import Adapter, BaseModel
from dataloader.adapt.utils import create_labels, flatten_fewshots
from dataloader.base import DatasetOptions
from segmenters import Segment


def _get_span_start_end(location: str) -> tuple[int, int]:
    """Get the start and end of a span."""
    if ";" in location:
        numbers = []
        groups = location.split(";")
        for group in groups:
            group_numbers = group.split()
            numbers.extend(map(int, group_numbers))
        return min(numbers), max(numbers)

    start, end = location.split(" ")
    return int(start), int(end)


class NmbeAnnotationModel(pydantic.BaseModel):
    """Model for a clinical patient note annotation."""

    pn_num: int
    case_num: int
    feature_num: list[int]
    annotation: list[list[str]]
    location: list[list[tuple[int, int]]]

    @pydantic.field_validator("location", mode="before")
    @classmethod
    def validate_location(cls, v: list[list[str]]) -> list[list[tuple[int, int]]]:
        """Split the location string into a list of tuples."""
        return [[_get_span_start_end(location) for location in locations] for locations in v]


class NmbePatientNoteModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE® Step 2 Clinical Skills exam."""

    pn_num: int
    case_num: int
    patient_note: str
    features: dict[int, str]
    case_features: dict[int, str]
    labels: NmbeAnnotationModel | None
    fewshots: None | list["NmbePatientNoteModel"] = None

    @pydantic.field_validator("features", "case_features", mode="before")
    @classmethod
    def validate_features(cls, v: dict | str) -> dict:
        """Ensure that features are always a dictionary."""
        if isinstance(v, dict):
            return v
        try:
            return ast.literal_eval(v)
        except ValueError:
            raise ValueError("Features must be a dictionary.")

    @pydantic.field_validator("labels", mode="after")
    @classmethod
    def validate_labels(cls, v: NmbeAnnotationModel | None) -> NmbeAnnotationModel | None:
        """Ensure that labels are always a NmbeAnnotationModel or None."""
        if v is None:
            return v
        if len(v.feature_num) != len(v.annotation) != len(v.location):
            raise ValueError("Feature numbers, annotations and locations must be of the same length.")
        return v


class NbmeAdapter(Adapter):
    """Adapter for the MedQA dataset."""

    input_model: typ.Type[NmbePatientNoteModel] = NmbePatientNoteModel
    output_model: typ.Type[BaseModel] = BaseModel

    @classmethod
    def translate_row(cls, row: dict[str, typ.Any], options: DatasetOptions) -> BaseModel:
        """Adapt a row."""

        def _format_row(row: dict[str, typ.Any], options: DatasetOptions) -> dict[str, typ.Any]:
            struct_row = cls.input_model(**row)
            segments: list[Segment] = list(options.segmenter(struct_row.patient_note))
            text_segments: list[str] = [chunk.text for chunk in segments]
            classes, targets = [], []
            if struct_row.labels is not None:
                classes, targets = create_labels(
                    segments=segments,
                    targets=struct_row.labels.feature_num,
                    spans=struct_row.labels.location,
                    classes=struct_row.case_features,
                    negatives=options.negatives,
                    seed=options.seed,
                )
            return {
                "aid": f"{struct_row.case_num}_{struct_row.pn_num}",
                "classes": classes,
                "segments": text_segments,
                "targets": targets,
            }

        formatted_row = _format_row(row, options)
        fewshots = None
        if "few_shot" in row:
            formatted_fewshots = [_format_row(row, options) for row in row["fewshots"]]
            fewshots = flatten_fewshots(formatted_fewshots, options.seed)

        return BaseModel(
            **formatted_row,
            fewshots=fewshots,
        )
