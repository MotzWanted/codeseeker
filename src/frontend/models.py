from datetime import datetime
import uuid
import pydantic
import pytz


class ChunkedTranscript(pydantic.BaseModel):
    """This class represents the chunked transcript."""

    chunks: list[str] = pydantic.Field(
        ...,
        description="A list of distinct and logically complete segments, each represented as a string. Each chunk should encapsulate a single, coherent idea or topic.",  # noqa: E501
    )


class Annotation(pydantic.BaseModel):
    """This class represents an annotation."""

    annotation_id: str = pydantic.Field(
        default_factory=lambda: str(uuid.uuid4()), description="The unique identifier for the annotation."
    )
    created_at: datetime = pydantic.Field(
        default_factory=lambda: str(datetime.now(pytz.timezone("Europe/Copenhagen"))),
        description="The timestamp of the alignment.",
    )
    sources: list[str] = pydantic.Field(..., description="The source data.")
    targets: list[str] = pydantic.Field(..., description="The target data.")
    matrix: list[list[float]] = pydantic.Field(..., description="The alignment matrix.")
    indices: list[list[int]] = pydantic.Field(..., description="The alignment indices.")


class AlignmentAnnotation(pydantic.BaseModel):
    """This class represents a data point for alignment."""

    alignment_id: uuid.UUID = pydantic.Field(..., description="The unique identifier for the alignment")
    annotations: Annotation = pydantic.Field(..., description="The annotation data.")
