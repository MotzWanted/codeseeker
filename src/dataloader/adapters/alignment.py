import abc
import ast
import hashlib
import random
import typing as typ
import pydantic
import sys

from intervaltree import Interval, IntervalTree
from loguru import logger

from segmenters import Segment, Segmenter

logger.remove()
logger.add(
    sys.stderr,
    format="{level} | {message}",
    level="DEBUG",
)


class AlignmentFewShotModel(pydantic.BaseModel):
    """Alignment few-shot model."""

    aid: str
    entities: list[str]
    segment: str
    labels: list[int]


class AlignmentModel(pydantic.BaseModel):
    """Alignment model."""

    aid: str
    entities: list[str]
    segments: list[str]
    labels: list[list[int]] | None = None
    fewshots: None | list["AlignmentFewShotModel"] = None

    @pydantic.field_validator("labels", mode="before")
    def validate_label_indices(cls, v: list[list[int]]) -> list[list[int]]:
        """Validate the labels."""
        if v is None:
            return v
        if 0 in v and len(v) > 1:
            raise ValueError("The zero index must be the only index if it is present.")
        return v

    @pydantic.model_validator(mode="after")
    def validate_labels(self):
        """Validate the labels."""
        if self.labels:
            max_value = max(max(inner_list) for inner_list in self.labels)
            if max_value > len(self.entities):
                raise ValueError("The maximum value in the labels must be less than the number of entities.")
        return self


class SyntheticAlignmentModel(pydantic.BaseModel):
    """Alignment model."""

    aid: str
    entities: list[str]
    segments: list[str]
    predictions: list[list[int]]
    sparse_matrix: list[list[float]]
    probabilities: list[list[float]]


class AlignmentModelForTraining(pydantic.BaseModel):
    """Alignment model."""

    aid: str
    entities: list[str]
    segment: str
    targets: list[int]
    probabilities: list[float] | None = None

    @pydantic.field_validator("segment", mode="before")
    def parse_query(cls, v: str) -> str:
        return v.replace("-", " ").strip()

    @pydantic.model_validator(mode="after")
    def validate_training_samples(self):
        """Validate the training samples."""
        if self.probabilities and all(self.probabilities) == 1.0:
            self.probabilities = [1.0] * len(self.targets)
        if self.probabilities and len(self.targets) != len(self.probabilities):
            raise ValueError("The number of targets must match the number of probabilities.")
        return self

    def parse_targets(self) -> str:
        """Parse the targets."""
        v = [int(el) for el in self.targets]
        return ", ".join(map(str, v))


class AlignmentAdapter(abc.ABC):
    """Adapter for alignment instances associated with multiple queries."""

    @abc.abstractmethod
    def is_compatible(self, row: dict[str, typ.Any]) -> bool:
        """Check if a row is compatible with this adapter."""
        ...

    @abc.abstractmethod
    def adapt(self, row: dict[str, typ.Any]) -> AlignmentModel:
        """Adapt a row to `AlignmentModel`."""
        ...

    def __call__(self, row: dict[str, typ.Any], **extras: typ.Any) -> dict[str, typ.Any]:
        """Adapt the row and add extra fields."""
        output = self.adapt(row).model_dump()
        output.update(extras)  # Add extra fields to the row
        return output


class NmbeAnnotationModel(pydantic.BaseModel):
    """Model for a clinical patient note annotation."""

    pn_num: int
    case_num: int
    feature_num: list[int]
    annotation: list[list[str]]
    location: list[list[str]]


class NmbePatientNoteModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE® Step 2 Clinical Skills exam."""

    pn_num: int
    case_num: int
    patient_note: str
    features: dict[int, str]
    case_features: dict[int, str]
    labels: NmbeAnnotationModel | None
    few_shot: None | list["NmbePatientNoteModel"] = None

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


class NbmeAdapter(AlignmentAdapter):
    """Adapter for the MedQA dataset."""

    input_model: typ.Type[NmbePatientNoteModel] = NmbePatientNoteModel

    def __init__(
        self,
        segmenter: Segmenter,
        negatives: int = 0,  # number of negative samples to include
        seed: int = 42,
    ) -> None:
        self.segmenter = segmenter
        self.negatives = negatives
        self.seed = seed

    def is_compatible(self, row: dict[str, typ.Any]) -> bool:
        """Check if a row is compatible with this adapter."""
        return self.input_model.model_validate(row) is not None

    @staticmethod
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

    def _map_features_to_note_segments(self, chunks: list[Segment], label: NmbeAnnotationModel) -> list[list[int]]:
        """Get the label indices."""
        tree = IntervalTree()
        for idx, chunk in enumerate(chunks, start=1):
            tree.add(Interval(chunk.start, chunk.end, idx))

        alignment_indices = []
        for i, (ann, loc) in enumerate(zip(label.annotation, label.location)):
            if not ann:
                # Add a zero index for features that was not found in the note
                alignment_indices.append([0])
                continue

            annotation_indices = set()
            for _, ann_loc in zip(ann, loc):
                ann_start, ann_end = self._get_span_start_end(ann_loc)
                # Find the index of the chunk that contains the annotation
                for interval in tree[ann_start:ann_end]:
                    annotation_indices.add(interval.data)

                if not annotation_indices:
                    chunk_spans = [(chunk.start, chunk.end) for chunk in chunks]
                    raise ValueError(
                        f"Annotation `{ann}` with location `{ann_loc}` not found in the note:", f"`{chunk_spans}`"
                    )

            if len(annotation_indices) > 1 and 0 in annotation_indices:
                raise ValueError("The zero index must be the only index if it is present.")
            alignment_indices.append(list(annotation_indices))

        if len(alignment_indices) != len(label.feature_num):
            raise ValueError("Annotation and feature number mismatch:", len(alignment_indices), len(label.feature_num))

        return alignment_indices

    def _map_sources_to_entities(self, sources: list[Segment], label: NmbeAnnotationModel) -> list[list[int]]:
        """Get the target label indices."""
        tree = IntervalTree()
        for feat_num, locations in zip(label.feature_num, label.location):
            for loc in locations:
                start, end = self._get_span_start_end(loc)
                tree.add(Interval(start, end, feat_num))

        target_features = []
        for segment in sources:
            entity_ids = set()
            for match in tree[segment.start : segment.end]:
                entity_ids.add(match.data)

            if entity_ids:
                target_features.append(list(entity_ids))
            else:
                target_features.append([])

        return target_features

    def _flatten_fewshots(self, fewshots: list[dict]) -> list[dict[str, str | list[int]]]:
        """Sample n targets and labels from fewshots data without replacement."""
        flatten_fewshots = []
        for shot in fewshots:
            for idx in range(len(shot["segments"])):
                flatten_fewshots.append(
                    {
                        "aid": f"{shot["aid"]}-{idx}",
                        "entities": shot["entities"],
                        "segment": shot["segments"][idx],
                        "labels": shot["labels"][idx],
                    }
                )
        return flatten_fewshots[:1000]  # putting a cap on the number of shots

    def _sample_negatives(self, features: dict[int, str], targets: list[list[int]]) -> dict[int, str]:
        """Sample negative features."""
        if self.negatives < 0:
            # Return all features if negatives is less than zero
            return features

        positive_features = {key for inner_list in targets for key in inner_list}
        if self.negatives == 0:
            # Return only positive features
            return {k: v for k, v in features.items() if k in positive_features}

        negative_features = {key for key in features if key not in positive_features}
        random.seed(self.seed)
        negative_samples = random.sample(list(negative_features), min(self.negatives, len(negative_features)))

        return {k: v for k, v in features.items() if k in set.union(positive_features, negative_samples)}

    def _shuffle_features(
        self, features: dict[int, str], targets: list[list[int]]
    ) -> tuple[list[str], list[list[int]]]:
        # Extract feature keys and values
        shuffled_keys = list(features.keys())
        random.seed(self.seed)
        random.shuffle(shuffled_keys)

        shuffled_values = [features[key] for key in shuffled_keys]

        # Create a mapping from feature id to new shuffled list index
        id_to_shuffled_index = {key: index for index, key in enumerate(shuffled_keys, start=1)}

        # Update nested list with new indices based on shuffled keys
        shuffled_targets = []
        for inner_list in targets:
            if inner_list:
                shuffled_targets.append([id_to_shuffled_index[key] for key in inner_list])
            else:
                shuffled_targets.append([0])

        return shuffled_values, shuffled_targets

    def _create_labels(
        self, labels: NmbeAnnotationModel | None, note_segments: list[Segment], features: dict[int, str]
    ) -> tuple[list[list[int]], list[str]]:
        if not labels:
            return []
        target_features = self._map_sources_to_entities(note_segments, labels)
        features = self._sample_negatives(features, target_features)
        shuffled_features, shuffled_targets = self._shuffle_features(features, target_features)
        return shuffled_features, shuffled_targets

    def _format_row(self, row: dict[str, typ.Any]) -> dict[str, typ.Any]:
        struct_row = self.input_model(**row)
        note_segments = list(self.segmenter(struct_row.patient_note))
        segments = [chunk.text for chunk in note_segments]
        entities, labels = self._create_labels(struct_row.labels, note_segments, struct_row.features)
        return {
            "aid": f"{struct_row.case_num}_{struct_row.pn_num}",
            "entities": entities,
            "segments": segments,
            "labels": labels,
        }

    def adapt(self, row: dict[str, typ.Any]) -> AlignmentModel:
        """Adapt a row."""
        formatted_row = self._format_row(row)
        fewshots = None
        if "few_shot" in row:
            formatted_fewshots = [self._format_row(row) for row in row["few_shot"]]
            fewshots = self._flatten_fewshots(formatted_fewshots)
            seed = int(hashlib.sha256(formatted_row["aid"].encode("utf-8")).hexdigest(), 16) % (2**32)
            random.seed(seed)
            random.shuffle(fewshots)

        return AlignmentModel(
            **formatted_row,
            fewshots=fewshots,
        )
