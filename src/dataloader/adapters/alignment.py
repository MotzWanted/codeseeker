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
    corpus: list[str]
    query: str
    labels: list[int]


class BinaryFewShotModel(pydantic.BaseModel):
    """Binary few-shot model."""

    aid: str
    query: str
    passage: str
    label: bool


class AlignmentModel(pydantic.BaseModel):
    """Alignment model."""

    aid: str
    corpus: list[str]
    queries: list[str]
    labels: list[list[int]] | None = None
    fewshots: None | list["AlignmentFewShotModel"] | list["BinaryFewShotModel"] = None

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
        if self.labels and len(self.labels) != len(self.corpus) and len(self.labels) != len(self.queries):
            raise ValueError("The number of labels must match the number of sources or targets.")
        return self


class SyntheticAlignmentModel(pydantic.BaseModel):
    """Alignment model."""

    aid: str
    corpus: list[str]
    queries: list[str]
    predictions: list[list[int]]
    sparse_matrix: list[list[float]]
    probabilities: list[list[float]]


class AlignmentModelForTraining(pydantic.BaseModel):
    """Alignment model."""

    aid: str
    corpus: list[str]
    query: str
    targets: list[int]
    probabilities: list[float] | None = None

    @pydantic.field_validator("query", mode="before")
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
    labels: NmbeAnnotationModel | None
    few_shot: None | list["NmbePatientNoteModel"] = None

    @pydantic.field_validator("features", mode="before")
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
        query_key: typ.Literal["patient_note", "features"],
        seed: int = 42,
        binary_fewshots: bool = False,
    ) -> None:
        self.segmenter = segmenter
        self.query_key = query_key  # determines the direction of the alignment
        self.seed = seed
        self.binary_fewshots = binary_fewshots

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

    def _map_features_to_note_segments(self, sources: list[Segment], label: NmbeAnnotationModel) -> list[list[int]]:
        """Get the label indices."""
        tree = IntervalTree()
        for idx, chunk in enumerate(sources, start=1):
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
                    chunk_spans = [(chunk.start, chunk.end) for chunk in sources]
                    raise ValueError(
                        f"Annotation `{ann}` with location `{ann_loc}` not found in the note:", f"`{chunk_spans}`"
                    )

            if len(annotation_indices) > 1 and 0 in annotation_indices:
                raise ValueError("The zero index must be the only index if it is present.")
            alignment_indices.append(list(annotation_indices))

        if len(alignment_indices) != len(label.feature_num):
            raise ValueError("Annotation and feature number mismatch:", len(alignment_indices), len(label.feature_num))

        return alignment_indices

    def _map_note_segments_to_features(
        self, sources: list[Segment], targets: list[str], source_labels: list[list[int]]
    ) -> tuple[list[str], list[list[int]]]:
        """Get the target label indices."""
        if len(targets) != len(source_labels):
            raise ValueError("The number of targets must match the number of source labels.")

        # Flatten the source_labels and count the number of zeros
        number_of_non_zeros_source = len([el for sublist in source_labels for el in sublist if el != 0])

        # Create a zero list with the same length as the number of sources
        target_indices = [[0] for _ in range(len(sources))]

        # Run over the source labels and populate the list with the target indices
        for target_idx, source_label in enumerate(source_labels, start=1):
            # Skip if the source_label is zero as the target is not present in the source
            if 0 in source_label:
                continue
            for source_idx in source_label:
                if 0 in target_indices[source_idx - 1]:
                    target_indices[source_idx - 1].pop(0)
                target_indices[source_idx - 1].append(target_idx)

        shuffled_targets, target_indices = self._shuffle_features(targets, target_indices)

        # Flatten target_indices and count the number of non-zeros
        number_of_non_zeros_target = len([el for sublist in target_indices for el in sublist if el != 0])

        if number_of_non_zeros_source != number_of_non_zeros_target:
            raise ValueError(
                "The number of non-zero elements in the target_indices must be the same as in the source_labels.",
            )

        return shuffled_targets, target_indices

    def _flatten_fewshots(self, fewshots: list[dict]) -> list[dict[str, str | list[int]]]:
        """Sample n targets and labels from fewshots data without replacement."""
        flatten_fewshots = []
        for shot in fewshots:
            if len(shot["queries"]) != len(shot["labels"]):
                raise ValueError("The number of labels and targets should be the same.")

            for idx in range(len(shot["queries"])):
                flatten_fewshots.append(
                    {
                        "aid": f"{shot["aid"]}-{idx}",
                        "corpus": shot["corpus"],
                        "query": shot["queries"][idx],
                        "labels": shot["labels"][idx],
                    }
                )
        if self.binary_fewshots:
            flatten_fewshots = self._expand_binary_fewshots(flatten_fewshots)
        return flatten_fewshots[:1000]  # putting a cap on the number of shots

    def _expand_binary_fewshots(self, fewshots: list[dict[str, typ.Any]]) -> list[dict[str, typ.Any]]:
        """Expand binary fewshots data."""
        false_fewshots = []
        true_fewshots = []
        for shot in fewshots:
            corpus = shot["corpus"]  # list of strings
            labels = shot["labels"]  # list of integers (indices in corpus)
            aid = shot["aid"]
            for idx, passage in enumerate(corpus, start=1):
                _fewshot = {
                    "aid": f"{aid}-{idx}",
                    "query": shot["query"],
                    "passage": passage,
                }
                if idx in labels:
                    _fewshot["label"] = True
                    true_fewshots.append(_fewshot)
                else:
                    _fewshot["label"] = False
                    false_fewshots.append(_fewshot)
        # Balance the classes by undersampling the majority class
        true_count = len(true_fewshots)
        balanced_false = (
            random.sample(false_fewshots, true_count) if len(false_fewshots) > true_count else false_fewshots
        )

        return true_fewshots + balanced_false

    def _set_corpus_and_queries(self, note_sentences: list[str], features: list[str]) -> tuple[list[str], list[str]]:
        return (features, note_sentences) if self.query_key == "patient_note" else (note_sentences, features)

    def _shuffle_features(self, features: list[str], indexes: list[list[int]]) -> tuple[list[str], list[list[int]]]:
        # Shuffle features and generate mapping
        shuffled_features = features[:]
        random.seed(self.seed)
        random.shuffle(shuffled_features)

        # Create mapping from old indices to new ones
        index_mapping = {i: shuffled_features.index(f) + 1 for i, f in enumerate(features, start=1)}

        # Update indexes list according to the shuffled mapping
        updated_indexes = []
        for sublist in indexes:
            updated_sublist = [index_mapping.get(i, 0) if i != 0 else 0 for i in sublist]
            updated_indexes.append(updated_sublist)

        for i in range(len(updated_indexes)):
            for old_index, new_index in zip(indexes[i], updated_indexes[i]):
                if old_index == new_index == 0:
                    continue
                if features[old_index - 1] != shuffled_features[new_index - 1]:
                    raise ValueError(
                        f"Feature mismatch: {features[old_index - 1]} != {shuffled_features[new_index - 1]}."
                    )

        return shuffled_features, updated_indexes

    def _create_labels(
        self, labels: NmbeAnnotationModel | None, note_segments: list[Segment], features: list[str]
    ) -> tuple[list[list[int]], list[str]]:
        if not labels:
            return []

        feat2note_links = self._map_features_to_note_segments(note_segments, labels)
        shuffled_features, note2feat_links = self._map_note_segments_to_features(
            note_segments, features, feat2note_links
        )
        return (feat2note_links, features) if self.query_key == "features" else (note2feat_links, shuffled_features)

    def _format_row(self, row: dict[str, typ.Any]) -> dict[str, typ.Any]:
        struct_row = self.input_model(**row)
        note_segments = list(self.segmenter(struct_row.patient_note))
        note_sentences = [chunk.text for chunk in note_segments]
        features = list(struct_row.features.values())
        labels = None
        if struct_row.labels:
            labels, features = self._create_labels(struct_row.labels, note_segments, features)
        corpus, queries = self._set_corpus_and_queries(note_sentences, features)
        return {
            "aid": f"{struct_row.case_num}_{struct_row.pn_num}",
            "corpus": corpus,
            "queries": queries,
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
