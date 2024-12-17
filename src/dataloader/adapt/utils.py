import random
import secrets
import typing

import datasets
from intervaltree import Interval, IntervalTree
from rich.table import Table

from dataloader.adapt.base import BaseDataModel, BaseFewshotModel
from segmenters.models import Segment


def dict_to_rich_table(data: dict, title: str) -> Table:
    """Convert dictionary to rich table for logging."""
    table = Table(title=title, show_header=True, header_style="bold blue")
    table.add_column("Feature", style="bold", no_wrap=False)
    table.add_column("Value", no_wrap=False)

    # For alternating row colors
    row_colors = ["black", "magenta"]

    for idx, (key, value) in enumerate(data.items()):
        # Add row with alternating colors for better readability
        table.add_row(str(key), str(value), style=row_colors[idx % 2])

    return table


def get_first_row(dataset: datasets.Dataset | datasets.DatasetDict) -> dict[str, typing.Any]:
    """Get the first row of a dataset."""
    if isinstance(dataset, datasets.DatasetDict):
        # Choose a random split from the DatasetDict
        split_names = list(dataset.keys())
        random_split = secrets.choice(split_names)
        dataset = dataset[random_split]
    return dataset[0]


def map_features_to_segments(
    self, chunks: list[Segment], annotations: list[list[str]], spans: list[list[tuple[int, int]]]
) -> list[list[int]]:
    """Get the label indices."""
    if len(annotations) != len(spans):
        raise ValueError("Annotations and spans must be of the same length.")

    tree = IntervalTree()
    for idx, chunk in enumerate(chunks, start=1):
        tree.add(Interval(chunk.start, chunk.end, idx))

    alignment_indices = []
    for ann, loc in zip(annotations, spans):
        if not ann:
            # Add a zero index for features that was not found in the note
            alignment_indices.append([0])
            continue

        annotation_indices = set()
        for _, ann_loc in zip(ann, loc):
            ann_start, ann_end = ann_loc
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

    return alignment_indices


def map_segments_to_targets(
    segments: list[Segment], targets: list[str | int], spans: list[list[tuple[int, int]]]
) -> list[list[str]]:
    """Get the target label indices."""
    tree = IntervalTree()
    for target, locations in zip(targets, spans):
        for location in locations:
            start, end = location
            tree.add(Interval(start, end, target))

    target_features = []
    for segment in segments:
        entity_ids = set()
        for match in tree[segment.start : segment.end]:
            entity_ids.add(match.data)

        if entity_ids:
            target_features.append(list(entity_ids))
        else:
            target_features.append([])

    return target_features


def sample_negatives(
    classes: dict[str, str], targets: list[list[int]], num_negatives: int, seed: int = 42
) -> dict[int, str]:
    """Sample negative features."""
    if num_negatives < 0:
        # Return all features if negatives is less than zero
        return classes

    positive_features = {key for inner_list in targets for key in inner_list}
    if num_negatives == 0:
        # Return only positive features
        return {k: v for k, v in classes.items() if k in positive_features}

    negative_features = {key for key in classes if key not in positive_features}
    random.seed(seed)
    negative_samples = random.sample(list(negative_features), min(num_negatives, len(negative_features)))

    return {k: v for k, v in classes.items() if k in set.union(positive_features, negative_samples)}


def flatten_fewshots(fewshots: list[BaseDataModel], seed: int) -> list[BaseFewshotModel]:
    """Sample n targets and labels from fewshots data without replacement."""
    flatten_fewshots = []
    for shot in fewshots:
        for idx in range(len(shot.segments)):
            flatten_fewshots.append(
                BaseFewshotModel(
                    aid=f"{shot.aid}-{idx}",
                    classes=shot.classes,
                    segment=shot.segments[idx],
                    targets=shot.targets[idx],
                )
            )
    # sample 1000 shots to avoid memory issues
    random.seed(seed)
    return random.sample(flatten_fewshots, min(1000, len(flatten_fewshots)))


def shuffle_classes(
    classes: dict[int | str, str], targets: list[list[int]], seed: int
) -> tuple[list[str], list[list[int]]]:
    # Extract feature keys and values
    shuffled_keys = list(classes.keys())
    random.seed(seed)
    random.shuffle(shuffled_keys)

    shuffled_values = [classes[key] for key in shuffled_keys]

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


def create_labels(
    segments: list[Segment],
    targets: list[int | str],
    spans: list[list[tuple[int, int]]],
    classes: dict[str, str],
    negatives: int,
    seed: int,
) -> tuple[list[list[int]], list[str]]:
    targets_index = map_segments_to_targets(segments=segments, targets=targets, spans=spans)
    output_space = sample_negatives(classes, targets_index, negatives)
    shuffled_features, shuffled_targets = shuffle_classes(output_space, targets_index, seed)
    return shuffled_features, shuffled_targets
