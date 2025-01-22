import pytest
from rich.table import Table

from dataloader.adapt.utils import (
    dict_to_rich_table,
    map_segments_to_targets,
    sample_negatives,
    shuffle_classes,
    sort_classes_alphabetically,
)
from dataloader.base import DatasetOptions
from segmenters.models import Segment


@pytest.fixture
def sample_mimic_row():
    return {
        "subject_id": 12345,
        "hadm_id": 67890,
        "note_id": "note_1",
        "text": "Patient presents with chest pain and shortness of breath.",
        "codes": ["428.0", "401.9"],  # Example ICD-9 codes
        "negatives": ["250.00", "272.4"],  # Example negative codes
        "classes": {
            "428.0": "Congestive heart failure",
            "401.9": "Hypertension",
            "250.00": "Diabetes mellitus",
            "272.4": "Hyperlipidemia",
        },
    }


@pytest.fixture
def sample_dataset_options():
    return DatasetOptions(
        negatives=2,
        seed=42,
        order="random",
        hard_negatives=0.5,
    )


@pytest.fixture
def sample_segments():
    return [
        Segment(start=0, end=35, text="Patient presents with chest pain"),
        Segment(start=36, end=60, text="and shortness of breath"),
    ]


@pytest.mark.parametrize(
    "num_negatives,targets,expected_length,hard_ratio,hard_negatives",
    [
        (2, [["428.0"]], 3, 0.5, ["250.00"]),  # 1 positive + 1 hard + 1 soft negative
        (0, [["428.0", "401.9"]], 2, 0.0, []),  # Only positives
        (-1, [["428.0"]], 4, 0.0, []),  # All classes
        (1, [["428.0"]], 2, 1.0, ["250.00"]),  # 1 positive + 1 hard negative
    ],
)
def test_sample_negatives_with_mimic_data(
    sample_mimic_row, num_negatives, targets, expected_length, hard_ratio, hard_negatives
):
    result = sample_negatives(
        classes=sample_mimic_row["classes"],
        targets=targets,
        num_negatives=num_negatives,
        seed=42,
        hard_ratio=hard_ratio,
        hard_negatives=hard_negatives,
    )
    assert len(result) == expected_length

    # Check that all positives are included
    for target_list in targets:
        for target in target_list:
            assert target in result
            assert result[target] == sample_mimic_row["classes"][target]


@pytest.mark.parametrize(
    "targets,spans,expected",
    [(["428.0", "401.9"], [[(0, 35)], [(36, 60)]], [["428.0"], ["401.9"]]), (["428.0"], [[(0, 35)]], [["428.0"], []])],
)
def test_map_segments_to_targets_with_mimic_data(sample_segments, targets, spans, expected):
    result = map_segments_to_targets(sample_segments, targets, spans)
    assert result == expected


def test_sort_classes_alphabetically_with_mimic_data(sample_mimic_row):
    targets = [["428.0", "401.9"], ["250.00"], []]
    keys, values, sorted_targets = sort_classes_alphabetically(sample_mimic_row["classes"], targets, seed=42)

    # Check if keys are sorted
    assert keys == sorted(sample_mimic_row["classes"].keys())

    # Check if values correspond to sorted keys
    assert values == [sample_mimic_row["classes"][k] for k in sorted(sample_mimic_row["classes"].keys())]

    # Check if targets are properly mapped
    assert len(sorted_targets) == len(targets)
    assert all(isinstance(t, list) for t in sorted_targets)


def test_shuffle_classes_with_mimic_data(sample_mimic_row):
    targets = [["428.0", "401.9"], ["250.00"], []]
    keys, values, shuffled_targets = shuffle_classes(sample_mimic_row["classes"], targets, seed=42)

    # Check that all original keys are present
    assert set(keys) == set(sample_mimic_row["classes"].keys())

    # Check that values correspond to shuffled keys
    assert all(sample_mimic_row["classes"][k] == v for k, v in zip(keys, values))

    # Check that targets are properly mapped
    assert len(shuffled_targets) == len(targets)
    assert all(isinstance(t, list) for t in shuffled_targets)

    # Empty lists should be mapped to [0]
    assert shuffled_targets[2] == [0]


def test_dict_to_rich_table_with_mimic_data(sample_mimic_row):
    # Convert a subset of the MIMIC data to a table
    display_data = {
        "Subject ID": sample_mimic_row["subject_id"],
        "Hospital Admission ID": sample_mimic_row["hadm_id"],
        "Diagnosis Codes": ", ".join(sample_mimic_row["codes"]),
    }

    table = dict_to_rich_table(display_data, "MIMIC Patient Data")
    assert isinstance(table, Table)
    assert table.title == "MIMIC Patient Data"
    assert table.columns[0].header == "Feature"
    assert table.columns[1].header == "Value"
