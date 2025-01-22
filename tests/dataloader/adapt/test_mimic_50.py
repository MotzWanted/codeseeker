import pytest

import dataloader
from dataloader.base import DatasetConfig, DatasetOptions
from dataloader.mimiciii.constants import MIMIC_III_50_PATH as mimiciii_50
from dataloader.mimiciv.constants import MIMIC_IV_50_PATH as mimiciv_50


@pytest.fixture
def mimic_iii_50_config():
    return DatasetConfig(identifier="mimic_iii_50", name_or_path=mimiciii_50, split="test")


@pytest.fixture
def mimic_iii_50_classes():
    return ["Congestive heart failure", "Hypertension", "Diabetes mellitus", "Hyperlipidemia"]


@pytest.fixture
def mimic_iv_50_config():
    return DatasetConfig(identifier="mimic_iv_50", name_or_path=mimiciv_50, split="test")


@pytest.fixture
def mimic_iii_50_dataset(dataset_options: DatasetOptions):
    mimic_iii_50_config.options = dataset_options
    return dataloader.load_dataset(mimic_iii_50_config)


@pytest.fixture
def mimic_iv_50_dataset(dataset_options: DatasetOptions):
    mimic_iv_50_config.options = dataset_options
    return dataloader.load_dxataset(mimic_iv_50_config)


@pytest.mark.parametrize(
    ("dataset_options", "expected_classes", "expected_order"),
    [
        (
            DatasetOptions(negatives=0, seed=42, order="alphabetical"),
            50,
        ),
    ],
)
def test_mimic_50(dataset_options: DatasetOptions, expected_classes: list[str], expected_order: list[str]):
    dset = mimic_iii_50_dataset(dataset_options)
    assert dset["classes"][0] == expected_classes
    for i, class_name in enumerate(expected_classes):
        assert dset["classes"][1][i] == class_name
