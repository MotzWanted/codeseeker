from dataloader.interface import load_dataset  # noqa: F401
from dataloader.mdace.constants import MDACE_INPATIENT_PATH as mdace_inpatient
from dataloader.meddec.constants import MEDDEC_PATH as meddec
from dataloader.mimiciii.constants import MIMIC_III_50_PATH as mimiciii_50
from dataloader.mimiciv.constants import MIMIC_IV_50_PATH as mimiciv_50
from dataloader.mimiciv.constants import MIMIC_IV_PATH as mimiciv
from dataloader.nbme.constants import NBME_PATH as nmbe  # noqa: F401
from dataloader.snomed.constants import SNOMED_PATH as snomed
from segmenters.base import factory

SEGMENTER = factory("document", spacy_model="en_core_web_lg")

DATASET_CONFIGS: dict[str, dict] = {
    "debug": {
        "identifier": "debug",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.0"],
        "split": "test",
        "options": {"subset_size": 10},
    },
    "meddec": {"identifier": "meddec", "name_or_path": meddec, "split": "validation"},
    "snomed": {"identifier": "snomed", "name_or_path": snomed, "split": "validation"},
    "mdace-icd10cm": {
        "identifier": "mdace-icd10cm",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm"],
        "options": {"adapter": "MdaceAdapter"},
    },
    "mimic-iv": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.0"],
        "options": {"subset_size": 300},
    },
    "mimiciv-cm-3.0": {
        "identifier": "mimiciv-cm-3.0",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.0"],
        "options": {"subset_size": 1000},
    },
    "mimiciv-cm-3.1": {
        "identifier": "mimiciv-cm-3.1",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.1"],
        "options": {"subset_size": 1000},
    },
    "mimiciv-cm-3.2": {
        "identifier": "mimiciv-cm-3.2",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.2"],
        "options": {"subset_size": 1000},
    },
    "mimiciv-cm-3.3": {
        "identifier": "mimiciv-cm-3.3",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.3"],
        "options": {"subset_size": 1000},
    },
    "mimiciv-cm-3.4": {
        "identifier": "mimiciv-cm-3.4",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.4"],
        "options": {"subset_size": 1000},
    },
    "mimic-iii-50": {
        "identifier": "mimic-iii-50",
        "name_or_path": mimiciii_50,
        "split": "test",
        "options": {"order": "alphabetical"},
    },
    "mimic-iv-50": {
        "identifier": "mimic-iv-50",
        "name_or_path": mimiciv_50,
        "split": "test",
        "subsets": ["icd10"],
        "options": {
            "order": "alphabetical",
        },
    },
}
