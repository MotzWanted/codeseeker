from dataloader.interface import load_dataset  # noqa: F401
from dataloader.mdace.constants import MDACE_INPATIENT_PATH as mdace_inpatient
from dataloader.meddec.constants import MEDDEC_PATH as meddec
from dataloader.mimiciii.constants import MIMIC_III_50_PATH as mimiciii_50  # noqa: F401
from dataloader.mimiciv.constants import MIMIC_IV_50_PATH as mimiciv_50  # noqa: F401
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
        "options": {"subset_size": 10, "segmenter": SEGMENTER},
    },
    "meddec": {
        "identifier": "meddec",
        "name_or_path": meddec,
        "split": "validation",
        "options": {"segmenter": SEGMENTER},
    },
    "snomed": {
        "identifier": "snomed",
        "name_or_path": snomed,
        "split": "validation",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.0": {
        "identifier": "mdace-icd10cm-3.0",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.0"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.1": {
        "identifier": "mdace-icd10cm-3.1",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.1"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.2": {
        "identifier": "mdace-icd10cm-3.2",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.2"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.3": {
        "identifier": "mdace-icd10cm-3.3",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.3"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-icd10cm-3.4": {
        "identifier": "mdace-icd10cm-3.4",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3.4"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-diagnosis-7": {
        "identifier": "mdace-diagnosis-7",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-7"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-diagnosis": {
        "identifier": "mdace-diagnosis",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3", "icd10cm-4", "icd10cm-5", "icd10cm-6", "icd10cm-7"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-procedure-4": {
        "identifier": "mdace-procedure-4",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10pcs-4"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-procedure-5": {
        "identifier": "mdace-procedure-5",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10pcs-5"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-procedure-6": {
        "identifier": "mdace-procedure-6",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10pcs-6"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-procedure-7": {
        "identifier": "mdace-procedure-7",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10pcs-7"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mimic-iv": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.0"],
        "options": {"subset_size": 300, "segmenter": SEGMENTER, "adapter": "MimicIvInferenceAdapter"},
    },
    "mimiciv-cm-3.0": {
        "identifier": "mimiciv-cm-3.0",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.0"],
        "options": {"subset_size": 1000, "segmenter": SEGMENTER, "adapter": "MimicIvInferenceAdapter"},
    },
    "mimiciv-cm-3.1": {
        "identifier": "mimiciv-cm-3.1",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.1"],
        "options": {"subset_size": 1000, "segmenter": SEGMENTER, "adapter": "MimicIvInferenceAdapter"},
    },
    "mimiciv-cm-3.2": {
        "identifier": "mimiciv-cm-3.2",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.2"],
        "options": {"subset_size": 1000, "segmenter": SEGMENTER, "adapter": "MimicIvInferenceAdapter"},
    },
    "mimiciv-cm-3.3": {
        "identifier": "mimiciv-cm-3.3",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.3"],
        "options": {"subset_size": 1000, "segmenter": SEGMENTER, "adapter": "MimicIvInferenceAdapter"},
    },
    "mimiciv-cm-3.4": {
        "identifier": "mimiciv-cm-3.4",
        "name_or_path": mimiciv,
        "split": "test",
        "subsets": ["icd10cm-3.4"],
        "options": {"subset_size": 1000, "segmenter": SEGMENTER, "adapter": "MimicIvInferenceAdapter"},
    },
}
