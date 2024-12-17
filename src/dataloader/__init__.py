from dataloader.meddec.constants import MEDDEC_PATH as meddec
from dataloader.snomed.constants import SNOMED_PATH as snomed
from dataloader.mdace.constants import MDACE_INPATIENT_PATH as mdace_inpatient
from dataloader.nbme.constants import NBME_PATH as nmbe  # noqa: F401
from dataloader.interface import load_dataset  # noqa: F401
from segmenters.base import factory

SEGMENTER = factory("document", spacy_model="en_core_web_lg")

DATASET_CONFIGS: dict[str, dict] = {
    "meddec": {
        "identifier": "meddec",
        "name_or_path": meddec,
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "snomed": {
        "identifier": "snomed",
        "name_or_path": snomed,
        "split": "validation",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-diagnosis-3": {
        "identifier": "mdace-diagnosis-3",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-3"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-diagnosis-4": {
        "identifier": "mdace-diagnosis-4",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-4"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-diagnosis-5": {
        "identifier": "mdace-diagnosis-5",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-5"],
        "split": "test",
        "options": {"segmenter": SEGMENTER},
    },
    "mdace-diagnosis-6": {
        "identifier": "mdace-diagnosis-6",
        "name_or_path": mdace_inpatient,
        "subsets": ["icd10cm-6"],
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
}
