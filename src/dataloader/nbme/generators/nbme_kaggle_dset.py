"""nbme: A public clinical patient note dataset from from the USMLE Step 2 Clinical Skills examination."""

# downloaded from https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/overview
import pydantic
import datasets
from loguru import logger
import typing as typ
import pandas as pd
import ast
from pathlib import Path
import numpy as np


class NmbeAnnotationModel(pydantic.BaseModel):
    """Model for a clinical patient note annotation."""

    pn_num: int
    case_num: int
    feature_num: list[int]
    annotation: list[list[str]]
    location: list[list[str]]


class NmbePatientNoteModel(pydantic.BaseModel):
    """Model for a clinical patient note from the USMLE Step 2 Clinical Skills exam."""

    pn_num: int
    case_num: int
    patient_note: str
    features: dict[int, str]
    case_features: dict[int, str]
    labels: NmbeAnnotationModel | None

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


_CITATION = """
@inproceedings{yaneva2024automated,
  title={Automated Scoring of Clinical Patient Notes: Findings From the Kaggle Competition and Their Translation into Practice},
  author={Yaneva, Victoria and Suen, King Yiu and Mee, Janet and Quranda, Milton and Harik, Polina and others},
  booktitle={Proceedings of the 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2024)},
  pages={87--98},
  year={2024}
}
"""  # noqa: E501

_DESCRIPTION = """
nbme-score-clinical-patient-notes: a corpus of 43,985 clinical patient notes (PNs) written by 35,156 examinees during
the high-stakes USMLE Step 2 Clinical Skills examination. In this exam, examinees interact with standardized patients
- people trained to portray simulated scenarios called clinical cases. For each encounter, an examinee writes a PN,
which is then scored by physician raters using a rubric of clinical concepts, expressions of which should be present in
the PN. The corpus features PNs from 10 clinical cases, as well as the clinical concepts from the case rubrics. A subset
of 2,840 PNs were annotated by 10 physician experts such that all 143 concepts from the case rubrics (e.g., shortness of
 breath) were mapped to 34,660 PN phrases (e.g., dyspnea, difficulty breathing). The corpus is available via a data
 sharing agreement with NBME and can be requested at https://www.nbme.org/services/data-sharing.
"""

# _URL = Path("~/Desktop/nbme-score-clinical-patient-notes/").expanduser()
_URL = Path("/nfs/nas/mlrd/datasets/nbme-patient-notes/")


class NbmePatientNotesConfig(datasets.BuilderConfig):
    """BuilderConfig for nbme-score-clinical-patient-notes."""

    def __init__(self, **kwargs: typ.Any):
        """BuilderConfig for MIMIC-III-50.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NbmePatientNotesConfig, self).__init__(**kwargs)


class NbmePatientNotes(datasets.GeneratorBasedBuilder):
    """nbme-score-clinical-patient-notes: a corpus of clinical patient notes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    BUILDER_CONFIGS = [
        NbmePatientNotesConfig(
            name="nbme-patient-notes",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "pn_num": datasets.Value("int64"),
                    "case_num": datasets.Value("int64"),
                    "patient_note": datasets.Value("string"),
                    "features": datasets.Value("string"),
                    "case_features": datasets.Value("string"),
                    "labels": datasets.Features(
                        {
                            "pn_num": datasets.Value("int64"),
                            "case_num": datasets.Value("int64"),
                            "feature_num": datasets.features.Sequence(datasets.Value("int64")),
                            "annotation": datasets.features.Sequence(
                                datasets.features.Sequence(datasets.Value("string"))
                            ),
                            "location": datasets.features.Sequence(
                                datasets.features.Sequence(datasets.Value("string"))
                            ),
                        }
                    ),
                }
            ),
            citation=_CITATION,
        )

    def _load_and_merge_data(self) -> pd.DataFrame:
        """Post-process the data."""
        patient_notes = pd.read_csv(_URL / "patient_notes.csv")
        features = pd.read_csv(_URL / "features.csv")
        train = pd.read_csv(_URL / "train.csv")

        # Merge features with train on feature_num and case_num to get feature_text in the annotations
        train_features = pd.merge(train, features, on=["feature_num", "case_num"], how="left")

        # Group the annotations and locations by pn_num and case_num
        grouped_annotations = (
            train_features.groupby(["pn_num", "case_num"])
            .agg(
                {
                    "feature_num": list,
                    "feature_text": list,
                    "annotation": list,
                    "location": list,
                }
            )
            .reset_index()
        )

        # Create a list of dictionaries for labels containing pn_num, feature_num, case_num, annotation, and location
        grouped_annotations["labels"] = grouped_annotations.apply(
            lambda x: {
                "pn_num": x["pn_num"],
                "case_num": x["case_num"],
                "feature_num": [el for el in x["feature_num"]],
                "annotation": [eval(el) if isinstance(el, str) else el for el in x["annotation"]],
                "location": [eval(el) if isinstance(el, str) else el for el in x["location"]],
            },
            axis=1,
        )

        # Merge grouped_annotations with patient_notes on pn_num and case_num
        merged_data = pd.merge(
            patient_notes,
            grouped_annotations[["pn_num", "case_num", "labels"]],
            on=["pn_num", "case_num"],
            how="left",
        )

        # Create a dictionary of feature_num and feature_text
        features_dict = features.set_index("feature_num")["feature_text"].to_dict()

        # Add the features dictionary to each row in merged_data
        merged_data["case_features"] = merged_data["case_num"].apply(
            lambda x: {fn: features_dict[fn] for fn in features[features["case_num"] == x]["feature_num"].values}
        )
        merged_data["features"] = merged_data.apply(lambda x: features_dict, axis=1)

        # Select and rename columns to match the desired output format
        final_data = merged_data[["pn_num", "case_num", "pn_history", "features", "case_features", "labels"]]
        final_data.columns = [
            "pn_num",
            "case_num",
            "patient_note",
            "features",
            "case_features",
            "labels",
        ]

        return final_data

    def _split_into_train_test(self, full_data: pd.DataFrame) -> typ.Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into train and test."""
        case_nums = full_data["case_num"].unique()
        np.random.shuffle(case_nums)  # Randomly shuffle the case numbers

        train_cases = case_nums[:5]
        test_cases = case_nums[5:]

        train_data = full_data[full_data["case_num"].isin(train_cases)]
        train_data.loc[:, "labels"] = train_data["labels"].apply(lambda x: None if pd.isna(x) else x)
        test_data = full_data[full_data["case_num"].isin(test_cases)]
        filtered_test_data = test_data[test_data["labels"].apply(lambda x: isinstance(x, dict) and len(x) > 0)]

        return train_data, filtered_test_data

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> typ.List[datasets.SplitGenerator]:  # type: ignore
        """Return SplitGenerators."""
        logger.info("Generating splits from `{}`.", _URL)
        full_data = self._load_and_merge_data()
        train, test = self._split_into_train_test(full_data)

        return [
            datasets.SplitGenerator(name=str(datasets.Split.TRAIN), gen_kwargs={"df": train}),
            datasets.SplitGenerator(name=str(datasets.Split.TEST), gen_kwargs={"df": test}),
        ]

    def _generate_examples(  # type: ignore
        self,
        df: pd.DataFrame,
    ) -> typ.Generator[typ.Tuple[typ.Hashable, typ.Dict[str, typ.Any]], None, None]:
        for i, row in df.iterrows():
            yield (
                i,
                NmbePatientNoteModel(
                    **{
                        "pn_num": row["pn_num"],
                        "case_num": row["case_num"],
                        "patient_note": row["patient_note"],
                        "features": row["features"],
                        "case_features": row["case_features"],
                        "labels": row["labels"],
                    }
                ).model_dump(),
            )
