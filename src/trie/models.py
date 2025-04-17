from collections import namedtuple
import typing
import pydantic
from pydantic_settings import SettingsConfigDict


class Root(pydantic.BaseModel):
    id: str
    name: str
    min: str
    max: str
    assignable: bool = False
    parent_id: str = pydantic.Field(default="")
    children_ids: list[str] = pydantic.Field(default_factory=list)

    model_config = SettingsConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="ignore"
    )


class Category(Root):
    id: str
    name: str
    min: str
    max: str
    description: str
    parent_id: str  # type: ignore
    children_ids: list[str] = pydantic.Field(default_factory=list)
    assignable: bool = False

    def __repr__(self) -> str:
        return f"{self.id} {self.min}-{self.max} {self.description}"

    def within(self, code: str) -> bool:
        return self.min[: len(code)] <= code[: len(self.max)] <= self.max[: len(code)]


class Code(Category):
    assignable: bool = True
    min: str = ""
    max: str = ""

    def __repr__(self) -> str:
        return f"{self.name} {self.description}"


class InstructionalNote(pydantic.BaseModel):
    """Represents an instructional note inspired by the ICD10CM"""

    name: str
    assignable: bool
    notes: list[str] | None = None
    includes: list[str] | None = None
    inclusion_term: list[str] | None = None
    excludes1: list[str] | None = None
    excludes2: list[str] | None = None
    use_additional_code: list[str] | None = None
    code_first: list[str] | None = None
    code_also: list[str] | None = None

    def is_empty(self) -> bool:
        """Check if the instructional note is empty."""
        return (
            not self.notes
            and not self.includes
            and not self.inclusion_term
            and not self.excludes1
            and not self.excludes2
            and not self.use_additional_code
            and not self.code_first
            and not self.code_also
        )


class Term(pydantic.BaseModel):
    """Represents a term in an alphabetic index."""

    id: str
    assignable: bool
    title: str
    code: str | None = None
    see: str | None = None
    see_also: str | None = None
    parent_id: str
    children_ids: list[str] = pydantic.Field(default_factory=list)


class ICDFileMap(pydantic.BaseModel):
    EXPECTED_FILES: typing.ClassVar[list[tuple[str, str]]] = [
        ("pcs_guidelines", "*pcs*guidelines*.pdf"),
        ("cm_guidelines", "*cm*guidelines*.pdf"),
        ("pcs_tabular", "*icd10pcs_tables*.xml"),
        ("pcs_index", "*icd10pcs_index*.xml"),
        ("cm_tabular", "icd10cm_tabular*.xml"),
        ("cm_neoplasm_index", "*icd10cm_neoplasm*.xml"),
        ("cm_disease_injuries_index", "*icd10cm_index*.xml"),
        ("cm_external_cause_index", "*icd10cm_eindex*.xml"),
        ("cm_drug_index", "*icd10cm_drug*.xml"),
    ]
    ALPHABETIC_INDEXES: typing.ClassVar[list[str]] = [
        "cm_neoplasm_index",
        "cm_disease_injuries_index",
        "cm_external_cause_index",
        "cm_drug_index",
        # "pcs_index",
    ]
    pcs_guidelines: pydantic.FilePath | None = pydantic.Field(
        default=None, description="Optional path to the PCS guidelines PDF"
    )
    cm_guidelines: pydantic.FilePath | None = pydantic.Field(
        default=None, description="Optional path to the CM guidelines PDF"
    )
    pcs_tabular: pydantic.FilePath | None = pydantic.Field(
        None, description="Path to the PCS tabular XML file"
    )
    pcs_index: pydantic.FilePath | None = pydantic.Field(
        default=None, description="Optional path to the PCS alphabetic index XML file"
    )
    cm_tabular: pydantic.FilePath = pydantic.Field(
        ..., description="Path to the CM tabular XML file"
    )
    cm_neoplasm_index: pydantic.FilePath | None = pydantic.Field(
        default=None, description="Optional path to the CM neoplasm index XML file"
    )
    cm_disease_injuries_index: pydantic.FilePath | None = pydantic.Field(
        default=None,
        description="Optional path to the CM disease and injuries index XML file",
    )
    cm_external_cause_index: pydantic.FilePath | None = pydantic.Field(
        default=None,
        description="Optional path to the CM external cause index XML file",
    )
    cm_drug_index: pydantic.FilePath | None = pydantic.Field(
        default=None, description="Optional path to the CM drug index XML file"
    )

    @classmethod
    def from_directory(cls, directory: pydantic.DirectoryPath) -> "ICDFileMap":
        def _get_best_match(self, file_pattern: str) -> pydantic.FilePath | None:
            """Return the updated version if `with_update` is True and available."""
            matches = list(directory.glob(file_pattern))
            if not matches:
                return None
            if len(matches) == 1:
                # return the file name with year match
                return matches[0]
            return matches[0]
            raise ValueError(
                f"Multiple files match the pattern '{file_pattern}'. Please specify a more specific pattern."
            )

        expected_files = {
            k: _get_best_match(directory, file_pattern=pattern)
            for k, pattern in cls.EXPECTED_FILES
        }

        missing = [k for k, v in expected_files.items() if not v]
        if missing:
            raise FileNotFoundError(
                f"Missing files. Check the download directory at {directory}. Missing: {', '.join(missing)}"
            )

        return cls(**expected_files)  # type: ignore


ICDCM_INDEX = [
    "cm_neoplasm_index",
    "cm_disease_injuries_index",
    "cm_external_cause_index",
    "cm_drug_index",
]


class PcsCategory(Category):
    """PCS Category."""


class PcsCode(Code):
    """PCS code."""


class PcsAxisLabel(pydantic.BaseModel):
    code: str
    label: str
    title: str


class PcsTableAxis(pydantic.BaseModel):
    pos: int
    code: str
    title: str
    label: str
    definition: str | None = None

    @pydantic.model_validator(mode="after")
    def combine_labels(self) -> "PcsTableAxis":
        if self.label and self.definition:
            self.label = f"{self.label} ({self.definition})"
        return self


class PcsAxis(pydantic.BaseModel):
    codes: int
    pos: int
    title: str
    labels: list[PcsAxisLabel]

    @pydantic.model_validator(mode="after")
    def validate_labels(self) -> "PcsAxis":
        if len(self.labels) != self.codes:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) does not match the expected number ({self.codes})."
            )
        return self


class PcsRow(pydantic.BaseModel):
    codes: int
    axes: list[PcsAxis]


class PcsTable(pydantic.BaseModel):
    table_id: str
    table_axes: list[PcsTableAxis]
    rows: list[PcsRow]


class CmCategory(Category, InstructionalNote):
    """CM Section."""


class CmCode(CmCategory, Code):
    """CM code."""

    assignable: bool = True
    min: str = ""
    max: str = ""
    etiology: bool
    manifestation: bool


SeventhCharacter = namedtuple("SeventhCharacter", ["character", "name", "parent_name"])


class CmElement(pydantic.BaseModel):
    notes: list[str] = []
    includes: list[str] = []
    inclusion_term: list[str] = []
    code_first: list[str] = []
    code_also: list[str] = []
    excludes1: list[str] = []
    excludes2: list[str] = []
    use_additional_code: list[str] = []


class CmDiag(CmElement):
    name: str
    desc: str
    seventh_characters: list[SeventhCharacter] = []
    children: list["CmDiag"] = []

    class Config:
        # allow forward reference
        arbitrary_types_allowed = True

    @pydantic.computed_field
    def manifestation(self) -> bool:
        return bool(self.code_first)

    @pydantic.computed_field
    def etiology(self) -> bool:
        return bool(self.use_additional_code)


class CmSection(CmElement):
    section_id: str
    description: str
    diags: list[CmDiag] = []
    first: str
    last: str

    @pydantic.model_validator(mode="after")
    def validate_section_id(self) -> "CmSection":
        if self.first == self.last:
            self.section_id = f"{self.first}-{self.last}"
        return self


class CmChapter(CmElement):
    chapter_id: str
    chapter_desc: str
    sections: list[CmSection] = []
    first: str
    last: str


class CmCell(pydantic.BaseModel):
    col: int
    heading: str
    code: str
    assignable: bool = True

    @pydantic.model_validator(mode="after")
    def validate_code(self) -> "CmCell":
        if isinstance(self.code, str):
            if "-" in self.code:
                self.assignable = False
            self.code = self.code.replace("-", "").rstrip(".")
        return self


class CmIndexTerm(pydantic.BaseModel):
    """Represents a <mainTerm> or a nested <term> in the ICD-10-CM Index."""

    title: str
    code: str | list[CmCell] | None = None
    see: str | None = None
    see_also: str | None = None
    sub_terms: list["CmIndexTerm"] = []
    assignable: bool = True

    @pydantic.model_validator(mode="after")
    def validate_code(self) -> "CmIndexTerm":
        if isinstance(self.code, str):
            if "-" in self.code:
                self.assignable = False
            self.code = self.code.replace("-", "").rstrip(".")
        return self


class CmLetter(pydantic.BaseModel):
    """Represents a <letter> block: e.g., <letter><title>A</title> ...</letter>."""

    letter: str
    main_terms: list[CmIndexTerm] = []
