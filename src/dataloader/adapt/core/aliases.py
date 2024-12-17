import pydantic

ID_ALIASES = pydantic.AliasChoices(
    "id",
    "uid",
)
TEXT_ALIASES = pydantic.AliasChoices(
    "transcript",
    "text",
)
TARGET_ALIASES = pydantic.AliasChoices(
    "target",
    "summary",
    "groundtruth_summary",
    "final_summary",
)
