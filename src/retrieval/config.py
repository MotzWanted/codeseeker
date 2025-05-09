import typing as typ

import pydantic

SearchBackend = typ.Literal["elasticsearch", "faiss", "qdrant"]


class BaseSearchFactoryConfig(pydantic.BaseModel):
    """Base config for all search engines."""

    backend: SearchBackend = pydantic.Field(..., description="Search backend to use.")
    subset_id_key: None | str = pydantic.Field(
        default="subset_id", description="Subset ID field to be indexed."
    )
    section_id_key: None | str = pydantic.Field(
        default="id", description="Section ID field to be indexed."
    )


class BaseSearchFactoryDiff(pydantic.BaseModel):
    """Relative search configs."""

    backend: SearchBackend
    group_key: None | str = None
    section_id_key: None | str = None


class QdrantFactoryDiff(BaseSearchFactoryDiff):
    """Configures a relative qdrant configuration."""

    backend: typ.Literal["qdrant"] = "qdrant"
    host: None | str = None
    port: None | int = None
    grpc_port: None | int = None
    persistent: None | bool = None
    exist_ok: None | bool = None
    qdrant_body: None | dict = None
    search_params: None | dict = None
    force_single_collection: None | bool = None


class QdrantFactoryConfig(BaseSearchFactoryConfig):
    """Configures the building of a Qdrant server."""

    host: str = "http://localhost"
    port: int = 6333
    grpc_port: None | int = 6334
    exist_ok: bool = True
