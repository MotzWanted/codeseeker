import pydantic
from qdrant_client import models as qdrm


class FactoryConfig(pydantic.BaseModel):
    """Configures the building of a Qdrant server."""

    host: str = "http://localhost"
    port: int = 6333
    grpc_port: None | int = 6334


class CollectionBody(pydantic.BaseModel):
    """Collection config."""

    vectors_config: dict[str, qdrm.VectorParams]
    sparse_vectors_config: dict[str, qdrm.SparseVectorParams]
    hnsw_config: qdrm.HnswConfigDiff | None = None
    quantization_config: qdrm.QuantizationConfig | None = None
