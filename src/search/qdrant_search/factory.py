from collections import defaultdict
import gc
import hashlib
from re import escape
import typing
import uuid

import blake3
import cbor2
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from grpc import StatusCode
from loguru import logger
import numpy as np
import qdrant_client
from qdrant_client import models as qdrm
from qdrant_client.http.models.models import QueryResponse
from qdrant_client.http import exceptions as qdrexc
from grpc._channel import _InactiveRpcError

from rich.progress import track
from sentence_transformers import SentenceTransformer

from search.qdrant_search.client import QdrantSearchService
from search.qdrant_search.models import CollectionBody
from throughster.factory import create_interface
from throughster.base import ModelInterface


class CustomTextEmbedding:
    """Custom text embedding class."""

    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self._dim = dim

    @property
    def dim(self) -> int:
        """Get embedding dimension."""
        if self._dim is None:
            raise ValueError("Embedding dimension not initialized")
        return self._dim

    def query_embed(self, query: list[float]) -> list[float]:
        """Embed query text."""
        # if len(query) != self.dim:
        #     raise ValueError(f"Query must be of length {self.dim}. Got {len(query)}")
        return query

    def passage_embed(self, texts: list[float]) -> list[float]:
        """Embed passage text."""
        # if len(texts) != self.dim:
        #     raise ValueError(f"Passage must be of length {self.dim}. Got {len(texts)}")
        return texts


class SentenceTransformerEmbedding:
    """Wrapper for SentenceTransformer models to make them compatible with fastembed interface."""

    def __init__(self, model_name: str, cuda: bool = False):
        self.model_name = model_name
        self.device = "cuda" if cuda else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        """Get embedding dimension."""
        if self._dim is None:
            raise ValueError("Embedding dimension not initialized")
        return self._dim

    def query_embed(self, query: str | list[str]) -> list[np.ndarray]:
        """Embed query text."""
        if isinstance(query, str):
            query = [query]
        return self.model.encode(query, convert_to_numpy=True).tolist()

    def passage_embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed passage text."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()


class VllmEmbeddingClient:
    """Base class for coding agents."""

    def __init__(
        self,
        model_name: str,
        dim: int,
        api_base: str,
        **kwargs,
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.dim = dim
        self._client = None

    @property
    def client(self) -> ModelInterface:
        if self._client is None:
            self._client = create_interface(
                provider="vllm",
                endpoint="embeddings",
                api_base=self.api_base,
                use_cache=True,
                model_name=self.model_name,
            )
        return self._client

    def query_embed(self, query: str | list[str]) -> list[np.ndarray]:
        """Embed query text."""
        if isinstance(query, str):
            query = [query]
        return self.client.embed_sync(texts=query)

    def passage_embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed passage text."""
        return self.client.embed_sync(texts=texts)


ModelType = (
    SparseTextEmbedding
    | TextEmbedding
    | LateInteractionTextEmbedding
    | SentenceTransformerEmbedding
    | VllmEmbeddingClient
    | CustomTextEmbedding
)


def _collection_name(collection_name: str, escape_rich: bool = True) -> str:
    """Format the collection name."""
    escape_fn = escape if escape_rich else lambda x: x
    return f"[bold cyan]Qdrant{escape_fn(f'[{collection_name}]')}[/bold cyan]"  # type: ignore


def format_passage_vectors(
    batch: list[dict[str, typing.Any]],
    text_key: str,
    models: list[ModelType],
) -> dict[str, list]:
    vector_dict = {}
    texts = [row[text_key] for row in batch]
    for embed_fn in models:
        if isinstance(embed_fn, TextEmbedding):
            vector_dict[embed_fn.model_name] = list(embed_fn.passage_embed(texts))
        elif isinstance(embed_fn, SparseTextEmbedding):
            vector_dict[embed_fn.model_name] = [
                v.as_object() for v in embed_fn.passage_embed(texts)
            ]
        elif isinstance(embed_fn, LateInteractionTextEmbedding):
            vector_dict[embed_fn.model_name] = list(embed_fn.passage_embed(texts))
        elif isinstance(embed_fn, SentenceTransformerEmbedding):
            vector_dict[embed_fn.model_name] = embed_fn.passage_embed(texts)
        elif isinstance(embed_fn, VllmEmbeddingClient):
            vector_dict[embed_fn.model_name] = embed_fn.passage_embed(texts)
        elif isinstance(embed_fn, CustomTextEmbedding):
            embeddings = [row["dense_embeddings"] for row in batch]
            vector_dict[embed_fn.model_name] = embed_fn.passage_embed(embeddings)  # type: ignore
        else:
            raise ValueError(f"Unsupported embedding function type: {type(embed_fn)}")
    return vector_dict


def format_query_vectors(
    row: dict[str, typing.Any],
    text_key: str,
    models: list[ModelType],
) -> dict[str, list]:
    vector_dict = defaultdict(list)
    for embed_fn in models:
        if isinstance(embed_fn, TextEmbedding):
            vector = embed_fn.query_embed(row[text_key])
            vector_dict[embed_fn.model_name].extend(list(vector))
        elif isinstance(embed_fn, SparseTextEmbedding):
            sparse_vec = embed_fn.query_embed(row[text_key])
            vector_dict[embed_fn.model_name].extend([v.as_object() for v in sparse_vec])
        elif isinstance(embed_fn, LateInteractionTextEmbedding):
            vectors = embed_fn.query_embed(row[text_key])
            vector_dict[embed_fn.model_name].extend([list(v) for v in vectors])
        elif isinstance(embed_fn, SentenceTransformerEmbedding):
            vector = embed_fn.query_embed(row[text_key])
            vector_dict[embed_fn.model_name].extend(vector)
        elif isinstance(embed_fn, VllmEmbeddingClient):
            vector = embed_fn.query_embed(row[text_key])
            vector_dict[embed_fn.model_name].extend(vector)
        elif isinstance(embed_fn, CustomTextEmbedding):
            vector = embed_fn.query_embed(row["dense_embeddings"])
            vector_dict[embed_fn.model_name].extend(vector)
        else:
            raise ValueError(f"Unsupported embedding function type: {type(embed_fn)}")
    return vector_dict


def _canon(obj: typing.Any) -> bytes:
    """Canonical CBOR dump — guarantees identical bytes across runs."""
    return cbor2.dumps(obj, canonical=True)


def _hash_rows(rows: list[dict[str, typing.Any]]) -> str:
    """
    Hash the *identity* of the rows, not their order.
    Uses the `id` field only (adjust if you prefer more).
    """
    h = blake3.blake3()
    for r in sorted(rows, key=lambda r: r["id"]):
        h.update(str(r["id"]).encode())
    return h.hexdigest()


def _hash_models(model_collection) -> str:
    """
    Stable digest for the embedding ensemble.
    Works for *your* concrete classes; tweak if you add fields.
    """
    serial = [
        {
            "cls": m.__class__.__name__,
            "model_name": m.model_name,
            **getattr(m, "_fingerprint_kwargs", {}),
        }
        for m in model_collection
    ]
    return hashlib.md5(_canon(serial)).hexdigest()


def make_index_fingerprint(
    *,
    data: list[dict[str, typing.Any]],
    model_collection,
    qdrant_body: dict[str, typing.Any],
    text_key: str,
    salt: bytes = b"codeseeker",
    dlen: int = 16,
) -> str:
    """
    Compute a 16‑byte (32‑hex) fingerprint that changes iff
    • models change
    • index blueprint changes
    • data *content* or payload schema changes
    """
    h = blake3.blake3()
    h.update(_canon({"model_spec": _hash_models(model_collection)}))
    h.update(_canon({"qdrant_body": qdrant_body}))
    h.update(_canon({"data_rows": _hash_rows(data)}))
    h.update(_canon({"text_key": text_key}))
    h.update(salt)
    return h.digest(length=dlen).hex()


def index_exists(
    service: QdrantSearchService,
    collection_name: str,
    exist_ok: bool = True,
) -> bool:
    """
    Check if a collection exists in Qdrant.
    """
    index_exist = _collection_exists(service.client, collection_name)

    if index_exist and not exist_ok:
        logger.info(f"Collection `{collection_name}` already exists. Dropping it.")
        service.drop_collection(collection_name)
        index_exist = False

    return index_exist


def build_qdrant_index(
    service: QdrantSearchService,
    collection_name: str,
    config: CollectionBody,
    points: list[qdrm.PointStruct],
    batch_size: int = 16,
) -> None:
    service.create_collection(collection_name=collection_name, body=config)
    # service.client.create_payload_index(
    #     collection_name=collection_name,
    #     field_name="aid",
    #     field_schema="keyword",  # type: ignore
    # )
    for j in track(
        range(0, len(points), batch_size),
        description=f"Building Qdrant Index `{collection_name}`",
        total=len(points) // batch_size,
    ):
        service.upsert_points(collection_name, points[j : j + batch_size])

    gc.collect()


def _collection_exists(
    client: qdrant_client.QdrantClient, collection_name: str
) -> bool:
    try:
        client.get_collection(collection_name=collection_name)
        index_exist = True
    except qdrexc.UnexpectedResponse as exc:
        if exc.status_code != 404:  # noqa: PLR2004
            raise Exception(f"Unexpected error: {exc}") from exc
        index_exist = False

    except _InactiveRpcError as exc:
        if exc.code() != StatusCode.NOT_FOUND:
            raise Exception(f"Unexpected error: {exc}") from exc
        index_exist = False
    return index_exist


def make_qdrant_body(
    model_collection: list[ModelType],
    distance: str,
    hnsw: dict[str, int] | None = None,
    **kwargs,
) -> CollectionBody:
    """Make the default Qdrant configuration body."""
    body = {}
    body["vectors_config"] = {}
    body["sparse_vectors_config"] = {}
    for model in model_collection:
        if isinstance(model, LateInteractionTextEmbedding):
            body["vectors_config"][model.model_name] = {
                "size": 128,  # type: ignore
                "distance": distance,
                "multivector_config": {"comparator": "max_sim"},
            }
        elif isinstance(model, SparseTextEmbedding):
            body["sparse_vectors_config"][model.model_name] = {"modifier": "idf"}
        else:
            body["vectors_config"][model.model_name] = {
                "size": int(model.dim),  # type: ignore
                "distance": distance,
            }
    if hnsw:
        body["hsnw_config"] = hnsw

    return CollectionBody(**body)


def format_qdrant_point(
    data: list[dict[str, typing.Any]],
    model_collection: list[ModelType],
    text_key: str,
    payload_keys: list[str] = [],
    batch_size: int = 64,
) -> list[qdrm.PointStruct]:
    points = []
    for j in track(
        range(0, len(data), batch_size),
        total=len(data) // batch_size,
        description="Formatting Qdrant point structs...",
    ):
        batch = data[j : j + batch_size]
        payloads = [
            {k: v for k, v in row.items() if k in payload_keys} for row in batch
        ]
        vectors = format_passage_vectors(batch, text_key, model_collection)
        if any(len(list(v)) != len(payloads) for v in vectors.values()):
            raise ValueError("Payloads and Vectors must have same length.")
        points.extend(
            [
                qdrm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={k: v[i] for k, v in vectors.items()},
                    payload=payloads[i],
                )
                for i in range(len(batch))
            ]
        )
    return points


def prefetch_iterator(
    dataset: typing.Iterable[dict],
    model_collection: list[ModelType],
    query_config: list[dict[str, str]],
    limit: int,
):
    for sample in dataset:
        query_vectors = defaultdict(list)
        for c in query_config:
            query_vectors.update(
                format_query_vectors(
                    sample,
                    c["query_key"],
                    [m for m in model_collection if m.model_name == c["model_name"]],
                )
            )
        yield [
            qdrm.Prefetch(query=v, using=model_key, limit=limit)
            for model_key, vecs in query_vectors.items()
            for v in vecs
        ]


def instantiate_models(model_config: list[dict[str, str]]) -> list[ModelType]:
    """Instantiate the models described in the config."""
    models = []
    for m in model_config:
        if m["type"] == "dense":
            models.append(TextEmbedding(m["model_name"], cuda=False))
        elif m["type"] == "sparse":
            models.append(SparseTextEmbedding(m["model_name"], cuda=False))
        elif m["type"] == "st":  # sentence-transformer
            models.append(SentenceTransformerEmbedding(m["model_name"]))
        elif m["type"] == "late_interaction":
            models.append(LateInteractionTextEmbedding(m["model_name"], cuda=False))
        elif m["type"] == "vllm":
            models.append(
                VllmEmbeddingClient(
                    model_name=m["model_name"],
                    api_base=m["api_base"],
                    dim=int(m["dim"]),
                )
            )
        elif m["type"] == "custom":
            models.append(
                CustomTextEmbedding(
                    model_name=m["model_name"],
                    dim=int(m["dim"]),
                )
            )
        else:
            raise ValueError(f"Unknown model type {m['type']}")
    return models


def ensure_qdrant_index(
    *,
    data: list[dict],
    text_key: str,
    model_cfg: list[dict[str, str]],
    hnsw_cfg: dict[str, int],
    distance: str,
    service: QdrantSearchService,
    payload_keys: list[str] | None = None,
) -> str:
    """Build the index if it is missing and return the fingerprint."""
    models = instantiate_models(model_cfg)
    qdrant_body = make_qdrant_body(
        model_collection=models, distance=distance, hnsw=hnsw_cfg
    )
    fingerprint = make_index_fingerprint(
        data=data,
        model_collection=models,
        qdrant_body=qdrant_body.model_dump(),
        text_key=text_key,
    )

    if not index_exists(service, fingerprint, exist_ok=True):
        points = format_qdrant_point(
            data=data,
            model_collection=models,
            text_key=text_key,
            payload_keys=payload_keys or [],
        )
        build_qdrant_index(service, fingerprint, qdrant_body, points)

    return fingerprint


def search(
    *,
    data: list[dict],
    model_cfg: list[dict[str, str]],
    service: QdrantSearchService,
    index_name: str,
    limit: int,
    fusion: qdrm.Fusion = qdrm.Fusion.RRF,
    timeout: int = 300,
) -> list[QueryResponse]:
    """Execute one query per element produced by `prefetch_iterator`.
    Returns the raw Qdrant responses so the caller can slice or post-process.
    """
    model_collection = instantiate_models(model_cfg)
    prefetch_iter = prefetch_iterator(
        data,
        model_collection,
        model_cfg,
        limit,
    )
    results: list[QueryResponse] = []
    for query in track(
        prefetch_iter,
        total=len(data),
        description=f"Querying Qdrant index {index_name}",
    ):
        res = service.client.query_points(
            index_name,
            prefetch=query,
            query=qdrm.FusionQuery(fusion=fusion),
            limit=limit,
            with_payload=True,
            timeout=timeout,
        )
        results.append(res)
    return results
