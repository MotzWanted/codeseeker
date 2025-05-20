import copy
import os
import pathlib
import subprocess
import time
import typing
from typing import Any

import loguru
import numpy as np
import qdrant_client
from grpc._channel import _InactiveRpcError

from qdrant_client import models as qdrm
from qdrant_client.http import exceptions as qdrexc

from retrieval.qdrant_search.models import CollectionBody


def _init_client(
    host: str, port: int, grpc_port: None | int, **kwargs: Any
) -> qdrant_client.QdrantClient:
    """Initialize the client."""
    try:
        return qdrant_client.QdrantClient(
            url=host,
            port=port,
            grpc_port=grpc_port or -1,
            prefer_grpc=grpc_port is not None,
        )
    except Exception as exc:
        raise Exception(
            f"Qdrant client failed to initialize. "
            f"Have you started the server at {f'{host}:{port}'}? "
            f"Start it with `docker run -p {port}:{port} qdrant/qdrant`"
        ) from exc


class F:
    """Building blocks for Qdrant filters"""

    @staticmethod
    def eq(field: str, value: typing.Any) -> qdrm.FieldCondition:
        return qdrm.FieldCondition(key=field, match=qdrm.MatchValue(value=value))

    @staticmethod
    def range(
        field: str, gte: float | int | None = None, lte: float | int | None = None
    ) -> qdrm.FieldCondition:
        return qdrm.FieldCondition(key=field, range=qdrm.Range(gte=gte, lte=lte))

    @staticmethod
    def and_(*conds: qdrm.FieldCondition) -> qdrm.Filter:
        return qdrm.Filter(must=list(conds))

    @staticmethod
    def or_(*conds: qdrm.FieldCondition) -> qdrm.Filter:
        return qdrm.Filter(should=list(conds))


class QdrantSearchService:
    """A client to interact with a search server."""

    _timeout: float = 300

    def __init__(
        self,
        host: str,
        port: int,
        grpc_port: None | int,
    ):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self._client = None
        self._spawn_service

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.host}:{self.port}]"

    def _make_cmd(self) -> list[str]:
        return [
            "docker",
            "run",
            "-p",
            f"{self.port}:{self.port}",
            "-p",
            f"{self.grpc_port}:{self.grpc_port}",
            "qdrant/qdrant",
        ]

    def _make_env(self) -> dict[str, typing.Any]:
        env = copy.copy(dict(os.environ))
        env["QDRANT__SERVICE__HTTP_PORT"] = str(self.port)
        if self.grpc_port is not None:
            env["QDRANT__SERVICE__GRPC_PORT"] = str(self.grpc_port)
        return env

    def _spawn_service(self) -> None | subprocess.Popen:
        cmd = self._make_cmd()
        env = self._make_env()
        stdout_file = pathlib.Path(f"{self.__repr__}.stdout.log")
        loguru.logger.debug(f"Writing stdout to `{stdout_file.absolute()}`")
        if stdout_file.exists():
            stdout_file.unlink()
        stderr_file = pathlib.Path(f"{self.__repr__}.stderr.log")
        loguru.logger.debug(f"Writing stderr to `{stderr_file.absolute()}`")
        if stderr_file.exists():
            stderr_file.unlink()
        server_proc = subprocess.Popen(
            cmd,  # noqa: S603
            env=env,
            stdout=stdout_file.open("w"),
            stderr=stderr_file.open("w"),
        )

        t0 = time.time()
        loguru.logger.info(f"Spawning {self.__repr__} ...")
        while not self.ping():
            time.sleep(0.1)
            if time.time() - t0 > self._timeout:
                server_proc.terminate()
                raise TimeoutError(
                    f"Couldn't ping the server after {self._timeout:.0f}s."
                )
        loguru.logger.debug(f"Spawned {self.__repr__} in {time.time() - t0:.1f}s.")
        return server_proc

    def size(self, index_name: str) -> int:
        """Return the number of vectors in the index."""
        return self.client.count(collection_name=index_name).count

    def ping(self) -> bool:
        """Ping the server."""
        try:
            self.client.get_collections()
            return True
        except qdrexc.UnexpectedResponse:
            return False
        except _InactiveRpcError:
            return False

    def __getstate__(self) -> object:
        """Return the state."""
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Set the state."""
        self.__dict__.update(state)

    @property
    def client(self) -> qdrant_client.QdrantClient:
        if self._client is None:
            self._client = _init_client(self.host, self.port, self.grpc_port)
        return self._client

    def create_collection(self, *, collection_name: str, body: CollectionBody) -> None:
        """Create a new collection."""
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=body.vectors_config,
            sparse_vectors_config=body.sparse_vectors_config,
            hnsw_config=body.hnsw_config,
            quantization_config=body.quantization_config,
        )

    def drop_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name=collection_name)

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def delete_points(self, collection_name: str, ids: list[int]) -> None:
        self.client.delete(
            collection_name=collection_name,
            points_selector=qdrm.PointIdsList(points=ids),  # type: ignore
        )

    def snapshot(
        self, collection_name: str, name: str | None = None
    ) -> dict[str, Any] | None:
        s = self.client.create_snapshot(collection_name, snapshot_name=name)
        return s.model_dump() if s else None  # REST returns JSON ↔ gRPC returns Proto

    def recover_snapshot(self, collection_name, snapshot_name: str) -> None:
        self.client.recover_snapshot(
            collection_name=collection_name,
            location=snapshot_name,
        )

    def set_optimizer(self, collection_name: str, **cfg: typing.Any) -> None:
        """
        Example:
            client.set_optimizer(
                indexing_threshold=20_000,
                m=32,              # HNSW
                ef_construct=256,
                payload_m=16       # HNSW for payload index
            )
        """
        self.client.update_collection(
            collection_name=collection_name,
            optimizers_config=qdrm.OptimizersConfigDiff(**cfg),
        )

    def upsert_points(
        self,
        collection_name: str,
        points: list[qdrm.PointStruct],
        wait: bool = True,
    ) -> None:
        """Upsert points into the index."""
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=wait,
        )

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int = 256,
        filt: qdrm.Filter | None = None,
        with_payload: bool = False,
    ) -> list[qdrm.Record]:
        scroll_res, _ = self.client.scroll(
            collection_name, limit=limit, filter=filt, with_payload=with_payload
        )
        return scroll_res

    def search(
        self,
        *,
        collection_name: str,
        vector: np.ndarray,
        limit: int = 10,
        filter_: qdrm.Filter | None = None,
        with_payload: bool = True,
        score_threshold: float | None = None,
        params: qdrm.SearchParams | None = None,
    ) -> list[qdrm.ScoredPoint]:
        return self.client.search(
            collection_name=collection_name,
            query_vector=vector.tolist(),
            limit=limit,
            query_filter=filter_,
            with_payload=with_payload,
            score_threshold=score_threshold,
            search_params=params,
        )

    def recommend(
        self,
        *,
        collection_name: str,
        positive: list[int | list[float]],
        negative: list[int | list[float]] | None = None,
        limit: int = 10,
        filter_: qdrm.Filter | None = None,
        with_payload: bool = True,
        strategy: qdrm.RecommendStrategy = qdrm.RecommendStrategy.AVERAGE_VECTOR,
        using: str | None = None,
        lookup_from: qdrm.LookupLocation | None = None,
    ) -> list[qdrm.ScoredPoint]:
        """Vector-only recommendation — lists of positives / negatives."""
        return self.client.recommend(
            collection_name=collection_name,
            positive=positive,
            negative=negative,
            limit=limit,
            filter=filter_,
            with_payload=with_payload,
            strategy=strategy,
            using=using,
            lookup_from=lookup_from,
        )

    def discover(
        self,
        *,
        collection_name: str,
        context: list[qdrm.ContextExamplePair],
        target: list[float] | None = None,
        limit: int = 10,
        filter_: qdrm.Filter | None = None,
        with_payload: bool = True,
        using: str | None = None,
    ) -> list[qdrm.ScoredPoint]:
        """NOTE: Discovery search requires Qdrant>=1.7

        contextis a list of ContextPair(positive, negative)
        target is optional provide it for discovery search, leave it None for pure context search.
        """
        return self.client.discover(
            collection_name=collection_name,
            context=context,
            target=target,
            limit=limit,
            filter=filter_,
            with_payload=with_payload,
            using=using,
        )
