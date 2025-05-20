from collections import OrderedDict
import json
import typing as typ

import pydantic
import rich
from loguru import logger
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from agents.analyse_agent import create_analyse_agent
from dataloader.base import DatasetConfig
import config as cnf
from qdrant_client.http.models.models import QueryResponse

import hashlib
from pathlib import Path

import datasets


from dataloader import load_dataset, DATASET_CONFIGS

import utils as exp_utils
from retrieval.qdrant_search import client as qdrant_client
from retrieval.qdrant_search import models as qdrant_models
from retrieval.qdrant_search import factory as qdrant_factory


class Arguments(pydantic.BaseModel):

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    base_model: dict[str, typ.Any] = {
        "provider": "vllm",
        "deployment": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "api_base": "http://localhost:6539/v1",
        "endpoint": "completions",
        "use_cache": True,
    }

    analyse_agent: dict[str, typ.Any] = {
        "agent_type": "base",
        "prompt_name": "analyse_agent/base_v1",
    }
    temperature: float = 0.0
    max_tokens: int = 10_000
    seed: int = 1  # e.g., "1:2:3:4:5"
    batch_size: int = 2
    num_workers: int = 4
    embed_config: list[dict[str, str]] = [
        {
            "type": "st",
            "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "query_key": "output",
        }
    ]

    qdrant_config: qdrant_models.FactoryConfig = qdrant_models.FactoryConfig()
    distance: str = "Cosine"
    hnsw: dict[str, int] = {"m": 32, "ef_construct": 256}

    rank: int = 25

    experiment_id: str = "search-terms-per-query"
    experiment_name: str = "S-PubMedBert-MS-MARCO"

    debug: bool = False

    def get_hash(self) -> str:
        """Create unique identifier for the arguments"""
        model_dict = self.model_dump(exclude={"experiment_id", "experiment_name"})
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    def get_experiment_folder(self) -> Path:
        """Get the experiment folder path."""
        return (
            cnf.DUMP_FOLDER
            / f"{self.experiment_id}/{self.experiment_name}/{self.get_hash()}"
        )


def format_eval_data(
    mdace: datasets.Dataset,
    retrieved_terms: list[QueryResponse],
) -> datasets.Dataset:
    """Format the evaluation data."""
    return datasets.Dataset.from_dict(
        {
            "retrieved_terms": [
                [p.payload for p in res.points] for res in retrieved_terms
            ],
            "targets": mdace["targets"],
            "note": mdace["note"],
            "note_type": mdace["note_type"],
            "aid": mdace["aid"],
        }
    )


def run(args: Arguments) -> None:
    rich.print(args)
    args.get_experiment_folder().mkdir(parents=True, exist_ok=True)
    with open(args.get_experiment_folder() / "config.json", "w") as f:  # type: ignore
        f.write(args.model_dump_json(indent=2))

    qdrant_service = qdrant_client.QdrantSearchService(
        **args.qdrant_config.model_dump()
    )

    xml_trie = exp_utils.build_icd_trie(year=2022)
    icd10cm = [code.name for code in xml_trie.get_root_codes("cm")]
    eval_trie: dict[str, int] = OrderedDict(
        {code: idx for idx, code in enumerate(sorted(icd10cm), start=1)}
    )
    mdace = load_dataset(DatasetConfig(**DATASET_CONFIGS["mdace-icd10cm"]))
    mdace = exp_utils.format_dataset(mdace, xml_trie, debug=args.debug)

    mdace = mdace.map(
        lambda x: {
            **x,
            "evidence": [
                " ".join(
                    x["note"][loc[0] : loc[-1] + 1] for loc in annotation["locations"]
                )
                for annotation in x["evidence_spans"]
            ],
        }
    )

    analyse_agent = create_analyse_agent(
        **args.analyse_agent,
        sampling_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    )

    task_maker = analyse_agent(
        init_client_fn=exp_utils._init_client_fn(**args.base_model),
        seed=args.seed,
    )

    analyse_mdace = mdace.map(
        task_maker,
        num_proc=args.num_workers,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=exp_utils._get_dataset(mdace).column_names,
        desc=f"[Analyse Agent] Generating search queries for seed {args.seed}.",
        load_from_cache_file=False,
    )

    assignable_terms = [
        term.model_dump() for term in xml_trie.index.values() if term.code
    ]
    assignable_terms.sort(key=lambda x: x["id"])
    logger.info(f"Fetched `{len(assignable_terms)}` assignable terms.")

    index_finger_print = qdrant_factory.ensure_qdrant_index(
        data=assignable_terms,
        text_key="path",
        model_cfg=args.embed_config,
        hnsw_cfg=args.hnsw,
        distance=args.distance,
        service=qdrant_service,
        payload_keys=["id", "code"],
    )

    assignable_terms_results = qdrant_factory.search(
        data=analyse_mdace.to_list(),
        model_cfg=args.embed_config,
        service=qdrant_service,
        index_name=index_finger_print,
        limit=args.rank,
    )

    y_true = mdace["targets"]
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    retrieved_codes = []
    for res in assignable_terms_results:
        accumulated_codes = set()
        terms = []
        for point in res.points:
            if not point.payload:
                continue
            terms.append(point.payload["id"])
            accumulated_codes.update(
                xml_trie.get_term_codes(point.payload["id"], subterms=False)
            )
        retrieved_codes.append(list(accumulated_codes))
    y_pred_bin_k = mlb.transform(retrieved_codes)

    metrics = {
        "precision": float(
            precision_score(y_true_bin, y_pred_bin_k, average="micro", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_bin, y_pred_bin_k, average="micro", zero_division=0)
        ),
    }
    rich.print(metrics)

    logger.info(f"Dumping retrieval results to {args.get_experiment_folder()}")
    with open(args.get_experiment_folder() / f"rank{args.rank}.json", "w") as f:  # type: ignore
        json.dump({"metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    args = Arguments()  # type: ignore
    run(args)
