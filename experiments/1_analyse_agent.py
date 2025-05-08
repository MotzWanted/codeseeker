import json
import typing as typ

import pydantic
import rich
from loguru import logger

from agents.analyse_agent import create_analyse_agent
from agents.search_agent import HfEmbeddingClient
from dataloader.base import DatasetConfig
import config as cnf
from qdrant_client.http.models.models import QueryResponse

import hashlib
from pathlib import Path

from functools import partial
import datasets


from dataloader import load_dataset, DATASET_CONFIGS

import utils as exp_utils
from search.qdrant_search import client as qdrant_client
from search.qdrant_search import models as qdrant_models
from search.qdrant_search import factory as qdrant_factory


class Arguments(pydantic.BaseModel):

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    embed_model: dict[str, typ.Any] = {
        "provider": "vllm",
        "deployment": "infly/inf-retriever-v1",
        "api_base": "http://localhost:6536/v1",
        "endpoint": "embeddings",
        "use_cache": True,
    }
    reasoning_model: dict[str, typ.Any] = {
        "provider": "vllm",
        "deployment": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "api_base": "http://localhost:6538/v1",
        "endpoint": "completions",
        "use_cache": True,
    }
    prompt_name: str = "analyse_agent/base_v2"
    agent_type: str = "base"
    temperature: float = 0.6
    max_tokens: int = 5_000
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    seed: int = 1  # e.g., "1:2:3:4:5"
    batch_size: int = 2
    num_workers: int = 4
    topk_lead_terms: int = 500
    topk_assignable_terms: int = 100

    lead_embedding_config: list[dict[str, str]] = [
        {
            "type": "st",
            "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "query_key": "output",
        },
        {
            "type": "custom",
            "model_name": "infly/inf-retriever-v1",
            "dim": "3584",
            "query_key": "embedding",
        },
        {"type": "sparse", "model_name": "Qdrant/bm25", "query_key": "output"},
    ]
    modifier_embed_config: list[dict[str, str]] = [
        {
            "type": "st",
            "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "query_key": "output",
        },
        {
            "type": "custom",
            "model_name": "infly/inf-retriever-v1",
            "dim": "3584",
            "query_key": "output_prompt",
        },
        # {"type": "sparse", "model_name": "Qdrant/bm25", "query_key": "output"},
    ]

    qdrant_config: qdrant_models.FactoryConfig = qdrant_models.FactoryConfig()
    distance: str = "Cosine"
    hnsw: dict[str, int] = {"m": 32, "ef_construct": 256}

    note_task: str = (
        "Given a clinical note, retrieve the clinical lead terms that are most relevant."
    )
    query_task: str = (
        "Given a clinical term, retrieve the clinical lead terms that are most relevant."
    )
    prompt_template: str = "Instruct: {task}\nQuery: {query}"
    ranks: list[int] = [25, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000]

    experiment_id: str = "search-terms"
    experiment_name: str = "inf-retriever-v1"

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


def run(args: Arguments) -> datasets.Dataset:
    rich.print(args)
    args.get_experiment_folder().mkdir(parents=True, exist_ok=True)
    with open(args.get_experiment_folder() / "config.json", "w") as f:  # type: ignore
        f.write(args.model_dump_json(indent=2))

    client = HfEmbeddingClient(
        prompt_key="path",
        init_client_fn=cnf._init_client_fn(**args.embed_model),
    )
    qdrant_service = qdrant_client.QdrantSearchService(
        **args.qdrant_config.model_dump()
    )

    ######### 0. RETRIEVE TOP K LEAD TERMS BASED ON NOTES #########

    xml_trie = exp_utils.build_icd_trie(year=2022)
    lead_terms = xml_trie.get_all_main_terms()
    terms = [
        term.model_dump() for term in lead_terms if term.assignable or term.children_ids
    ]
    terms_dset = datasets.Dataset.from_list(terms)
    terms_dset = terms_dset.map(
        client,
        num_proc=args.num_workers,
        batched=True,
        batch_size=1024,
        desc="Embedding lead terms for pre retrieval.",
        load_from_cache_file=False,
    )

    mdace = load_dataset(DatasetConfig(**DATASET_CONFIGS["mdace-icd10cm"]))
    mdace = exp_utils.format_dataset(mdace, xml_trie)

    ##### 1. RETRIEVE TOP K LEAD TERMS BASED ON LLM-GENERATED TERMS #####
    # index_finger_print = qdrant_factory.ensure_qdrant_index(
    #     data=terms_dset.to_list(),
    #     text_key="path",
    #     model_cfg=args.lead_embedding_config,
    #     hnsw_cfg=args.hnsw,
    #     distance=args.distance,
    #     service=qdrant_service,
    #     payload_keys=["id"],
    # )

    agent = create_analyse_agent(
        agent_type=args.agent_type,
        prompt_name=args.prompt_name,
        sampling_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "seed": args.seed,
        },
    )

    task_maker = agent(
        init_client_fn=cnf._init_client_fn(**args.reasoning_model),
        seed=args.seed,
    )

    mdace = mdace.map(
        task_maker,
        num_proc=args.num_workers,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=cnf._get_dataset(mdace).column_names,
        desc=f"Generating search queries for seed {args.seed}.",
        load_from_cache_file=False,
    )

    task_fn_2 = partial(
        exp_utils.get_detailed_instruct,
        task_description=args.query_task,
        prompt_template=args.prompt_template,
    )
    mdace = mdace.map(
        lambda row: {
            "output_prompt": [task_fn_2(query=q) for q in row["output"]],
            **row,
        },
        desc="Formatting task prompts.",
    )

    client.prompt_key = "output_prompt"
    mdace = mdace.map(
        client,
        num_proc=args.num_workers,
        batched=True,
        batch_size=1024,
        remove_columns=cnf._get_dataset(mdace).column_names,
        desc="Embedding LLM-generated terms for retrieval.",
        load_from_cache_file=False,
    )

    # lead_term_results = qdrant_factory.search(
    #     data=mdace.to_list(),
    #     model_cfg=args.lead_embedding_config,
    #     service=qdrant_service,
    #     index_name=index_finger_print,
    #     limit=max(args.ranks),
    # )

    # top_k_lead_terms = [
    #     [
    #         xml_trie.index[p.payload["id"]].model_dump()
    #         for p in res.points[: args.topk_lead_terms]
    #         if p.payload
    #     ]
    #     for res in lead_term_results
    # ]

    # eval_data = format_eval_data(mdace, lead_term_results)

    # metrics = exp_utils.analyse_agent_metrics(eval_data.to_list(), xml_trie, args.ranks)
    # logger.info(f"Dumping retrieval results to {args.get_experiment_folder()}")
    # with open(args.get_experiment_folder() / "lead_term_results.json", "w") as f:  # type: ignore
    #     json.dump({"metrics": metrics}, f, indent=2)

    ##### 2. RETRIEVE ASSIGNABLE TERMS BASED ON LLM-GENERATED TERMS #####
    # logger.info(f"Filtering top {args.topk_lead_terms} lead terms.")
    # top_lead_terms = {
    #     lead["id"]: lead for retrieved in mdace["terms"] for lead in retrieved
    # }
    # logger.info(f"Fetching sub terms for `{len(top_lead_terms)}` lead terms.")
    assignable_terms = [term.model_dump() for term in xml_trie.get_assignable_terms()]
    assignable_terms.sort(key=lambda x: x["id"])
    logger.info(f"Fetched `{len(assignable_terms)}` assignable terms.")

    assignable_terms_index = datasets.Dataset.from_list(assignable_terms)
    client.prompt_key = "path"
    assignable_terms_index = assignable_terms_index.map(
        client,
        num_proc=args.num_workers,
        batched=True,
        batch_size=1024,
        desc="Embedding lead terms for post retrieval.",
        load_from_cache_file=False,
    )

    logger.info(f"Fectched {len(assignable_terms_index)} assignable terms.")
    index_finger_print = qdrant_factory.ensure_qdrant_index(
        data=assignable_terms_index.to_list(),
        text_key="path",
        model_cfg=args.modifier_embed_config,
        hnsw_cfg=args.hnsw,
        distance=args.distance,
        service=qdrant_service,
        payload_keys=["id"],
    )

    assignable_terms_results = qdrant_factory.search(
        data=mdace.to_list(),
        model_cfg=args.modifier_embed_config,
        service=qdrant_service,
        index_name=index_finger_print,
        limit=max(args.ranks),
    )

    eval_data = format_eval_data(mdace, assignable_terms_results)

    metrics = exp_utils.analyse_agent_metrics(
        eval_data.to_list(), xml_trie, args.ranks, strict=True
    )
    logger.info(f"Dumping retrieval results to {args.get_experiment_folder()}")
    with open(args.get_experiment_folder() / "sub_term_results.json", "w") as f:  # type: ignore
        json.dump({"metrics": metrics}, f, indent=2)

    return eval_data


if __name__ == "__main__":
    args = Arguments()  # type: ignore
    run(args)
