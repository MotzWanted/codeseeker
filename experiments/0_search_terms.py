from collections import defaultdict
import json
import typing as typ

import pydantic
import rich
from loguru import logger

from agents.analyse_agent import create_analyse_agent
from agents.search_agent import HfEmbeddingClient
from dataloader.base import DatasetConfig
import config as cnf
from trie.base import Trie
from trie.icd import ICD10Trie
from rich.progress import track

import hashlib
from pathlib import Path

from functools import partial
import datasets


from dataloader import load_dataset, DATASET_CONFIGS

import utils as exp_utils
from search.qdrant_search import client as qdrant_client
from search.qdrant_search import models as qdrant_models
from search.qdrant_search import factory as qdrant_factory

from qdrant_client import models as qdrm


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

    post_embedding_config: list[dict[str, str]] = [
        {
            "type": "custom",
            "model_name": "infly/inf-retriever-v1",
            "dim": "3584",
            "query_key": "note_prompt",
        },
        {"type": "sparse", "model_name": "Qdrant/bm25", "query_key": "note"},
    ]
    topk_lead_terms: int = 500

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
        {"type": "sparse", "model_name": "Qdrant/bm25", "query_key": "note"},
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


def eval_retrieval_metrics(
    eval_data: list[dict[str, typ.Any]],
    xml_trie: Trie,
    ranks: list[int],
    strict: bool = False,
) -> tuple[dict[str, float], tuple[str, float]]:
    """Evaluate retrieval metrics (recall and precision) at different rank cutoffs."""
    sum_recall = defaultdict(float)
    sum_precision = defaultdict(float)
    n_examples = len(eval_data)

    for row in track(eval_data, total=len(eval_data), description="Evaluating metrics"):
        row_dict = dict(row)
        term_ids = [t["id"] for t in row_dict["retrieved_terms"]]
        target_codes = set(row_dict["targets"])

        accumulated_codes = set()
        accumulated_lead_terms = set()
        tp_at_k = {}
        seen_at_k = {}
        terms_per_lead_at_k = {}

        # iterate over retrieved term IDs and update the true‑positive tally
        for i, term_id in enumerate(term_ids, start=1):
            accumulated_lead_terms.add(term_id.split(".")[0])
            if strict:
                term = xml_trie.index[term_id]
                if term.code:
                    accumulated_codes.add(term.code)
                if term.manifestation_code:
                    accumulated_codes.add(term.manifestation_code)
                # if term.parent_id:
                #     accumulated_codes.update(xml_trie.get_all_term_codes(term_id))
            else:
                accumulated_codes.update(xml_trie.get_all_term_codes(term_id))

            if i in ranks:
                tp_at_k[i] = len(accumulated_codes & target_codes)
                seen_at_k[i] = len(accumulated_codes)
                terms_per_lead_at_k[i] = i / len(accumulated_lead_terms)

        # fill missing ranks with the last computed value
        last_tp, last_seen = 0, 1
        for k in sorted(ranks):
            tp_here = tp_at_k.get(k, last_tp)
            seen_here = seen_at_k.get(k, last_seen)
            negatives_per_positives = seen_here - tp_here
            sum_recall[f"recall@{k}"] += tp_here / len(target_codes)
            sum_precision[f"precision@{k}"] += tp_here / seen_here
            sum_precision[f"neg_per_pos@{k}"] += negatives_per_positives
            sum_precision[f"terms_per_lead@{k}"] += terms_per_lead_at_k.get(k, 0.0)
            last_tp, last_seen = tp_here, seen_here

    # convert sums → means
    metrics_results = {
        **{k: v / n_examples for k, v in sum_recall.items()},
        **{k: v / n_examples for k, v in sum_precision.items()},
    }

    # print metrics
    for k in sorted(ranks):
        rich.print(f"\nRetrieval metrics at rank {k}:")
        rich.print(f"  Recall: {metrics_results[f'recall@{k}']:.4f}")
        rich.print(f"  Precision: {metrics_results[f'precision@{k}']:.4f}")
        rich.print(
            f"  Negatives per positives: {metrics_results[f'neg_per_pos@{k}']:.4f}"
        )
        rich.print(f"  Terms per lead: {metrics_results[f'terms_per_lead@{k}']:.4f}")

    best_recall = max(
        ((k, v) for k, v in metrics_results.items() if k.startswith("recall")),
        key=lambda kv: kv[1],
    )
    return metrics_results, best_recall


def run(args: Arguments):
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

    xml_trie = ICD10Trie.from_cms(year=2022)
    xml_trie.parse()
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
    mdace = datasets.concatenate_datasets(mdace.values())  # type: ignore
    # mdace = mdace.select(range(10))

    # task_fn_1 = partial(
    #     exp_utils.get_detailed_instruct,
    #     task_description=args.note_task,
    #     prompt_template=args.prompt_template,
    # )
    # mdace = mdace.map(
    #     lambda row: {
    #         "note_prompt": task_fn_1(query=row["note"]),
    #         **row,
    #     },
    #     desc="Formatting note task prompts.",
    # )
    # client.prompt_key = "note_prompt"
    # mdace = mdace.map(
    #     client,
    #     num_proc=args.num_workers,
    #     batched=True,
    #     batch_size=1024,
    #     remove_columns=cnf._get_dataset(mdace).column_names,
    #     desc="Embedding notes for post retrieval.",
    #     load_from_cache_file=False,
    # )

    # post_emb_collection = qdrant_factory.instantiate_models(args.post_embedding_config)
    # qdrant_body = qdrant_factory.make_qdrant_body(
    #     post_emb_collection, **args.model_dump()
    # )
    # index_finger_print = qdrant_factory.make_index_fingerprint(
    #     data=terms_dset.to_list(),
    #     model_collection=post_emb_collection,
    #     qdrant_body=qdrant_body.model_dump(),
    #     text_key="title",
    # )
    # if not qdrant_factory.index_exists(
    #     qdrant_service,
    #     index_finger_print,
    #     exist_ok=True,
    # ):
    #     points = qdrant_factory.format_qdrant_point(
    #         data=terms_dset.to_list(),
    #         model_collection=post_emb_collection,
    #         text_key="title",
    #         payload_keys=["id"],
    #     )
    #     qdrant_factory.build_qdrant_index(
    #         qdrant_service, index_finger_print, qdrant_body, points
    #     )
    # prefetch_iter = qdrant_factory.prefetch_iterator(
    #     mdace, post_emb_collection, args.post_embedding_config, max(args.ranks)
    # )
    # top_k_results = []
    # top_k_lead_terms = []
    # for query in track(
    #     prefetch_iter,
    #     total=len(mdace),
    #     description="Querying index to retrieve post lead terms...",
    # ):
    #     result = qdrant_service.client.query_points(
    #         index_finger_print,
    #         prefetch=query,
    #         query=qdrm.FusionQuery(fusion=qdrm.Fusion.RRF),
    #         limit=max(args.ranks),
    #         with_payload=True,
    #         timeout=300,
    #     )
    #     top_k_results.append(result)
    #     top_k_lead_terms.append(
    #         xml_trie.index[p.payload["id"]].model_dump()
    #         for p in result.points[: args.topk_lead_terms]
    #         if p.payload
    #     )

    # eval_data = datasets.Dataset.from_dict(
    #     {
    #         "retrieved_terms": [
    #             [p.payload for p in res.points] for res in top_k_results
    #         ],
    #         "targets": mdace["targets"],
    #         "note": mdace["note"],
    #         "note_type": mdace["note_type"],
    #         "aid": mdace["aid"],
    #     }
    # )

    # metrics, _ = eval_retrieval_metrics(
    #     eval_data.to_list(), xml_trie, args.ranks, strict=False
    # )

    ##### 1. RETRIEVE TOP K LEAD TERMS BASED ON LLM-GENERATED TERMS #####
    lead_emb_collection = qdrant_factory.instantiate_models(args.lead_embedding_config)
    qdrant_body = qdrant_factory.make_qdrant_body(
        lead_emb_collection, **args.model_dump()
    )
    index_finger_print = qdrant_factory.make_index_fingerprint(
        data=terms_dset.to_list(),
        model_collection=lead_emb_collection,
        qdrant_body=qdrant_body.model_dump(),
        text_key="path",
    )
    if not qdrant_factory.index_exists(
        qdrant_service,
        index_finger_print,
        exist_ok=True,
    ):
        points = qdrant_factory.format_qdrant_point(
            data=terms_dset.to_list(),
            model_collection=lead_emb_collection,
            text_key="path",
            payload_keys=["id"],
        )
        qdrant_factory.build_qdrant_index(
            qdrant_service, index_finger_print, qdrant_body, points
        )

    del terms
    del terms_dset

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

    prefetch_iter = qdrant_factory.prefetch_iterator(
        mdace, lead_emb_collection, args.lead_embedding_config, max(args.ranks)
    )
    top_k_results = []
    top_k_lead_terms = []
    for query in track(
        prefetch_iter,
        total=len(mdace),
        description="Querying index to retrieve post lead terms...",
    ):
        result = qdrant_service.client.query_points(
            index_finger_print,
            prefetch=query,
            query=qdrm.FusionQuery(fusion=qdrm.Fusion.RRF),
            limit=max(args.ranks),
            with_payload=True,
            timeout=300,
        )
        top_k_results.append(result)
        top_k_lead_terms.append(
            xml_trie.index[p.payload["id"]].model_dump()
            for p in result.points[: args.topk_lead_terms]
            if p.payload
        )

    eval_data = datasets.Dataset.from_dict(
        {
            "retrieved_terms": [
                [p.payload for p in res.points] for res in top_k_results
            ],
            "targets": mdace["targets"],
            "note": mdace["note"],
            "note_type": mdace["note_type"],
            "aid": mdace["aid"],
        }
    )

    metrics, max_recall = eval_retrieval_metrics(
        eval_data.to_list(), xml_trie, args.ranks
    )
    logger.info(f"Dumping retrieval results to {args.get_experiment_folder()}")
    with open(args.get_experiment_folder() / "lead_term_results.json", "w") as f:  # type: ignore
        json.dump({"metrics": metrics}, f, indent=2)

    sorted_top_k_lead_terms = [
        list(sorted(terms, key=lambda x: x["id"])) for terms in top_k_lead_terms
    ]

    mdace = mdace.add_column("terms", sorted_top_k_lead_terms)

    ##### 2. RETRIEVE ASSIGNABLE TERMS BASED ON LLM-GENERATED TERMS #####
    logger.info(f"Filtering top {args.topk_lead_terms} lead terms.")
    top_lead_terms: dict[str, dict] = {}
    term2aid: dict[str, set[str]] = defaultdict(set)
    for aid, retrieved in zip(mdace["aid"], mdace["terms"]):
        for lead in retrieved:
            term2aid[lead["id"]].add(aid)
            top_lead_terms.setdefault(lead["id"], {**lead})
    logger.info(f"Fetching sub terms for `{len(top_lead_terms)}` lead terms.")
    all_terms = []
    for lead_term_id in top_lead_terms:
        all_terms.append(
            {"aid": list(term2aid[lead_term_id]), **top_lead_terms[lead_term_id]}
        )
        sub_terms = xml_trie.get_all_term_children(lead_term_id)
        for sub_term in sub_terms:
            term2aid[sub_term.id].update(term2aid[lead_term_id])
            all_terms.append(
                {"aid": list(term2aid[lead_term_id]), **sub_term.model_dump()}
            )
    logger.info(f"Fectched {len(all_terms)} unique sub terms.")

    assignable_terms_index = [term for term in all_terms if term.get("code")]
    assignable_terms_index.sort(key=lambda x: x["id"])

    del all_terms
    del top_lead_terms

    assignable_terms_index = datasets.Dataset.from_list(assignable_terms_index)
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
    modifier_emb_collection = qdrant_factory.instantiate_models(
        args.modifier_embed_config
    )
    index_finger_print = qdrant_factory.make_index_fingerprint(
        data=assignable_terms_index.to_list(),
        model_collection=modifier_emb_collection,
        qdrant_body=qdrant_body.model_dump(),
        text_key="path",
    )
    if not qdrant_factory.index_exists(
        qdrant_service,
        index_finger_print,
        exist_ok=False,
    ):
        points = qdrant_factory.format_qdrant_point(
            data=assignable_terms_index.to_list(),
            model_collection=modifier_emb_collection,
            text_key="path",
            payload_keys=["id"],
        )
        qdrant_factory.build_qdrant_index(
            qdrant_service, index_finger_print, qdrant_body, points
        )

    del assignable_terms_index

    prefetch_iter = qdrant_factory.prefetch_iterator(
        mdace, modifier_emb_collection, args.modifier_embed_config, max(args.ranks)
    )
    assignable_terms_results = []
    for idx, query in track(
        enumerate(prefetch_iter),
        total=len(mdace),
        description="Querying index to retrieve assignable terms...",
    ):
        result = qdrant_service.client.query_points(
            index_finger_print,
            prefetch=query,
            query=qdrm.FusionQuery(fusion=qdrm.Fusion.RRF),
            limit=max(args.ranks),
            with_payload=True,
            timeout=800,
        )
        # result.points = [
        #     point
        #     for point in result.points
        #     if mdace["aid"][idx] in term2aid[point.payload["id"]]  # type: ignore
        # ]
        assignable_terms_results.append(result)

    eval_data = datasets.Dataset.from_dict(
        {
            "retrieved_terms": [
                [p.payload for p in res.points] for res in assignable_terms_results
            ],
            "targets": mdace["targets"],
            "note": mdace["note"],
            "note_type": mdace["note_type"],
            "aid": mdace["aid"],
        }
    )

    metrics, _ = eval_retrieval_metrics(
        eval_data.to_list(), xml_trie, args.ranks, strict=True
    )
    logger.info(f"Dumping retrieval results to {args.get_experiment_folder()}")
    with open(args.get_experiment_folder() / "sub_term_results.json", "w") as f:  # type: ignore
        json.dump({"metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    args = Arguments()  # type: ignore
    run(args)
