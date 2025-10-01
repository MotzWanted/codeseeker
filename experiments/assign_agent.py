from collections import OrderedDict
import hashlib
from pathlib import Path
import typing as typ

import datasets
import pydantic
import rich
import config as cnf
from loguru import logger

from agents.assign_agent import create_assign_agent
from agents.base import HfBaseAgent
import dataloader
from dataloader.base import DatasetConfig
from dataloader.interface import load_dataset
from trie.base import Trie
import utils as exp_utils

from retrieval.qdrant_search import client as qdrant_client
from retrieval.qdrant_search import models as qdrant_models
from retrieval.qdrant_search import factory as qdrant_factory


class Arguments(pydantic.BaseModel):
    """Args for the script."""

    experiment_id: str = "assign-agent"
    experiment_name: str = "plmicd-recall25"

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    base_model: dict[str, typ.Any] = {
        "provider": "vllm",
        "deployment": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "api_base": "http://localhost:6539/v1",
        "endpoint": "completions",
        "use_cache": True,
    }

    prompt_name: str = "assign_agent/reasoning_v5"
    agent_type: str = "reasoning"
    temperature: float = 0.0
    max_tokens: int = 5_000

    dataset: str = "mdace-icd10cm"  # "mimic-iii-50" | "mimic-iv" | "mdace-icd10cm"
    seed: int = 1

    num_workers: int = 4
    batch_size: int = 1

    qdrant_config: qdrant_models.FactoryConfig = qdrant_models.FactoryConfig()
    distance: str = "Cosine"
    hnsw: dict[str, int] = {"m": 32, "ef_construct": 256}
    rank: int = 10

    embed_config: list[dict[str, str]] = [
        {
            "type": "st",
            "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "query_key": "evidence",
        },
    ]

    debug: bool = False

    use_cache: bool = True  # whether to cache on request level

    def get_hash(self) -> str:
        """Create unique identifier for the arguments"""
        model_dict = self.model_dump(exclude={"experiment_id", "experiment_name"})
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    def get_experiment_folder(self) -> Path:
        """Get the experiment folder path."""
        experiment_folder = (
            cnf.DUMP_FOLDER
            / f"{self.experiment_id}/{self.experiment_name}/{self.get_hash()}"
        )
        experiment_folder.mkdir(parents=True, exist_ok=True)
        return experiment_folder


def retrieve_codes(
    embed_config: list[dict[str, str]],
    trie: Trie,
    dataset: datasets.Dataset,
    qdrant_service: qdrant_client.QdrantSearchService,
    rank: int,
) -> datasets.Dataset:
    """Run the pipeline for the analysis agent."""
    terms = [
        {
            **term.model_dump(),
            "group": exp_utils.make_group_id(term, cross_references=True),
        }
        for term in trie.index.values()
    ]

    logger.info(f"Fetched `{len(terms)}` terms from the Trie.")

    index_finger_print = qdrant_factory.ensure_qdrant_index(
        data=terms,
        text_key="path",
        model_cfg=embed_config,
        hnsw_cfg=args.hnsw,
        distance=args.distance,
        service=qdrant_service,
        payload_keys=["id", "group"],
        recreate=False,
    )

    logger.info("Searching for cross-references...")
    queries = dataset.to_list()
    cross_reference_results = qdrant_factory.search_by_group(
        data=dataset.to_list(),
        model_cfg=embed_config,
        service=qdrant_service,
        index_name=index_finger_print,
        limit=rank,
        group_size=1,
        group_key="group",
    )

    for idx, res in enumerate(cross_reference_results):
        cross_references = []
        for i, point in enumerate(res.points, start=1):
            if not point.payload:
                continue
            term = trie.index[point.payload["id"]]
            if term.see:
                cross_references.append(term.see)
            if term.see_also:
                cross_references.append(term.see_also)
        queries[idx]["evidence"].extend(cross_references)

    code_terms = [
        {
            **term.model_dump(),
            "group": exp_utils.make_group_id(term, cross_references=False),
        }
        for term in trie.index.values()
    ]

    # Create and ensure index
    index_finger_print = qdrant_factory.ensure_qdrant_index(
        data=code_terms,
        text_key="path",
        model_cfg=embed_config,
        hnsw_cfg=args.hnsw,
        distance=args.distance,
        service=qdrant_service,
        payload_keys=["id", "group"],
        recreate=False,
    )

    assignable_terms_results = qdrant_factory.search_by_group(
        data=queries,
        model_cfg=embed_config,
        service=qdrant_service,
        index_name=index_finger_print,
        limit=rank,
        group_size=1,
        group_key="group",
    )

    retrieved_codes = []
    retrieved_code_objects = []
    for res in assignable_terms_results:
        unique_codes = set()
        for point in res.points:
            if not point.payload:
                continue
            term = trie.index[point.payload["id"]]
            if not term.code:
                continue
            unique_codes.update(trie.get_term_codes(term.id, subterms=False))
        retrieved_codes.append(list(unique_codes))
        retrieved_code_objects.append(
            [trie[code].model_dump() for code in unique_codes]
        )

    return datasets.Dataset.from_dict(
        {
            "codes": retrieved_code_objects,
            "output": retrieved_codes,
            "targets": dataset["targets"],
            "subset_targets": [
                [code for code in targets if code in retrieved]
                for targets, retrieved in zip(dataset["targets"], retrieved_codes)
            ],
            "note": dataset["note"],
            "note_type": dataset["note_type"],
            "aid": dataset["aid"],
        }
    )


def pipe(
    agent: HfBaseAgent,
    dataset: datasets.Dataset,
    trie: Trie,
    num_workers: int,
    batch_size: int,
    seed: int,
):
    """Run the agent on the dataset."""
    assign_dataset = dataset.map(
        lambda x: {
            **x,
            "instructional_notes": trie.get_instructional_notes(x["output"]),
            "guidelines": [
                trie.guidelines["IB"].model_dump(),
                trie.guidelines["II"].model_dump(),
                trie.guidelines["III"].model_dump(),
            ],
        },
        desc="[Assign Agent] Fetching ICD guideline data for codes.",
    )
    assign_dataset = assign_dataset.map(
        agent,
        num_proc=num_workers,
        batched=True,
        batch_size=batch_size,
        desc=f"Predicting with seed `{seed}`.",
        remove_columns=exp_utils._get_dataset(dataset).column_names,
        load_from_cache_file=False,
    )

    assign_eval_data = assign_dataset.map(
        lambda x: {
            **x,
            "output": [
                x["codes"][subset_idx - 1]["code"]
                for subset_idx in x["output"]
                if len(x["codes"]) >= subset_idx > 0
            ],
        },
        desc="[Assign Agent] Decoding output predictions",
        load_from_cache_file=False,
    )
    return assign_eval_data


def run(args: Arguments) -> None:
    rich.print(args)
    with open(args.get_experiment_folder() / "config.json", "w") as f:  # type: ignore
        f.write(args.model_dump_json(indent=2))
    qdrant_service = qdrant_client.QdrantSearchService(
        **args.qdrant_config.model_dump()
    )
    xml_trie = exp_utils.build_icd_trie(year=2022)
    mdace = load_dataset(DatasetConfig(**dataloader.DATASET_CONFIGS["mdace-icd10cm"]))
    mdace = exp_utils.format_dataset(mdace, xml_trie, args.debug)
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
    icd10cm = [code.name for code in xml_trie.get_root_codes("cm")]
    eval_trie: dict[str, int] = OrderedDict(
        {code: idx for idx, code in enumerate(sorted(icd10cm), start=1)}
    )

    mdace_with_retrieved_terms = retrieve_codes(
        embed_config=args.embed_config,
        trie=xml_trie,
        dataset=mdace,
        qdrant_service=qdrant_service,
        rank=args.rank,
    )

    exp_utils.evaluate_and_dump_metrics(
        mdace_with_retrieved_terms,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix=f"{args.experiment_name}_upper_bound",
    )

    agent = create_assign_agent(
        agent_type=args.agent_type,
        prompt_name=args.prompt_name,
        sampling_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    )

    task_maker = agent(
        init_client_fn=exp_utils._init_client_fn(**args.base_model),
        seed=args.seed,
    )

    assign_eval_data = pipe(
        agent=task_maker,
        dataset=mdace_with_retrieved_terms,
        trie=xml_trie,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    exp_utils.evaluate_and_dump_metrics(
        assign_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix=args.experiment_name,
    )


if __name__ == "__main__":
    args = Arguments()  # type: ignore
    run(args)
