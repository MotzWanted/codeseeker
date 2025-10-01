from collections import OrderedDict
import typing as typ

import pydantic
import rich
from loguru import logger

from agents.base import HfBaseAgent
from agents.verify_agent import create_verify_agent
import dataloader
from dataloader.base import DatasetConfig
import config as cnf

import hashlib
from pathlib import Path

import datasets


from dataloader.interface import load_dataset
from trie.base import Trie
import utils as exp_utils
from retrieval.qdrant_search import client as qdrant_client
from retrieval.qdrant_search import models as qdrant_models
from retrieval.qdrant_search import factory as qdrant_factory


def decode_predictions(row: dict[str, typ.Any]) -> dict[str, typ.Any]:
    """Decode the predictions from the model output."""
    merged_codes = {}
    for idx, group in enumerate(row["output"]):
        for code_idx in group:
            if 0 < code_idx <= len(row["codes"][idx]):
                code_entry = row["codes"][idx][code_idx - 1]
                code = code_entry["code"]

                if code not in merged_codes:
                    merged_codes[code] = {
                        "code": code,
                        "description": code_entry["description"],
                        "id": code_entry["id"],
                        "paths": [code_entry["path"]],
                    }
                else:
                    merged_codes[code]["paths"].append(code_entry["path"])

    # If no codes were merged, return original codes
    if not merged_codes:
        for idx, group in enumerate(row["codes"]):
            for code_entry in group:
                if code_entry["code"] not in merged_codes:
                    merged_codes[code_entry["code"]] = {
                        "code": code_entry["code"],
                        "description": code_entry["description"],
                        "id": code_entry["id"],
                        "paths": [code_entry["path"]],
                    }
                else:
                    merged_codes[code_entry["code"]]["paths"].append(code_entry["path"])

    # Build output dictionary and merged list
    merged_codes_list = []
    for code, data in merged_codes.items():
        merged_entry = {
            "code": code,
            "description": data["description"],
            "id": data["id"],
            "path": ", ".join(sorted(data["paths"])),
        }
        merged_codes_list.append(merged_entry)

    return {
        **row,
        "output": list(merged_codes.keys()),
        "codes": merged_codes_list,
    }


class Arguments(pydantic.BaseModel):

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    base_model: dict[str, typ.Any] = {
        "provider": "vllm",
        "deployment": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "api_base": "http://localhost:6539/v1",
        "endpoint": "completions",
        "use_cache": True,
    }
    prompt_name: str = "verify_agent/one_per_term_v4"
    agent_type: str = "reasoning"
    temperature: float = 0.0
    max_tokens: int = 10_000
    seed: int = 1  # e.g., "1:2:3:4:5"
    batch_size: int = 1
    num_workers: int = 4
    embed_config: list[dict[str, str]] = [
        {
            "type": "st",
            "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "query_key": "evidence",
        },
    ]

    qdrant_config: qdrant_models.FactoryConfig = qdrant_models.FactoryConfig()
    distance: str = "Cosine"
    hnsw: dict[str, int] = {"m": 32, "ef_construct": 256}
    rank: int = 5

    debug: bool = False

    experiment_id: str = "verify-agent"
    experiment_name: str = "v2"

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


def retrieve_terms(
    embed_config: list[dict[str, str]],
    xml_trie: Trie,
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
        for term in xml_trie.index.values()
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
            term = xml_trie.index[point.payload["id"]]
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
        for term in xml_trie.index.values()
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

    def group_by_rank(x: list[typ.Any], size: int) -> list[list[typ.Any]]:
        """Group a list of strings by size."""
        groups = []
        for i in range(0, len(x), size):
            groups.append(x[i : i + size])
        return groups

    retrieved_codes = []
    grouped_codes = []
    for res in assignable_terms_results:
        unique_codes = set()
        grouped_points = group_by_rank(res.points, rank)
        group_codes = []
        for group in grouped_points:
            tmp_codes = set()
            for point in group:
                if not point.payload:
                    continue
                term = xml_trie.index[point.payload["id"]]
                if not term.code:
                    continue
                tmp_codes.update(xml_trie.get_term_codes(term.id, subterms=False))
            group_codes.append(list(tmp_codes))
            unique_codes.update(tmp_codes)
        grouped_codes.append(group_codes)
        retrieved_codes.append(list(unique_codes))

    return datasets.Dataset.from_dict(
        {
            "output": retrieved_codes,
            "codes": grouped_codes,
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
) -> datasets.Dataset:
    """Run the pipeline for the analysis agent."""
    verify_dataset = dataset.map(
        lambda x: {
            **x,
            "codes": [
                sorted(
                    [
                        exp_utils.CandidateCode(
                            **{**trie[term["code"]].model_dump(), **term}
                        ).model_dump()
                        for term in term_group
                    ],
                    key=lambda x: x["id"],
                )
                for term_group in x["terms"]
                if term_group
            ],
            "instructional_notes": [
                trie.get_instructional_notes(code_group)
                for code_group in x["codes"]
                if code_group
            ],
            "guidelines": [
                trie.get_guidelines(code_group)
                for code_group in x["codes"]
                if code_group
            ],
        },
        desc="Fetching ICD tabular data for codes.",
    )

    verify_dataset = verify_dataset.map(
        agent,
        num_proc=num_workers,
        batched=True,
        batch_size=batch_size,
        remove_columns=exp_utils._get_dataset(dataset).column_names,
        desc=f"[Verify Agent] Verifying code for seed {seed}.",
        load_from_cache_file=False,
    )
    verify_eval_data = verify_dataset.map(
        decode_predictions,
        desc="[Verify Agent] Decoding output predictions.",
        load_from_cache_file=False,
    )

    average_length = sum(len(row) for row in verify_eval_data["output"]) / len(
        verify_eval_data
    )
    logger.info(
        f"[Verify Agent] Average number of verified codes: {average_length:.2f}"
    )

    return verify_eval_data


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

    mdace_with_retrieved_terms = retrieve_terms(
        embed_config=args.embed_config,
        xml_trie=xml_trie,
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

    agent = create_verify_agent(
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

    verify_eval_data = pipe(
        agent=task_maker,
        dataset=mdace_with_retrieved_terms,
        trie=xml_trie,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    exp_utils.evaluate_and_dump_metrics(
        verify_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix=args.experiment_name,
    )


if __name__ == "__main__":
    args = Arguments()  # type: ignore
    run(args)
