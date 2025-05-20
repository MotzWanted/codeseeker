from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import pathlib
import typing as typ

import datasets
import pydantic
import rich
from loguru import logger
import torch

from agents.analyse_agent import create_analyse_agent
from agents.assign_agent import create_assign_agent
from agents.locate_agent import create_locate_agent
from agents.verify_agent import create_verify_agent
import dataloader
from dataloader.base import DatasetConfig
from dataloader.interface import load_dataset
import utils as exp_utils
import config as exp_config
from retrieval.qdrant_search import models as qdrant_models
from retrieval.qdrant_search import client as qdrant_client
from retrieval.qdrant_search import factory as qdrant_factory


def evaluate_and_dump_metrics(
    eval_data: datasets.Dataset,
    trie: OrderedDict[str, int],
    dump_path: Path,
    file_prefix: str,
):
    """Compute, print, and optionally dump evaluation metrics."""
    overall_monitor = exp_config.TrieClassificationMonitor(trie=trie)
    contextual_monitor = exp_config.TrieClassificationMonitor(trie=trie)

    overall_monitor.update(
        target_inputs=eval_data["targets"], pred_inputs=eval_data["output"]
    )
    contextual_monitor.update(
        target_inputs=eval_data["subset_targets"], pred_inputs=eval_data["output"]
    )

    overall_metrics = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in overall_monitor.get().items()
    }
    contextual_metrics = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in contextual_monitor.get().items()
    }

    rich.print(f"[blue][{file_prefix}]Evaluation Metrics.[/blue]")
    rich.print(f"[blue][Overall] {overall_metrics}[/blue]")
    rich.print(f"[blue][Contextual] {contextual_metrics}[/blue]")

    for metrics in [overall_metrics, contextual_metrics]:
        with open(dump_path / f"{file_prefix}_metrics.json", "w") as f:
            json.dump(metrics, f)

    with open(dump_path / "responses.json", "w") as f:
        cols_to_remove = set(exp_utils._get_dataset(eval_data).column_names) - set(
            ["aid", "note_type", "output", "targets", "response"]
        )
        dump_data = eval_data.remove_columns(list(cols_to_remove))
        json.dump(dump_data.to_list(), f)

    return metrics


class Arguments(pydantic.BaseModel):
    """Args for the script."""

    experiment_id: str = "agentic-system"
    experiment_name: str = "debug"

    dataset: str = "mdace-icd10cm"  # "mimic-iii-50" | "mimic-iv" | "mdace-icd10cm"
    seed: str = "1"  # e.g., "1:2:3:4:5"
    n_samples: int = 1

    base_model: dict[str, typ.Any] = {
        "provider": "vllm",
        "deployment": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "api_base": "http://localhost:6539/v1",
        "endpoint": "completions",
        "use_cache": True,
    }
    temperature: float = 0.0
    max_tokens: int = 10_000

    analyse_agent: dict[str, typ.Any] = {
        "agent_type": "base",
        "prompt_name": "analyse_agent/base_v1",
    }
    locate_agent: dict[str, typ.Any] = {
        "agent_type": "snippet",
        "prompt_name": "locate_agent/locate_few_terms",
    }
    verify_agent: dict[str, typ.Any] = {
        "agent_type": "reasoning",
        "prompt_name": "verify_agent/one_per_term_v2",
    }
    assign_agent: dict[str, typ.Any] = {
        "agent_type": "reasoning",
        "prompt_name": "assign_agent/reasoning_v4",
    }

    batch_size: int = 1
    num_workers: int = 4

    topk_assignable_terms: int = 10
    embed_config: list[dict[str, str]] = [
        {
            "type": "st",
            "model_name": "pritamdeka/S-PubMedBert-MS-MARCO",
            "query_key": "output",
        },
    ]

    qdrant_config: qdrant_models.FactoryConfig = qdrant_models.FactoryConfig()
    distance: str = "Cosine"
    hnsw: dict[str, int] = {"m": 32, "ef_construct": 256}

    debug: bool = False

    use_cache: bool = True  # whether to cache on request level

    def get_hash(self) -> str:
        """Create unique identifier for the arguments"""
        model_dict = self.model_dump(exclude={"experiment_id", "experiment_name"})
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    def get_experiment_folder(self) -> pathlib.Path:
        """Get the experiment folder path."""
        path = (
            exp_config.DUMP_FOLDER
            / f"{self.experiment_id}/{self.experiment_name}/{self.get_hash()}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path


def run(args: Arguments):
    """Run the script."""
    # exp = mlflow.set_experiment(args.experiment_id)
    rich.print(args)
    with open(args.get_experiment_folder() / "config.json", "w") as f:  # type: ignore
        f.write(args.model_dump_json(indent=2))
    qdrant_service = qdrant_client.QdrantSearchService(
        **args.qdrant_config.model_dump()
    )
    xml_trie = exp_utils.build_icd_trie(year=2022)
    mdace = load_dataset(DatasetConfig(**dataloader.DATASET_CONFIGS["mdace-icd10cm"]))
    mdace = exp_utils.format_dataset(mdace, xml_trie, args.debug)
    mdace_codes = set(code for row in mdace["targets"] for code in row)
    icd10cm = [code.name for code in xml_trie.get_root_codes("cm")]
    eval_trie: dict[str, int] = OrderedDict(
        {code: idx for idx, code in enumerate(sorted(icd10cm), start=1)}
    )
    assignable_terms = [
        term.model_dump()
        for term in xml_trie.index.values()
        if term.code and term.code in mdace_codes
    ]
    logger.info(f"Fectched {len(assignable_terms)} assignable terms.")

    ###### 1. Analyze Agent ######
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

    index_finger_print = qdrant_factory.ensure_qdrant_index(
        data=assignable_terms,
        text_key="path",
        model_cfg=args.embed_config,
        hnsw_cfg=args.hnsw,
        distance=args.distance,
        service=qdrant_service,
        payload_keys=["id"],
    )

    assignable_terms_results = qdrant_factory.search(
        data=analyse_mdace.to_list(),
        model_cfg=args.embed_config,
        service=qdrant_service,
        index_name=index_finger_print,
        limit=args.topk_assignable_terms,
        merge_search=False,
    )

    retrieved_codes = []
    retrieved_terms = []
    for res in assignable_terms_results:
        unique_codes = set()
        terms = []
        for point in res.points:
            if not point.payload:
                continue
            terms.append(point.payload["id"])
            unique_codes.update(
                xml_trie.get_term_codes(point.payload["id"], subterms=False)
            )
        retrieved_codes.append(list(unique_codes))
        retrieved_terms.append(terms)

    analyse_eval_data = datasets.Dataset.from_dict(
        {
            "output": retrieved_codes,
            "terms": retrieved_terms,
            "query": analyse_mdace["output"],
            "targets": mdace["targets"],
            "subset_targets": [
                [code for code in targets if code in retrieved]
                for targets, retrieved in zip(mdace["targets"], retrieved_codes)
            ],
            "note": mdace["note"],
            "note_type": mdace["note_type"],
            "aid": mdace["aid"],
        }
    )

    evaluate_and_dump_metrics(
        eval_data=analyse_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix="analyze",
    )

    def group_by_size(x: list[str], size: int) -> list[list[str]]:
        """Group a list of strings by size."""
        groups = []
        for i in range(0, len(x), size):
            groups.append(x[i : i + size])
        return groups

    ###### 2. Locate Agent ######
    locate_dset = analyse_eval_data.map(
        lambda x: {
            **x,
            "terms": [
                group for group in group_by_size(x["terms"], args.topk_assignable_terms)
            ],
        },
        desc="Grouping assignable terms by lead term.",
    )

    def fetch_term_data(term_ids: list[str]) -> list[str]:
        """Get unique codes from a list of strings."""
        terms = [xml_trie.index[term] for term in term_ids]
        codes_seen = set()
        unique_terms = []
        for term in terms:
            if term.code in codes_seen:
                continue
            unique_terms.append(term)
            codes_seen.add(term.code)
        return [term.model_dump() for term in unique_terms]

    locate_dset = locate_dset.map(
        lambda x: {
            **x,
            "terms": [fetch_term_data(group) for group in x["terms"]],
        },
        desc="Fetching ICD alphabetical index data for terms.",
    )

    locate_agent = create_locate_agent(
        **args.locate_agent,
        sampling_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    )

    task_maker = locate_agent(
        init_client_fn=exp_utils._init_client_fn(**args.base_model),
        seed=args.seed,
    )

    locate_mdace = locate_dset.map(
        task_maker,
        num_proc=args.num_workers,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=exp_utils._get_dataset(locate_dset).column_names,
        desc=f"[Locate Agent] Locating terms for seed {args.seed}.",
        load_from_cache_file=False,
    )

    locate_mdace = locate_mdace.map(
        lambda x: {
            **x,
            "output": list(
                [
                    x["terms"][idx][term_idx - 1]["id"]
                    for term_idx in group
                    if len(x["terms"][idx]) >= term_idx > 0
                ]
                for idx, group in enumerate(x["output"])
            ),
        },
        desc="Decoding output terms.",
        load_from_cache_file=False,
    )

    located_codes = []
    codes_to_verify = []
    for row in locate_mdace["output"]:
        unique_codes = set()
        grouped_codes = []
        for group in row:
            codes = []
            for term_id in group:
                codes.extend(xml_trie.get_term_codes(term_id, subterms=False))
            unique_codes.update(codes)
            grouped_codes.append(codes)
        codes_to_verify.append(grouped_codes)
        located_codes.append(list(unique_codes))

    locate_eval_data = datasets.Dataset.from_dict(
        {
            "codes": codes_to_verify,
            "output": located_codes,
            "targets": locate_mdace["targets"],
            "subset_targets": [
                [code for code in targets if code in retrieved]
                for targets, retrieved in zip(locate_mdace["targets"], located_codes)
            ],
            "note": locate_mdace["note"],
            "note_type": locate_mdace["note_type"],
            "aid": locate_mdace["aid"],
        }
    )

    evaluate_and_dump_metrics(
        eval_data=locate_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix="locate",
    )

    ###### 3. Verify Agent ######
    verify_dset = locate_eval_data.map(
        lambda x: {
            **x,
            "codes": [
                sorted(
                    [xml_trie[code].model_dump() for code in set(code_group)],
                    key=lambda x: x["id"],
                )
                for code_group in x["codes"]
                if code_group
            ],
            "instructional_notes": [
                xml_trie.get_instructional_notes(code_group)
                for code_group in x["codes"]
                if code_group
            ],
            "guidelines": [
                xml_trie.get_guidelines(code_group)
                for code_group in x["codes"]
                if code_group
            ],
        },
        desc="Fetching ICD tabular data for codes.",
    )

    verify_agent = create_verify_agent(
        **args.verify_agent,
        sampling_params={
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
    )
    task_maker = verify_agent(
        init_client_fn=exp_utils._init_client_fn(**args.base_model),
    )
    verify_dset = verify_dset.map(
        task_maker,
        num_proc=args.num_workers,
        batched=True,
        batch_size=args.batch_size,
        desc=f"[Verify Agent] Predicting with seed `{args.seed}`.",
        remove_columns=exp_utils._get_dataset(verify_dset).column_names,
        load_from_cache_file=False,
    )

    verify_eval_data = verify_dset.map(
        lambda x: {
            **x,
            "output": [
                x["codes"][idx][code_idx - 1]["name"]
                for idx, group in enumerate(x["output"])
                for code_idx in group
                if len(x["codes"][idx]) >= code_idx > 0
            ],
            "codes": [
                x["codes"][idx][code_idx - 1]
                for idx, group in enumerate(x["output"])
                for code_idx in group
                if len(x["codes"][idx]) >= code_idx > 0
            ],
        },
        desc="Decoding output predictions",
        load_from_cache_file=False,
    )

    evaluate_and_dump_metrics(
        eval_data=verify_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix="verify",
    )

    ###### 4. Assign Agent ######

    assign_dset = verify_eval_data.map(
        lambda x: {
            **x,
            "instructional_notes": xml_trie.get_instructional_notes(x["output"]),
            "guidelines": [
                xml_trie.guidelines["IB"].model_dump(),
                xml_trie.guidelines["II"].model_dump(),
                xml_trie.guidelines["III"].model_dump(),
            ],
        },
        desc="Fetching ICD guideline data for codes.",
    )

    assign_agent = create_assign_agent(
        **args.assign_agent,
        sampling_params={
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
        },
    )
    task_maker = assign_agent(
        init_client_fn=exp_utils._init_client_fn(**args.base_model),
    )
    assign_dset = assign_dset.map(
        task_maker,
        num_proc=args.num_workers,
        batched=True,
        batch_size=args.batch_size,
        desc=f"[Assign Agent] Predicting with seed `{args.seed}`.",
        remove_columns=exp_utils._get_dataset(assign_dset).column_names,
        load_from_cache_file=False,
    )

    assign_eval_data = assign_dset.map(
        lambda x: {
            **x,
            "output": [
                x["codes"][subset_idx - 1]["name"]
                for subset_idx in x["output"]
                if len(x["codes"]) >= subset_idx > 0
            ],
        },
        desc="Decoding output predictions",
        load_from_cache_file=False,
    )

    evaluate_and_dump_metrics(
        eval_data=assign_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix="assign",
    )


if __name__ == "__main__":
    args = Arguments()
    run(args)
