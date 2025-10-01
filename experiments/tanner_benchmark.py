from collections import OrderedDict
import hashlib
import pathlib
import typing as typ

import pydantic
import rich

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

import analyse_agent as step1
import locate_agent as step2
import verify_agent as step3
import assign_agent as step4


class Arguments(pydantic.BaseModel):
    """Args for the script."""

    experiment_id: str = "agentic-system"
    experiment_name: str = "tanner"

    dataset: str = "mdace-icd10cm"  # "mimic-iii-50" | "mimic-iv" | "mdace-icd10cm"
    seed: int = 1
    n_samples: int = 1

    base_model: dict[str, typ.Any] = {
        "provider": "vllm",
        "deployment": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "api_base": "http://localhost:6538/v1",
        "endpoint": "completions",
        "use_cache": True,
    }
    temperature: float = 0.6
    max_tokens: int = 10_000

    analyse_agent: dict[str, typ.Any] = {
        "agent_type": "base",
        "prompt_name": "analyse_agent/strict_v3",
    }
    locate_agent: dict[str, typ.Any] = {
        "agent_type": "split",
        "prompt_name": "locate_agent/locate_few_terms_v1",
    }
    verify_agent: dict[str, typ.Any] = {
        "agent_type": "reasoning",
        "prompt_name": "verify_agent/few_per_term_v2",
    }
    assign_agent: dict[str, typ.Any] = {
        "agent_type": "reasoning",
        "prompt_name": "assign_agent/reasoning_v6",
    }

    batch_size: int = 1
    num_workers: int = 4
    all_codes: bool = True  # whether to use all codes in ICd

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
    xml_trie = exp_utils.build_icd_trie(year=2024)
    mdace = load_dataset(DatasetConfig(**dataloader.DATASET_CONFIGS[args.dataset]))
    mdace = exp_utils.format_dataset(mdace, xml_trie, args.debug)
    if args.all_codes:
        eval_trie: dict[str, int] = OrderedDict(
            {code: idx for idx, code in enumerate(sorted(xml_trie.lookup), start=1)}
        )
    else:
        icd10cm = [code.name for code in xml_trie.get_root_codes("cm")]
        eval_trie: dict[str, int] = OrderedDict(
            {code: idx for idx, code in enumerate(sorted(icd10cm), start=1)}
        )
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

    analyse_eval_data = step1.pipe(
        agent=task_maker,
        embed_config=args.embed_config,
        trie=xml_trie,
        dataset=mdace,
        qdrant_service=qdrant_service,
        rank=args.topk_assignable_terms,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.seed,
        hnsw=args.hnsw,
        distance=args.distance,
        all_codes=args.all_codes,
    )

    exp_utils.evaluate_and_dump_metrics(
        eval_data=analyse_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix="analyze",
    )

    ###### 2. Locate Agent ######

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

    locate_eval_data = step2.pipe(
        agent=task_maker,
        trie=xml_trie,
        dataset=analyse_eval_data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    exp_utils.evaluate_and_dump_metrics(
        eval_data=locate_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix="locate",
    )

    ###### 3. Verify Agent ######
    verify_agent = create_verify_agent(
        **args.verify_agent,
        sampling_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    )
    task_maker = verify_agent(
        init_client_fn=exp_utils._init_client_fn(**args.base_model),
        seed=args.seed,
    )
    verify_eval_data = step3.pipe(
        agent=task_maker,
        dataset=locate_eval_data,
        trie=xml_trie,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    exp_utils.evaluate_and_dump_metrics(
        eval_data=verify_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix="verify",
    )

    ###### 4. Assign Agent ######
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
    assign_eval_data = step4.pipe(
        agent=task_maker,
        dataset=verify_eval_data,
        trie=xml_trie,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    exp_utils.evaluate_and_dump_metrics(
        eval_data=assign_eval_data,
        trie=eval_trie,
        dump_path=args.get_experiment_folder(),
        file_prefix="assign",
    )


if __name__ == "__main__":
    args = Arguments()
    run(args)
