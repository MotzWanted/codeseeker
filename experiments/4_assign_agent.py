from collections import OrderedDict
import json
import typing as typ

import datasets
import pydantic
import rich
from loguru import logger
import torch

from agents.assign_agent import create_assign_agent
import dataloader
from dataloader.base import DatasetConfig
import config as cnf
from trie.icd import ICD10Trie


class Arguments(cnf.BaseArguments):
    """Args for the script."""

    experiment_id: str = "assign-agent"
    experiment_name: str = "per-code"

    api_base: str = "http://localhost:6538/v1"
    deployment: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"

    prompt_name: str = (
        "assign_agent/per_code_v1:assign_agent/per_code_v2:assign_agent/per_code_v3"  # lookup in `src/alignment/templates`
    )

    agent_type: str = "mock-per-code"
    temperature: float = 0.0

    dataset: str = "mdace-icd10cm"  # "mimic-iii-50" | "mimic-iv" | "mdace-icd10cm"
    negatives: str = (
        "1:2:5:10:20"  # number of negative samples to include in the prompt
    )
    seed: str = "1"  # e.g., "1:2:3:4:5"
    n_samples: int = 1

    num_workers: int = 2
    batch_size: int = 1

    use_cache: bool = True  # whether to cache on request level

    @pydantic.computed_field
    def _negatives(self) -> list[int]:
        """Get the negatives."""
        return (
            [int(x) for x in self.negatives.split(":")]
            if ":" in self.negatives
            else [int(self.negatives)]
        )


def run(args: Arguments):
    """Run the script."""
    # exp = mlflow.set_experiment(args.experiment_id)
    rich.print(args)
    xml_trie = ICD10Trie.from_cms(year=2022)
    xml_trie.parse()
    for dataset in args._datasets:  # type: ignore
        dset_config: DatasetConfig = DatasetConfig(
            **dataloader.DATASET_CONFIGS[dataset]
        )
        dset_config.options.prep_map_kws = {
            "load_from_cache_file": False,
            "num_proc": args.num_workers,
        }
        logger.info(f"Running {dataset} with seeds `{args._seeds}`.")
        for prompt_name in args._prompts:  # type: ignore
            for num_negs in args._negatives:  # type: ignore
                for seed in args._seeds:  # type: ignore
                    dset_config.options.seed = seed
                    dset = dataloader.load_dataset(dset_config)
                    if isinstance(dset, datasets.DatasetDict):
                        dset = datasets.concatenate_datasets(dset.values())  # type: ignore
                    logger.info(f"Predicting with {num_negs} negatives...")
                    agent = create_assign_agent(
                        agent_type=args.agent_type,
                        prompt_name=prompt_name,
                        seed=seed,
                        sampling_params={
                            "temperature": args.temperature,
                            "seed": seed,
                        },
                    )
                    # dset = dset.select(range(0, 10))
                    task_maker = agent(
                        init_client_fn=cnf._init_client_fn(**args.model_dump()),
                        trie=xml_trie,
                        n_samples=num_negs,
                    )
                    eval_data = dset.map(
                        task_maker,
                        num_proc=args.num_workers,
                        batched=True,
                        batch_size=args.batch_size,
                        desc=f"Predicting with seed `{seed}`.",
                        remove_columns=cnf._get_dataset(dset).column_names,
                        load_from_cache_file=False,
                    )

                    unique_codes: set[str] = set()
                    for codes in eval_data["codes"]:
                        unique_codes.update(codes)
                    trie: dict[str, int] = OrderedDict(
                        {
                            code: idx
                            for idx, code in enumerate(sorted(unique_codes), start=1)
                        }
                    )

                    monitor = cnf.TrieClassificationMonitor(trie=trie)
                    monitor.update(
                        target_inputs=eval_data["targets"],
                        pred_inputs=eval_data["predictions"],
                    )
                    metrics = {
                        k: v.item()
                        for k, v in monitor.get().items()
                        if isinstance(v, torch.Tensor)
                    }
                    rich.print(f"[yellow]{metrics}[/yellow]")
                    save_folder = (
                        cnf.DUMP_FOLDER
                        / str(args.experiment_folder)
                        / dataset
                        / f"{prompt_name}_neg{str(num_negs)}_seed{str(seed)}"
                    )

                    logger.info(f"Dumping results to {save_folder}")
                    save_folder.mkdir(parents=True, exist_ok=True)
                    with open(save_folder / "responses.json", "w") as f:
                        cols_to_remove = set(
                            cnf._get_dataset(eval_data).column_names
                        ) - set(["aid", "classes", "indexes", "targets", "response"])
                        dump_data = eval_data.remove_columns(list(cols_to_remove))
                        json.dump(dump_data.to_list(), f)
                    with open(save_folder / "averages.json", "w") as f:
                        json.dump(metrics, f)
                    with open(save_folder / "config.json", "w") as f:
                        json.dump(args.model_dump_json(), f)


if __name__ == "__main__":
    args = Arguments()
    run(args)
