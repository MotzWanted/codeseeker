import json
import pathlib
import typing as typ
from functools import partial

import datasets
import pydantic
import rich
from loguru import logger
from throughster.factory import create_interface
import torch

import dataloader
from agents.aligners.llm import create_llm_aligner
from agents.ops import HfAlignment
from dataloader.base import DatasetConfig
from config import DUMP_FOLDER, BaseArguments, _get_dataset
from finetune.helpers import TrieClassificationMonitor


class Arguments(BaseArguments):
    """Args for the script."""

    experiment_id: str = "assign-agent"
    experiment_name: str = "one-condition"

    api_base: str = "http://localhost:6538/v1"
    deployment: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"

    prompt_name: str = "icdcm1_struct:icdcm2_struct:icdcm3_struct:icdcm4_struct"  # lookup in `src/alignment/templates`

    aligner_type: str = "regex"  # "structured" | "regex" | "unconstrained"
    temperature: float = 0.0
    token_limit: int = 128000

    dataset: str = "mdace-icd10cm"  # "mimic-iii-50" | "mimic-iv" | "mdace-icd10cm"
    negatives: str = "0:20:40:60:80:100"  # number of negative samples to include in the prompt
    seed: str = "1"  # e.g., "1:2:3:4:5"
    n_samples: int = 1

    num_workers: int = 4
    batch_size: int = 2

    response_wait_time: int = 0  # seconds; useful to prevent RateLimitErrors
    use_cache: bool = True  # whether to cache on request level

    @pydantic.computed_field
    def _negatives(self) -> list[int]:
        """Get the negatives."""
        return [int(x) for x in self.negatives.split(":")] if ":" in self.negatives else [int(self.negatives)]


def run(args: Arguments):
    """Run the script."""
    # exp = mlflow.set_experiment(args.experiment_id)
    rich.print(args)
    for dataset in args._datasets:
        dset_config: DatasetConfig = DatasetConfig(**dataloader.DATASET_CONFIGS[dataset])
        dset_config.options.prep_map_kws = {"load_from_cache_file": False, "num_proc": args.num_workers}
        logger.info(f"Running {dataset} with seeds `{args._seeds}`.")
        for prompt_name in args._prompts:
            for num_negs in args._negatives:
                dset_config.options.negatives = num_negs
                for seed in args._seeds:
                    dset_config.options.seed = seed
                    dset = dataloader.load_dataset(dset_config)
                    if isinstance(dset, datasets.DatasetDict):
                        dset = datasets.concatenate_datasets(dset.values())
                    logger.info(f"Predicting with {num_negs} negatives...")
                    init_client = partial(
                        create_interface,
                        provider=args.provider,
                        api_base=args.api_base,
                        endpoint=args.endpoint,
                        model_name=args.deployment,
                        use_cache=args.use_cache,
                        cache_dir=str(pathlib.Path(f"~/.cache/throughster/{args.deployment}").expanduser()),
                    )
                    aligner = create_llm_aligner(
                        aligner_type=args.aligner_type,
                        prompt_name=prompt_name,
                        token_limit=args.token_limit,
                        seed=seed,
                        sampling_params={
                            "temperature": args.temperature,
                            "n": args.n_samples,
                        },
                    )
                    task_maker = HfAlignment(
                        init_client_fn=init_client,
                        aligner=aligner,
                        wait_time=args.response_wait_time,
                    )
                    eval_data = dset.map(
                        task_maker,
                        num_proc=args.num_workers,
                        batched=True,
                        batch_size=args.batch_size,
                        desc=f"Predicting with seed `{seed}` for {num_negs} negatives.",
                        remove_columns=_get_dataset(dset).column_names,
                        load_from_cache_file=False,
                    )

                    unique_codes = set()
                    for row in eval_data:
                        unique_codes.update([code["name"] for code in row["classes"]])
                    trie = {code: idx for idx, code in enumerate(sorted(unique_codes))}

                    monitor = TrieClassificationMonitor(trie=trie)
                    monitor.update(
                        target_input_ids=eval_data["targets"],
                        preds_input_ids=eval_data["indexes"],
                        list_of_classes=eval_data["classes"],
                    )
                    metrics = {k: v.item() for k, v in monitor.get().items() if isinstance(v, torch.Tensor)}
                    rich.print(f"[yellow]{metrics}[/yellow]")
                    save_folder = (
                        DUMP_FOLDER
                        / str(args.experiment_folder)
                        / dataset
                        / f"{prompt_name}_neg{str(num_negs)}_seed{str(seed)}"
                    )

                    logger.info(f"Dumping results to {save_folder}")
                    save_folder.mkdir(parents=True, exist_ok=True)
                    with open(save_folder / "responses.json", "w") as f:
                        cols_to_remove = set(_get_dataset(eval_data).column_names) - set(
                            ["aid", "classes", "indexes", "targets", "response"]
                        )
                        dump_data = eval_data.remove_columns(list(cols_to_remove))
                        json.dump(dump_data.to_list(), f)
                    with open(save_folder / "averages.json", "w") as f:
                        json.dump(metrics, f)
                    with open(save_folder / "config.json", "w") as f:
                        json.dump(args.model_dump_json(), f)


if __name__ == "__main__":
    args = Arguments()
    run(args)
