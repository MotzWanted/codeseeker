from functools import partial
import hashlib
import json
import pathlib
import datasets
import typing as typ
from loguru import logger
import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict

import rich

from alignment.aligners.llm import create_llm_aligner
from alignment.ops import HfAlignment
import dataloader
from dataloader.base import DatasetConfig
from throughster.factory import create_interface

from finetune.helpers import ClassificationMonitor


# MLFOW_TRACKING_URI = "http://172.16.40.132:5101"

# mlflow.set_tracking_uri(str(MLFOW_TRACKING_URI))
# mlflow_client = mlflow.MlflowClient(tracking_uri=str(MLFOW_TRACKING_URI))
dump_folder = pathlib.Path("~/research/entityseeker/benchmark-data").expanduser()


class Arguments(BaseSettings):
    """Args for the script."""

    experiment_id: str = "mimiciv-icd10cm"
    experiment_name: str = "constrained-decoding-lora"

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    api_base: str = "http://localhost:6538/v1"
    deployment: str = "/root/models/hard-neg-meta-llama-Llama-3.2-3B-Instruct-align-merged"
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"

    prompt_name: str = "icdcm_v2"  # lookup in `src/alignment/templates`

    aligner_type: str = "regex"  # "structured" | "regex" | "unconstrained"
    temperature: float = 0.0
    token_limit: int = 128000

    dataset: str = "mimiciv-cm-3.0:mimiciv-cm-3.1:mimiciv-cm-3.2:mimiciv-cm-3.3:mimiciv-cm-3.4"
    fewshots: str = "0"  # e.g., "1:5:10:20:50"
    negatives: str = "-1"  # number of negative samples to include in the prompt
    seed: str = "1:2:3"  # e.g., "1:2:3:4:5"
    n_samples: int = 1

    num_workers: int = 8
    batch_size: int = 1

    response_wait_time: int = 0  # seconds; useful to prevent RateLimitErrors
    use_cache: bool = True  # whether to cache on request level

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)

    @pydantic.computed_field
    def _deployment_name(self) -> str:
        """Get the model name."""
        return self.deployment.split("/")[-1]

    @pydantic.computed_field
    def _hash(self) -> str:
        """Create unique identifier for the arguments"""
        model_dict = self.model_dump(
            exclude={"_hash", "_deployment_name", "experiment_folder", "experiment_name", "seed", "fewshots"}
        )
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    @pydantic.computed_field
    def experiment_folder(self) -> str:
        """Get the experiment name."""
        return f"{self._deployment_name}/{self.experiment_name}/{self._hash}"

    @pydantic.computed_field
    def _negatives(self) -> list[int]:
        """Get the negatives."""
        return [int(x) for x in self.negatives.split(":")] if ":" in self.negatives else [int(self.negatives)]

    @pydantic.computed_field
    def _num_shots(self) -> list[int]:
        """Get the num_shots."""
        return [int(x) for x in self.fewshots.split(":")] if ":" in self.fewshots else [int(self.fewshots)]

    @pydantic.computed_field
    def _seeds(self) -> list[int]:
        """Get the seeds."""
        return [int(x) for x in self.seed.split(":")] if ":" in self.seed else [int(self.seed)]

    @pydantic.computed_field
    def _datasets(self) -> list[str]:
        """Get the datasets."""
        return [x for x in self.dataset.split(":")] if ":" in self.dataset else [self.dataset]


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


def run(args: Arguments):
    """Run the script."""
    # exp = mlflow.set_experiment(args.experiment_id)
    rich.print(args)
    for dataset in args._datasets:
        dset_config: DatasetConfig = DatasetConfig(**dataloader.DATASET_CONFIGS[dataset])
        dset_config.options.prep_map_kws = {"load_from_cache_file": False, "num_proc": args.num_workers}
        logger.info(f"Running {dataset} with seeds `{args._seeds}`.")
        for num_shots in args._num_shots:
            dset_config.options.shots = num_shots
            for num_negs in args._negatives:
                dset_config.options.negatives = num_negs
                for seed in args._seeds:
                    dset_config.options.seed = seed
                    dset: datasets.Dataset = dataloader.load_dataset(dset_config)
                    classes = dset[0]["classes"]
                    logger.info(f"Predicting with {len(classes)} codes...")
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
                        prompt_name=args.prompt_name,
                        num_shots=num_shots,
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
                        desc=f"Predicting {num_shots} shots with seed `{seed}` for {len(classes)} classes.",
                        remove_columns=_get_dataset(dset).column_names,
                        load_from_cache_file=False,
                    )
                    monitor = ClassificationMonitor(classes=len(classes))
                    monitor.update(target_input_ids=eval_data["targets"], preds_input_ids=eval_data["indexes"])
                    metrics = {k: v.item() for k, v in monitor.get().items()}
                    rich.print(f"[yellow]{metrics}[/yellow]")
                    save_folder = dump_folder / str(args.experiment_folder) / f"{dataset}" / f"seed{str(seed)}"

                    logger.info(f"Dumping results to {save_folder}")
                    save_folder.mkdir(parents=True, exist_ok=True)
                    with open(save_folder / "responses.json", "w") as f:
                        cols_to_remove = set(_get_dataset(eval_data).column_names) - set(
                            ["aid", "classes", "indexes", "targets", "index2code", "note_type"]
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
