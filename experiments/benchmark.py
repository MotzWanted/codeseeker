from collections import Counter
from functools import partial
import hashlib
import itertools
import json
import pathlib
import datasets
import typing as typ
from loguru import logger
import numpy as np
import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict

import rich

from alignment.aligners.llm import create_llm_aligner
from alignment.metrics import AlignmentMetrics
from alignment.ops import HfAlignment
import dataloader
from dataloader.base import DatasetConfig
from throughster.factory import create_interface

# MLFOW_TRACKING_URI = "http://172.16.40.132:5101"

# mlflow.set_tracking_uri(str(MLFOW_TRACKING_URI))
# mlflow_client = mlflow.MlflowClient(tracking_uri=str(MLFOW_TRACKING_URI))
dump_folder = pathlib.Path("~/research/entityseeker/benchmark-data").expanduser()


class Arguments(BaseSettings):
    """Args for the script."""

    experiment_id: str = "alignment-benchmark"
    experiment_name: str = "zero-shot-negative-sampling"

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    api_base: str = "http://localhost:6538/v1"
    deployment: str = "meta-llama/Llama-3.3-70B-Instruct"
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"

    prompt_name: str = "icd"  # lookup in `src/alignment/templates`

    aligner_type: str = "regex"  # "structured" | "regex"
    temperature: float = 0.0
    token_limit: int = 128000

    dataset: str = "mdace-diagnosis-3:mdace-diagnosis-4:mdace-diagnosis-5:mdace-diagnosis-6:mdace-diagnosis-7"
    fewshots: str = "0"  # e.g., "1:5:10:20:50"
    negatives: str = "450:350:250:150:50:0"  # number of negative samples to include in the prompt
    seed: str = "1"  # e.g., "1:2:3:4:5"
    n_samples: int = 1

    num_workers: int = 4
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


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


def _count_nested_list_distribution(list_of_lists: list[list[int]]) -> tuple[dict[int, int], dict[str, int]]:
    # Flatten the list for integers
    flat_integers = list(itertools.chain.from_iterable(itertools.chain.from_iterable(list_of_lists)))
    integer_counter = Counter(flat_integers)

    # Flatten list of lists as tuples for counting distribution of lists
    flattened_lists = list(map(tuple, itertools.chain.from_iterable(list_of_lists)))
    list_counter = Counter(flattened_lists)

    # Convert Counters to JSON-serializable format
    return dict(integer_counter), {str(k): v for k, v in list_counter.items()}


def run(args: Arguments):
    """Run the script."""
    # exp = mlflow.set_experiment(args.experiment_id)
    rich.print(args)
    seeds_ = [int(x) for x in args.seed.split(":")] if ":" in args.seed else [int(args.seed)]
    negatives_ = [int(x) for x in args.negatives.split(":")] if ":" in args.negatives else [int(args.negatives)]
    num_shots_ = [int(x) for x in args.fewshots.split(":")] if ":" in args.fewshots else [int(args.fewshots)]
    datasets_ = [x for x in args.dataset.split(":")] if ":" in args.dataset else [args.dataset]
    for dataset in datasets_:
        dset_config: DatasetConfig = DatasetConfig(**dataloader.DATASET_CONFIGS[dataset])
        dset_config.options.prep_map_kws = {"load_from_cache_file": False, "num_proc": args.num_workers}
        logger.info(f"Running {dataset} with seeds `{seeds_}` and num_shots `{num_shots_}`.")
        for num_shots in num_shots_:
            dset_config.options.shots = num_shots
            for num_negs in negatives_:
                dset_config.options.negatives = num_negs
                for seed in seeds_:
                    logger.info(f"Running with seed `{seed}`, num_shots `{num_shots}` and num negatives `{num_negs}`.")
                    dset_config.options.seed = seed
                    dset: datasets.Dataset = dataloader.load_dataset(dset_config)
                    classes = dset[0]["classes"]
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
                    results = eval_data.map(
                        AlignmentMetrics(labels_key="targets", predictions_key="indexes"),
                        num_proc=args.num_workers,
                        desc="Computing metrics...",
                        remove_columns=_get_dataset(eval_data).column_names,
                        load_from_cache_file=False,
                    )
                    averages = {}
                    for metric, values in results.to_dict().items():  # type: ignore
                        flat_values = [v[0] for v in values]
                        averages[metric] = np.mean(flat_values)
                    rich.print(f"[yellow]{averages}[/yellow]")
                    save_folder = (
                        dump_folder
                        / str(args.experiment_folder)
                        / f"{dataset}"
                        / f"{num_shots}shots"
                        / f"{num_negs}negatives"
                        / f"seed{str(seed)}"
                    )

                    preds_dist, pred_dist = _count_nested_list_distribution(eval_data["indexes"])
                    targets_dist, target_dist = _count_nested_list_distribution(eval_data["targets"])

                    logger.info(f"Dumping results to {save_folder}")
                    save_folder.mkdir(parents=True, exist_ok=True)
                    # if seed == 0:
                    #     with open(save_folder / "responses.json", "w") as f:
                    #         data_list = eval_data.to_list()
                    #         json.dump(data_list, f)
                    with open(save_folder / "distributions.json", "w") as f:
                        json.dump(
                            {
                                "num_classes": len(classes),
                                "classes": classes,
                                "predictions": pred_dist,
                                "targets": target_dist,
                                "predictions_flat": preds_dist,
                                "targets_flat": targets_dist,
                            },
                            f,
                        )
                    with open(save_folder / "averages.json", "w") as f:
                        json.dump(averages, f)
                    with open(save_folder / "config.json", "w") as f:
                        json.dump(args.model_dump_json(), f)


if __name__ == "__main__":
    args = Arguments()
    run(args)
