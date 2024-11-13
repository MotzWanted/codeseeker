from functools import partial
import hashlib
import json
import pathlib
import datasets
import typing as typ
from loguru import logger
import pandas as pd
import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict

import mlflow
import rich

from alignment.aligners.llm import create_llm_aligner
from alignment.ops import HfAlignment
from dataloader.loaders.nbme_notes import NbmeDatasetLoader
from dataloader.adapters.alignment import NbmeAdapter
from throughster.factory import create_interface
from segmenters import factory, Segmenter
from tools import mlflow as mlflow_utils

USER_PATH = pathlib.Path.home()
MLFOW_TRACKING_URI = USER_PATH / "mlruns"

mlflow.set_tracking_uri(str(MLFOW_TRACKING_URI))
mlflow_client = mlflow.MlflowClient(tracking_uri=str(MLFOW_TRACKING_URI))
dump_folder = pathlib.Path("~/research/patient-note-scoring/benchmark-data").expanduser()


class Arguments(BaseSettings):
    """Args for the script."""

    experiment_id: str = "shuffle-experiments"

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    api_base: str = "http://localhost:6538/v1"
    deployment: str = "meta-llama/Llama-3.1-70B-Instruct"
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"

    prompt_name: str = "note2feature"  # lookup in `src/alignment/templates`

    aligner_type: str = "structured"
    temperature: float = 0.0
    token_limit: int = 128000

    split: str = "test"
    size: int = 300
    segmenter: str = "nbme"  # spacy | nbme | regex
    spacy_model: str = "en_core_web_lg"
    fewshots: str = "1:3:5:10:15:25:50"  # e.g., "1:5:10:20:50"
    shots_from_same_patient: bool = False
    seed: str = "1:2:3"  # e.g., "1:2:3:4:5"
    n_samples: int = 1

    corpus_key: str = "features"  # the corpus indices to predict
    query_key: str = "patient_note"  # the queries given for predicting indices, e.g., transcript or patient note

    num_workers: int = 4
    batch_size: int = 5

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
        model_dict = self.model_dump(exclude={"_hash", "_deployment_name", "experiment_folder"})
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    @pydantic.computed_field
    def experiment_folder(self) -> str:
        """Get the experiment name."""
        return f"{self._deployment_name}/{self.prompt_name}/{self._hash}"


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


def run(args: Arguments):
    """Run the script."""
    exp = mlflow.set_experiment(args.experiment_id)
    loader = NbmeDatasetLoader()
    raw_data: datasets.Dataset = loader(
        split="test", size=args.size, shots_from_same_patient=args.shots_from_same_patient
    )  # type: ignore
    segmenter: Segmenter = factory(args.segmenter, args.spacy_model)  # noqa: F821
    seeds_ = [int(x) for x in args.seed.split(":")] if ":" in args.seed else [int(args.seed)]
    num_shots_ = [int(x) for x in args.fewshots.split(":")] if ":" in args.fewshots else [int(args.fewshots)]
    for num_shots in num_shots_:
        for seed in seeds_:
            adapter = NbmeAdapter(
                segmenter=segmenter, query_key=args.query_key, seed=seed, binary_fewshots=args.aligner_type == "binary"
            )
            adapted_data = raw_data.map(
                adapter,
                num_proc=args.num_workers,
                desc=f"Adapting dataset to `AlignmentModel` using `{NbmeAdapter.__name__}`.",
                remove_columns=_get_dataset(raw_data).column_names,
                load_from_cache_file=False,
            )
            with mlflow.start_run(
                experiment_id=exp.experiment_id, run_name=f"{args._deployment_name}-{args.prompt_name}-{num_shots}shots"
            ):
                mlflow.log_params(
                    {
                        "prompt_name": args.prompt_name,
                        "seed": seed,
                        "num_shots": num_shots,
                        "segmenter": args.segmenter,
                        **args.model_dump(include={"deployment", "temperature", "split", "n_samples"}),
                    }
                )
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
                eval_data = adapted_data.map(
                    task_maker,
                    num_proc=args.num_workers,
                    batched=True,
                    batch_size=args.batch_size,
                    desc=f"Predicting {num_shots} shots with seed `{seed}`.",
                    remove_columns=_get_dataset(adapted_data).column_names,
                    load_from_cache_file=False,
                )
                df = pd.DataFrame(
                    {
                        "corpus": eval_data["corpus"],
                        "queries": eval_data["queries"],
                        "predictions": eval_data["indexes"],
                        "labels": eval_data["labels"],
                    }
                )
                results = mlflow.evaluate(
                    partial(
                        mlflow_utils.model_fn,
                        seed=seed,
                        num_shots=num_shots,
                        prompt_name=args.prompt_name,
                        client=mlflow_client,
                    ),
                    df,
                    predictions="predictions",
                    targets="labels",
                    extra_metrics=mlflow_utils.DEFAULT_METRICS,
                )
                rich.print(f"[yellow]{results.metrics}")
                save_folder = dump_folder / str(args.experiment_folder) / f"{num_shots}shots" / f"seed{str(seed)}"
                logger.info(f"Dumping results to {save_folder}")
                save_folder.mkdir(parents=True, exist_ok=True)
                with open(save_folder / "responses.json", "w") as f:
                    data_list = eval_data.to_list()
                    json.dump(data_list, f)
                with open(save_folder / "results.json", "w") as f:
                    json.dump(results.metrics, f)
                with open(save_folder / "config.json", "w") as f:
                    json.dump(args.model_dump_json(), f)


if __name__ == "__main__":
    args = Arguments()
    run(args)
