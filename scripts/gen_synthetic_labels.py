from functools import partial
import hashlib
import json
import pathlib
import datasets
import typing as typ
from loguru import logger
import numpy as np
import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict


from alignment.aligners.llm import create_llm_aligner
from alignment.ops import HfSyntheticAlignment
from dataloader.loaders.nbme_notes import NbmeDatasetLoader
from dataloader.adapters.alignment import NbmeAdapter
from throughster.factory import create_interface
from segmenters import factory, Segmenter


dump_folder = pathlib.Path("~/research/patient-note-scoring/synthetic-data").expanduser()


class Arguments(BaseSettings):
    """Args for the script."""

    provider: str = "vllm"  # "azure" | "vllm" | "mistral"
    api_base: str = "http://localhost:6538/v1"
    deployment: str = "meta-llama/Meta-Llama-3.1-70B-instruct"
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"

    dset_name: str = "nbme-train-subset"
    prompt_name: str = "loft"  # lookup in `src/alignment/templates`

    aligner_type: str = "long-context"
    temperature: float = 0.0
    token_limit: int = 128000

    split: str = "train"
    size: int | None = None
    segmenter: str = "nbme"  # spacy | nbme | regex
    spacy_model: str = "en_core_web_lg"
    fewshots: int = 50
    seed: str = "1"  # e.g., "1:2:3:4:5"
    n_samples: int = 1
    threshold: float = 0.0

    corpus_key: str = "features"  # the corpus indices to predict
    query_key: str = "patient_note"  # the queries given for predicting indices, e.g., transcript or patient note

    num_workers: int = 16
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
        model_dict = self.model_dump(exclude={"_hash", "_deployment_name", "experiment_folder"})
        return hashlib.md5(str(model_dict).encode()).hexdigest()

    @pydantic.computed_field
    def experiment_folder(self) -> str:
        """Get the experiment name."""
        return f"{self._deployment_name}/{self.dset_name}/{self._hash}"


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


class FilterPredictions:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, row: dict[str, typ.Any]) -> dict[str, typ.Any]:
        predictions = row["predictions"]
        sparse_matrix = np.array(row["sparse_matrix"])
        probabilities = np.array(row["probabilities"])

        if probabilities is None:
            raise ValueError("Probabilities must be provided in the row.")

        mask = probabilities >= self.threshold
        filtered_matrix = sparse_matrix * mask

        filtered_predictions = [
            [idx for idx in idx_list if idx > 0 and mask[row_idx, idx - 1]]
            for row_idx, idx_list in enumerate(predictions)
        ]

        filtered_predictions = [idx_list if idx_list else [0] for idx_list in predictions]

        return {
            **row,
            "predictions": filtered_predictions,
            "sparse_matrix": filtered_matrix,
            "probabilities": probabilities,
        }


def run(args: Arguments):
    """Run the script."""
    loader = NbmeDatasetLoader()
    eval_data: datasets.Dataset = loader(split=args.split, size=args.size, num_proc=args.num_workers)  # type: ignore
    segmenter: Segmenter = factory(args.segmenter, args.spacy_model)  # noqa: F821
    adapter = NbmeAdapter(segmenter=segmenter, query_key=args.query_key, binary_fewshots=args.aligner_type == "binary")
    eval_data = eval_data.map(
        adapter,
        num_proc=args.num_workers,
        desc=f"Adapting dataset to `AlignmentModel` using `{NbmeAdapter.__name__}`.",
        remove_columns=_get_dataset(eval_data).column_names,
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
        num_shots=args.fewshots,
        token_limit=args.token_limit,
        seed=args.seed,
        sampling_params={
            "temperature": args.temperature,
            "n": args.n_samples,
        },
    )
    task_maker = HfSyntheticAlignment(
        init_client_fn=init_client,
        aligner=aligner,
        wait_time=args.response_wait_time,
    )
    data = eval_data.map(
        task_maker,
        num_proc=args.num_workers,
        desc=f"Generating synthetic labels with threshold {args.threshold}...",
        remove_columns=_get_dataset(eval_data).column_names,
    )
    filtered_data = data.map(
        FilterPredictions(threshold=args.threshold),
        num_proc=args.num_workers,
        desc="Filtering predictions...",
        remove_columns=_get_dataset(data).column_names,
    )
    save_folder = dump_folder / str(args.experiment_folder)
    logger.info(f"Dumping results to {save_folder}")
    save_folder.mkdir(parents=True, exist_ok=True)
    filtered_data.save_to_disk(f"{save_folder}/synthetic_data")
    with open(save_folder / "config.json", "w") as f:
        json.dump(args.model_dump_json(), f)


if __name__ == "__main__":
    args = Arguments()
    run(args)
