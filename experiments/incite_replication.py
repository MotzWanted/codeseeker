import os
import pathlib
import mlflow
from mlflow.metrics import make_metric
import datasets
import typing as typ
import pandas as pd
from pydantic_settings import SettingsConfigDict, BaseSettings

import rich

from alignment.aligners.sparse import ChainMatcher, LexiconMatcher, WordMatchAligner
from alignment.ops import HfAlignment
from dataloader.loaders.nbme_notes import NbmeDatasetLoader
from dataloader.adapters.alignment import NmbeAdapter
from segmenters import factory, Segmenter
from tools import mlflow as mlflow_utils
from tools import lexicon

USER_PATH = pathlib.Path.home()
MLFOW_TRACKING_URI = USER_PATH / "mlruns"

mlflow.set_tracking_uri(str(MLFOW_TRACKING_URI))
mlflow_client = mlflow.MlflowClient(tracking_uri=str(MLFOW_TRACKING_URI))


class Arguments(BaseSettings):
    """Args for the script."""

    experiment_id: str = "incite-replication"
    run_name: str = "lexicon_matching"

    split: str = "test"
    segmenter: str = "nbme"  # spacy | sentence | regex
    spacy_model: str = "en_core_web_lg"
    n_samples: int = 300

    corpus_key: str = "patient_note"  # the corpus indices to predict
    query_key: str = "features"  # the queries given for predicting indices, e.g., transcript or patient note

    num_workers: int = 12

    response_wait_time: int = 0  # seconds; useful to prevent RateLimitErrors
    use_cache: bool = True  # whether to cache on request level

    model_config = SettingsConfigDict(cli_parse_args=True)


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


def model_fn(*args: tuple) -> dict:
    df = typ.cast(pd.DataFrame, args[0])
    return pd.Series(df["predictions"], index=df.index)


def run(args: Arguments):
    """Run the script."""
    exp = mlflow.set_experiment(args.experiment_id)
    eval_data: datasets.Dataset = NbmeDatasetLoader().load_dataset(split="test", size=args.n_samples)  # type: ignore
    segmenter: Segmenter = factory(args.segmenter, args.spacy_model)  # noqa: F821
    adapter = NmbeAdapter(segmenter=segmenter, query_key=args.query_key)
    eval_data = eval_data.map(
        adapter,
        num_proc=args.num_workers,
        desc=f"Adapting dataset to `AlignmentModel` using `{NmbeAdapter.__name__}`.",
        remove_columns=_get_dataset(eval_data).column_names,
    )

    # with mlflow.start_run(experiment_id=exp.experiment_id, run_name=args.run_name):
    #     mlflow.log_params(
    #         {
    #             "segmenter": args.segmenter,
    #         }
    #     )
    word_matcher = ChainMatcher(LexiconMatcher(lexicon=lexicon.UMLS(os.environ.get("UMLS_API_KEY"))))
    task_maker = HfAlignment(client=None, aligner=WordMatchAligner(word_matcher))
    data = eval_data.map(
        task_maker,
        num_proc=args.num_workers,
        desc="Predicting with FuzzyMatching.",
        remove_columns=_get_dataset(eval_data).column_names,
        load_from_cache_file=False,
    )
    df = pd.DataFrame(
        {
            "corpus": data["corpus"],
            "queries": data["queries"],
            "predictions": data["predictions"],
            "labels": data["labels"],
        }
    )
    results = mlflow.evaluate(
        model_fn,
        df,
        predictions="predictions",
        targets="labels",
        extra_metrics=[
            make_metric(
                eval_fn=mlflow_utils.recall,
                greater_is_better=True,
            ),
            make_metric(
                eval_fn=mlflow_utils.precision,
                greater_is_better=True,
            ),
            make_metric(
                eval_fn=mlflow_utils.f1_score,
                greater_is_better=True,
            ),
            make_metric(
                eval_fn=mlflow_utils.exact_match,
                greater_is_better=True,
            ),
            make_metric(
                eval_fn=mlflow_utils.positive_ratio,
                greater_is_better=False,
            ),
            make_metric(
                eval_fn=mlflow_utils.negative_ratio,
                greater_is_better=False,
            ),
        ],
    )
    rich.print(f"[yellow]{results.metrics}")
    del df
    del results


if __name__ == "__main__":
    args = Arguments()
    run(args)
