import json

import datasets
import rich
from loguru import logger

from agents.locate_agent import create_locate_agent
import dataloader
from dataloader.base import DatasetConfig
from config import DUMP_FOLDER, BaseArguments, _get_dataset, _init_client_fn
from trie.icd import ICD10Trie


def precision_recall(predicted: list[list[str]], ground_truth: list[list[str]]):
    total_true_positives = 0
    total_predicted = 0
    total_actual = 0

    for pred, gold in zip(predicted, ground_truth):
        pred_set = set(pred)
        gold_set = set(gold)

        true_positives = len(pred_set & gold_set)
        total_true_positives += true_positives
        total_predicted += len(pred_set)
        total_actual += len(gold_set)

    precision = total_true_positives / total_predicted if total_predicted else 0
    recall = total_true_positives / total_actual if total_actual else 0

    return precision, recall


class Arguments(BaseArguments):
    """Args for the script."""

    experiment_id: str = "locate-agent"
    experiment_name: str = "long-context"

    api_base: str = "http://localhost:6538/v1"
    deployment: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    prompt_name: str = "locate_1"  # lookup in `src/alignment/templates`
    temperature: float = 0.0

    dataset: str = "mdace-icd10cm"  # "mimic-iii-50" | "mimic-iv" | "mdace-icd10cm"
    seed: str = "1"  # e.g., "1:2:3:4:5"

    num_workers: int = 1
    batch_size: int = 1


def run(args: Arguments):
    """Run the script."""
    rich.print(args)
    xml_trie = ICD10Trie.from_cms(year=2022)
    xml_trie.parse()
    for dataset in args._datasets:
        dset_config: DatasetConfig = DatasetConfig(**dataloader.DATASET_CONFIGS[dataset])
        dset_config.options.prep_map_kws = {"load_from_cache_file": False, "num_proc": args.num_workers}
        logger.info(f"Running {dataset} with seeds `{args._seeds}`.")
        for prompt_name in args._prompts:
            for seed in args._seeds:
                dset_config.options.seed = seed
                dset = dataloader.load_dataset(dset_config)
                if isinstance(dset, datasets.DatasetDict):
                    dset = datasets.concatenate_datasets(dset.values())
                agent = create_locate_agent(
                    agent_type="long-context",
                    prompt_name=prompt_name,
                    seed=seed,
                    sampling_params={
                        "temperature": args.temperature,
                        "seed": seed,
                    },
                )
                dset = dset.select(range(0, 10))
                task_maker = agent(init_client_fn=_init_client_fn(**args.model_dump()), trie=xml_trie)
                eval_data = dset.map(
                    task_maker,
                    num_proc=args.num_workers,
                    batched=True,
                    batch_size=args.batch_size,
                    desc=f"Predicting with seed `{seed}`.",
                    remove_columns=_get_dataset(dset).column_names,
                    load_from_cache_file=False,
                )

                precision, recall = precision_recall(eval_data["codes"], eval_data["targets"])
                metrics = {"precision": precision, "recall": recall}
                rich.print(f"[yellow]{metrics}[/yellow]")
                save_folder = DUMP_FOLDER / str(args.experiment_folder) / dataset / f"{prompt_name}_seed{str(seed)}"

                logger.info(f"Dumping results to {save_folder}")
                save_folder.mkdir(parents=True, exist_ok=True)
                with open(save_folder / "responses.json", "w") as f:
                    cols_to_remove = set(_get_dataset(eval_data).column_names) - set(
                        ["aid", "note", "targets", "codes", "response"]
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
