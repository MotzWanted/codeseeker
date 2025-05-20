from collections import OrderedDict
import json
import typing as typ

import pydantic
import rich
from loguru import logger
import torch

from agents.assign_agent import create_assign_agent
from agents.verify_agent import create_verify_agent
import dataloader
from dataloader.base import DatasetConfig
from retrieval.negative_sampler import NegativeSampler
from retrieval.plm_icd import PLMICDRetriever
import utils as exp_utils
import config as exp_config


class Arguments(exp_config.BaseArguments):
    """Args for the script."""

    experiment_id: str = "strict-verify-agent"
    experiment_name: str = "plmicd-recall25"

    api_base: str = "http://localhost:6539/v1"
    deployment: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    endpoint: typ.Literal["chat/completions", "completions"] = "completions"

    prompt_name: str = "assign_agent/per_code_v4"
    assign_prompt_name: str = "assign_agent/reasoning_v4"

    pretrained_model_path: str = (
        "/nfs/nas/mlrd/models/medical_coding/plmicd_mdace/100k_steps"
    )
    agent_type: str = "reasoning"
    temperature: float = 0.0
    max_tokens: int = 10_000

    dataset: str = "mdace-icd10cm"  # "mimic-iii-50" | "mimic-iv" | "mdace-icd10cm"
    seed: str = "1"  # e.g., "1:2:3:4:5"
    n_samples: int = 1

    retriever: typ.Literal["plmicd", "sampler"] = "sampler"
    recall: str = "5:10:25:40:55"  # number of negative samples to include in the prompt

    num_workers: int = 4
    batch_size: int = 1

    debug: bool = False

    use_cache: bool = True  # whether to cache on request level

    @pydantic.computed_field
    def _recall(self) -> list[int]:
        """Get the negatives."""
        return (
            [int(x) for x in self.recall.split(":")]
            if ":" in self.recall
            else [int(self.recall)]
        )


def run(args: Arguments):
    """Run the script."""
    # exp = mlflow.set_experiment(args.experiment_id)
    rich.print(args)
    xml_trie = exp_utils.build_icd_trie(year=2022)
    icd10cm = [code.name for code in xml_trie.get_root_codes("cm")]
    eval_trie: dict[str, int] = OrderedDict(
        {code: idx for idx, code in enumerate(sorted(icd10cm), start=1)}
    )
    if args.retriever == "sampler":
        sampler = NegativeSampler(
    plm_icd = PLMICDRetriever(
        pretrained_model_path=args.pretrained_model_path,
        valid_labels=[
            code.name for code in xml_trie.get_root_codes("cm") if code.assignable
        ],
    )
    for dataset in args._datasets:  # type: ignore
        dset_config: DatasetConfig = DatasetConfig(
            **dataloader.DATASET_CONFIGS[dataset]
        )
        dset_config.options.prep_map_kws = {
            "load_from_cache_file": False,
            "num_proc": args.num_workers,
        }
        dset = dataloader.load_dataset(dset_config)
        dset = exp_utils.format_dataset(dset, xml_trie, args.debug)
        logger.info(f"Running {dataset} with seeds `{args._seeds}`.")
        for prompt_name in args._prompts:  # type: ignore
            for r in args._recall:  # type: ignore
                plm_icd.top_k = r
                for seed in args._seeds:  # type: ignore
                    dset = dset.map(
                        plm_icd,
                        batched=True,
                        batch_size=16,
                        desc=f"Retrieving codes for rank@{r}.",
                    )
                    retrieval_monitor = exp_config.TrieClassificationMonitor(
                        trie=eval_trie
                    )
                    retrieval_monitor.update(
                        target_inputs=dset["targets"],
                        pred_inputs=dset["codes"],
                    )
                    retrieval_metrics = {
                        k: v.item()
                        for k, v in retrieval_monitor.get().items()
                        if isinstance(v, torch.Tensor)
                    }
                    rich.print("[blue]Retrieval evaluation.[/blue]")
                    rich.print(f"[blue]{retrieval_metrics}[/blue]")
                    dset = dset.map(
                        lambda x: {
                            **x,
                            "codes": xml_trie.group_by_category(x["codes"]),
                            "subset_targets": [
                                code for code in x["codes"] if code in x["targets"]
                            ],
                        },
                        desc="Grouping codes by category.",
                    )
                    dset = dset.map(
                        lambda x: {
                            **x,
                            "codes": [
                                sorted(
                                    [
                                        xml_trie[code].model_dump()
                                        for code in code_group
                                    ],
                                    key=lambda x: x["id"],
                                )
                                for code_group in x["codes"]
                            ],
                            "instructional_notes": [
                                xml_trie.get_instructional_notes(code_group)
                                for code_group in x["codes"]
                            ],
                            "guidelines": [
                                xml_trie.get_guidelines(code_group)
                                for code_group in x["codes"]
                            ],
                        },
                        desc="Fetching ICD tabular data for codes.",
                    )
                    logger.info(f"Predicting with recall@{r} ...")

                    agent = create_verify_agent(
                        agent_type=args.agent_type,
                        prompt_name=prompt_name,
                        seed=seed,
                        sampling_params={
                            "temperature": args.temperature,
                            "seed": seed,
                            "max_tokens": args.max_tokens,
                        },
                    )
                    task_maker = agent(
                        init_client_fn=exp_utils._init_client_fn(**args.model_dump()),
                    )
                    eval_data = dset.map(
                        task_maker,
                        num_proc=args.num_workers,
                        batched=True,
                        batch_size=args.batch_size,
                        desc=f"Predicting with seed `{seed}`.",
                        remove_columns=exp_utils._get_dataset(dset).column_names,
                        load_from_cache_file=False,
                    )

                    eval_data = eval_data.map(
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

                    golden_monitor = exp_config.TrieClassificationMonitor(
                        trie=eval_trie
                    )
                    ceiled_monitor = exp_config.TrieClassificationMonitor(
                        trie=eval_trie
                    )
                    golden_monitor.update(
                        target_inputs=eval_data["targets"],
                        pred_inputs=eval_data["output"],
                    )
                    ceiled_monitor.update(
                        target_inputs=eval_data["subset_targets"],
                        pred_inputs=eval_data["output"],
                    )
                    golden_metrics = {
                        k: v.item()
                        for k, v in golden_monitor.get().items()
                        if isinstance(v, torch.Tensor)
                    }
                    ceiled_metrics = {
                        k: v.item()
                        for k, v in ceiled_monitor.get().items()
                        if isinstance(v, torch.Tensor)
                    }
                    rich.print(
                        "[yellow][Verify Agent] Golden truth evaluation.[/yellow]"
                    )
                    rich.print(f"[yellow]{golden_metrics}[/yellow]")
                    rich.print(
                        "[blue][Verify Agent] Evaluation conditioned on subset.[/blue]"
                    )
                    rich.print(f"[blue]{ceiled_metrics}[/blue]")
                    save_folder = (
                        exp_config.DUMP_FOLDER
                        / str(args.experiment_folder)
                        / dataset
                        / f"{prompt_name}_neg{str(r)}_seed{str(seed)}"
                    )

                    logger.info(f"Dumping results to {save_folder}")
                    save_folder.mkdir(parents=True, exist_ok=True)
                    with open(save_folder / "responses.json", "w") as f:
                        cols_to_remove = set(
                            exp_utils._get_dataset(eval_data).column_names
                        ) - set(
                            ["aid", "note_type", "predictions", "targets", "response"]
                        )
                        dump_data = eval_data.remove_columns(list(cols_to_remove))
                        json.dump(dump_data.to_list(), f)
                    with open(save_folder / "golden_metrics.json", "w") as f:
                        json.dump(golden_metrics, f)
                    with open(save_folder / "ceiled_metrics.json", "w") as f:
                        json.dump(ceiled_metrics, f)
                    with open(save_folder / "retrieval_metrics.json", "w") as f:
                        json.dump(retrieval_metrics, f)
                    with open(save_folder / "config.json", "w") as f:
                        json.dump(args.model_dump_json(), f)

                    eval_data = eval_data.map(
                        lambda x: {
                            **x,
                            "instructional_notes": xml_trie.get_instructional_notes(
                                x["output"]
                            ),
                            "guidelines": [
                                xml_trie.guidelines["IB"].model_dump(),
                                xml_trie.guidelines["II"].model_dump(),
                                xml_trie.guidelines["III"].model_dump(),
                            ],
                        },
                        desc="Fetching ICD guideline data for codes.",
                    )

                    assign_agent = create_assign_agent(
                        agent_type=args.agent_type,
                        prompt_name=args.assign_prompt_name,
                        seed=seed,
                        sampling_params={
                            "temperature": args.temperature,
                            "seed": seed,
                            "max_tokens": args.max_tokens,
                        },
                    )
                    task_maker = assign_agent(
                        init_client_fn=exp_utils._init_client_fn(**args.model_dump()),
                    )
                    eval_data = eval_data.map(
                        task_maker,
                        num_proc=args.num_workers,
                        batched=True,
                        batch_size=args.batch_size,
                        desc=f"Predicting with seed `{seed}`.",
                        remove_columns=exp_utils._get_dataset(dset).column_names,
                        load_from_cache_file=False,
                    )

                    eval_data = eval_data.map(
                        lambda x: {
                            **x,
                            "output": [
                                x["codes"][subset_idx - 1]["name"]
                                for subset_idx in x["output"]
                                if r >= subset_idx > 0
                            ],
                        },
                        desc="Decoding output predictions",
                        load_from_cache_file=False,
                    )
                    golden_monitor = exp_config.TrieClassificationMonitor(
                        trie=eval_trie
                    )
                    ceiled_monitor = exp_config.TrieClassificationMonitor(
                        trie=eval_trie
                    )
                    golden_monitor.update(
                        target_inputs=eval_data["targets"],
                        pred_inputs=eval_data["output"],
                    )
                    ceiled_monitor.update(
                        target_inputs=eval_data["subset_targets"],
                        pred_inputs=eval_data["output"],
                    )
                    golden_metrics = {
                        k: v.item()
                        for k, v in golden_monitor.get().items()
                        if isinstance(v, torch.Tensor)
                    }
                    ceiled_metrics = {
                        k: v.item()
                        for k, v in ceiled_monitor.get().items()
                        if isinstance(v, torch.Tensor)
                    }
                    rich.print(
                        "[yellow][Assign Agent] Golden truth evaluation.[/yellow]"
                    )
                    rich.print(f"[yellow]{golden_metrics}[/yellow]")
                    rich.print(
                        "[blue][Assign Agent] Evaluation conditioned on subset.[/blue]"
                    )
                    rich.print(f"[blue]{ceiled_metrics}[/blue]")
                    with open(save_folder / "assign_golden_metrics.json", "w") as f:
                        json.dump(golden_metrics, f)


if __name__ == "__main__":
    args = Arguments()
    run(args)
