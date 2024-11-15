"""Trains the alignment model."""

from collections import defaultdict
import copy
import functools
import json
from pathlib import Path
import re
import typing as typ

import datasets
import numpy as np
import pydantic
from pydantic_settings import BaseSettings
import rich
import torch
import transformers
from loguru import logger
from prompt_poet import Prompt
from rich.table import Table
from torch.utils import data as torch_data
from wandb.integration.lightning.fabric import WandbLogger

from dataloader.adapters.alignment import AlignmentModelForTraining
from finetune import helpers, loop
from finetune.callback import PrintCallback
from finetune.monitor import MeanMonitor
from tools.exception import dump_exceptions_to_file
from tools.json_logger import JsonLogger
from tools.pbar import IterProgressBar
from tools.pprint import human_format_nb, pprint_class_balance, pprint_parameters_stats

from alignment.aligners import templates


class FilterSyntheticData:
    def __init__(self, prob_threshold: float) -> None:
        self.prob_threshold = prob_threshold

    def __call__(self, batch: dict[str, list[typ.Any]]) -> dict[str, list[typ.Any]]:
        output = defaultdict(list)
        batch_size = len(batch[list(batch.keys())[0]])
        # Iterate through each example in the batch
        for idx in range(batch_size):
            # Skip if the segments has not been splitted properly
            if len(batch["segments"][idx]) < 2:
                continue
            np_targets = np.array(batch["targets"][idx])
            np_probs = np.array(batch["probabilities"][idx])

            # Ensure the sizes of targets and probabilities match
            if np_probs.size != np_targets.size:
                continue

            # Create a mask where probabilities are above the threshold
            mask = np_probs >= self.prob_threshold

            # Skip if no targets are above the threshold
            if not mask.any():
                continue

            output["aid"].append(batch["aid"][idx])
            output["entities"].append(batch["entities"][idx])
            output["segment"].append(batch["segment"][idx])
            output["targets"].append(list(np_targets[mask]))
            output["probabilities"].append(list(np_probs[mask]))
            output["extras"].append(batch["extras"][idx])
        return output


class Arguments(BaseSettings):
    """Script arguments."""

    project_name: str = "nbme-align"
    data_path: str = "/home/amo/data/patient-note-alignment/silver-labels/v2/"

    version: str = "v1"
    backbone: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_path: str = "/mnt/md0/amo/models/{version}-{backbone}-align"
    debug: int = 0
    # Training details
    prompt_name: str = "fine-tune-it"  # lookup in `src/alignment/templates`
    train_size: int = 50_000
    batch_size: int = 32
    eval_batch_size: int = 8
    micro_batch_size: int = 1

    lr: float = 1e-5
    weight_decay: float = 1e-1
    prompt_loss_weight: float = 0
    num_epochs: int = 1
    seed: int = 1
    # Dataset filtering
    negative_ratio: float = 0.3  # the ratio of negative examples in the training set
    prob_threshold: float = 0.0  # the probability threshold for filtering silver labels in training samples
    # Training strategy - lightning
    strategy: str = "ddp"  # "deepspeed" | "dpp" | "fsdp" | "ddp_spawn"
    devices: list[int] | int | str = [6, 7]
    precision: str = "bf16-true"  # bf16-mixed
    quantize: str = "nf4"  # none | nf4
    attn: str = "flash_attention_2"
    ckpt: int = 1
    # Generation config: transformers.GenerationConfig
    do_sample: bool = False
    # temperature: float = 0.0
    # top_p: float = 0.0
    # Peft
    peft: str = "LORA"  # none | LORA
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = ":all"
    # Loop details
    track_metric: str = "f1"
    tack_mode: str = "max"
    num_workers: int = 16
    log_freq: int = 20
    eval_freq: int = 100
    generate_freq: int = -1
    max_eval: int = 1_024
    pbar_keys: str = r"(target_loss)|(f1)"
    # Tokenizer
    max_new_tokens: int = 64
    max_length: int = 8_096
    filter_longer_than: int = 8_000
    padding: str = "longest"
    truncation: int = 1


ONLY_DIGITS = re.compile(r"[^0-9]")


def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    return json.dumps(value, ensure_ascii=False)


def _parse_one(x: str) -> None | int:
    try:
        return int(ONLY_DIGITS.sub("", x))
    except ValueError:
        return None


def parse_prediction(pred: str) -> set[int]:
    """Cast the generated prediction to a set of integers."""
    out = (_parse_one(x) for x in pred.split(","))
    return {x for x in out if x is not None}


def run(args: Arguments) -> None:  # noqa: C901, PLR0915
    """Train a Multi-segment classification model."""
    helpers.setup_env()
    output_path = helpers.setup_output_dir(args.output_path, args)
    if helpers.is_global_zero():
        rich.print(args)

    # Load the Tokenizer
    tokenizer = helpers.load_tokenizer(args.backbone, max_length=args.max_length)

    # Load the dataset
    data = _load_dataset(args, tokenizer=tokenizer)

    # Load Lightning Fabric
    fabric = helpers.init_fabric(
        strategy=args.strategy,
        precision=args.precision,
        devices=args.devices,
        seed=args.seed,
        loggers=[JsonLogger(output_path, remove_existing=True), WandbLogger(project=args.project_name)],  # type: ignore
        callbacks=[
            helpers.SavePretrainedModelCallback(
                track_metric=args.track_metric,
                track_mode=args.tack_mode,
                args=args,
                output_path=output_path,
                tokenizer=tokenizer,
                template=args.prompt_name,
                copy_files=[templates.__file__],
            ),
            *((PrintCallback(),) if args.debug else ()),
        ],
    )

    # Load the Pretrained Language Model
    # NOTE: this needs to happen after launching Fabric, else some parameters are not placed on the right device.
    model = helpers.load_pretrained_lm(
        args.backbone,
        resize_token_embeddings=len(tokenizer),
        quantize=args.quantize,
        precision=args.precision,
        attn=args.attn,
        activation_checkpointing=args.ckpt > 0,
        peft=args.peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=helpers.get_layer_names(args.lora_target_modules),
    )
    if fabric.is_global_zero:
        rich.print(model)

    # Wrap the model
    wrapped_model: transformers.PreTrainedModel = fabric.setup_module(model)  # type: ignore
    if fabric.is_global_zero:
        pprint_parameters_stats(wrapped_model)

    # Setup the Optimizer and Wrap it
    wrapped_optimizer, scheduler = helpers.init_and_wrap_optimizer(
        model=wrapped_model,
        fabric=fabric,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_iters=(len(data["train"]) * args.num_epochs) // args.batch_size,
        use_paged_optim=args.quantize != "none",
    )

    # Define the batch-size per device
    train_batch_size = -(-args.batch_size // fabric.world_size)
    (logger.info if args.batch_size % fabric.world_size == 0 else logger.warning)(
        "Batch size: {train_batch_size}/GPU (total={global_batch})",
        train_batch_size=train_batch_size,
        global_batch=train_batch_size * fabric.world_size,
    )

    # Build the collate fns ; list[dict] -> dict[str, torch.Tensor]
    collate_fn: typ.Callable[..., _AlignCollateFn] = functools.partial(
        _AlignCollateFn,
        template=args.prompt_name,
        tokenizer=tokenizer,
        padding=args.padding,
        truncation=bool(args.truncation),
    )  # type: ignore
    train_dataloader, valid_dataloader, test_dataloader = fabric.setup_dataloaders(
        *(
            torch_data.DataLoader(
                data[split],  # type: ignore
                batch_size=bs,
                num_workers=args.num_workers,
                collate_fn=collate_fn(),
                pin_memory=True,
                shuffle=True,
            )
            for split, bs in [
                ("train", train_batch_size),
                ("validation", args.eval_batch_size),
                ("test", args.eval_batch_size),
            ]
        )
    )

    # Define the train/valid monitors
    train_monitor = MeanMonitor(keys=["loss", "prompt_loss", "target_loss"])
    eval_monitor = helpers.ClassificationMonitor(
        keys=["loss", "prompt_loss", "target_loss"],
        tokenizer=tokenizer,
        parse_labels_fn=parse_prediction,
    )

    # Run the training loop
    run_status = loop.training(
        fabric=fabric,
        model=wrapped_model,
        tokenizer=tokenizer,
        generation_config=transformers.GenerationConfig(do_sample=args.do_sample),
        optimizer=wrapped_optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        train_monitor=train_monitor.to(fabric.device, dtype=torch.float64),
        valid_monitor=eval_monitor.to(fabric.device, dtype=torch.float64),
        num_epochs=args.num_epochs,
        micro_batch_size=args.micro_batch_size,
        pbar_keys=args.pbar_keys,
        lm_prompt_weight=args.prompt_loss_weight,
        eval_freq=args.eval_freq,
        generate_freq=args.generate_freq,
        log_freq=args.log_freq,
        max_eval=args.max_eval,
        max_new_tokens=args.max_new_tokens,
        stopping_criteria=helpers.PatternStoppingCriteria(
            tokenizer=tokenizer,
            patterns=[tokenizer.eos_token],
        ),
    )

    # Run on the test set
    with IterProgressBar(disable=not fabric.is_global_zero) as pbar:
        test_metrics = loop.evaluate(
            step=-1,
            fabric=fabric,
            model=wrapped_model,
            tokenizer=tokenizer,
            generation_config=transformers.GenerationConfig(do_sample=args.do_sample),
            dataloader=test_dataloader,
            monitor=eval_monitor.to(fabric.device, dtype=torch.float64),
            max_eval=args.max_eval,
            max_new_tokens=args.max_new_tokens,
            pbar=pbar,
            task_type="test",
            generate=True,
            stopping_criteria=helpers.PatternStoppingCriteria(
                tokenizer=tokenizer,
                patterns=[tokenizer.eos_token],
            ),
        )
    with open(output_path / "test_metrics.json", "w") as f:
        test_metrics = {k: _cast_torch(v) for k, v in test_metrics.items()}
        json.dump(test_metrics, f, indent=2)
    rich.print(test_metrics)

    # Flush logs
    for log in fabric.loggers:
        log.finalize(run_status)

    logger.info(
        "Training completed, status={status}, model_path={path}", path=output_path.absolute(), status=run_status
    )


def _cast_torch(x: typ.Any) -> typ.Any:  # noqa: ANN401
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().mean().item()
    return x


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


def _load_dataset(
    args: Arguments,
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> datasets.DatasetDict:
    """Load and filter the dataset.

    Steps:
    1. Load & validate using `AlignmentModelForTraining`
    2. Clean and filter the dataset based on a probability threshold
    3. Sample dataset if needed
    4. Filter by length: compose the prompts+targets and check that the total lengths falls within limits
    5. Compute stats
    """

    def _validate(data: datasets.DatasetDict, model: typ.Type[pydantic.BaseModel], n_samples: int = 3) -> None:
        """Validate the data."""
        for dataset in data.values():
            for i, row in enumerate(dataset):
                model(**row)
                if i >= n_samples:
                    break

    # Load the datasets & validate
    data: datasets.DatasetDict = datasets.load_from_disk(args.data_path)  # type: ignore
    _validate(data, AlignmentModelForTraining)

    adapted_data: datasets.DatasetDict = copy.deepcopy(data)
    # Filter the dataset based on probability threshold
    adapted_data["train"] = adapted_data["train"].filter(
        FilterSyntheticData(prob_threshold=args.prob_threshold),
        num_proc=args.num_workers,
        batched=True,
        batch_size=args.batch_size,
        desc=f"Filtering target predictions with probabilities lower than {args.prob_threshold}",
    )

    # Check the ratio of negatives and maybe reduce the number of negative examples
    all_negative_indices = [i for i, pred in enumerate(adapted_data["train"]["targets"]) if pred == [0]]
    negative_ratio = np.round(len(all_negative_indices) / len(adapted_data["train"]), 2)
    if negative_ratio > args.negative_ratio:
        logger.info(
            f"Reducing the ratio negatives in train from {negative_ratio} to {args.negative_ratio}.",
        )
        n_negatives = int(args.negative_ratio * len(adapted_data["train"]))
        np.random.shuffle(all_negative_indices)
        less_negative_indices = all_negative_indices[:n_negatives]
        positive_indices = [i for i in range(len(adapted_data["train"])) if i not in all_negative_indices]
        selected_indices = less_negative_indices + positive_indices
        adapted_data["train"] = adapted_data["train"].select(selected_indices)

    # Filter examples that exceed the maximum length
    if args.filter_longer_than > 0:
        m = sum(len(dset) for dset in adapted_data.values())
        adapted_data = adapted_data.filter(
            functools.partial(
                _tokenize_and_check_length,
                prompt=_PromptWrapper(args.prompt_name),
                tokenizer=tokenizer,
                model_=AlignmentModelForTraining,
                max_length=args.filter_longer_than,
            ),
            num_proc=args.num_workers,
            desc=f"Filtering examples longer than {args.filter_longer_than} tokens",
        )
        n = sum(len(dset) for dset in adapted_data.values())
        logger.info(
            "Filtered {n_filtered}/{n_initial} ({frac:.2%}) examples longer than {max_tokens} tokens",
            n_filtered=m - n,
            n_initial=m,
            frac=(m - n) / m,
            max_tokens=args.filter_longer_than,
        )

    adapted_data["train"] = adapted_data["train"].shuffle()
    if len(adapted_data["train"]) > args.train_size:
        adapted_data["train"] = adapted_data["train"].select(range(args.train_size))

    # Collect stats
    _fmt = functools.partial(human_format_nb, precision=1)
    stats = [
        *({"name": "data", "split": split, "n": _fmt(len(dset))} for split, dset in data.items()),
        *({"name": "filtered_data", "split": split, "n": _fmt(len(dset))} for split, dset in adapted_data.items()),
    ]
    class_stats = [
        *(
            {"name": "data", "split": split, "class_ballance": pprint_class_balance(dset["targets"])}
            for split, dset in data.items()
        ),
        *(
            {"name": "filtered_data", "split": split, "class_ballance": pprint_class_balance(dset["targets"])}
            for split, dset in adapted_data.items()
        ),
    ]

    # Print the stats
    if helpers.is_global_zero():
        _plot_rich_table(stats, header="Dataset", add_section=lambda i, _: i % len(data) == 0)
        _plot_rich_table(class_stats, header="Dataset", add_section=lambda i, _: i % len(data) == 0)

    return adapted_data


D = typ.TypeVar("D", bound=datasets.Dataset | datasets.DatasetDict)

M = typ.TypeVar("M", bound=pydantic.BaseModel)


def _tokenize_and_check_length(
    row: dict[str, typ.Any],
    *,
    prompt: typ.Callable[[M], tuple[str, str]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model_: typ.Type[M] = AlignmentModelForTraining,
    max_length: int,
) -> bool:
    """Tokenize the row and check the length."""
    prompt_text, target_text = prompt(model_(**row))
    encoded = tokenizer([prompt_text], [target_text], return_tensors="pt", truncation=False)
    return encoded.input_ids.shape[-1] <= max_length


def _plot_rich_table(
    data: list[dict[str, typ.Any]],
    header: str,
    add_section: None | typ.Callable[[int, str], bool] = None,
) -> None:
    """Plot a rich table."""
    table = Table(title=header)
    columns: set[str] = {k for row in data for k in row}
    for i, col in enumerate(columns):
        table.add_column(col, justify="left", style="bold magenta" if i == 0 else "cyan")
    for j, row in enumerate(data):
        if add_section is not None and add_section(j, row["name"]):
            table.add_section()
        table.add_row(*[str(row.get(col, "--")) for col in columns])

    rich.print(table)


class _PromptWrapper:
    """Wrapper around the prompt."""

    PATH_TO_TEMPLATES = Path(templates.__file__).parent

    def __init__(self, prompt_name: str):
        self.template_path = self.PATH_TO_TEMPLATES / f"{prompt_name}.yml.j2"
        self._core = functools.partial(Prompt, template_path=self.template_path)

    def __call__(self, row: AlignmentModelForTraining) -> tuple[str, str]:
        """Make a training example and format the task as a prompt."""
        targets = row.parse_targets()
        prompt = self._core(template_data={**row.model_dump(), "targets": targets, "custom_tojson": custom_tojson})
        return prompt.string, targets


class _AlignCollateFn:
    """Collate function for training an alignment model.

    This functions gathers multiple dataset rows, and format the task as a prompt.

    Output (dict[str, torch.Tensor]):
        - input_ids: torch.Tensor
        - attention_mask: torch.Tensor
        - token_type_ids: torch.Tensor
        - prompt_input_ids: torch.Tensor
        - prompt_attention_mask: torch.Tensor
        - target_input_ids: torch.Tensor
        - target_attention_mask: torch.Tensor
    """

    prompt: typ.Callable[[AlignmentModelForTraining], tuple[str, str]]

    def __init__(  # noqa: PLR0913
        self,
        *,
        tokenizer: transformers.PreTrainedTokenizerBase,
        template: str,
        padding: str = "longest",
        truncation: bool = True,
        seed: None | int = None,
    ) -> None:
        self.prompt = _PromptWrapper(template)
        self.tokenize_fn = helpers.SafeTokenize(
            tokenizer=tokenizer,
            padding=padding,
            truncation=truncation,
            strict="warn",
        )
        self.seed = seed
        self._lookup = None
        self._rng = None

    @property
    def eos_token(self) -> str:
        """Get the EOS token."""
        return self.tokenize_fn.tokenizer.eos_token

    def __getstate__(self) -> dict:
        """Get the state."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:
        """Set the state."""
        self.__dict__.update(state)

    @dump_exceptions_to_file
    def __call__(self, inputs: list[dict[str, typ.Any]]) -> dict[str, torch.Tensor]:
        """Collate the inputs."""
        prompt_texts: list[str] = []
        target_texts: list[str] = []
        inputs_ = [AlignmentModelForTraining(**r) for r in inputs]  # type: ignore
        for i, row_i in enumerate(inputs_):
            # Format the sample as a prompt
            prompt, target = self.prompt(row_i)
            prompt_texts.append(prompt)
            target_texts.append(target)

        # Add EOS tokens
        target_texts = [f"{target}{self.eos_token}" for target in target_texts]

        # Tokenize the texts & return
        return {
            **dict(
                self.tokenize_fn(
                    prompt_texts,
                    target_texts,
                    return_token_type_ids=True,
                )
            ),
            **{f"prompt_{k}": v for k, v in self.tokenize_fn(prompt_texts).items()},
            **{f"target_{k}": v for k, v in self.tokenize_fn(target_texts).items()},
        }


if __name__ == "__main__":
    args = Arguments()
    run(args)
