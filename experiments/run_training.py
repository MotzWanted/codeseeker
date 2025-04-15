"""Trains the alignment model."""

from collections import OrderedDict
import functools
import json
import re
import typing as typ
from pathlib import Path


import datasets
import pydantic
import rich
import torch
import transformers
from jinja2 import Environment, FileSystemLoader
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.profilers import PyTorchProfiler
from loguru import logger
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
from peft.tuners.lora.layer import LoraLayer
from prompt_poet import Prompt
from pydantic_settings import BaseSettings
from rich.table import Table
from torch import nn
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.utils import data as torch_data
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import wandb
from wandb.integration.lightning.fabric import WandbLogger

import dataloader
from agents.aligners import templates
from dataloader import mimiciii_50, mimiciv, mimiciv_50
from dataloader.adapt.base import BaseTrainingModel
from dataloader.base import DatasetConfig
from finetune import helpers, loop
from finetune.callback import PrintCallback
from finetune.monitor import MeanMonitor
from tools.exception import dump_exceptions_to_file
from tools.json_logger import JsonLogger
from tools.pbar import IterProgressBar
from tools.pprint import human_format_nb, pprint_parameters_stats

SEGMENTER = dataloader.factory("document", spacy_model="en_core_web_lg")

DATASET_CONFIGS: dict[str, dict] = {
    "debug": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv_50,
        "subsets": ["icd10"],
        "options": {
            "subset_size": 300,
            "adapter": "MimicForTrainingAdapter",
            "order": "alphabetical",
        },
    },
    "mimic-iv-3.0": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv,
        "subsets": ["icd10cm-3.0"],
        "options": {
            "negatives": 100,
            "adapter": "MimicForTrainingAdapter",
        },
    },
    "mimic-iv-3.0-hard-neg": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv,
        "subsets": ["icd10cm-3.0"],
        "options": {
            "negatives": 100,
            "hard_negatives": 1.0,
            "adapter": "MimicForTrainingAdapter",
        },
    },
    "mimic-iii-50": {
        "identifier": "mimic-iii-50",
        "name_or_path": mimiciii_50,
        "options": {"adapter": "MimicForTrainingAdapter", "order": "alphabetical"},
    },
    "mimic-iv-50": {
        "identifier": "mimic-iv-50",
        "name_or_path": mimiciv_50,
        "subsets": ["icd10"],
        "options": {
            "adapter": "MimicForTrainingAdapter",
            "order": "alphabetical",
            "negatives": 50,
        },
    },
}


class Arguments(BaseSettings):
    """Script arguments."""

    project_name: str = "mimic-50"
    dataset: str = "mimic-iv-50"

    version: str = "v0"
    backbone: str = "meta-llama/Llama-3.2-1B"
    output_path: str = Path("~/research/models/{version}-{backbone}-{dataset}-align").expanduser().as_posix()
    debug: int = 1
    # Training details
    prompt_name: str = "icdcm_v2_it"  # lookup in `src/alignment/templates`
    logits_processor: functools.partial[OutlinesLogitsProcessor] | None = None
    # functools.partial(
    #     RegexLogitsProcessor, regex_string=r"(?:[1-9]\d{0,3})(?:,(?:[1-9]\d{0,3})){0,39}"
    # )
    train_size: int = 7_500
    val_size: int = 2_000
    test_size: int = 2_000
    batch_size: int = 1
    eval_batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8

    lr: float = 1e-5
    weight_decay: float = 1e-1
    prompt_loss_weight: float = 0
    num_epochs: int = 1
    seed: int = 1
    # Training strategy - lightning
    strategy: str = "ddp"  # "deepspeed" | "dpp" | "fsdp" | "ddp_spawn"
    devices: list[int] | int | str = [4, 5]
    precision: str = "bf16-true"  # bf16-mixed
    quantize: str = "none"  # "nf4"  # none | nf4
    attn: str = "flash_attention_2"
    ckpt: int = 1
    # Generation config: transformers.GenerationConfig
    do_sample: bool = False
    # Peft
    peft: str = "LORA"  # none | LORA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Loop details
    track_metric: str = "f1_macro"
    tack_mode: str = "max"
    num_workers: int = 16
    log_freq: int = 20
    eval_freq: int = 250
    generate_freq: int = -1
    max_eval: int = 1_024
    pbar_keys: str = r"(target_loss)|(f1_micro)"
    # Tokenizer
    max_new_tokens: int = 64
    max_length: int = 8_094
    filter_longer_than: int = 8_094
    padding: str = "longest"
    truncation: int = 1


ONLY_DIGITS = re.compile(r"[^0-9]")


# Specific to FSDP
def lora_auto_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int, **kwargs) -> bool:
    """Auto-wrap policy for FSDP to wrap only LoRA-specific layers.

    Args:
        module (nn.Module): The module being evaluated for wrapping.
        recurse (bool): Whether to recurse into the module's submodules.
        nonwrapped_numel (int): The number of unwrapped parameters in the current module.
        **kwargs (Any): Additional arguments for future compatibility.

    Returns:
        bool: True if the module should be wrapped by FSDP, False otherwise.
    """
    return isinstance(module, LoraLayer)


LAYERS = {LlamaDecoderLayer}
# Let's make this a config later on or just throw them
# into the Arguments class
FSDP_DEFAULTS = {
    "sharding_strategy": "FULL_SHARD",
    "state_dict_type": "sharded",
    "sync_module_states": True,
    "use_orig_params": True,
    "auto_wrap_policy": functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=LAYERS),
    "activation_checkpointing_policy": LAYERS,  # enables activation checkpointing for the given layers
    "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
    "forward_prefetch": False,
    "cpu_offload": True,
    "limit_all_gathers": True,
}


def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    def sanitize_value(val):
        # Recursively sanitize strings within nested structures
        if isinstance(val, str):
            # Replace non-printable characters with a space
            return re.sub(r"[^\x20-\x7E]", " ", val)
        return val

    sanitized_value = sanitize_value(value)
    return json.dumps(sanitized_value, ensure_ascii=False)


def run(args: Arguments) -> None:
    """Train a Multi-segment classification model."""
    # Raise an error if the strategy is FSDP and quantization is enabled
    if args.strategy == "fsdp" and args.quantize != "none":
        raise ValueError("FSDP does not (currently) support quantization.")

    helpers.setup_env()
    output_path = helpers.setup_output_dir(args.output_path, args)
    if helpers.is_global_zero():
        rich.print(args)

    # Load the Tokenizer
    tokenizer = helpers.load_tokenizer(args.backbone, max_length=args.max_length)

    # Wrap tokenizer and logits processor
    logits_processor = helpers.maybe_wrap_logits_processor(args.logits_processor, tokenizer)
    # Load the dataset
    data, classes = _load_dataset(args, tokenizer=tokenizer)

    if args.strategy == "fsdp":
        strategy = FSDPStrategy(**FSDP_DEFAULTS)
    else:
        strategy = args.strategy

    # Load Lightning Fabric
    fabric = helpers.init_fabric(
        strategy=strategy,
        precision=args.precision,
        devices=args.devices,
        seed=args.seed,
        loggers=[
            JsonLogger(output_path, remove_existing=True),
            WandbLogger(project=args.project_name, config=args.model_dump()),
        ],  # type: ignore
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
            *(
                (
                    PyTorchProfiler(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=args.log_freq),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profile"),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                        with_modules=True,
                    ),
                )
                if args.debug > 1
                else ()
            ),
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
        lora_target_modules=args.lora_target_modules,
    )
    if fabric.is_global_zero:
        rich.print(model)
    # Check if the model uses GQA
    if not model.config.num_key_value_heads < model.config.num_attention_heads:
        logger.warning("Model does not use GQA, which may lead to higher memory consumption.")

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
    train_batch_size = args.micro_batch_size
    effective_batch_size = train_batch_size * args.gradient_accumulation_steps * fabric.world_size
    (logger.info if args.batch_size == effective_batch_size else logger.warning)(
        "Batch size: {train_batch_size}/GPU, Gradient Accumulation: {grad_accum}, Total: {effective_batch}",
        train_batch_size=train_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        effective_batch=effective_batch_size,
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
    eval_monitor = helpers.TrieClassificationMonitor(
        trie=classes, keys=["loss", "prompt_loss", "target_loss"], tokenizer=tokenizer
    )

    # Run the training loop
    run_status = loop.training(
        fabric=fabric,
        model=wrapped_model,
        tokenizer=tokenizer,
        logits_processors=logits_processor,
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
            logits_processors=logits_processor,
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
    if fabric.is_global_zero:
        samples_table = wandb.Table(columns=["prompt", "targets", "predictions"]) if fabric.is_global_zero else None
        per_class_table = wandb.Table(columns=["idx", "code", "desc", "freq", "tp", "fp", "fn", "f1_macro"])
        records_data = test_metrics.pop("sample-predictions")
        for prompt, target, prediction in zip(
            records_data["prompt"], records_data["targets"], records_data["predictions"]
        ):
            samples_table.add_data(fabric.loggers[1].name, prompt, target, prediction)
        fabric.log("test/samples", samples_table)
        table = test_metrics.pop("table")
        metrics_to_add = [table[col] for col in per_class_table.columns]
        for row in zip(*metrics_to_add):
            per_class_table.add_data(*row)
        fabric.log(
            "test/per-class-metrics",
            per_class_table,
        )
        fabric.log_dict({f"test/{k}": v for k, v in test_metrics.items()})
    rich.print(test_metrics)
    with open(output_path / "test_metrics.json", "w") as f:
        test_metrics = {k: _cast_torch(v) for k, v in test_metrics.items()}
        json.dump(test_metrics, f, indent=2)

    # Flush logs
    for log in fabric.loggers:
        log.finalize(run_status)

    logger.info(
        "Training completed, status={status}, model_path={path}", path=output_path.absolute(), status=run_status
    )


def _cast_torch(x: typ.Any) -> typ.Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().mean().item()
    return x


def _load_dataset(
    args: Arguments,
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> tuple[datasets.DatasetDict, OrderedDict[str, str]]:
    """Load and filter the dataset.

    Steps:
    1. Load & validate using `AlignmentModelForTraining`
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
    dset_config: DatasetConfig = DatasetConfig(**DATASET_CONFIGS[args.dataset])
    dset_config.options.prep_map_kws = {"num_proc": args.num_workers}
    data: datasets.DatasetDict = dataloader.load_dataset(dset_config)  # type: ignore
    _validate(data, BaseTrainingModel)
    classes = data["train"]["classes"][0]

    # Filter examples that exceed the maximum length
    if args.filter_longer_than > 0:
        m = sum(len(dset) for dset in data.values())
        data = data.filter(
            functools.partial(
                _tokenize_and_check_length,
                prompt=_PromptWrapper(args.prompt_name),
                tokenizer=tokenizer,
                model_=BaseTrainingModel,
                max_length=args.filter_longer_than,
            ),
            num_proc=args.num_workers,
            desc=f"Filtering examples longer than {args.filter_longer_than} tokens",
        )
        n = sum(len(dset) for dset in data.values())
        logger.info(
            "Filtered {n_filtered}/{n_initial} ({frac:.2%}) examples longer than {max_tokens} tokens",
            n_filtered=m - n,
            n_initial=m,
            frac=(m - n) / m,
            max_tokens=args.filter_longer_than,
        )

    data["train"] = data["train"].shuffle()
    if len(data["train"]) > args.train_size:
        data["train"] = data["train"].select(range(args.train_size))

    if len(data["validation"]) > args.val_size:
        data["validation"] = data["validation"].select(range(args.val_size))

    if len(data["test"]) > args.test_size:
        data["test"] = data["test"].select(range(args.test_size))

    if args.dataset == "debug":
        data["test"] = data["test"].select(range(25))
        data["validation"] = data["validation"].select(range(25))

    # Collect stats
    _fmt = functools.partial(human_format_nb, precision=1)
    stats = [
        *({"name": "data", "split": split, "n": _fmt(len(dset))} for split, dset in data.items()),
        *({"name": "filtered_data", "split": split, "n": _fmt(len(dset))} for split, dset in data.items()),
    ]

    # Print the stats
    if helpers.is_global_zero():
        _plot_rich_table(stats, header="Dataset", add_section=lambda i, _: i % len(data) == 0)

    return data, classes


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


D = typ.TypeVar("D", bound=datasets.Dataset | datasets.DatasetDict)

M = typ.TypeVar("M", bound=pydantic.BaseModel)


def _tokenize_and_check_length(
    row: dict[str, typ.Any],
    *,
    prompt: typ.Callable[[M], tuple[str, str]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model_: typ.Type[M] = BaseTrainingModel,
    max_length: int,
) -> bool:
    """Tokenize the row and check the length."""
    prompt_text, target_text = prompt(model_(**row))
    encoded = tokenizer([prompt_text], [target_text], return_tensors="pt", truncation=False)
    return encoded.input_ids.shape[-1] <= max_length


class _PromptWrapper:
    """Wrapper around the prompt."""

    PATH_TO_TEMPLATES = Path(templates.__file__).parent

    def __init__(self, prompt_name: str, shuffle_targets: bool = False) -> None:
        env = Environment(loader=FileSystemLoader(self.PATH_TO_TEMPLATES))
        loader = typ.cast(FileSystemLoader, env.loader)
        self.raw_template, self.template_path, _ = loader.get_source(env, f"{prompt_name}.yml.j2")
        self.shuffle_targets = shuffle_targets

    def __call__(self, row: BaseTrainingModel) -> tuple[str, str]:
        """Make a training example and format the task as a prompt."""
        classes, targets = row.parse_targets()
        prompt = Prompt(
            raw_template=self.raw_template,
            template_data={**row.model_dump(), "classes": classes, "targets": targets, "custom_tojson": custom_tojson},
        )
        return prompt.string, targets


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

    prompt: typ.Callable[[BaseTrainingModel], tuple[str, str]]

    def __init__(
        self,
        *,
        tokenizer: transformers.PreTrainedTokenizerBase,
        template: str,
        padding: str = "longest",
        truncation: bool = True,
        shuffle_targets: bool = False,
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
        inputs_ = [BaseTrainingModel(**r) for r in inputs]  # type: ignore
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
