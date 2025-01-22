import functools
import json
import re
import typing as typ
from pathlib import Path

import datasets
import pydantic
import rich
import transformers
import wandb
from jinja2 import Environment, FileSystemLoader
from loguru import logger
from prompt_poet import Prompt
from pydantic_settings import BaseSettings
from rich.table import Table
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

import dataloader
from alignment.aligners import templates
from dataloader import mimiciv
from dataloader.adapt.base import BaseTrainingModel
from dataloader.base import DatasetConfig
from finetune import helpers
from tools.pprint import human_format_nb

SEGMENTER = dataloader.factory("document", spacy_model="en_core_web_lg")

DATASET_CONFIGS: dict[str, dict] = {
    "debug": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv,
        "subsets": ["icd10cm-3.0"],
        "options": {
            "negatives": 100,
            "subset_size": 300,
            "segmenter": SEGMENTER,
            "adapter": "MimicIvForTrainingAdapter",
            "order": "alphabetical",
        },
    },
    "mimic-iv-3.0": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv,
        "subsets": ["icd10cm-3.0"],
        "options": {
            "negatives": 100,
            "segmenter": SEGMENTER,
            "adapter": "MimicIvForTrainingAdapter",
        },
    },
    "mimic-iv-3.0-hard-neg": {
        "identifier": "mimic-iv",
        "name_or_path": mimiciv,
        "subsets": ["icd10cm-3.0"],
        "options": {
            "negatives": 100,
            "hard_negatives": 1.0,
            "segmenter": SEGMENTER,
            "adapter": "MimicIvForTrainingAdapter",
        },
    },
}


class Arguments(BaseSettings):
    """Script arguments."""

    project_name: str = "unsloth"
    run_name: str = "debug"
    dataset: str = "debug"

    version: str = "v1"
    backbone: str = "unsloth/llama-3-8b-bnb-4bit"
    output_path: str = Path("~/research/models/{version}-{backbone}-align").expanduser().as_posix()
    debug: int = 0
    # Training details
    prompt_name: str = "icdcm_v2_it"  # lookup in `src/alignment/templates`

    train_size: int = 5_000
    batch_size: int = 1
    eval_batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-1
    prompt_loss_weight: float = 0
    num_epochs: int = 1
    seed: int = 1
    lr_scheduler_type: str = "linear"
    # Unsloth arguments
    dtype: str | None = None
    load_in_4bit: bool = True
    optimizer: str = "adamw_8bit"

    max_seq_len: int = 8096

    # LoRa arguments
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Loop details
    track_metric: str = "f1_macro"
    tack_mode: str = "max"
    num_workers: int = 16
    log_freq: int = 20
    eval_freq: int = 10
    save_steps: int = 100
    generate_freq: int = -1
    max_eval: int = 1_024
    pbar_keys: str = r"(target_loss)|(f1_micro)"
    # Tokenizer
    test_samples: int = 5
    max_new_tokens: int = 64
    max_length: int = 128_000
    filter_longer_than: int = -1
    padding: str = "longest"
    truncation: int = 1


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


def run(args: Arguments):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.backbone,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.prompt_name,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    dataset, num_classes = _load_dataset(
        dataset=args.dataset,
        train_size=args.train_size,
        num_workers=args.num_workers,
        prompt_name=args.prompt_name,
        filter_longer_than=args.filter_longer_than,
        tokenizer=tokenizer,
    )

    monitor = helpers.ClassificationMonitor(keys=["loss"], classes=num_classes, tokenizer=tokenizer)

    wandb.init(project=args.project_name, config=args.model_dump())

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=args.max_seq_len,
        dataset_num_proc=args.num_workers,
        packing=False,
        callbacks=[helpers.ClassificationMonitorCallback(monitor)],
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            fp16=False,
            bf16=True,
            logging_steps=args.log_freq,
            eval_steps=args.eval_freq,
            logging_strategy="steps",
            eval_strategy="steps",
            optim=args.optimizer,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=args.output_path,
            report_to="wandb",
            save_steps=args.save_steps,
            run_name=args.run_name,
        ),
    )

    trainer.train()

    wandb.finish()

    model.save_pretrained_merged(args.output_path, tokenizer, save_method="lora")


def _load_dataset(
    dataset: str,
    num_workers: int,
    prompt_name: str,
    train_size: int,
    filter_longer_than: int,
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> tuple[datasets.DatasetDict, int]:
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
    dset_config: DatasetConfig = DatasetConfig(**DATASET_CONFIGS[dataset])
    dset_config.options.prep_map_kws = {"num_proc": num_workers}
    data: datasets.DatasetDict = dataloader.load_dataset(dset_config)  # type: ignore
    _validate(data, BaseTrainingModel)
    num_classes = len(data["train"]["classes"][0])

    data = data.map(
        _PromptWrapper(prompt_name),
        num_proc=num_workers,
        desc="Formatting prompts",
        remove_columns=_get_dataset(data).column_names,
    )

    # Filter examples that exceed the maximum length
    if filter_longer_than > 0:
        m = sum(len(dset) for dset in data.values())
        data = data.filter(
            functools.partial(
                _tokenize_and_check_length,
                tokenizer=tokenizer,
                max_length=filter_longer_than,
            ),
            num_proc=num_workers,
            desc=f"Filtering examples longer than {filter_longer_than} tokens",
        )
        n = sum(len(dset) for dset in data.values())
        logger.info(
            "Filtered {n_filtered}/{n_initial} ({frac:.2%}) examples longer than {max_tokens} tokens",
            n_filtered=m - n,
            n_initial=m,
            frac=(m - n) / m,
            max_tokens=filter_longer_than,
        )

    data["train"] = data["train"].shuffle()
    if len(data["train"]) > train_size:
        data["train"] = data["train"].select(range(train_size))

    # Collect stats
    _fmt = functools.partial(human_format_nb, precision=1)
    stats = [
        *({"name": "data", "split": split, "n": _fmt(len(dset))} for split, dset in data.items()),
        *({"name": "filtered_data", "split": split, "n": _fmt(len(dset))} for split, dset in data.items()),
    ]

    # Print the stats
    if helpers.is_global_zero():
        _plot_rich_table(stats, header="Dataset", add_section=lambda i, _: i % len(data) == 0)
        # _plot_rich_table(class_stats, header="Dataset", add_section=lambda i, _: i % len(data) == 0)

    return data, num_classes


D = typ.TypeVar("D", bound=datasets.Dataset | datasets.DatasetDict)

M = typ.TypeVar("M", bound=pydantic.BaseModel)


def _tokenize_and_check_length(
    row: dict[str, typ.Any],
    *,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_length: int,
) -> bool:
    """Tokenize the row and check the length."""
    encoded = tokenizer(row["prompt"], ["targets"], return_tensors="pt", truncation=False)
    return encoded.input_ids.shape[-1] <= max_length


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


class _PromptWrapper:
    """Wrapper around the prompt."""

    PATH_TO_TEMPLATES = Path(templates.__file__).parent

    def __init__(self, prompt_name: str):
        env = Environment(loader=FileSystemLoader(self.PATH_TO_TEMPLATES))
        loader = typ.cast(FileSystemLoader, env.loader)
        self.raw_template, self.template_path, _ = loader.get_source(env, f"{prompt_name}.yml.j2")

    def __call__(self, row: dict[str, typ.Any]) -> tuple[str, str]:
        """Make a training example and format the task as a prompt."""
        _m = BaseTrainingModel(**row)
        targets = _m.parse_targets()
        prompt = Prompt(
            raw_template=self.raw_template,
            template_data={**_m.model_dump(), "targets": targets, "custom_tojson": custom_tojson},
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


if __name__ == "__main__":
    args = Arguments()
    run(args)
