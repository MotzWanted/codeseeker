import json
import os
import pathlib
import re
import sys
import typing as typ

import lightning as L
import pydantic
import torch
import transformers
from lightning.fabric.loggers.logger import Logger
from lightning.fabric.wrappers import is_wrapped
from loguru import logger
from peft import mapping as peft_mapping
from peft import utils as peft_utils
from peft.tuners import lora
from transformers.generation import stopping_criteria as generate_stops

from finetune.monitor import MeanAggregator, Monitor, SumAggregator

from .callback import Callback

T = typ.TypeVar("T")


def load_tokenizer(
    backbone: str,
    max_length: int = -1,
) -> transformers.PreTrainedTokenizerBase:
    """Load the tokenizer."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(backbone)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if max_length > 0:
        tokenizer.model_max_length = max_length
    return tokenizer


def load_pretrained_lm(
    backbone: str,
    *,
    quantize: str = "none",
    precision: str = "bf16-true",
    attn: str = "flash_attention_2",
    activation_checkpointing: bool = False,
    peft: str = "none",
    lora_r: int = 64,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: None | str | list[str] = None,
    resize_token_embeddings: int = -1,
) -> transformers.PreTrainedModel:
    """Load a pretrained language model. Apply quantization and PEFT if requested."""
    if quantize != "none":
        try:
            import bitsandbytes as bnb  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError("Please install `pip install bitsandbytes scipy`") from exc

        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=quantize,
            bnb_4bit_compute_dtype=_prec_to_dtype(precision),
        )
        logger.info("Using quantization config: {quant_config}", quant_config=quant_config)
    else:
        quant_config = None

    # Load the Model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        backbone,
        quantization_config=quant_config,
        torch_dtype=_prec_to_dtype(precision) if quant_config is None else None,
        attn_implementation=attn,
    )
    if resize_token_embeddings:
        model.resize_token_embeddings(resize_token_embeddings)

    # Cast parameters and register hooks to enable checkpointing
    if quantize != "none":
        model = peft_utils.other.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=activation_checkpointing,
        )

    # Apply activation checkpointing
    if activation_checkpointing:
        logger.info("Activating gradient checkpointing")
        model.gradient_checkpointing_enable()

    # Preparing the model for PEFT
    if peft != "none":
        logger.info("Preparing model for PEFT `{peft}`", peft=peft)
        model = peft_mapping.get_peft_model(
            model,
            peft_config=peft_mapping.get_peft_config(
                {
                    "peft_type": peft,
                    "r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "target_modules": lora_target_modules,
                    "task_type": "CAUSAL_LM",
                }
            ),
        )

    return model  # type: ignore


def init_and_wrap_optimizer(
    *,
    model: torch.nn.Module,
    fabric: L.Fabric,
    lr: float = 1e-5,
    min_lr_frac: float = 0.1,
    weight_decay: float = 1e-1,
    max_iters: int = 1_000_000,
    use_paged_optim: bool = False,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Initialize the optimizer and scheduler."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if use_paged_optim:
        from bitsandbytes.optim import PagedAdamW

        optimizer = PagedAdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    wrapped_optimizer: torch.optim.Optimizer = fabric.setup_optimizers(optimizer)  # type: ignore

    # Setup the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=lr * min_lr_frac)  # type: ignore
    return wrapped_optimizer, scheduler


def init_fabric(
    *,
    strategy: str,
    precision: str,
    devices: list[int] | int | str = "auto",
    seed: None | int = None,
    loggers: None | list[Logger] = None,
    callbacks: None | list[Callback] = None,
) -> L.Fabric:
    """Initialize the lightning's Fabric."""
    fabric = L.Fabric(
        strategy=strategy,
        accelerator="auto",
        devices=devices,
        precision=precision,  # type: ignore
        loggers=loggers,
        callbacks=callbacks,
    )
    fabric.launch()
    if seed is not None:
        fabric.seed_everything(seed)
    return fabric


def setup_env() -> None:
    """Setup the torch and logging environment."""
    torch.set_float32_matmul_precision("high")
    silent_loggers_on_slave_nodes()


def setup_output_dir(output_path: str, args: pydantic.BaseModel) -> pathlib.Path:
    """Setup the output directory."""
    params = {k: _as_safe_filename(str(v)) for k, v in args.model_dump().items()}
    output_path_ = pathlib.Path(output_path.format(**params))
    if not output_path_.exists():
        output_path_.mkdir(parents=True, exist_ok=True)
    return output_path_


def save_pretrained_model(
    model: transformers.PreTrainedModel,
    *,
    args: pydantic.BaseModel,
    output_path: pathlib.Path,
    tokenizer: transformers.PreTrainedTokenizerBase,
    template: None | str = None,
    metrics: None | dict[str, float | torch.Tensor] = None,
    step: None | int = None,
) -> None:
    """Save the model as a Huggingface `transformers.PreTrainedModel`.

    NOTE: this doesn't handle merging LoRA adapters nor sharded models.
    """
    logger.info("Saving model to {path}", path=output_path.absolute())
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    if template is not None:
        (output_path / "template.txt").write_text(template)
    (output_path / "args.json").write_text(json.dumps(args.model_dump()))
    if metrics is not None:
        metrics = {k: v.mean().item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        if step is not None:
            metrics["step"] = step
        (output_path / "metrics.json").write_text(json.dumps(metrics))


def is_global_zero() -> bool:
    """Check if the current process is the global zero."""
    return (os.environ.get("NODE_RANK", "0") == "0") and (os.environ.get("LOCAL_RANK", "0") == "0")


def unwrapped_model(model: T) -> T:
    """Get the unwrapped model."""
    if is_wrapped(model):
        return model.module  # type: ignore
    return model


def _as_safe_filename(s: T) -> T:
    """Convert a string to a safe filename."""
    if isinstance(s, str):
        return re.sub(r"[^a-zA-Z0-9_\-\.]", "-", s)

    return s


def _prec_to_dtype(x: str) -> torch.dtype:
    _dtypes = {
        "bf16-mixed": torch.bfloat16,
        "bf16-true": torch.bfloat16,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "32": torch.float32,
        "fp32": torch.float32,
        "64": torch.float64,
        "fp64": torch.float64,
    }
    try:
        return _dtypes[x]
    except KeyError as exc:
        raise RuntimeError(f"Unknown precision `{x}`. Choose from `{list(_dtypes.keys())}`.") from exc


def silent_loggers_on_slave_nodes(slave_level: str = "ERROR") -> None:
    """Only allow logging from the global zero rank."""
    if not is_global_zero():
        logger.remove()
        logger.add(sys.stderr, level=slave_level)


class SavePretrainedModelCallback(Callback):
    """Callback to save the model."""

    def __init__(
        self,
        *,
        track_metric: str = "loss",
        track_mode: str = "min",
        args: pydantic.BaseModel,
        output_path: pathlib.Path,
        tokenizer: transformers.PreTrainedTokenizerBase,
        template: None | str = None,
        copy_files: None | list[str | pathlib.Path] = None,
    ) -> None:
        self.track_metric = re.compile(track_metric)
        self.track_mode = track_mode
        self.best = float("inf") if track_mode == "min" else -float("inf")
        # Attrs to save
        self.args = args
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.template = template
        self.copy_files = copy_files or []

    def on_fit_start(self, **kws: typ.Any) -> None:
        """Called at the beginning of the training loop."""
        self.best = float("inf") if self.track_mode == "min" else -float("inf")

        # Copy files
        for file in self.copy_files:
            file_ = pathlib.Path(file)
            if file_.is_file():
                dest = pathlib.Path(self.output_path, file_.name)
                dest.write_text(file_.read_text())

    def _search_metric(self, metrics: typ.Iterable[str]) -> None | str:
        """Search for the metric."""
        matches = []
        for k in metrics:
            if self.track_metric.search(k):
                matches.append(k)

        if len(matches) == 0:
            return None
        if len(matches) > 1:
            raise RuntimeError(f"Found multiple matches for `{self.track_metric}`: {matches}")
        return matches[0]

    def on_validation_end(
        self, *, metrics: dict[str, typ.Any], model: transformers.PreTrainedModel, step: None | int = None
    ) -> None:
        """Called at the end of the validation loop."""
        matched_metric = self._search_metric(metrics)
        if matched_metric is None:
            logger.warning(
                f"Could not find `{self.track_metric}` in `{metrics}`. "
                f"Skipping `{type(self).__name__}.on_validation_end()`"
            )
            return
        value = metrics[matched_metric]
        if (self.track_mode == "min" and value < self.best) or (self.track_mode == "max" and value > self.best):
            self.best = value
            logger.info(
                "Saving model at step {step} with {metric}={value} - path={output_path}",
                step=step,
                metric=self.track_metric,
                value=value.mean().detach().cpu().item() if isinstance(value, torch.Tensor) else value,
                output_path=self.output_path.absolute(),
            )
            # Unwrap the model and save it
            save_pretrained_model(
                model=unwrapped_model(model),
                step=step,
                args=self.args,
                output_path=self.output_path,
                tokenizer=self.tokenizer,
                template=self.template,
                metrics=metrics,
            )


class PatternStoppingCriteria(generate_stops.StoppingCriteria):
    """Stopping criteria based on a list of `input_ids` patterns."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase, patterns: list[str]):
        pattern_token_ids = [tokenizer.encode(p, add_special_tokens=False, return_tensors="pt") for p in patterns]
        self.pattern_token_ids: list[torch.LongTensor] = pattern_token_ids  # type: ignore

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: typ.Any) -> bool:
        """Check if the generated sequence contains one of the patterns."""
        self.pattern_token_ids = [p.to(input_ids.device) for p in self.pattern_token_ids]  # type: ignore
        for pattern in self.pattern_token_ids:
            matches = input_ids[:, -pattern.shape[-1] :] == pattern[None, :]
            if matches.all():
                return True

        return False


class ClassificationMonitor(Monitor):
    """Monitor for classification tasks."""

    def __init__(
        self,
        keys: list[str],
        tokenizer: transformers.PreTrainedTokenizerBase,
        parse_labels_fn: typ.Callable[[str], set[int]],
    ) -> None:
        super().__init__()
        self.keys = keys
        self.tokenizer = tokenizer
        self.parse_labels_fn = parse_labels_fn
        self.aggregators = torch.nn.ModuleDict(
            {
                **{k: SumAggregator() for k in ["tp", "fp", "fn", "tn", "pos_count", "neg_count"]},
                **{k: MeanAggregator() for k in [*self.keys, "_hit"]},
            }  # type: ignore
        )

    def get(self) -> dict[str, torch.Tensor]:
        """Get values from all aggregators."""
        output = {key: self.aggregators[key].get() for key in self.keys}
        tp = self.aggregators["tp"].get()
        fp = self.aggregators["fp"].get()
        fn = self.aggregators["fn"].get()
        tn = self.aggregators["tn"].get()
        pos_targets = self.aggregators["pos_count"].get()
        neg_targets = self.aggregators["neg_count"].get()
        if tp == fp == fn == tn == 0:
            return output
        output["accuracy"] = self.aggregators["_hit"].get()
        output["precision"] = tp / (tp + fp)
        output["recall"] = tp / (tp + fn)
        output["f1"] = 2 * (output["precision"] * output["recall"]) / (output["precision"] + output["recall"])
        output["pos_ratio"] = (tp + fp) / pos_targets
        output["neg_ratio"] = (tn + fn) / neg_targets
        return output

    def update(
        self,
        *,
        target_input_ids: None | torch.Tensor = None,
        preds_input_ids: None | torch.Tensor = None,
        **kws: typ.Any,
    ) -> None:
        """Update the metrics."""
        for key in self.keys:
            self.aggregators[key].update(kws[key])

        if target_input_ids is None or preds_input_ids is None:
            return

        # Compute the true/false negatives/positives
        conf_matrix = self._make_conf_matrix(
            self.tokenizer,
            target_input_ids,
            preds_input_ids,
            self.parse_labels_fn,
        )
        for k, v in conf_matrix.items():
            self.aggregators[k].update(v)

    @staticmethod
    def _make_conf_matrix(
        tokenizer: transformers.PreTrainedTokenizerBase,
        target_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        parse_labels_fn: typ.Callable[[str], set[int]],
    ) -> dict[str, torch.Tensor]:
        """Compute the true/false positives."""
        preds_strs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        target_strs = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
        preds: list[set[int]] = [parse_labels_fn(x) for x in preds_strs]
        targets: list[set[int]] = [parse_labels_fn(x) for x in target_strs]

        if os.environ.get("DEBUG", "0") == "1":
            import rich

            rich.print(
                {
                    "preds_strs": preds_strs,
                    "target_strs": target_strs,
                }
            )

        def _tp(pred: set[int], target: set[int]) -> int:
            """True positives."""
            if target == {0}:
                return 0
            return len(pred & target)

        def _fp(pred: set[int], target: set[int]) -> int:
            """False positives."""
            if target != {0}:
                return 0
            return len(pred - target)

        def _fn(pred: set[int], target: set[int]) -> int:
            """False negatives."""
            if pred == {0} and sum(target) > 0:
                return 1
            return 0

        def _tn(pred: set[int], target: set[int]) -> int:
            """True negatives."""
            if pred == target == {0}:
                return 1
            return 0

        def _pos(target: set[int]) -> int:
            """Count positives"""
            if target != {0}:
                return len(target)
            return 0

        def _neg(target: set[int]) -> float:
            """Count negatives"""
            return int(target == {0})

        def _hit(pred: set[int], target: set[int]) -> int:
            """Hit."""
            return int(pred == target)

        return {
            "tp": torch.tensor([_tp(p, t) for p, t in zip(preds, targets)]),
            "fp": torch.tensor([_fp(p, t) for p, t in zip(preds, targets)]),
            "fn": torch.tensor([_fn(p, t) for p, t in zip(preds, targets)]),
            "tn": torch.tensor([_tn(p, t) for p, t in zip(preds, targets)]),
            "pos_count": torch.tensor([_pos(t) for t in targets]),
            "neg_count": torch.tensor([_neg(t) for t in targets]),
            "_hit": torch.tensor([_hit(p, t) for p, t in zip(preds, targets)]),
        }


class SafeTokenize:
    """A wrapper for truncating and padding the inputs while warning the user."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        *,
        truncation: bool = False,
        return_tensors: str = "pt",
        strict: typ.Literal["pass", "warn", "raise"] = "pass",
        **kwargs: typ.Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.truncation = truncation
        self.kwargs = kwargs
        if return_tensors != "pt":
            raise RuntimeError(f"Can only handle `return_tensors='pt'` but got `{return_tensors}`.")
        self.return_tensors = return_tensors
        self.strict = strict

    def __call__(self, *args: typ.Any, truncation: None | bool = None, **kwargs: typ.Any) -> dict[str, torch.Tensor]:
        """Tokenize the inputs."""
        kws = {**self.kwargs, **kwargs}
        truncation = truncation or self.truncation
        outputs: dict[str, torch.Tensor] = dict(self.tokenizer(*args, **kws, return_tensors=self.return_tensors))
        if self.strict != "pass":
            input_ids = outputs["input_ids"]
            if input_ids.shape[-1] > self.tokenizer.model_max_length:
                msg = (
                    f"Tokenized length {input_ids.shape[-1]} is greater than "
                    f"the maximum length {self.tokenizer.model_max_length}."
                )
                if self.strict == "raise":
                    raise RuntimeError(msg)
                if self.strict == "warn":
                    logger.warning(msg)
                else:
                    raise RuntimeError(f"Unknown strictness `{self.strict}`.")

        if truncation:
            if self.tokenizer.padding_side == "left":
                outputs = {k: v[:, -self.tokenizer.model_max_length :] for k, v in outputs.items()}
            elif self.tokenizer.padding_side == "right":
                outputs = {k: v[:, : self.tokenizer.model_max_length] for k, v in outputs.items()}
            else:
                raise RuntimeError(f"Unknown padding side `{self.tokenizer.padding_side}`.")
        return outputs


def merge_lora(model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:
    """Merge LoRA adapters."""
    for module in model.modules():
        if isinstance(module, lora.LoraLayer):
            module.merge(safe_merge=True)

    return model


def merge_and_unload_lora(model: transformers.PreTrainedModel) -> transformers.PreTrainedModel:
    """Merge LoRA adapters and get rif of the LoRA weights."""
    for module in model.modules():
        if isinstance(module, lora.LoraLayer):
            module.merge_and_unload()
            assert module.merged

    return model


MISTRAL_TARGET_ALL_LAYERS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]

MISTRAL_TARGET_ATTN_LAYERS = [  # TODO
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def get_layer_names(pattern: str) -> str | list[str]:
    """Get the layer names."""
    if pattern.startswith(":"):
        pattern = pattern[1:]
        if pattern == "all":
            return MISTRAL_TARGET_ALL_LAYERS
        if pattern == "attn":
            return MISTRAL_TARGET_ATTN_LAYERS
        raise RuntimeError(f"Unknown pattern `{pattern}`.")

    return pattern
