import functools
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
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
from outlines.models.transformers import TransformerTokenizer

from finetune.monitor import ClassAggregator, MeanAggregator, Monitor

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
        from bitsandbytes.optim import PagedAdamW  # type: ignore

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

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: typ.Any) -> bool:  # type: ignore
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
        classes: int,
        keys: list[str] = [],
        tokenizer: transformers.PreTrainedTokenizerBase | None = None,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.tokenizer = tokenizer
        self.classes = classes
        self.class_aggregators = torch.nn.ModuleDict(
            {
                **{k: ClassAggregator(classes) for k in ["tp", "fp", "fn", "tn"]},
            }
        )
        self.aggregators = torch.nn.ModuleDict(
            {
                **{k: ClassAggregator(classes) for k in ["tp", "fp", "fn", "tn"]},
                **{k: MeanAggregator() for k in [*self.keys, "_hit", "pos_ratio"]},
            }
        )

    def get(self) -> dict[str, torch.Tensor]:
        """Get values from all aggregators."""
        output = {key: self.aggregators[key].get() for key in self.keys}
        tp = self.aggregators["tp"].get()
        fp = self.aggregators["fp"].get()
        fn = self.aggregators["fn"].get()

        # Micro F1
        micro_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
        micro_recall = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
        output["f1_micro"] = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-10)

        # Macro F1 (ignoring TN-only classes)
        precision_per_class = tp / (tp + fp + 1e-10)
        recall_per_class = tp / (tp + fn + 1e-10)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-10)

        # Exclude classes where TP + FP + FN = 0 (TN-only classes)
        valid_classes = (tp + fp + fn) > 0
        if valid_classes.any():  # Ensure there are valid classes
            macro_f1 = f1_per_class[valid_classes].mean()
        else:
            macro_f1 = torch.tensor(0.0)  # Handle edge case where no valid classes exist

        output["f1_macro"] = macro_f1

        # Classification Metrics
        output["accuracy"] = self.aggregators["_hit"].get()
        output["positive_ratio"] = self.aggregators["pos_ratio"].get()
        return output

    def update(
        self,
        *,
        target_input_ids: None | torch.Tensor | list[set[int]] = None,
        preds_input_ids: None | torch.Tensor | list[set[int]] = None,
        **kws: typ.Any,
    ) -> None:
        """Update the metrics."""
        for key in self.keys:
            self.aggregators[key].update(kws[key])

        if target_input_ids is None or preds_input_ids is None:
            return

        if isinstance(target_input_ids, torch.Tensor) and isinstance(preds_input_ids, torch.Tensor):
            target_tokens = self._tokenize_fn(self.tokenizer, target_input_ids)
            preds_tokens = self._tokenize_fn(self.tokenizer, preds_input_ids)
            preds_input_ids = self._parse_tokens_fn(preds_tokens)
            target_input_ids = self._parse_tokens_fn(target_tokens)

        prediction_matrix = list2tensor_vectorized(len(preds_input_ids), self.classes, preds_input_ids)
        target_matrix = list2tensor_vectorized(len(target_input_ids), self.classes, target_input_ids)

        if os.environ.get("DEBUG", "0") == "1":
            import rich

            rich.print(
                {
                    "predictions": preds_input_ids,
                    "targets": target_input_ids,
                }
            )

        # Compute the true/false negatives/positives
        conf_matrix = self._make_conf_matrix(target_matrix, prediction_matrix)
        for k, v in conf_matrix.items():
            self.aggregators[k].update(v)

    @staticmethod
    def _tokenize_fn(tokenizer: transformers.PreTrainedTokenizerBase, input_ids: torch.Tensor) -> list[str]:
        """Tokenize the input."""
        return tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    @staticmethod
    def _parse_tokens_fn(tokens: list[str]) -> list[set[int]]:
        """Parse the tokens."""

        def _parse_one_element(input_string: str) -> set[int]:
            """Parse a single element."""
            return {int(num.strip()) for num in input_string.split(",") if num.strip().isdigit()}

        return [_parse_one_element(x) for x in tokens]

    @staticmethod
    def _make_conf_matrix(targets: torch.Tensor, preds: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the true/false positives."""
        _targets = targets.bool()
        _preds = preds.bool()

        return {
            "tp": (_targets & _preds).sum(dim=0),
            "fp": (_preds & ~_targets).sum(dim=0),
            "fn": (_targets & ~_preds).sum(dim=0),
            "tn": (~_targets & ~_preds).sum(dim=0),
            "_hit": (~(_targets ^ _preds)).all(dim=1).float(),  # counting row wise exact matches
            "pos_ratio": preds.sum() / targets.sum(),
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
        outputs: dict[str, torch.Tensor] = dict(self.tokenizer(*args, **kws, return_tensors=self.return_tensors))  # type: ignore
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


def maybe_wrap_logits_processor(
    processor_fn: functools.partial[OutlinesLogitsProcessor] | None,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> transformers.LogitsProcessorList | None:
    """Wrap the logits processor."""
    if processor_fn is None:
        return None
    _tokenizer = TransformerTokenizer(tokenizer=tokenizer)
    return transformers.LogitsProcessorList([processor_fn(tokenizer=_tokenizer)])


def list2tensor_vectorized(dim_x: int, dim_y: int, indices: list[set[int | float]]) -> torch.Tensor:
    """Convert a list of indices to a sparse tensor."""
    row_indices = []
    col_indices = []
    values = []

    for i, preds in enumerate(indices):
        preds = torch.tensor(list(preds), dtype=torch.float32)  # Convert the set to a PyTorch tensor
        pred_signs = torch.where(preds < 0, -1, 1)  # Determine the sign
        pred_indices = torch.abs(preds) - 1  # Get absolute indices (0-based)

        # Filter valid indices (within bounds)
        valid_mask = (pred_indices >= 0) & (pred_indices < dim_y)
        valid_count = int(valid_mask.sum().item())  # Explicitly convert to Python int
        row_indices.extend([i] * valid_count)  # Repeat row index for valid preds
        col_indices.extend(pred_indices[valid_mask].to(torch.int).tolist())  # Valid column indices
        values.extend(pred_signs[valid_mask].tolist())  # Valid signs

    # Convert row_indices, col_indices, and values to PyTorch tensors
    row_indices = torch.tensor(row_indices, dtype=torch.long)
    col_indices = torch.tensor(col_indices, dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float32)

    # Create the sparse tensor
    sparse_tensor = torch.zeros((dim_x, dim_y), dtype=torch.float32)
    sparse_tensor[row_indices, col_indices] = values

    return sparse_tensor
