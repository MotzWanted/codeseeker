from collections import defaultdict
import functools
import math
import re
import time
import typing as typ

import lightning as L
import torch
import transformers
import wandb
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.wrappers import is_wrapped
from loguru import logger
from rich import progress
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from transformers.generation import stopping_criteria as generate_stops

from tools.chrono import Chrono
from tools.pbar import IterProgressBar

from finetune.monitor import Monitor

T = typ.TypeVar("T")


def summon_params_if_fsdp(func):
    @functools.wraps(func)
    def wrapper(model, *args, **kwargs):
        # Only summon params if it's a generate() call and model is in eval mode
        if is_fsdp(model) and not model.training and func.__name__ == "_lm_generate":
            with FSDP.summon_full_params(model, offload_to_cpu=True):
                return func(model, *args, **kwargs)
        return func(model, *args, **kwargs)

    return wrapper


def unwrap_model_if_wrapped(func):
    @functools.wraps(func)
    def wrapper(model, *args, **kwargs):
        if is_wrapped(model):
            model = model.module
        return func(model, *args, **kwargs)

    return wrapper


def is_fsdp(model: T) -> bool:
    """Check if the model is wrapped in FSDP."""
    return isinstance(model._strategy, FSDPStrategy)


def training(
    *,
    fabric: L.Fabric,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    logits_processors: transformers.LogitsProcessorList | None = None,
    generation_config: transformers.GenerationConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: None | torch.optim.lr_scheduler.LRScheduler = None,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader | dict[str, DataLoader],
    train_monitor: Monitor,
    valid_monitor: Monitor,
    num_epochs: int,
    max_steps: None | int = None,
    micro_batch_size: None | int = None,
    gradient_accumulation_steps: int = 1,
    pbar_keys: str = r"(target_loss)|(accuracy)",
    lm_prompt_weight: float = 0,
    eval_freq: int = 100,
    generate_freq: int = -1,
    log_freq: int = 5,
    max_eval: None | int = None,
    max_new_tokens: None | int = None,
    stopping_criteria: None | generate_stops.StoppingCriteria | list[generate_stops.StoppingCriteria] = None,
) -> typ.Literal["completed", "interrupted", "failed"]:
    """Main training loop.

    Trains a model for `num_epochs` epochs on the given dataloaders.
    Test it evert `eval_freq` steps on the validation dataloader and log the metrics every `log_freq` steps.

    NOTE: Make sure the collate functions output batches with keys:
        - input_ids
        - attention_mask
        - token_type_ids
        - prompt_token_ids
        - prompt_attention_mask
        - target_token_ids
        - target_attention_mask

    Args:
        fabric (L.Fabric): The lightning fabric module.
        model (transformers.PreTrainedModel): The model to train, must already be wrapped with fabric.
        optimizer (torch.optim.Optimizer): Torch optimizer wrapped with fabric.
        scheduler (None | torch.optim.lr_scheduler.LRScheduler, optional): Learning rate scheduler.
        train_dataloader (torch.utils.data.DataLoader): Training dataloader.
        valid_dataloader (torch.utils.data.DataLoader | dict): Validation dataloader(s).
        train_monitor (Monitor): Monitors the training metrics
        valid_monitor (Monitor): Monitors the validation metrics
        num_epochs (int): Number of epochs to train for.
        max_steps (None | int, optional): Maximum number of steps to train for.
        micro_batch_size (None | int, optional): Micro-batch size to use for each forward pass.
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before stepping.
        pbar_keys (str , optional): Keys to display in the progress bar. Regex pattern.
        lm_prompt_weight (float, optional): Weight of prompt in the language modeling loss.
        eval_freq (int, optional): Evaluate every `eval_freq` steps.
        generate_freq (int, optional): Generate every `generate_freq` steps.
        log_freq (int, optional): Log the metrics every `log_freq` steps.
        max_eval (None | int, optional): Maximum number of points to evaluate on.
        max_new_tokens (None | int, optional): Maximum number of new tokens to generate in each eval step.
        stopping_criteria (None | generate_stops.StoppingCriteria | list[generate_stops.StoppingCriteria], optional):
            Custom stopping criteria for the generation.

    Returns:
        str: The status of the training loop: "completed", "interrupted" or "failed".
    """
    if not is_wrapped(model):
        raise ValueError(
            "The model must be wrapped with the lightning fabric. Use `fabric.setup_module(model)` to do so."
        )
    if not is_wrapped(optimizer):
        raise ValueError(
            "The optimizer must be wrapped with the lightning fabric. Use `fabric.setup_optimizers(optimizer)` to do so"
        )

    generate_freq = eval_freq if generate_freq < 0 else generate_freq
    if not generate_freq % eval_freq == 0:
        raise ValueError("`generate_freq` must be a multiple of `eval_freq`.")

    micro_batch_size = _strictly_pos_or_none(micro_batch_size)
    max_steps = _strictly_pos_or_none(max_steps)
    step: int = 0
    train_metrics = {}
    eval_metrics = {}
    _pbar_info = functools.partial(_format_pbar_info, pattern=re.compile(pbar_keys))
    _run_status = "completed"
    _chrono = Chrono(buffer_size=10)
    train_monitor.reset()
    fabric.call("on_fit_start", step=step, model=model)
    epoch_length = len(train_dataloader)
    samples_table = (
        wandb.Table(columns=["run", "step", "prompt", "targets", "predictions"]) if fabric.is_global_zero else None
    )
    per_class_table = (
        wandb.Table(columns=["idx", "code", "desc", "freq", "tp", "fp", "fn", "f1_macro"])
        if fabric.is_global_zero
        else None
    )

    try:
        for epoch in range(num_epochs):
            fabric.call("on_epoch_start", step=step)
            _chrono.reset()
            epoch_n_steps = epoch_length if max_steps is None else min(epoch_length, max_steps - step)
            with IterProgressBar(disable=not fabric.is_global_zero) as pbar:
                train_task = pbar.add_task(
                    f"Epoch {1+epoch}/{num_epochs}",
                    total=epoch_n_steps,
                    info=_pbar_info(f"0/{epoch_n_steps} (0)", train_metrics, eval_metrics),
                )

                for it, batch in enumerate(train_dataloader):
                    if it > epoch_n_steps:
                        break
                    fabric.call("on_train_batch_start", batch=batch, step=step)

                    # Forward & backward pass with gradient accumulation
                    n_micro_steps = (
                        math.ceil(batch["input_ids"].shape[0] / micro_batch_size) if micro_batch_size is not None else 1
                    )
                    _chrono.start()

                    accumulated_loss = 0.0
                    for m, micro_batch in enumerate(_iterate_micro_batches(batch, micro_batch_size)):
                        output = _lm_forward(model, micro_batch, lm_prompt_weight)
                        loss = output["loss"] / gradient_accumulation_steps  # Scale loss
                        accumulated_loss += output["loss"].item()

                        with fabric.no_backward_sync(model, enabled=m < n_micro_steps - 1):
                            fabric.backward(loss)

                        fabric.call("on_train_batch_end", batch=batch, output=output, step=step)
                        train_monitor.update(**batch, **output)

                    # Optimizer step after accumulating gradients
                    if (it + 1) % gradient_accumulation_steps == 0 or (it + 1) == len(train_dataloader):
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

                    step += 1
                    _chrono.stop()

                    # Evaluate
                    if step % eval_freq == 0:
                        batch = None
                        torch.cuda.empty_cache()
                        model.eval()
                        fabric.call("on_validation_start", step=step)
                        evaluate_fn = functools.partial(
                            evaluate,
                            step=step,
                            fabric=fabric,
                            model=model,
                            logits_processors=logits_processors,
                            max_eval=max_eval,
                            pbar=pbar,
                            lm_prompt_weight=lm_prompt_weight,
                            max_new_tokens=max_new_tokens,
                            stopping_criteria=stopping_criteria,
                            monitor=valid_monitor,
                            generate=step % generate_freq == 0,
                        )
                        if isinstance(valid_dataloader, dict):
                            eval_metrics_ = {}
                            for task_name, dataloader in valid_dataloader.items():
                                eval_metrics_[task_name] = evaluate_fn(
                                    tokenizer=tokenizer,
                                    dataloader=dataloader,
                                    generation_config=generation_config,
                                    task_type=f"Valid({task_name})",
                                )
                            eval_metrics_ = _flatten_dict(eval_metrics_)
                        else:
                            eval_metrics_ = evaluate_fn(
                                tokenizer=tokenizer, dataloader=valid_dataloader, generation_config=generation_config
                            )
                        if fabric.is_global_zero:
                            records_data = eval_metrics_.pop("sample-predictions")
                            for prompt, target, prediction in zip(
                                records_data["prompt"], records_data["targets"], records_data["predictions"]
                            ):
                                samples_table.add_data(fabric.loggers[1].name, step, prompt, target, prediction)
                            fabric.log("eval/samples", samples_table)
                            table = eval_metrics_.pop("table")
                            metrics_to_add = [table[col] for col in per_class_table.columns]
                            for row in zip(*metrics_to_add):
                                per_class_table.add_data(*row)
                            fabric.log(
                                "eval/per-class-metrics",
                                per_class_table,
                            )
                            # multi_line_chart = wandb.plot.line_series(
                            #     xs=per_class_table.get_column("Step"),
                            #     ys=per_class_table.get_column("f1_macro"),
                            #     keys=per_class_table.get_column("Code"),
                            #     title="Per Class F1-Macro",
                            #     xname="Step",
                            # )
                            # fabric.log("eval/f1_macro_multi", multi_line_chart)
                            fabric.log_dict({f"eval/{k}": v for k, v in eval_metrics_.items()}, step=step)
                        eval_metrics.update(eval_metrics_)
                        fabric.call("on_validation_end", step=step, model=model, metrics=eval_metrics)
                        model.train()

                    # Update progress bar
                    pbar.update(
                        train_task,
                        completed=1 + it,
                        info=_pbar_info(
                            step=f"{1+it}/{epoch_n_steps} ({step})",
                            train_metrics={**train_metrics, "loss": accumulated_loss / gradient_accumulation_steps},
                            eval_metrics=eval_metrics,
                            avg_elapsed_time=_chrono.get_avg_laps_per_second(),
                        ),
                        refresh=True,
                    )

                    # Compute, reset, and log metrics
                    if step % log_freq == 0:
                        train_metrics.update(train_monitor.compute())
                        if fabric.is_global_zero:
                            fabric.log_dict(
                                {
                                    **{f"train/{k}": v for k, v in train_metrics.items()},
                                    **_fetch_lrs(optimizer),
                                    "epoch": epoch,
                                },
                                step=step,
                            )

                pbar.remove_task(train_task)
                fabric.call("on_epoch_end", step=step)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupted.")
        _run_status = "interrupted"
    except Exception as e:
        logger.exception(e)
        _run_status = "failed"

    fabric.call("on_fit_end", step=step, status=_run_status, model=model)
    return _run_status


@torch.no_grad()
def evaluate(
    *,
    step: int,
    fabric: L.Fabric,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    logits_processors: None | transformers.LogitsProcessorList = None,
    generation_config: transformers.GenerationConfig,
    dataloader: DataLoader,
    monitor: Monitor,
    max_eval: None | int = None,
    pbar: progress.Progress,
    lm_prompt_weight: float = 0,
    max_new_tokens: None | int = None,
    stopping_criteria: None | generate_stops.StoppingCriteria | list[generate_stops.StoppingCriteria] = None,
    task_type: str = "validation",
    generate: bool = True,
) -> dict[str, typ.Any]:
    """Evaluation loop.

    Args:
        step (int): Current training step.
        fabric (L.Fabric): Lightning fabric module.
        model (transformers.PreTrainedModel): Model to evaluate, must be wrapped in fabric.
        dataloader (torch.utils.data.DataLoader): Dataloader to evaluate on.
        monitor (Monitor): Compute the metrics.
        max_eval (None | int, optional): Maximum number of points to evaluate on.
        pbar (progress.Progress): Rich progress bar.
        lm_prompt_weight (float, optional): Weight of prompt in the language modeling loss.
        max_new_tokens (None | int, optional): Maximum number of new tokens to generate in each eval step.
        stopping_criteria (None | generate_stops.StoppingCriteria | list[generate_stops.StoppingCriteria], optional):
            Custom stopping criteria for the generation.
        task_type (str, optional): Type of task to evaluate on.
        generate (bool, optional): Whether to generate completions for the evaluation.

    Returns:
        dict[str, typ.Any]: The computed metrics aggregated over the evaluation set.
    """
    if isinstance(stopping_criteria, generate_stops.StoppingCriteria):
        stopping_criteria = [stopping_criteria]
    if isinstance(stopping_criteria, list):
        stopping_criteria = generate_stops.StoppingCriteriaList(stopping_criteria)

    max_eval = _strictly_pos_or_none(max_eval)
    n_eval_steps = (
        len(dataloader) if max_eval is None else -(-max_eval // (fabric.world_size * (dataloader.batch_size or 1)))
    )
    valid_task = pbar.add_task(task_type, total=n_eval_steps, info=f"0/{n_eval_steps}")
    monitor.reset()
    start_time = time.perf_counter()
    generated_samples = defaultdict(list)
    for it, batch in enumerate(dataloader):
        if it > n_eval_steps:
            break
        fabric.call(f"on_{task_type}_batch_start", batch=batch, step=step)  # TODO: Fix this

        # Generate new tokens based on the prompt tokens. This enables end-to-end testing of the model.
        # NOTE: Unwrapping the model required to avoid problems with accessing other methods than `.forward()`
        #       This might not work with sharded strategies.
        if generate:
            preds_input_ids = _lm_generate(
                model,
                batch["prompt_input_ids"],
                batch["prompt_attention_mask"],
                generation_config=generation_config,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                logits_processors=logits_processors,
            )
        else:
            preds_input_ids = None
        # Run a forward pass
        output = _lm_forward(model=model, batch=batch, lm_prompt_weight=lm_prompt_weight)
        if preds_input_ids is not None:
            output["preds_input_ids"] = preds_input_ids

        # Do something with the model output
        fabric.call(f"on_{task_type}_batch_end", batch=batch, output=output, step=step)  # TODO: Fix this
        if fabric.is_global_zero and it <= 10:
            tmp_samples = _get_samples_record(tokenizer, batch, output)
            for k, v in tmp_samples.items():
                generated_samples[k].extend(v)
        monitor.update(**batch, **output)

        # Update the progress bar
        pbar.update(valid_task, completed=it, info=f"{1+it}/{n_eval_steps}")
    # Remove the progress bar
    pbar.remove_task(valid_task)

    # Compute the metrics and return
    metrics = monitor.compute()
    metrics["sample-predictions"] = generated_samples
    metrics["elapsed_time"] = time.perf_counter() - start_time  # type: ignore
    return metrics


@summon_params_if_fsdp
def _lm_forward(
    model: transformers.PreTrainedModel,
    batch: dict[str, typ.Any],
    lm_prompt_weight: float = 0,
) -> dict[str, typ.Any]:
    try:
        input_ids = batch["input_ids"][:, :-1]
        attention_mask = batch["attention_mask"][:, :-1]
        labels = batch["input_ids"][:, 1:]
        labels_mask = batch["attention_mask"][:, 1:]
        token_type_ids = batch["token_type_ids"][:, 1:]
    except KeyError as exc:
        raise ValueError(
            f"Missing key `{exc}` in the batch. Make sure to include [input_ids, attention_mask, token_type_ids]"
        ) from exc

    # Forward pass with the language model
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    prompt_loss = _masked_lm_loss(labels, (labels_mask > 0) & (token_type_ids == 0), output["logits"])
    target_loss = _masked_lm_loss(labels, token_type_ids > 0, output["logits"])
    loss = lm_prompt_weight * prompt_loss + target_loss
    output = {**output, "prompt_loss": prompt_loss, "target_loss": target_loss, "loss": loss}
    return output


@summon_params_if_fsdp
@unwrap_model_if_wrapped
def _lm_generate(
    model: transformers.PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_config: transformers.GenerationConfig,
    pad_token_id: None | int = None,
    max_new_tokens: None | int = None,
    stopping_criteria: None | generate_stops.StoppingCriteriaList = None,
    logits_processors: None | transformers.LogitsProcessorList = None,
) -> torch.Tensor:
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,  # NOTE: https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
        logits_processor=logits_processors,
    )
    generated_ids = generated_ids[:, input_ids.shape[-1] :]
    return generated_ids


def _masked_lm_loss(labels: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Compute the loss for tokens with mask > 0."""
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    )
    # Mask the loss and compute the mean
    loss = (mask * loss.view(*mask.shape)).sum(dim=-1) / mask.sum(dim=-1)
    return loss.mean()


def _iterate_micro_batches(
    batch: dict[str, torch.Tensor], micro_batch_size: None | int
) -> typ.Iterator[dict[str, torch.Tensor]]:
    """Iterate over the micro batches."""
    if micro_batch_size is None or micro_batch_size <= 0:
        yield batch
    else:
        for i in range(0, batch["input_ids"].shape[0], micro_batch_size):
            yield {k: v[i : i + micro_batch_size] for k, v in batch.items()}


def _fetch_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Fetch the learning rates from the optimizer."""
    return {f"lr/{i}": group["lr"] for i, group in enumerate(optimizer.param_groups)}


def _format_pbar_info(
    step: int | str,
    train_metrics: dict[str, torch.Tensor],
    eval_metrics: dict[str, torch.Tensor],
    pattern: re.Pattern,
    style: str = "yellow",
    avg_elapsed_time: None | float = None,
) -> str:
    parts = []
    for key in train_metrics:
        if pattern.search(key):
            parts.append(f"train/{key}={train_metrics[key].item():.2f}")
    for key in eval_metrics:
        if pattern.search(key):
            parts.append(f"eval/{key}={eval_metrics[key].item():.2f}")
    desc = " ".join(parts) if len(parts) > 0 else "--"
    desc = f"{step} • [{style}]{desc}[/{style}]"
    if avg_elapsed_time is not None:
        if avg_elapsed_time > 1:
            desc += f" • [cyan]{avg_elapsed_time:.2f} opt/s[/cyan]"
        else:
            desc += f" • [cyan]{1/avg_elapsed_time:.2f} s/opt[/cyan]"

    return desc


def _strictly_pos_or_none(x: None | int) -> None | int:
    """Return None if x is None or x > 0."""
    return None if x is None or x <= 0 else x


def _flatten_dict(x: dict[str, dict]) -> dict[str, typ.Any]:
    """Flatten a nested dict. Recursive function."""
    out = {}
    for k, v in x.items():
        if isinstance(v, dict):
            out.update({f"{k}/{k_}": v_ for k_, v_ in _flatten_dict(v).items()})
        else:
            out[k] = v

    return out


def _get_samples_record(
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
) -> list[dict[str, typ.Any]]:
    """Get the samples record."""
    prompts = tokenizer.batch_decode(batch["prompt_input_ids"], skip_special_tokens=True)
    targets = tokenizer.batch_decode(batch["target_input_ids"], skip_special_tokens=True)
    predictions = tokenizer.batch_decode(output["preds_input_ids"], skip_special_tokens=True)
    return {"prompt": prompts, "targets": targets, "predictions": predictions}


def _get_occurences_table(predictions: list[int], targets: list[int]) -> wandb.Table:
    """Get the occurrences table."""
    # Prepare data for two bars per class
    data = []
    for cls_id, (t_count, p_count) in enumerate(zip(targets, predictions)):
        data.append([cls_id, "Targets", t_count])
        data.append([cls_id, "Predictions", p_count])

    # Create the table
    return wandb.Table(data=data, columns=["Class Index", "Type", "Count"])


def custom_grouped_bar_chart(predictions: list[int], targets: list[int]) -> None:
    """Plots a grouped bar chart using W&B and a custom Vega-Lite spec."""
    # Generate the data table
    table = _get_occurences_table(predictions, targets)
    fields = {"label": "Class Index", "category": "Type", "value": "Count"}

    return wandb.plot_table(
        vega_spec_name="carey/new_chart",
        data_table=table,
        fields=fields,
        string_fields={"title": "Occurrences of Each Class (Targets vs Predictions)"},
    )
