import json
from pathlib import Path
import random
import typing

import datasets
import numpy as np
from prompt_poet import Prompt
from jinja2 import Environment, FileSystemLoader
import wandb
from dataloader.adapt.base import BaseTrainingModel
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

PatchFastRL("GRPO", FastLanguageModel)
# ignore E402 error
from trl import GRPOConfig, GRPOTrainer  #  noqa: E402
import re  #  noqa: E402
from dataloader import mimiciv_50  #  noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]

DSET_CONFIG = {
    "identifier": "mimic-iv-50",
    "name_or_path": mimiciv_50,
    "subsets": "icd10",
    "options": {
        "adapter": "MimicForTrainingAdapter",
        "order": "alphabetical",
    },
}

############################## REWARD FUNCTIONS ##############################


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


def categorical_cross_entropy(predictions: list[int], targets: list[int], classes: dict[str, str]) -> float:
    """
    Compute the categorical cross-entropy loss between true labels and predicted probabilities.

    Args:
        y_true: One-hot encoded true labels (list of integers).
        y_pred: One-hot encoded predictions (numpy array of shape (num_classes,)).

    Returns:
        A float representing the categorical cross-entropy loss.
    """

    def list2matrix(dim_y: int, list_indices: list[int]) -> np.ndarray:
        sparse_matrix = np.zeros(dim_y, dtype=np.float32)
        for idx in list_indices:
            pred_sign = 0 if idx <= 0 else 1
            pred_idx = abs(idx) - 1
            if 0 <= pred_idx < dim_y:
                sparse_matrix[pred_idx] = pred_sign
        return sparse_matrix

    num_classes = len(classes)

    # Convert lists to one-hot encoded vectors
    preds_sparse_vector = list2matrix(num_classes, predictions)
    target_sparse_vector = list2matrix(num_classes, targets)

    epsilon = 1e-9  # Avoid log(0) issues
    y_pred = np.clip(preds_sparse_vector.astype(float), epsilon, 1.0 - epsilon)

    # Compute categorical cross-entropy loss
    return -float(np.sum(target_sparse_vector.astype(float) * np.log(y_pred)))


def inverse_loss_reward_scaled(loss, alpha=0.1, min_val=0, max_val=2) -> float:
    base_reward = 1 / (1 + alpha * loss)  # Original inverse loss reward
    return min_val + (max_val - min_val) * base_reward  # Scale to [0,2]


def extract_xml_answer(text: str) -> list[int]:
    pattern = r"<answer>(.*?)</answer>"  # Match only content inside <answer>...</answer>
    answers = re.findall(pattern, text, re.DOTALL)
    return [n for n in answers if n.is_digit() and 1 <= n <= 50]


class _PromptWrapper:
    """Wrapper around the prompt."""

    PATH_TO_TEMPLATES = Path(__file__).parent

    def __init__(self, prompt_name: str, shuffle_targets: bool = False) -> None:
        env = Environment(loader=FileSystemLoader(self.PATH_TO_TEMPLATES))
        loader = typing.cast(FileSystemLoader, env.loader)
        self.raw_template, self.template_path, _ = loader.get_source(env, f"{prompt_name}.yml.j2")
        self.shuffle_targets = shuffle_targets

    def __call__(self, row: BaseTrainingModel) -> tuple[Prompt, str]:
        """Make a training example and format the task as a prompt."""
        targets = row.parse_targets()
        prompt = Prompt(
            raw_template=self.raw_template,
            template_path=None,
            template_data={
                **row.model_dump(),
                "targets": targets,
                "custom_tojson": custom_tojson,
            },
        )
        return prompt, targets


train_dset = datasets.load_dataset("parquet", data_files=f"{PROJECT_DIR}/data/mimic-iv-sublime-new/train.parquet")[
    "train"
]
val_dset = datasets.load_dataset("parquet", data_files=f"{PROJECT_DIR}/data/mimic-iv-sublime-new/validation.parquet")[
    "train"
]
test_dset = datasets.load_dataset("parquet", data_files=f"{PROJECT_DIR}/data/mimic-iv-sublime-new/test.parquet")[
    "train"
]


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    targets = [int(t) for t in answer.split(",")]
    extracted_answers = [extract_xml_answer(r) for r in responses]

    losses = [categorical_cross_entropy(preds, targets) for preds in extracted_answers]
    rewards = [inverse_loss_reward_scaled(loss) for loss in losses]

    if random.randint(1, 32) == 1:
        print(
            "-" * 20,
            f"Question:\n{q}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_answers[0]}",
            f"\nAnswer:\n{answer[0]}",
            f"\nLoss: {losses[0]:.3f}",
            f"\nReward: {rewards[0]:.3f}",
        )

    return rewards


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.1 if match else 0.0 for match in matches]


def xmlcount_reward_func(completions, codes, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    thinking_lengths = [len(re.findall(pattern, c)) for c in contents]
    true_lenght = len(codes)
    return [1.0 - abs(tl - true_lenght) / true_lenght for tl in thinking_lengths]


############################## TRAINING ##############################
model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"
version = "v1"
output_path: str = Path("~/research/models/{model_name}-grpo-lora-{version}").expanduser().as_posix()
max_seq_length = 768  # Can increase for longer reasoning traces
lora_rank = 32  # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.4,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)

training_args = GRPOConfig(
    use_vllm=True,  # use vLLM for fast inference!
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Increase to 4 for smoother training
    num_generations=8,  # Decrease if out of memory
    max_prompt_length=6096,
    max_completion_length=max_seq_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=800,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",  # Can use Weights & Biases
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dset,
    eval_dataset=val_dset,
)
trainer.train()

wandb.finish()

model.save_pretrained_merged(output_path, tokenizer, save_method="lora")
