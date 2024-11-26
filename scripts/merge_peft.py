"""Script for merging PEFT LoRA weights with the base model.

Adapted from https://github.com/eugenepentland/landmark-attention-qlora/blob/main/llama/merge_peft.py
"""

import json
import pathlib
import shutil

import transformers
from loguru import logger
from peft import peft_model
from rich import status

from pydantic_settings import BaseSettings, SettingsConfigDict


class Args(BaseSettings):
    """Arguments for the script."""

    input_dir: str = "/mnt/md0/vli/data/docgen/models/v11-en-mistralai-Mistral-7B-v0.1-align"
    output_dir: str = "{input_dir}-merged"
    device: str = "cpu"

    model_config = SettingsConfigDict(cli_parse_args=True)


def main(args: Args) -> None:
    """Main function."""
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir.format(**args.model_dump()))
    if args.device == "auto":
        device_arg = {"device_map": "auto"}
    else:
        device_arg = {"device_map": {"": args.device}}

    config_file = input_dir / "adapter_config.json"
    if not config_file.exists():
        raise ValueError(f"`{config_file}` not found. Found: {list(output_dir.iterdir())}")
    with status.Status(f"Loading configuration: {config_file}"):
        peft_config = json.loads(config_file.read_text())
        base_model = peft_config["base_model_name_or_path"]

    with status.Status(f"Loading base model: {base_model}"):
        base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model, **device_arg)

    with status.Status(f"Loading PEFT: {args.input_dir}"):
        model = peft_model.PeftModel.from_pretrained(base_model, args.input_dir)

    with status.Status("Merging and unloading PEFT layers"):
        model = model.merge_and_unload()

    with status.Status(f"Saving unloaded model to {output_dir.absolute()}"):
        model.save_pretrained(output_dir)
        # copy all files that don't start with `adapter_` from the input dir to the output dir
        for file in input_dir.iterdir():
            if file.name.startswith("adapter_"):
                continue
            shutil.copy2(file, output_dir)

    logger.info(f"Success - saved to {output_dir.absolute()}")


if __name__ == "__main__":
    args = Args()
    main(args)
