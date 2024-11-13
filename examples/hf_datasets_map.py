"""Example of using the `llm-client` for offline processing with Hugging Face `datasets.map()` or `datasets.filter()`."""  # noqa: E501

import argparse
from functools import partial

import datasets
import typing as typ

import rich

from throughster.base import ModelInterface
from throughster.factory import create_interface
from throughster.hf_datasets import HfOperation, transform
from throughster.core.models import BaseResponse
from throughster.prompt import Prompt

SYSTEM_PROMPT = [{"role": "system", "content": "You are a helpful translator."}]
USER_PROMPT = [
    {
        "role": "user",
        "content": """Please translate the following text from {{ source }} to {{ target }}:
"{{ text }}""",
    }
]

SUPPORTED_LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
}

LanguageCode = typ.Literal["en", "de", "fr", "es", "it"]

NUM_WORKERS = 2


class TranslateOp(HfOperation):
    """Make translation tasks."""

    def __init__(
        self,
        init_client_fn: typ.Callable[..., ModelInterface],
        translate_from: LanguageCode,
        translate_to: LanguageCode,
        prompt: Prompt,
        user_prompt: list[dict[str, str]],
        sampling_params: dict[str, typ.Any],
        text_key: str = "text",
    ) -> None:
        self.init_client_fn = init_client_fn
        self.translate_from = SUPPORTED_LANGUAGES[translate_from]
        self.translate_to = SUPPORTED_LANGUAGES[translate_to]
        self._client = None
        self.prompt = prompt
        self.user_prompt = user_prompt
        self.text_key = text_key
        self.sampling_params = sampling_params

    def __call__(self, row: dict[str, typ.Any], idx: None | list[int] = None) -> dict[str, list[typ.Any]]:
        """Translate the input batch."""
        request = self.create_requests(row)
        # Wrapper function to run the async operation synchronously
        response = self.translate(request)
        translations = [choices.content for choices in response.choices]
        return {"text": row[self.text_key], "translation": translations}

    def create_requests(self, batch: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Create translation requests."""
        text = batch[self.text_key]
        return {
            "messages": self.prompt(
                prompt=self.user_prompt,
                prompt_variables={"text": text, "source": self.translate_from, "target": self.translate_to},
            ),
            **self.sampling_params,
        }

    def translate(self, request: dict[str, typ.Any]) -> BaseResponse:
        return self.client.sync_call(request)


def run(args):
    data = datasets.load_dataset("EleutherAI/lambada_openai", args.source, split="test[:100]")

    if not isinstance(data, (datasets.Dataset, datasets.DatasetDict)):
        raise ValueError("The dataset must be a `datasets.Dataset` or `datasets.DatasetDict`.")

    sampling_params = {"temperature": 0.5, "top_p": 0.95, "max_tokens": 512}
    init_client = partial(
        create_interface,
        "vllm",
        api_base="http://localhost:6538/v1",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        use_cache=False,
    )
    translate_op = TranslateOp(
        init_client_fn=init_client,
        translate_from=args.source,
        translate_to=args.target,
        prompt=Prompt(system_prompt=SYSTEM_PROMPT),
        user_prompt=USER_PROMPT,
        sampling_params=sampling_params,
    )

    rich.print(f"Original data: {data}", f"{data[0]}", sep="\n\n")
    data = transform(
        data,
        translate_op,
        operation="map",
        # HuggingFace specific parameters
        desc="Translating text...",
        num_proc=NUM_WORKERS,
        load_from_cache_file=False,
    )
    rich.print(f"Translated data: {data}", f"{data[0]}", sep="\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of using async requests with the llm-client.")
    parser.add_argument("--provider", type=str, choices=["vllm"], default="vllm", help="The provider to use.")
    parser.add_argument(
        "--source", type=str, choices=SUPPORTED_LANGUAGES.keys(), default="en", help="The source language."
    )
    parser.add_argument(
        "--target", type=str, choices=SUPPORTED_LANGUAGES.keys(), default="de", help="The target language."
    )
    args = parser.parse_args()
    run(args)
