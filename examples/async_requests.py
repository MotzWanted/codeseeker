"""Example of using multiple async requests with the llm-client.

NOTE: Requires you to define sufficient environment variables for the provider you want to use.
"""

import argparse
import asyncio


import rich
from throughster import create_interface, Prompt

SYSTEM_PROMPT = [{"role": "system", "content": "You are a world class linguist."}]
USER_PROMPT = [{"role": "user", "content": """Translate this "{{ text }}" to {{ language }}."""}]
LANGUAGES = ["French", "Spanish", "Italian"]


async def main(args):
    throughster = create_interface(args.provider, api_base="http://localhost:6538/v1", use_cache=False)
    TranslatePrompt = Prompt(system_prompt=SYSTEM_PROMPT)
    sampling_params = {"temperature": 0.5}

    text_to_translate = "Happy birthday!"
    requests = [
        {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": TranslatePrompt(USER_PROMPT, {"text": text_to_translate, "language": language}),
            **sampling_params,
        }
        for language in LANGUAGES
    ]
    results = await throughster.batch_call(requests)
    for idx, result in enumerate(results):
        rich.print(f"Translation to {LANGUAGES[idx]}:")
        rich.print(result.model_dump())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of using async requests with the llm-client.")
    parser.add_argument(
        "--provider", type=str, choices=["vllm", "azure", "mistral"], default="vllm", help="The provider to use."
    )
    args = parser.parse_args()
    asyncio.run(main(args))
