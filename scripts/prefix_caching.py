import asyncio
from time import time

from throughster.base import ModelInterface
from throughster.core.models import BaseResponse
from throughster.factory import create_interface


async def generate(client: ModelInterface, prompts: list[dict], sampling_params: dict) -> list[str]:
    """Generate texts from a list of prompts."""
    coroutines = [client.call({"messages": prompt, **sampling_params}) for prompt in prompts]
    results: list[BaseResponse] = await asyncio.gather(*coroutines)
    return [c.content for res in results for c in res.choices]


prefix = {
    "role": "system",
    "content": """You are an expert school principal, skilled in effectively managing faculty and staff. 
    Draft 10-15 questions for a potential first grade Head Teacher for""",
}

# Sample prompts.
prompts = [
    {"role": "user", "content": "Hello, my name is:"},
    {"role": "user", "content": "The president of the United States is:"},
    {"role": "user", "content": "The capital of France is:"},
    {"role": "user", "content": "The future of AI is:"},
]

generating_prompts = [[prefix, prompt] for prompt in prompts]

# Create a sampling params object.
sampling_params = {"temperature": 0.0}

# Create an LLM.
regular_llm = create_interface(
    "vllm",
    api_base="http://localhost:6539/v1",
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
)
prefix_cached_llm = create_interface(
    "vllm",
    api_base="http://localhost:6538/v1",
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
)


# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
async def run():
    print("Results without `enable_prefix_caching`")
    start_time_regular = time()
    regular_generated_texts = await generate(regular_llm, generating_prompts, sampling_params)
    duration_regular = time() - start_time_regular

    # Print the outputs.
    for generated_text in regular_generated_texts:
        print(f"Generated text: {generated_text!r}")

    print("-" * 80)

    # Warmup so that the shared prompt's KV cache is computed.
    await generate(prefix_cached_llm, generating_prompts[:1], sampling_params)

    # Generate with prefix caching.
    start_time_cached = time()
    cached_generated_texts = await generate(prefix_cached_llm, generating_prompts, sampling_params)
    duration_cached = time() - start_time_cached

    print("Results with `enable_prefix_caching`")

    # Print the outputs. You should see the same outputs as before.
    for generated_text in cached_generated_texts:
        print(f"Generated text: {generated_text!r}")

    print("-" * 80)

    # Compare the results and display the speedup
    generated_same = all([regular_generated_texts[i] == cached_generated_texts[i] for i in range(len(prompts))])
    print(f"Generated answers are the same: {generated_same}")

    speedup = round(duration_regular / duration_cached, 2)
    print(f"Speed up of cached generation compared to the regular is: {speedup}")


if __name__ == "__main__":
    asyncio.run(run())
