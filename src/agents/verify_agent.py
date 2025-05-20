from collections import defaultdict
from functools import partial
import re
import typing as typ


from agents.base import HfBaseAgent
from agents.errors import StructuredError

ANSWER_PATTERN = r"<answer>.*?(\b[0-9]\d{0,3}(?:\s*,\s*[1-9]\d{0,3})*\b).*?<\/answer>"


class VerifyAgent(HfBaseAgent):
    """A dummy assign agent that simulates the candidate space"""

    def __call__(
        self, batch: dict[str, list[typ.Any]], *args, **kwargs
    ) -> dict[str, list[typ.Any]]:
        """Process a row of agent tasks from a HuggingFace datasets.map()."""
        batch_size = len(batch[list(batch.keys())[0]])
        if batch_size > 1:
            raise ValueError(
                f"Batch size must be 1, but got {batch_size}. Please use a batch size of 1."
            )
        self._warm_up_prefix_cache(
            {"note": batch["note"][0], "codes": [], "instructional_notes": []}
        )
        # flatten the batch where instructional_notes and codes are nested
        # should increase the batch size to the length of the codes
        new_batch = defaultdict(list)
        new_batch["codes"] = [group for group in batch["codes"][0]]
        new_batch["instructional_notes"] = [
            group for group in batch["instructional_notes"][0]
        ]
        new_batch["guidelines"] = [group for group in batch["guidelines"][0]]
        new_batch["note"] = [batch["note"][0]] * len(new_batch["codes"])

        output = super().__call__(new_batch, *args, **kwargs)

        if not isinstance(output, dict):
            raise ValueError(
                f"Expected output to be a dictionary, but got {type(output)}"
            )

        if "output" not in output:
            return {
                **batch,
                "output": [[]],  # output is guaranteed by parser to be a list
                "reasoning": [
                    "No output found in the response."
                ],  # reasoning is guaranteed to be a string
            }

        # reshape the output to match the original batch size
        return {
            **batch,
            "output": [output["output"]],  # output is guaranteed by parser to be a list
            "reasoning": [
                "\n\n".join(output["reasoning"])
            ],  # reasoning is guaranteed to be a string
        }

    def _warm_up_prefix_cache(self, row: dict[str, typ.Any]) -> None:
        """Warm up the prefix cache."""
        request = self.format_request(**row)
        request["max_tokens"] = 1
        self.client.sync_call(request)

    def parser(self, content: str) -> dict[str, typ.Any]:
        """Compress the choices."""
        content = content.replace("IDs:", "").replace("ID:", "")
        answer_match = re.search(ANSWER_PATTERN, content, re.DOTALL)
        if not answer_match:
            raise StructuredError(
                f"Could not find answer tags in the response: {content[-250:]}"
            )

        output = [int(num.strip()) for num in answer_match.group(1).split(",")]
        if not output:
            raise StructuredError(
                f"Found answer tags but no valid numbers inside: {answer_match.group(1)}"
            )

        return {"reasoning": content, "output": output}


def create_verify_agent(
    agent_type: str,
    prompt_name: str,
    sampling_params: dict[str, typ.Any],
    seed: int = 42,
) -> typ.Callable[..., HfBaseAgent]:
    """
    Factory method to create an AssignAgent instance based on the specified type.
    """
    if agent_type == "reasoning":
        return partial(
            VerifyAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
