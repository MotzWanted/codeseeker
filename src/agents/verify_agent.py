from functools import partial
import re
import typing as typ


from agents.base import HfBaseAgent
from agents.errors import StructuredError

ANSWER_PATTERN = r"<answer>.*?(\b[1-9]\d{0,3}(?:\s*,\s*[1-9]\d{0,3})*\b).*?<\/answer>"


class VerifyAgent(HfBaseAgent):
    """A dummy assign agent that simulates the candidate space"""

    def __call__(
        self, batch: dict[str, list[typ.Any]], *args, **kwargs
    ) -> dict[str, list[typ.Any]]:
        return super().__call__(batch, *args, **kwargs)

    def _validate_input(self, row: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Format the input."""
        if "note" not in row or "codes" not in row or "instructional_notes" not in row:
            raise ValueError(
                f"Missing `note`, `code` or `instruction_notes` in input data: {row}"
            )
        if not isinstance(row["codes"], list) and not isinstance(row["codes"][0], list):
            raise ValueError(f"Expected `codes` to be a nested list: {row['codes']}")
        return row

    def parser(self, content: str) -> dict[str, typ.Any]:
        """Compress the choices."""
        content = content.replace("IDs:", "").replace("ID:", "")
        answer_match = re.search(ANSWER_PATTERN, content, re.DOTALL)
        output = (
            [int(num.strip()) for num in answer_match.group(1).split(",")]
            if answer_match
            else []
        )
        if not output:
            raise StructuredError(
                f"Could not find any relevant answer in the response: {content[-250:]}"
            )
        return {"reasoning": content, "output": output}


def create_assign_agent(
    agent_type: str,
    prompt_name: str,
    sampling_params: dict[str, typ.Any],
    seed: int = 42,
) -> typ.Callable[..., HfBaseAgent]:
    """
    Factory method to create an AssignAgent instance based on the specified type.
    """
    if agent_type == "mock":
        return partial(
            VerifyAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
