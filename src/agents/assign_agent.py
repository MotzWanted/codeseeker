from functools import partial
import re
import typing as typ


from agents.base import HfBaseAgent
from agents.errors import StructuredError

ANSWER_PATTERN = r"<answer>.*?(\b[1-9]\d{0,3}(?:\s*,\s*[1-9]\d{0,3})*\b).*?<\/answer>"


class AssignAgent(HfBaseAgent):
    """A dummy assign agent that simulates the candidate space"""

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


class StructuredAssignAgent(AssignAgent):
    """A structured assign agent that simulates the candidate space."""

    CODE_LIST = r"(?:[1-9]\d{0,3})(?:,(?:[1-9]\d{0,3})){0,19}\n"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_params["guided_regex"] = self.CODE_LIST

    def parser(self, content: str) -> dict[str, typ.Any]:
        """Compress the choices into a single."""
        # match comma separated list of integers with regex
        output = [int(num.strip()) for num in content.split(",")]

        return {"reasoning": "", "output": output}


def create_assign_agent(
    agent_type: str,
    prompt_name: str,
    sampling_params: dict[str, typ.Any],
    seed: int = 42,
) -> typ.Callable[..., HfBaseAgent]:
    """
    Factory method to create an AssignAgent instance based on the specified type.
    """
    if agent_type == "structured":
        return partial(
            StructuredAssignAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )

    elif agent_type == "reasoning":
        return partial(
            AssignAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
