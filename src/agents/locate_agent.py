from collections import defaultdict
from functools import partial
import re
import typing as typ


from agents.base import HfBaseAgent
from agents.errors import StructuredError

ANSWER_PATTERN = r"<answer>.*?(\b[0-9]\d{0,3}(?:\s*,\s*[1-9]\d{0,3})*\b).*?<\/answer>"


class LocateAgent(HfBaseAgent):
    """A dummy assign agent that simulates the candidate space"""

    def parser(self, content: str) -> dict[str, typ.Any]:
        """Compress the choices."""
        content = (
            content.replace("IDs:", "").replace("ID:", "").replace("ID", "").strip()
        )
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


class LocateSplitAgent(LocateAgent):
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
        new_batch["terms"] = [group for group in batch["terms"][0]]
        new_batch["note"] = [batch["note"][0]] * len(new_batch["terms"])

        output = super().__call__(new_batch, *args, **kwargs)
        # reshape the output to match the original batch size
        return {
            **batch,
            "output": [output["output"]],
            "reasoning": ["\n\n".join(output["reasoning"])],
        }

    def _warm_up_prefix_cache(self, row: dict[str, typ.Any]) -> None:
        """Warm up the prefix cache."""
        request = self.format_request(**row)
        request["max_tokens"] = 1
        self.client.sync_call(request)


class LocateSnippetAgent(LocateAgent):
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
        # flatten the batch where instructional_notes and codes are nested
        # should increase the batch size to the length of the codes
        new_batch = defaultdict(list)
        new_batch["terms"] = [group for group in batch["terms"][0]]
        new_batch["query"] = [q for q in batch["query"][0]]

        output = super().__call__(new_batch, *args, **kwargs)
        # reshape the output to match the original batch size
        return {
            **batch,
            "output": [output["output"]],
            "reasoning": ["\n\n".join(output["reasoning"])],
        }


class StructuredLocateAgent(LocateAgent):
    """A structured assign agent that simulates the candidate space."""

    TERM_LIST = r"(?:[0-9]\d{0,1})\n"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_params["guided_regex"] = self.TERM_LIST

    def parser(self, content: str) -> dict[str, typ.Any]:
        """Compress the choices into a single."""
        # match comma separated list of integers with regex
        output = list(set(int(num.strip()) for num in content.split(",")))

        return {"reasoning": "", "output": output}


def create_locate_agent(
    agent_type: str,
    prompt_name: str,
    sampling_params: dict[str, typ.Any],
    seed: int = 42,
) -> typ.Callable[..., HfBaseAgent]:
    """
    Factory method to create an AssignAgent instance based on the specified type.
    """
    if agent_type == "base":
        return partial(
            LocateAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )
    elif agent_type == "split":
        return partial(
            LocateSplitAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )
    elif agent_type == "snippet":
        return partial(
            LocateSnippetAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )
    elif agent_type == "structured":
        return partial(
            StructuredLocateAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
