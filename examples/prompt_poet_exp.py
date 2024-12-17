from collections import defaultdict
import json
import datasets
import typing as typ
from prompt_poet import Prompt
import rich

from dataloader.nbme.nbme_notes import NbmeDatasetLoader
from dataloader.adapt.alignment import NbmeAdapter
from segmenters.base import WindowSegmenter


def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    return json.dumps(value, ensure_ascii=False)


raw_template = """
- name: system instructions
  role: system
  content: |
    You are a medical expert fact-checking medical documentation.
    Your goal is to identify and select the documentation indices that entail the hypothesis.

{% if not few_shots %}
- name: instruction guidelines
  role: system
  content: |
    # Instruction guidelines:
    1. A hypothesis is entailed only if all aspects of the hypothesis are supported by the documentation.
    2. For compound hypotheses, ensure that all aspects are supported by the documentation indice(s).
    3. If the hypothesis is not entailed by the documentation, output a zero list: `[0]`.
    4. If a compound hypothesis is entailed, list all document indices that support the hypothesis, e.g., `[1, 3, 7]`.
    5. Else output the single document index that entails the hypothesis: `[5]`.
    6. Note that fewer indices are better for alignment.

    # Output
    Please output in JSON format, e.g., `{"indices": [3, 5]}`. If the hypothesis is not supported, output a zero list: `{"indices": [0]}`.
{% endif %}

{% for shot in few_shots %}
- name: few shot (k={{ loop.index }})
  role: user
  truncation_priority: {{ loop.index }}
  content: |
    ====== Example case {{ loop.index }} ======
    # Documentation:
    {% for option in shot.sources %}
    [{{ loop.index }}] "{{ custom_tojson(option) }}"
    {% endfor %}
    {% for data in shot.targets %}
    # Example {{ loop.index }}
    Hypothesis: "{{ data.fact | escape }}"
    Response: `{"indices": {{ data.index }}}`
    {% endfor %}

{% endfor %}

- name: new input
  role: user
  content: |
    ====== Now it's your turn! ======
    # Documentation:
    {% for option in input_list %}
    [{{ loop.index }}] "{{ custom_tojson(option) }}"
    {% endfor %}
    # New example
    Hypothesis: "{{ fact | escape }}"
    Response:
"""  # noqa: E501


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


class PromptFormatter:
    def __init__(
        self, raw_template: str, source_key: str, target_key: str, fewshots_key: str, token_limit: int
    ) -> None:
        self.source_key = source_key
        self.target_key = target_key
        self.fewshots_key = fewshots_key
        self.raw_template = raw_template
        self.token_limit = token_limit

    def __call__(self, batch: dict[str, list[typ.Any]], idx: list[int] | None = None) -> dict[str, list[typ.Any]]:
        self._validate_input(batch)

        sources = batch[self.source_key]
        targets = batch[self.target_key]
        fewshots = batch[self.fewshots_key]
        output = defaultdict(list)
        for i in range(len(sources)):
            prompts = []
            for fact in targets[i]:
                prompts.append(self._format_prompt(sources[i], fact, fewshots[i]))
            output["messages"].append(prompts)

        return output

    def _format_prompt(self, source: str, target: str, fewshots: list[dict[str, typ.Any]]) -> list[dict | str]:
        """Format the prompt."""
        fewshots = self._format_fewshots(fewshots)
        prompt = Prompt(
            raw_template=self.raw_template,
            template_data={"few_shots": fewshots, "input_list": source, "fact": target, "custom_tojson": custom_tojson},
            token_limit=self.token_limit,
            truncation_step=1,
        )
        prompt.tokenize()
        prompt.truncate()
        return prompt.messages

    def _format_fewshots(self, fewshots: list[dict[str, typ.Any]]) -> list[dict[str, typ.Any]]:
        """Format the fewshots."""
        formatted_fewshots = []
        for shot in fewshots:
            formatted_fewshot = {
                "sources": shot["sources"],
                "targets": [{"fact": fact, "index": index} for fact, index in zip(shot["targets"], shot["labels"])],
            }
            formatted_fewshots.append(formatted_fewshot)
        return formatted_fewshots

    def _validate_input(self, batch: dict[str, typ.Any]) -> None:
        if self.source_key not in batch:
            raise ValueError(f"Missing key: {self.source_key}")
        if self.target_key not in batch:
            raise ValueError(f"Missing key: {self.target_key}")
        if not isinstance(batch[self.source_key], list):
            raise ValueError(f"Invalid type for key {self.source_key}: {type(batch[self.source_key])}")
        if not isinstance(batch[self.target_key], list):
            raise ValueError(f"Invalid type for key {self.target_key}: {type(batch[self.target_key])}")


def run():
    data: datasets.Dataset = NbmeDatasetLoader().load_dataset(split="test", size=300)  # type: ignore

    adapter = NbmeAdapter(segmenter=WindowSegmenter())
    data = data.map(
        adapter,
        num_proc=1,
        desc=f"Adapting dataset to `AlignmentModel` using `{NbmeAdapter.__name__}`.",
        remove_columns=_get_dataset(data).column_names,
        # load_from_cache_file=False,
    )

    task = PromptFormatter(
        raw_template=raw_template,
        source_key="sources",
        target_key="targets",
        fewshots_key="fewshots",
        token_limit=4096,
    )
    data = data.map(
        task,
        num_proc=64,
        batched=True,
        batch_size=1,
        desc=f"Formatting prompts using `{PromptFormatter.__name__}`.",
        remove_columns=_get_dataset(data).column_names,
    )
    rich.print(data)


if __name__ == "__main__":
    run()
