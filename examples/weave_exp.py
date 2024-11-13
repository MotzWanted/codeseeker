import asyncio
import json
import weave
import datasets
import typing as typ

from throughster.prompt import Prompt
from dataloader.loaders.nbme_notes import NbmeDatasetLoader
from dataloader.adapters.alignment import NmbeAdapter, SentenceSegmenter
from throughster.base import ModelInterface
from alignment.models import FactualAlignment
from throughster.factory import create_interface

# We create a model class with one predict function.
# All inputs, predictions and parameters are automatically captured for easy inspection.


def _get_dataset(dset: datasets.Dataset | datasets.DatasetDict) -> datasets.Dataset:
    """Get a `datasets.Dataset`."""
    if isinstance(dset, datasets.Dataset):
        return dset
    return next(iter(dset.values()))


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


def custom_tojson(value):
    # Use json.dumps with ensure_ascii=False to avoid unnecessary escaping
    return json.dumps(value, ensure_ascii=False)


class AlignmentModel(weave.Model):
    provider: str
    model_name: str
    prompt_template: str
    temperature: float

    @weave.op()
    async def predict(self, sources: list[str], targets: list[str], fewshots: list = None) -> dict:
        client = create_interface(
            self.provider,
            model_name=self.model_name,
        )

        alignment_data = await asyncio.gather(
            *[self.generate_alignment(client=client, sources=sources, target=t, fewshots=fewshots) for t in targets]
        )

        predictions = [a.indices for a in alignment_data]

        return {"model_output": predictions}

    @weave.op(name="generate-alignment")
    async def generate_alignment(
        self, client: ModelInterface, sources: list[str], target: str, fewshots: list[dict] | None
    ) -> FactualAlignment:
        """Generate an alignment for a task."""
        request = {
            "messages": self._format_prompt(sources, target, fewshots),
            "temperature": self.temperature,
            "stop": ["\n\n\n"],
        }
        result = await client.structured_call(request, schema=FactualAlignment, max_attempts=5)
        return result.validated_schema  # type: ignore

    def _format_prompt(
        self, sources: list[str], target: str, fewshots: list[dict[str, typ.Any]]
    ) -> list[dict[str, str]]:
        """Format the prompt."""
        fewshots = self._format_fewshots(fewshots[: self.num_shots])
        prompt = Prompt(
            raw_template=self.raw_template,
            template_data={
                "few_shots": fewshots,
                "input_list": sources,
                "fact": target,
                "custom_tojson": custom_tojson,
            },
            token_limit=self.token_limit,
            truncation_step=1,
        )
        return prompt.messages


# We call init to begin capturing data in the project, intro-example.
weave.init("intro-example")

# We create our model with our system prompt.
model = AlignmentModel(
    name="alignment-model",
    provider="vllm",
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
    prompt_template=raw_template,
    temperature=0.1,
)
data: datasets.Dataset = NbmeDatasetLoader().load_dataset(split="test", size=300)  # type: ignore

adapter = NmbeAdapter(segmenter=SentenceSegmenter())
data = data.map(
    adapter,
    num_proc=12,
    desc=f"Adapting dataset to `AlignmentModel` using `{NmbeAdapter.__name__}`.",
    remove_columns=_get_dataset(data).column_names,
)

data_list = data.to_list()[:10]


# We define a scoring functions to compare our model predictions with a ground truth label.
@weave.op()
def exact_match(labels: list[list[int]], model_output: list[list[int]]) -> dict:
    exact_matches = sum(1 for y, y_hat in zip(labels, model_output) if set(y) == set(y_hat))
    return {"exact_match": exact_matches / len(labels)}


# Finally, we run an evaluation of this model.
# This will generate a prediction for each input example, and then score it with each scoring function.
evaluation = weave.Evaluation(
    name="test_evaluation",
    dataset=data_list,
    scorers=[exact_match],
)
print(asyncio.run(evaluation.evaluate(model)))
# if you're in a Jupyter Notebook, run:
# await evaluation.evaluate(model)
