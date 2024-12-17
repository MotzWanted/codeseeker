import os
import typing as typ

from flask import Flask, render_template, request

from prompt_poet import Prompt
from alignment.aligners.llm import StructuredLLMAligner
from alignment.metrics import AlignmentMetrics
from throughster.factory import create_interface, ModelInterface

from frontend import config
from frontend import models

from frontend.initial_data import nbme_data, entry_count

os.environ["VLLM_API_BASE"] = "http://localhost:6538/v1"


class Clients:
    """Clients class."""

    gpt_4_turbo: ModelInterface
    mistral_large: ModelInterface
    local_llm: ModelInterface
    gpt_35_turbo: ModelInterface

    def get(self, name: str) -> ModelInterface:
        """Get the client."""
        return getattr(self, name)


throughsters = Clients()
aligner = StructuredLLMAligner(
    prompt_name="note2feature",
    num_shots=0,
    token_limit=128000,
    seed=1,
    sampling_params=config.SAMPLING_PARAMS,
)

app = Flask(__name__)


def evaluation_metrics(labels: list[list[int]], predictions: list[list[int]]) -> dict[str, float]:
    """Calculate the alignment metrics."""
    return {
        **AlignmentMetrics.recall_precision(labels, predictions),
        **AlignmentMetrics.exact_match(labels, predictions),
        **AlignmentMetrics.positive_ratio(labels, predictions),
        **AlignmentMetrics.negative_ratio(labels, predictions),
    }


@app.before_request
async def get_event_loop():
    # hack to get the event loop in the context of the request
    # this should not be necessary but the event loop is closed somewhere when it shouldn't
    throughsters.gpt_4_turbo = create_interface("azure", model_name="gpt-4-turbo-0409-1", use_cache=True)
    throughsters.mistral_large = create_interface("mistral", use_cache=True)
    throughsters.local_llm = create_interface(
        "vllm", endpoint="completions", model_name="meta-llama/Llama-3.1-8B-Instruct", use_cache=True
    )
    throughsters.gpt_35_turbo = create_interface("azure", model_name="gpt-35-turbo-1", use_cache=True)


def _flatten_list(data: dict) -> list[str]:
    """Flatten the list."""
    return [val for sublist in data.values() for val in sublist if val.strip()]


async def chunk_transcript(
    transcript: str,
    sampling_params: dict[str, typ.Any],
    prompt_template: str | list[typ.Dict[str, typ.Any]] = config.CHUNKING_TEMPLATE,
) -> list[str]:
    """Chunk the transcript."""
    requests = {
        "messages": Prompt(prompt_template),
        "template_data": {"transcript": transcript},
        **sampling_params,
    }

    client = throughsters.gpt_35_turbo
    results = await client.structured_call(requests, schema=models.ChunkedTranscript, max_attempts=config.MAX_ATTEMPTS)

    return results.validated_schema.chunks  # type: ignore


def get_match_count(labels: list[list[int]], predictions: list[list[int]]) -> list[str]:
    match_counts = []

    for y, y_hat in zip(labels, predictions):
        intersection = len(set(y) & set(y_hat))  # Count of matching elements
        labels_count = len(y)  # Count of actual label elements

        # To handle cases where there are no labels, but predictions were made
        if labels_count == 0 and len(y_hat) == 0:
            match_count = "1/1"  # Both are empty, treat as full match
        elif labels_count == 0:
            match_count = f"0/{len(y_hat)}"  # No labels but predictions exist
        else:
            match_count = f"{intersection}/{labels_count}"  # Proportion of matches relative to the number of labels

        match_counts.append(match_count)

    return match_counts


@app.route("/")
async def home() -> str:
    """Render the home page."""
    entry_index = int(request.args.get("entry_index", 0))
    model = request.args.get("model", "gpt_4_turbo")
    client = throughsters.get(model)

    classes = nbme_data[entry_index]["classes"]
    segments = nbme_data[entry_index]["segments"]
    targets = nbme_data[entry_index]["targets"]

    alignment_data = await aligner.predict(client=client, entities=classes, segments=segments)

    metrics = evaluation_metrics(targets, alignment_data.indexes)
    match_counts = get_match_count(targets, alignment_data.indexes)

    return render_template(
        "index-with-labels.html",
        match_counts=match_counts,
        sources=classes,
        targets=segments,
        labels=targets,
        selected_entry=entry_index,
        num_entries=entry_count,
        model=model,
        alignment_matrix=alignment_data.matrix.tolist(),
        alignment_indices=alignment_data.indexes,
        metrics=metrics,
    )


@app.route("/align", methods=["GET", "POST"])
async def align() -> str:
    """Align the summary with the transcript."""
    if request.method == "POST":
        input_type = request.form.get("input_type")
        target_type = request.form.get("target_type")

        model = request.form.get("model", "gpt_4_turbo")
        client = throughsters.get(model)

        output = {}
        for _key, _type in {"sources": input_type, "targets": target_type}.items():
            if _type == "transcript":
                output[_key] = await chunk_transcript(
                    transcript=request.form[f"{_key}_transcript"], sampling_params=config.SAMPLING_PARAMS
                )
            elif _type == "summary":
                output[_key] = {}
                category_headers = request.form.getlist(f"{_key}_headers[]")
                for i, header in enumerate(category_headers):
                    category_items = request.form.getlist(f"{_key}_box_{i + 1}[]")
                    output[_key][header] = category_items

        alignment_data = await aligner.predict(
            client=client,
            entities=output["targets"] if isinstance(output["targets"], list) else _flatten_list(output["targets"]),
            segments=output["sources"] if isinstance(output["sources"], list) else _flatten_list(output["sources"]),
        )

        match_counts = [sum(1 if val == 1 or val == -1 else 0 for val in sublist) for sublist in alignment_data.matrix]

        return render_template(
            "index-without-labels.html",
            match_counts=match_counts,
            sources=output["sources"],
            targets=output["targets"],
            selected_entry=0,
            num_entries=entry_count,
            target=target_type,
            input=input_type,
            model=model,
            alignment_matrix=alignment_data.matrix.tolist(),
            alignment_indices=alignment_data.indexes,
        )

    return render_template("align.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
