from abc import abstractmethod
import asyncio
from collections import defaultdict
from functools import partial
import re
import typing as typ
from loguru import logger
import torch
from transformers import AutoTokenizer

from prompt_poet import Prompt
from jinja2 import Environment, FileSystemLoader
import pydantic
from throughster.base import ModelInterface
from throughster.core.models import ResponseChoice, BaseResponse
from throughster.hf_datasets import HfOperation

from agents.base import PATH_TO_TEMPLATES, custom_tojson
from trie.base import Trie
from trie import models
from models.plmicd import PLMICDModel



class InputModel(pydantic.BaseModel):
    """Input model for the Locate Agent."""

    note: str
    terms: list[models.Term]


class OutputModel(pydantic.BaseModel):
    """Output model for the Locate Agent."""

    terms: list[models.Term]
    codes: list[models.Code]
    evidence: list[tuple[str, str]] | None = None
    response: str | None = None


class PLMICDLocateAgent(HfOperation):

    def __init__(
        self,
        pretrained_model_path: str,
        device: str = "cpu",
        top_k: int = 1000,
        note_max_length: int = 4000,
        *args,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path + "/tokenizer")
        self.model = PLMICDModel.from_pretrained(pretrained_model_path + "/model")
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.id2label = self.model.config.id2label
        self.top_k = top_k
        self.note_max_length = note_max_length
        super().__init__(init_client_fn=lambda _: None, *args, **kwargs)

    def __call__(self, batch: dict[str, list[typ.Any]], *args, **kwargs) -> dict[str, list[typ.Any]]:
        batch_size = len(batch[list(batch.keys())[0]])
        batch_rows = [{key: value[i] for key, value in batch.items()} for i in range(batch_size)]
        batch_rows = [InputModel(
            terms=[],
            note=row["note"],
        ) for row in batch_rows]
        batch_notes = [row.note for row in batch_rows]

        # Tokenize the data
        tokenized_inputs = self.tokenizer(
            batch_notes,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.note_max_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)

        # Sort the output logits for each entry in the batch and get the top k labels
        logits = outputs["logits"].sigmoid()
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        # Convert the top_k indices to labels
        # for item in batch... for index in item
        top_k_codes = [[self.id2label[str(idx.item())] for idx in indices] for indices in top_k_indices]

        return {
            **batch,
            "terms": [[] for _ in range(batch_size)],
            "codes": top_k_codes,
            "evidence": [[] for _ in range(batch_size)],
            "response": [[] for _ in range(batch_size)],
        }

class HfLocateAgent(HfOperation):
    def __init__(self, trie: Trie, init_client_fn: typ.Callable[..., ModelInterface]):
        self.trie: Trie = trie
        self.init_client_fn = init_client_fn

    def __call__(self, batch: dict[str, list[typ.Any]], *args, **kwargs) -> dict[str, list[typ.Any]]:
        """Process a row of alignment tasks from a HuggingFace datasets.map()."""
        batch_size = len(batch[list(batch.keys())[0]])

        main_terms = self.trie.get_all_main_terms()
        batch_rows = [{key: value[i] for key, value in batch.items()} for i in range(batch_size)]
        batch_rows = [self._format_input({**row, "terms": main_terms}) for row in batch_rows]

        responses = asyncio.run(*[self.predict(row) for row in batch_rows])

        batch = {key: [row[key] for row in batch_rows] for key in batch_rows[0].keys()}

        output = defaultdict(list)
        for i, r in enumerate(responses):
            alignment_data = r.model_dump()
            for key, value in alignment_data.items():
                output[key].append(value)

        return {
            **batch,
            **output,
        }

    @staticmethod
    def _format_input(rows: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Format the targets."""
        return InputModel(**rows)

    @abstractmethod
    async def predict(self, inputs: list[InputModel]) -> list[OutputModel]:
        """Handle a batch of alignment tasks."""
        NotImplementedError
        

class LLMLocateAgent(HfLocateAgent):
    def __init__(self, prompt_name: str, seed: int, sampling_params: dict[str, typ.Any], *args, **kwargs):
        env = Environment(loader=FileSystemLoader(PATH_TO_TEMPLATES), autoescape=False)
        loader = typ.cast(FileSystemLoader, env.loader)
        self.raw_template, self.template_path, _ = loader.get_source(env, f"{prompt_name}.yml.j2")
        self.prompt_name = prompt_name
        self.seed = seed
        self.sampling_params = sampling_params
        super().__init__(*args, **kwargs)

    async def predict(self, input: InputModel) -> OutputModel:
        """Handle a batch of alignment tasks."""
        request = self.format_request(**input.model_dump())
        resp: BaseResponse = await self.client.call(request=request)

        preds, response = self.compress_choices(resp.choices)
        predicted_terms = [input.terms[i - 1] for i in preds]
        term_codes = set()
        for term in predicted_terms:
            term_codes.update(self.trie.get_all_term_codes(term.id))

        return OutputModel(
            terms=predicted_terms,
            codes=[self.trie.tabular[self.lookup[c]] for c in term_codes],
            evidence=None,
            response=response,
        )

    def format_request(self, **kwargs) -> Prompt:
        """Format the prompt."""
        prompt_template = Prompt(
            raw_template=self.raw_template,
            template_data={"custom_tojson": custom_tojson, **kwargs},
        )
        prompt = self.prompt_messages_or_string(self.client, prompt_template)
        return {
            "prompt": prompt if self.client.endpoint == "completions" else None,
            "messages": prompt if self.client.endpoint == "chat/completions" else None,
            "seed": self.seed,
            "max_tokens": 5000,
            **self.sampling_params,
        }

    @staticmethod
    def prompt_messages_or_string(client: ModelInterface, prompt: Prompt) -> str | list[dict[str, str]]:
        if client.endpoint == "chat/completions":
            return prompt.messages
        return prompt.string

    def compress_choices(self, choices: list[ResponseChoice]) -> tuple[list[int], str]:
        """Compress the choices."""
        c = choices[0]
        answer_match = re.search(self.ANSWER_PATTERN, c.content)
        preds = [int(num.strip()) for num in answer_match.group(1).split(",")] if answer_match else []
        if not preds:
            logger.warning(f"Could not find any relevant tokens in the response: {c.content[-250:]}")
        return preds, c.content


def create_locate_agent(
    agent_type: str,
    sampling_params: dict[str, typ.Any] = {},
    prompt_name: str | None = None,
    pretrained_model_path: str | None = None,
    seed: int = 42,
) -> HfLocateAgent:
    """
    Factory method to create an LLMAligner instance based on the specified type.

    Args:
        aligner_type (str): The type of aligner to create ("binary" or "long_context").
        prompt_name (str): The name of the prompt template to use.
        num_shots (int): Number of few-shot examples.
        token_limit (int): Token limit for the prompts.
        seed (int): Seed for random operations.
        sampling_params (dict[str, Any]): Sampling parameters for the LLM.

    Returns:
        LLMAligner: An instance of either BinaryLLMAligner or LongContextLLMAligner.
    """
    if not prompt_name and not pretrained_model_path:
        raise ValueError("Either prompt_name or pretrained_model_path must be provided.")
    if agent_type == "long-context":
        return partial(
            LLMLocateAgent,
            prompt_name=prompt_name,
            seed=seed,
            sampling_params=sampling_params,
        )
    elif agent_type == "plmicd":
        return partial(
            PLMICDLocateAgent,
            pretrained_model_path=pretrained_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            top_k=sampling_params.get("top_k", 1000),
            note_max_length=sampling_params.get("note_max_length", 4000),
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
