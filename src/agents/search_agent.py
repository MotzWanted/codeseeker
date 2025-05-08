import asyncio
import typing as typ

import numpy as np
from throughster.base import ModelInterface

from throughster.hf_datasets import HfOperation

Text = str
FlatBatch = list[Text]
Nested = list[Text] | list[list[Text]]
Embeds = list[np.ndarray]


class HfEmbeddingClient(HfOperation):
    """Base class for coding agents."""

    def __init__(
        self,
        prompt_key: str,
        init_client_fn: typ.Callable[..., ModelInterface],
    ):
        self.prompt_key = prompt_key
        self.init_client_fn = init_client_fn
        super().__init__(init_client_fn=init_client_fn)

    def __call__(
        self, batch: dict[str, list[typ.Any]], *args, **kwargs
    ) -> dict[str, list[typ.Any]]:
        """Process a row of agent tasks from a HuggingFace datasets.map()."""
        raw_inputs: Nested = batch[self.prompt_key]

        flat_inputs, regroup = self._flatten(raw_inputs)  # ① flatten
        flat_embeds: Embeds = asyncio.run(
            self._async_call_wrapper(flat_inputs)  # ② embed once
        )
        nested_embeds: Nested = regroup(flat_embeds)  # ③ regroup

        return {
            "dense_embeddings": nested_embeds,
            **batch,  # keep original fields
        }

    async def _async_call_wrapper(self, batch_inputs: list[str]) -> list[np.ndarray]:
        """"""
        return await self.client.embed(texts=batch_inputs)

    @staticmethod
    def _flatten(raw: Nested) -> tuple[FlatBatch, typ.Callable[[Embeds], Nested]]:
        """Flatten `raw` into a single `list[str]` and return a function that can
        restore the original nesting for a list of embeddings of equal length.
        """
        # Fast path: already flat
        if not raw or isinstance(raw[0], str):
            flat_batch = typ.cast(FlatBatch, raw)
            return flat_batch, lambda embeds: typ.cast(Nested, embeds)

        # Nested case
        lengths: list[int] = [len(sub) for sub in raw]  # store segment sizes
        flat: FlatBatch = [t for sub in raw for t in sub]  # flatten

        def regroup(embeds: Embeds) -> Nested:
            out, i = [], 0
            for n in lengths:
                out.append(embeds[i : i + n])
                i += n
            return out

        return flat, regroup
