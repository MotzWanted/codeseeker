import typing
import numpy as np
import torch

from trie.base import Trie


class NegativeSampler:

    def __init__(
        self,
        negatives: dict[str, list[str]],
        trie: Trie,
        num_negatives_per_code: int = 5,
        top_k: int | None = 200,
        seed: int | None = 42,
        device: str = "cpu",
        within_chapter: bool | None = None,
    ) -> None:
        # load negatives
        self.negatives: dict[str, list[str]] = negatives
        self.trie = trie
        self.within_chapter = within_chapter
        self.num_negatives_per_code = num_negatives_per_code
        self.rng = np.random.RandomState(seed)
        self.top_k = top_k
        if top_k is None:
            top_k = len(next(iter(negatives.values())))
        self.weights = np.exp(-0.5 * np.arange(top_k))
        self.weights /= self.weights.sum()

        self.device = torch.device(device)

    def _sample_negatives(self, code_id: str) -> list[str]:
        """Sample negatives for a given code (with capped population)."""
        if self.within_chapter is None:
            population = self.negatives[code_id][: self.top_k]
        else:
            operator = np.equal if self.within_chapter else np.not_equal
            chapter_id = self.trie.get_chapter_id(code_id)
            population = [
                code
                for code in self.negatives[code_id]
                if operator(self.trie.get_chapter_id(code), chapter_id)
            ][: self.top_k]
        n = min(len(population), self.num_negatives_per_code)
        if n == 0:
            return []
        return self.rng.choice(
            population, size=n, replace=False, p=self.weights
        ).tolist()

    def __call__(
        self, batch: dict[str, list[typing.Any]], *args, **kwargs
    ) -> dict[str, list[typing.Any]]:
        if "targets" not in batch:
            raise ValueError(
                "Batch must contain a 'targets' field (list of list of codes)"
            )

        batch_codes = [
            self._sample_negatives(code) + [code]
            for codes in batch["targets"]
            for code in codes
        ]

        return {**batch, "codes": batch_codes}
