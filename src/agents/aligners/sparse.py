import abc
import os
import re
import typing as typ

import pydantic
from agents.base import Aligner
from agents.models import Alignment
from segmenters.models import Segment
from tools import lexicon as um
import ahocorasick
from intervaltree import IntervalTree


class Match(pydantic.BaseModel):
    """A text chunk."""

    text: str
    start: int
    end: int
    idx: int

    # Makes the model immutable which will automatically generate __hash__ and __eq__ allowing instances to be hashed.
    model_config = pydantic.ConfigDict(frozen=True)

    @pydantic.field_validator("text", mode="before")
    @classmethod
    def _validate_text(cls, v: str) -> str:
        v = v.strip()
        if v:
            return v
        raise ValueError("Text cannot be empty")


def default_corpus_clean_fn(text: str) -> str:
    """Remove unnecessary newline characters and lowercase the text."""
    text = text.replace("\n", " ")
    text = text.replace("\n\r", " ")
    return text.strip().lower()


def default_query_clean_fn(text: str) -> str:
    """Remove unnecessary newline characters and lowercase the text."""
    text = text.replace("-", " ")
    return text.strip().lower()


class WordMatcher(abc.ABC):
    """Segments a sequence."""

    def __init__(
        self,
        clean_corpus_fn: typ.Callable[[str], str] = default_corpus_clean_fn,
        clean_query_fn: typ.Callable[[str], str] = default_query_clean_fn,
    ):
        """Initialize the WordMatcher class."""
        self.clean_corpus_fn = clean_corpus_fn
        self.clean_query_fn = clean_query_fn

    @abc.abstractmethod
    def __call__(self, queries: list[str], corpus: str) -> typ.Iterable[Match]:
        """Run the pipeline for matching."""
        ...


class ExactMatcher(WordMatcher):
    def __call__(self, queries: list[str], corpus: str) -> typ.Iterable[Match]:
        """Run the pipeline for exact matching."""
        corpus = self.clean_corpus_fn(corpus)
        queries = [self.clean_query_fn(query) for query in queries]
        matches = self.exact_matching(queries, corpus)
        return matches

    def exact_matching(self, queries: list[str], corpus: str) -> list[Match]:
        """Perform exact matching and return a list of matched spans."""
        # Build the Aho-Corasick automaton (see: https://pyahocorasick.readthedocs.io/en/latest/)
        automaton = ahocorasick.Automaton()
        for idx, query in enumerate(queries):
            automaton.add_word(query, (idx, query))
        automaton.make_automaton()
        # Search for all queries in the corpus in a single pass
        matches = set()
        for end_index, (idx, query) in automaton.iter(corpus):
            start_index = end_index - len(query) + 1
            matches.add(Match(start=start_index, end=end_index + 1, text=query, idx=idx))

        return list(matches)


class LexiconMatcher(WordMatcher):
    def __init__(self, lexicon: um.UMLS):
        """Initialize the LexiconMatcher class."""
        self.lexicon = lexicon
        super().__init__()

    def __call__(self, queries: list[str], corpus: str) -> typ.Iterable[Match]:
        """Run the pipeline for lexicon matching."""
        corpus = self.clean_query_fn(corpus)
        queries = [self.clean_query_fn(query) for query in queries]
        query_expansion = [self.lookup_synonyms(query) for query in queries]
        if not any(query_expansion):
            return []
        matches = set()
        matches.update(self.synonym_matching(query_expansion, corpus))

        return list(matches)

    def lookup_synonyms(self, query: str) -> list[str]:
        """Perform lexical matching and return a list of matched spans."""
        cui = self.lexicon.search(query, k=1)
        if not cui:
            return []
        return self.lexicon.synonyms(cui[0].ui)

    def synonym_matching(self, queries: list[list[str]], corpus: str) -> list[Match]:
        """Perform exact matching and return a list of matched spans."""
        # Build the Aho-Corasick automaton (see: https://pyahocorasick.readthedocs.io/en/latest/)
        automaton = ahocorasick.Automaton()
        for idx, synonyms in enumerate(queries):
            if not synonyms:
                continue
            for synonym in synonyms:
                automaton.add_word(synonym, (idx, synonym))
        automaton.make_automaton()
        # Search for all synonyms in the corpus in a single pass
        matches = set()
        for end_index, (idx, query) in automaton.iter(corpus):
            start_index = end_index - len(query) + 1
            matches.add(Match(start=start_index, end=end_index + 1, text=query, idx=idx))

        return list(matches)


class FuzzyMatcher(LexiconMatcher):
    def __init__(self, initial_threshold: float = 0.8, dynamic_k: int = 2, lexicon: bool = False):
        """Initialize the FuzzyMatching class."""
        self.initial_threshold = initial_threshold
        self.dynamic_k = dynamic_k
        if lexicon:
            self.lexicon = um.UMLS(os.environ.get("UMLS_API_KEY"))

    def __call__(self, queries: list[str], corpus: str) -> typ.Iterable[Match]:
        """Run the pipeline for fuzzy matching."""
        corpus = self.clean_query_fn(corpus)
        queries = [self.clean_query_fn(query) for query in queries]
        matches = []
        matches.extend(self.fuzzy_similarity_matching(q, corpus) for q in queries)
        return matches

    def fuzzy_similarity_matching(self, query: str, corpus: str) -> list[Match]:
        """Perform fuzzy similarity and dynamic threshold matching, returning a list of Segments."""
        # Use regex to split the corpus into words and get their character positions
        matches = list(re.finditer(r"\S+", corpus))
        corpus_words = [match.group() for match in matches]
        word_starts = [match.start() for match in matches]
        word_ends = [match.end() for match in matches]
        corpus_length = len(corpus_words)

        query_words = query.split()
        query_length = len(query_words)
        window_sizes = range(max(1, query_length - 1), query_length + 3)

        segments = []

        for window_size in window_sizes:
            for i in range(corpus_length - window_size + 1):
                window_span_words = corpus_words[i : i + window_size]
                window_span = " ".join(window_span_words)
                ratio = self.levenshtein_ratio(query, window_span)
                dt = self.dynamic_threshold(query_length)
                if ratio >= dt:
                    # Map word indices to character indices in the corpus
                    start = word_starts[i]
                    end = word_ends[i + window_size - 1]
                    segment_text = corpus[start:end]
                    segment = Segment(text=segment_text, start=start, end=end)
                    segments.append(segment)

        return segments

    # Helper methods
    def dynamic_threshold(self, n: int) -> float:
        """Calculate the dynamic threshold based on the length in words of the query."""
        dt = self.initial_threshold * (self.dynamic_k / n)
        return dt

    def levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Compute the Levenshtein ratio between two strings."""
        lensum = len(s1) + len(s2)
        lendist = self.levenshtein_distance(s1, s2)
        ratio = (lensum - lendist) / lensum
        return ratio

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


class ChainMatcher(WordMatcher):
    def __init__(self, *matchers: WordMatcher):
        """Initialize the ChainMatcher class."""
        self.matchers = matchers

    def __call__(self, queries: list[str], corpus: str) -> typ.Iterable[Match]:
        """Run the pipeline for chain matching."""
        matches = []
        for matcher in self.matchers:
            matches.extend(matcher(queries, corpus))
            ids = {match.idx for match in matches}
            # Remove queries that have already been matched
            queries = [q for i, q in enumerate(queries) if i not in ids]
        return matches


class WordMatchAligner(Aligner):
    def __init__(self, matcher: WordMatcher):
        """Initialize the FuzzyMatchAligner class."""
        self.aligner = matcher

    async def predict(self, corpus: list[str], queries: list[str], **kwargs) -> Alignment:
        """Align a list of queries to a corpus list."""
        # Concatenate corpus chunks
        concatenated_corpus = ""
        offsets = IntervalTree()
        cursor = 0
        # indices start at 1 to account for no match "[0]""
        for idx, chunk in enumerate(corpus, 1):
            concatenated_corpus += chunk + " "
            end = cursor + len(chunk)
            offsets[cursor:end] = idx
            cursor = end + 1
        concatenated_corpus = concatenated_corpus.strip()

        # Use the matcher to find matches for all queries
        matches: typ.Iterable[Match] = self.aligner(queries, concatenated_corpus)

        indices = self._decode_indices(matches, len(queries), offsets)
        return Alignment(indexes=indices)

    def _decode_indices(self, matches: typ.Iterable[Match], num_queries: int, offsets: IntervalTree) -> list[list[int]]:
        """Build the indices matrix from matches."""
        # Initialize a list to hold the corpus indices for each query
        indices = [[] for _ in range(num_queries)]

        # Update the indices list with the corpus indices for each match
        for match in matches:
            # Get the corpus indices (chunk indices) that overlap with the match interval
            overlapping_intervals = offsets[match.start : match.end]
            corpus_idxs = {interval.data for interval in overlapping_intervals}
            query_idx = match.idx
            # Update the indices list for this query
            indices[query_idx].extend(corpus_idxs)

        # Remove duplicates, sort the indices, and handle queries with no matches
        indices = [
            sorted(set(idx_list)) if idx_list else [0]  # [0] indicates no match for the query
            for idx_list in indices
        ]
        return indices
