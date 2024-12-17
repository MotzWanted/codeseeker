import abc
import functools
import re
import typing as typ
from spacy import Language
import numpy as np
import spacy
from spacy.cli.download import download as spacy_download

from segmenters.models import Segment
from segmenters.patterns import EndOfSentencePattern, IsInlinePunctuation, TitlePattern


def _load_spacy_model(model_name: str) -> Language:
    try:
        return spacy.load(model_name)
    except OSError:
        spacy_download(model_name)
        return spacy.load(model_name)


# @numba.njit(cache=True, nogil=True, fastmath=True)
def _is_within_bounds(x: int, spans: np.ndarray) -> bool:
    for s in spans:
        if x >= s[0] and x <= s[1]:
            return True
    return False


class Segmenter(abc.ABC):
    """Segments a sequence."""

    @abc.abstractmethod
    def __call__(self, text: str) -> typ.Iterable[Segment]:
        """Chunk a sequence into chunks."""
        ...


class NbmeSegmenter(Segmenter):
    """Chunks a Nbme patient note into sentences."""

    sentence_pattern = re.compile(r"(?<=[.!?])\s+|\r\n|\n")
    bullet_pattern = re.compile(r"(\b[A-Z]{1,}\b:)([\s\S]*?)(?=\r\n[A-Z]{1,}\b:|\Z)", re.DOTALL)

    def __init__(self, min_length: int = 6) -> None:
        self.min_length = min_length

    def __call__(self, text: str) -> typ.Iterable[Segment]:
        """Chunk a sequence into sentences."""
        titles_list = [(match.start(), match.end()) for match in TitlePattern.finditer(text)]
        titles = np.array(titles_list) if len(titles_list) > 0 else None
        bullet_strings = [match.group(0) for match in self.bullet_pattern.finditer(text)]
        bullet_list = [(match.start(), match.end()) for match in self.bullet_pattern.finditer(text)]
        bullets = np.array(bullet_list) if len(bullet_list) > 0 else None
        cursor = 0
        for match_start in sorted({match.start() for match in self.sentence_pattern.finditer(text)}):
            # Check if a bullet is split between the two chunks; if yes, join the segments.
            chunk = text[cursor:match_start].strip()
            if (
                bullets is not None
                and _is_within_bounds(match_start, bullets)
                and any(bullet in chunk for bullet in bullet_strings)
            ):
                yield Segment(
                    text=chunk,
                    start=cursor,
                    end=match_start,
                )
                cursor = match_start
            elif titles is not None and _is_within_bounds(match_start, titles):
                continue
            elif len(chunk) < self.min_length:
                continue
            else:
                yield Segment(
                    text=chunk,
                    start=cursor,
                    end=match_start,
                )
                cursor = match_start

        # Last segment
        if cursor < len(text):
            chunk = text[cursor:].strip()
            if self.min_length < len(chunk):
                yield Segment(
                    text=chunk,
                    start=cursor,
                    end=len(text),
                )


class WindowSegmenter(Segmenter):
    """Chunks a sequence into windows."""

    def __init__(self, size: int = 100, overlap: int = 0):
        self.size = size
        self.overlap = min(overlap, self.size - 1)

    def __call__(self, text: str) -> typ.Iterable[Segment]:
        """Chunk a sequence into windows."""
        cursor = 0
        while cursor < len(text):
            end_char = min(cursor + self.size, len(text))
            chunk = text[cursor:end_char].strip()
            if chunk:
                yield Segment(
                    text=chunk,
                    start=cursor,
                    end=end_char,
                )
            cursor += self.size - self.overlap


class DocumentSegmenter(Segmenter):
    """Chunks a sequence into documents."""

    def __call__(self, text: str) -> typ.Iterable[Segment]:
        """Chunk a sequence into documents."""
        cursor = 0
        yield Segment(
            text=text,
            start=cursor,
            end=len(text),
        )


class SpacySegmenter(Segmenter):
    """Chunks a sequence into sentences using spaCy."""

    def __init__(self, model_name: str = "en_core_web_sm", min_length: int = 5):
        self.model_name = model_name
        self.nlp = _load_spacy_model(model_name)
        self.min_length = min_length

    def __call__(self, text: str) -> typ.Iterable[Segment]:
        """Chunk a sequence into sentences."""
        buffer = ""
        cursor = 0
        for sent in self.nlp(text).sents:
            buffer += sent.text + " "
            chunk = buffer.strip()
            if len(chunk) >= self.min_length:
                yield Segment(text=chunk, start=cursor, end=sent.end_char)
                buffer = ""
                cursor = sent.end_char

        # Last segment
        if buffer:
            yield Segment(text=buffer.strip(), start=cursor, end=len(text))


class GateSegmenter(Segmenter):
    """Implements a gating mechanism."""

    def __init__(
        self,
        condition: typ.Callable[[str], typ.Any],
        on_true: None | Segmenter = None,
        on_false: None | Segmenter = None,
    ):
        self.condition = condition
        self.on_true = on_true
        self.on_false = on_false

    def __call__(self, text: str) -> typ.Iterable[Segment]:
        """Chunk a sequence into sentences."""
        match = self.condition(text)
        if match and self.on_true is not None:
            yield from self.on_true(text)
        elif not match and self.on_false is not None:
            yield from self.on_false(text)
        else:
            yield Segment(text=text, start=0, end=len(text))


class RegexSegmenter(Segmenter):
    """Shorten segments that are too long."""

    def __init__(
        self,
        pattern: re.Pattern,
        min_length: int = 1,
    ):
        self.pattern = pattern
        self.min_length = min_length

    def __call__(self, text: str) -> typ.Iterable[Segment]:
        """Process the output segments and make them shorter."""
        cursor = 0
        for match_start in sorted({match.start() for match in self.pattern.finditer(text)}):
            chunk = text[cursor:match_start].strip()
            if len(chunk) < self.min_length:
                continue
            else:
                yield Segment(
                    text=chunk,
                    start=cursor,
                    end=match_start,
                )
                cursor = match_start

        # Last segment
        chunk = text[cursor:].strip()
        if cursor < len(text):
            if len(chunk) < self.min_length:
                return
            yield Segment(
                text=chunk,
                start=cursor,
                end=len(text),
            )


class ChainSegmenters(Segmenter):
    """Chain multiple segmenters together."""

    def __init__(
        self,
        *segmenters: None | Segmenter,
        filter_fn: None | typ.Callable[[str], typ.Any] = None,
        clean_fn: None | typ.Callable[[str], str] = None,
    ):
        self.segmenters = [seg for seg in segmenters if seg is not None]
        self.filter_fn = filter_fn
        self.clean_fn = clean_fn

    @staticmethod
    def _apply_segmenter(segment: Segment, segmenter: Segmenter) -> typ.Iterable[Segment]:
        for s in segmenter(segment.text):
            yield Segment(
                text=s.text,
                start=s.start + segment.start,
                end=s.end + segment.start,
            )

    def __call__(self, text: str) -> typ.Iterable[Segment]:
        """Chunk a sequence into sentences."""
        segments: typ.Iterable[Segment] = [Segment(text=text, start=0, end=len(text))]
        # Recursively apply the segmenters
        for segmenter in self.segmenters:
            segments = [s for segment in segments for s in self._apply_segmenter(segment, segmenter)]

        # Yield the segments
        for segment in segments:
            if self.clean_fn is not None:
                segment.text = self.clean_fn(segment.text)
            if self.filter_fn is not None and not self.filter_fn(segment.text):
                continue
            yield segment


def factory(type_: str, spacy_model: str) -> Segmenter:
    """Return a segmenter."""
    if type_ == "spacy":
        segmenter = SpacySegmenter(spacy_model, min_length=5)
    elif type_ == "document":
        segmenter = DocumentSegmenter()
    elif type_ == "nbme":
        segmenter = NbmeSegmenter()
    elif type_ == "sentence":
        segmenter = RegexSegmenter(EndOfSentencePattern, min_length=5)
    elif type_ == "window":
        segmenter = WindowSegmenter()
    else:
        raise ValueError(f"Invalid segmenter type: `{type_}`")
    return segmenter


class StripSegmenter(Segmenter):
    """Strip punctuation."""

    def __init__(self, chars: str = r",\*\.;#\t\n\- "):
        self.left_pattern = re.compile(rf"^[{chars}]+")
        self.right_pattern = re.compile(rf"[{chars}]+$")

    def __call__(self, text: str) -> typ.Iterable[Segment]:
        """Process the output segments and make them shorter."""
        start_length = len(text)
        text = self.left_pattern.sub("", text)
        n_chars_removed_left = start_length - len(text)
        text = self.right_pattern.sub("", text)
        yield Segment(
            text=text,
            start=n_chars_removed_left,
            end=n_chars_removed_left + len(text),
        )


def _longer_than(text: str, length: float = 1) -> bool:
    return len(text) >= length


def pipeline(
    type_: str = "spacy",
    spacy_model: str = "en_core_web_md",
    shorten_segments_above: float = 120,
) -> Segmenter:
    """Segmentation pipeline."""
    base_segmenter = factory(type_.replace("raw_", ""), spacy_model)
    if type_.startswith("raw_"):
        return ChainSegmenters(RegexSegmenter(re.compile(r"(\n\n)|(---)")), base_segmenter)
    triggered_with_length = functools.partial(_longer_than, length=shorten_segments_above)
    return ChainSegmenters(
        StripSegmenter(),
        base_segmenter,
        StripSegmenter(),
        GateSegmenter(triggered_with_length, on_true=RegexSegmenter(IsInlinePunctuation)),
    )
