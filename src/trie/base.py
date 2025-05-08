from abc import ABC, abstractmethod
from collections import defaultdict
import pathlib

import numpy as np
from rapidfuzz import fuzz, utils, process

from trie import models

TRIE_CACHE_DIR = pathlib.Path("~/.cache/trie").expanduser()
TRIE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class Trie(ABC):
    """Trie data structure for storing and retrieving Codes."""

    def __init__(self) -> None:
        self.roots: list[str] = []
        self.tabular: dict[str, models.Code] = {}
        self.lookup: dict[str, str] = {}
        self.index: dict[str, models.Term] = {}
        self.index_lookup: dict[str, str] = {}

    def __repr__(self) -> str:
        return str(self.tabular)

    def __getitem__(self, code: str) -> models.Code:
        code_id = self.lookup[code]
        return self.tabular[code_id]

    def get_all_tabular_parents(self, code_id: str) -> list[models.Code]:
        parents = []
        code = self.tabular[code_id]
        while code.parent_id:
            code = self.tabular[code.parent_id]
            parents.append(code)
        return parents

    def get_all_index_parents(self, term_id: str) -> list[models.Term]:
        parents = []
        code = self.index[term_id]
        while code.parent_id:
            code = self.index[code.parent_id]
            parents.append(code)
        return parents

    def get_all_tabular_children(self, code_id: str) -> list[models.Code]:
        children = []
        code = self.tabular[self.lookup[code_id]]
        if code.children_ids:
            for c_id in code.children_ids:
                children.append(c_id)
                children.extend(self.get_all_tabular_children(c_id))
        return children

    def get_chapter_id(self, code: str) -> str:
        parents = self.get_all_tabular_parents(code)
        return parents[-1].id

    def get_category_id(self, code: str) -> str:
        parents = self.get_all_tabular_parents(code)
        return parents[-2].id

    def get_all_main_terms(self) -> list[models.Term]:
        """Get all main terms in the index."""
        return [term for term in self.index.values() if not term.parent_id]

    def get_all_term_children(self, term_id: str) -> list[models.Term]:
        """Get all term children in the index."""
        if term_id not in self.index:
            raise ValueError(f"Term with id {term_id} does not exist in the index.")
        term = self.index[term_id]
        children = []
        if term.children_ids:
            for c_id in term.children_ids:
                children.append(self.index[c_id])
                children.extend(self.get_all_term_children(c_id))
        return children

    def get_assignable_terms(self) -> list[models.Term]:
        """Get all assignable terms in the index."""
        return [term for term in self.index.values() if term.assignable]

    def get_all_term_codes(self, term_id: str) -> list[str]:
        """Get all term codes in the index."""
        codes = set()
        if term_id not in self.index:
            raise ValueError(f"Term with id {term_id} does not exist in the index.")
        term = self.index[term_id]

        if term.assignable and term.code:
            codes.add(term.code)
            codes.update(self.get_all_tabular_children(term.code))
            if term.manifestation_code:
                codes.add(term.manifestation_code)
        elif term.code and not term.assignable:
            codes.update(self.get_all_tabular_children(term.code))

        sub_terms = self.get_all_term_children(term.id)
        for sub_term in sub_terms:
            if sub_term.assignable and sub_term.code:
                codes.add(sub_term.code)
                codes.update(self.get_all_tabular_children(sub_term.code))
                if term.manifestation_code:
                    codes.add(sub_term.manifestation_code)
            elif sub_term.code and not sub_term.assignable:
                codes.update(self.get_all_tabular_children(sub_term.code))

        # Return both primary codes and manifestation codes, with manifestation codes last
        return list(codes)

    def get_leaves(self) -> list[models.Code]:
        return [n for n in self.tabular.values() if n.assignable]

    def get_instructional_notes(
        self, codes: list[str]
    ) -> list[models.InstructionalNote]:
        """Get all instructional notes for a given code."""
        included_parents = set()
        instructional_notes = defaultdict(list)
        for c_id in codes:
            c = self[c_id]
            parents = [
                p for p in self.get_all_tabular_parents(c_id) if p.id not in self.roots
            ]
            chapter = parents.pop(-1)
            if chapter.name not in instructional_notes:
                instructional_notes[chapter.name].append(
                    models.InstructionalNote(**chapter.model_dump()).model_dump(
                        exclude_none=True
                    )
                )
            for parent in reversed([c] + parents):
                if parent.id in included_parents:
                    continue
                instruct_note = models.InstructionalNote(**parent.model_dump())
                # if all fields are empty or None, skip
                if instruct_note.is_empty():
                    continue
                instructional_notes[chapter.id].append(
                    instruct_note.model_dump(exclude_none=True)
                )  # Prepend guideline_data to guidelines
                included_parents.add(parent.id)
        return [note for group in instructional_notes.values() for note in group]

    def get_root_leaves(self, root: str) -> list[models.Code]:
        """Get all codes under a specific root."""
        if root not in self.roots:
            raise ValueError(
                f"Root {root} does not exist in the trie. Available roots: {self.roots}"
            )
        root_node = self.tabular[root]
        return [
            self.tabular[code_id]
            for code_id in root_node.children_ids
            if self.tabular[code_id].assignable
        ]

    def insert_to_tabular(
        self, node: models.Root | models.Category | models.Code
    ) -> None:
        """Insert a node into the trie."""
        if node.id in self.tabular:
            raise ValueError(f"Node with id {node.id} already exists in the trie.")

        self.tabular[node.id] = node  # type: ignore
        if isinstance(node, models.Code):
            if node.id in self.lookup:
                raise ValueError(f"Code {node.name} already exists in the lookup.")
            if node.parent_id not in self.tabular:
                raise ValueError(f"Parent {node.parent_id} does not exist in the trie.")
            self.lookup[node.name] = node.id

        if node.parent_id:
            parents = self.get_all_tabular_parents(node.id)
            for parent in parents:
                parent.children_ids.append(node.id)

    def insert_to_index(self, node: models.Term) -> None:
        """Insert a node into the index."""
        if node.id in self.index:
            raise ValueError(f"Node with id {node.id} already exists in the index.")
        self.index[node.id] = node
        self.index_lookup[node.id] = node.title
        node.path = node.title  # default path is the title
        if node.parent_id:
            parent = self.index[node.parent_id]  # only the direct parent
            if node.id not in parent.children_ids:  # avoid duplicates
                parent.children_ids.append(node.id)

            # build the path (now just follow parent links)
            parents = self.get_all_index_parents(node.id)[::-1]  # root → … → parent
            node.path = ", ".join([p.title for p in parents] + [node.title])

    def find_terms(
        self, queries: list[str], main_terms: bool, limit: int = 1
    ) -> list[models.Term]:
        """Find a term in the index."""
        if main_terms:
            choices = []
            ids = []
            for term in self.get_all_main_terms():
                choices.append(term.title)
                ids.append(term.id)
        else:
            choices = list(self.index_lookup.values())
            ids = list(self.index_lookup.keys())

        fuzzy_matrix: np.ndarray = process.cdist(
            queries,
            choices,
            scorer=fuzz.QRatio,
            processor=utils.default_process,
        )
        indices = np.argsort(fuzzy_matrix, axis=1)[:, -limit:]
        return [self.index[idx] for idx in np.array(ids)[indices].flatten()]

    @abstractmethod
    def parse(self, *args, **kwargs) -> "Trie":
        """Abstract method to parse the trie. Must be implemented by subclasses."""
        ...
