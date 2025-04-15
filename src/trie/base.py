from abc import ABC, abstractmethod
import pathlib

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

    def get_all_index_parents(self, term_id: str) -> list[models.Code]:
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

    def get_chapter_name(self, code: str) -> str:
        parents = self.get_all_tabular_parents(code)
        return parents[-1].name

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

    def get_all_term_codes(self, term_id: str) -> list[str]:
        """Get all term codes in the index."""
        codes = set()
        if term_id not in self.index:
            raise ValueError(f"Term with id {term_id} does not exist in the index.")
        term = self.index[term_id]
        if term.code:
            codes.add(term.code)
        sub_terms = self.get_all_term_children(term.id)
        for sub_term in sub_terms:
            if sub_term.code:
                if sub_term.assignable:
                    codes.add(sub_term.code)
                else:
                    codes.update(self.get_all_tabular_children(sub_term.code))
        return list(codes)

    def get_leaves(self) -> list[models.Code]:
        return [n for n in self.tabular.values() if n.assignable]

    def get_root_leaves(self, root: str) -> list[models.Code]:
        """Get all codes under a specific root."""
        if root not in self.roots:
            raise ValueError(f"Root {root} does not exist in the trie. Available roots: {self.roots}")
        root_node = self.tabular[root]
        return [self.tabular[code_id] for code_id in root_node.children_ids if self.tabular[code_id].assignable]

    def insert_to_tabular(self, node: models.Root | models.Category | models.Code) -> None:
        """Insert a node into the trie."""
        if node.id in self.tabular:
            raise ValueError(f"Node with id {node.id} already exists in the trie.")

        self.tabular[node.id] = node
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
        if node.parent_id:
            parents = self.get_all_index_parents(node.id)
            for parent in parents:
                parent.children_ids.append(node.id)

    @abstractmethod
    def parse(self, *args, **kwargs) -> "Trie":
        """Abstract method to parse the trie. Must be implemented by subclasses."""
        NotImplementedError

    # def get_guidelines(self, codes: list[str]) -> dict[str, typing.Any]:
    #     guideline_fields = [
    #         "notes",
    #         "includes",
    #         "excludes1",
    #         "excludes2",
    #         "use_additional_code",
    #         "code_first",
    #         "code_also",
    #         "inclusion_term",
    #     ]
    #     guidelines = []
    #     included_parents = set()
    #     for c in codes:
    #         code = self[c]
    #         for parent in reversed([code] + self.get_all_parents(code.id)):
    #             if "icd10cm" in parent.name or parent.id in included_parents:
    #                 continue
    #             guideline_data = {k: v for k, v in parent.model_dump(include=guideline_fields).items() if v}
    #             if not guideline_data:
    #                 continue
    #             guideline_data["code"] = parent.name
    #             guideline_data["assignable"] = parent.assignable
    #             guidelines.append(guideline_data)  # Prepend guideline_data to guidelines
    #             included_parents.add(parent.id)

    #     return guidelines
