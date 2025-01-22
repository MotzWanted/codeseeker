import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from random import shuffle
from typing import Dict, List, Optional

import dill
import polars as pl
from rich.progress import track


@dataclass
class Node:
    id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    assignable: bool = True
    desc: str = ""
    code: str = ""
    min: str = ""
    max: str = ""
    inclusion_notes: List[str] = field(default_factory=list)

    def within(self, code: str) -> bool:
        raise NotImplementedError("Method not implemented")

    def full_desc(self, nodes: Dict[str, "Node"]) -> str:
        if self.parent_id:
            parent = nodes[self.parent_id]
            return f"{parent.full_desc(nodes)} {self.desc}".strip()
        else:
            return self.desc

    def list_desc(self, nodes: Dict[str, "Node"]) -> List[str]:
        if self.parent_id:
            parent = nodes[self.parent_id]
            return parent.list_desc(nodes) + [self.desc]
        else:
            return []

    def is_leaf(self) -> bool:
        return not self.children_ids

    def __repr__(self) -> str:
        return f"{self.code} {self.desc}"


class Category(Node):
    min: str
    max: str

    def __repr__(self) -> str:
        return f"{self.id} {self.min}-{self.max} {self.desc}"

    def within(self, code: str) -> bool:
        return self.min[: len(code)] <= code[: len(self.max)] <= self.max[: len(code)]


class ICD(Node):
    def __repr__(self) -> str:
        return f"{self.id} {self.code} {self.desc}"

    def within(self, code: str) -> bool:
        return code[: len(self.code)] == self.code[: len(code)]


class Trie:
    _cache_dir = Path("~/.cache/tries").expanduser()
    _cache_dir.mkdir(parents=True, exist_ok=True)

    def __reduce__(self):
        # Define how to reconstruct the object during unpickling
        return (self.__class__, (), self.__dict__)

    @classmethod
    def set_cache_dir(cls, path: str | Path) -> None:
        """Set the cache directory for Trie instances."""
        cls._cache_dir = Path(path)
        cls._cache_dir.mkdir(parents=True, exist_ok=True)

    def save_to_cache(self, cache_key: str) -> None:
        """Save the trie instance to cache."""
        cache_path = self._cache_dir / f"{cache_key}.pkl"
        with open(cache_path, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load_from_cache(cls, cache_key: str) -> Optional["Trie"]:
        """Load a trie instance from cache if it exists."""
        cache_path = cls._cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return dill.load(f)
        return None

    def __init__(self) -> None:
        self.roots: List[str] = []
        self.all: Dict[str, Node] = {}
        self.lookup: Dict[str, str] = {}

    def __repr__(self) -> str:
        return str(self.all)

    def get_all_parents(self, node_id: str) -> List[Node]:
        parents = []
        node = self.all[node_id]
        while node.parent_id:
            node = self.all[node.parent_id]
            parents.append(node)
        return parents

    def get_all_children(self, node_id: str) -> List[Node]:
        children = []
        node = self.all[node_id]
        if node.children_ids:
            for c_id in node.children_ids:
                children.append(self.all[c_id])
                children.extend(self.get_all_children(c_id))
        return children

    def insert(self, node: Node, root_char: str) -> None:
        root_id = None
        for root in self.roots:
            if self.all[root].code == root_char:
                root_id = root
                break

        is_root = False
        if not root_id:
            root_id = f"{root_char}"
            root_node = Category(id=root_id, code=root_char, desc="", min=root_char, max=root_char)
            self.roots.append(root_id)
            self.all[root_id] = root_node
            is_root = True

        root = self.all[root_id]

        if node.code and node.code[0] != root.code:
            node.code = root.code + node.code

        if is_root:
            self.all[node.id] = node
            self.lookup[node.code] = node.id
            return

        self._insert(root, node)
        self.all[node.id] = node
        self.lookup[node.code] = node.id

    def _insert(self, candidate: Node, node: Node):
        if candidate.children_ids:
            for c_id in candidate.children_ids:
                c = self.all[c_id]
                if c.within(node.code):
                    return self._insert(c, node)

        node.parent_id = candidate.id
        candidate.children_ids.append(node.id)

    def get_leaves(self) -> List[ICD]:
        return [n for n in self.all.values() if not n.children_ids and isinstance(n, ICD)]

    def __getitem__(self, code: str) -> Node:
        node_id = self.lookup[code]
        return self.all[node_id]

    @staticmethod
    def _create_node(code: str, desc: str) -> Node:
        if code:
            return ICD(id=code, code=code, desc=desc.strip())
        else:
            if "Kap." in desc:
                desc = " ".join(desc.split(":")[1:])
            min, max = desc.split("[")[-1].replace("]", "").split("-")
            desc = desc.split("[")[0]
            return Category(
                id=min.strip() + "-" + max.strip(),
                max=max.strip(),
                min=min.strip(),
                desc=desc.strip(),
                code=min.strip(),
            )

    @staticmethod
    def from_sks_raw(files: Dict[str, str]) -> "Trie":
        trie = Trie()
        for root, f in files.items():
            r = csv.reader(open(f), delimiter=";")

            for x in r:
                code, desc = x
                node = Trie._create_node(code, desc)
                trie.insert(node, root)
        return trie


class XMLTrie(Trie):
    def insert(self, node: Node, root_char: str) -> None:
        root_id = None
        for root in self.roots:
            if self.all[root].code == root_char:
                root_id = root
                break

        is_root = False
        if not root_id:
            root_id = f"{root_char}"
            root_node = self.all[root_id]
            self.roots.append(root_id)
            self.all[root_id] = root_node
            is_root = True

        root = self.all[root_id]

        if is_root:
            self.all[node.id] = node
            self.lookup[node.code] = node.id
            return

        self._insert(root, node)
        self.all[node.id] = node
        self.lookup[node.code] = node.id

    @staticmethod
    def from_xml(root: ET.Element, coding_system: str) -> "XMLTrie":
        trie = XMLTrie()
        code_root_id = f"{coding_system}"
        if "icd10cm" in coding_system:
            return XMLTrie.parse_tabular(trie, root, code_root_id)
        elif "icd10pcs" in coding_system:
            return XMLTrie.parse_table(trie, root, code_root_id)
        else:
            raise ValueError(f"Unknown coding system: {coding_system}")

    @staticmethod
    def parse_table(trie: Trie, root: ET.Element, root_node_id: str) -> "XMLTrie":
        """Parse PCS tables and insert them into the trie."""
        num_tables = len(root.findall("pcsTable"))
        code_root = Category(
            id=f"{root_node_id}", code=f"{root_node_id}", desc="", min="1", max=f"{num_tables}", assignable=False
        )
        trie.roots.append(code_root.id)
        trie.all[code_root.id] = code_root
        trie.lookup[code_root.code] = code_root.id
        pcs_tables = root.findall("pcsTable")
        for table_index, pcs_table in track(
            enumerate(pcs_tables), description="Parsing PCS tables", total=len(pcs_tables)
        ):
            table_id = f"{root_node_id}_Table_{table_index + 1}"
            num_rows = len(pcs_table.findall("pcsRow"))
            table_node = Category(
                id=table_id,
                code=table_id,
                desc=f"PCS Table {table_index + 1}",
                min="1",
                max=f"{num_rows}",
                assignable=False,
            )
            trie.insert(table_node, root_char=root_node_id)

            fixed_axes = []
            for axis in sorted(pcs_table.findall("axis"), key=lambda x: int(x.get("pos", 0))):
                fixed_axes.append(axis)

            for pcs_row in pcs_table.findall("pcsRow"):
                row_id = f"{table_id}_Row_{pcs_row.get('codes')}"
                axis_pos = [axis.attrib["pos"] for axis in pcs_row.findall("axis")]
                row_node = Category(
                    id=row_id,
                    code=row_id,
                    desc=f"PCS Table {table_index + 1} PCS Row {pcs_row.get('codes')}",
                    min=min(axis_pos),
                    max=max(axis_pos),
                    assignable=False,
                )
                trie.insert(row_node, root_char=table_id)
                XMLTrie._parse_pcs_row(trie, pcs_row, fixed_axes, parent_code=row_id)

        return trie

    @staticmethod
    def parse_tabular(trie: Trie, root: ET.Element, root_node_id: str) -> "XMLTrie":
        num_chapters = len(root.findall("chapter"))
        code_root = Category(
            id=f"{root_node_id}", code=f"{root_node_id}", desc="", min="1", max=f"{num_chapters}", assignable=False
        )
        trie.roots.append(code_root.id)
        trie.all[code_root.id] = code_root
        trie.lookup[code_root.code] = code_root.id
        for chapter in track(root.findall("chapter"), description="Parsing ICD-10-CM chapters"):
            for section in chapter.findall("section"):
                section_id = section.get("id", "")
                section_desc = section.findtext("desc", "")
                node = Category(
                    id=f"{section_id}",
                    code=section_id,
                    desc=section_desc,
                    min=section_id.split("-")[0],
                    max=section_id.split("-")[-1],
                    assignable=False,
                )
                trie.insert(node, root_char=root_node_id)
                for diag in section.findall("diag"):
                    XMLTrie._parse_diag(trie, diag, parent_code=section_id)
        return trie

    @staticmethod
    def from_xml_file(file_path: str, coding_system: str, use_cache: bool = True) -> "XMLTrie":
        cache_key = f"xmltrie_{coding_system}_{Path(file_path).stem}"
        cache_path = XMLTrie._cache_dir / f"{cache_key}.pkl"
        if use_cache and cache_path.exists():
            # Create a cache key based on the file path and coding system
            cache_key = f"xmltrie_{coding_system}_{Path(file_path).stem}"
            cached_trie = XMLTrie.load_from_cache(cache_key)
            if cached_trie is not None:
                return cached_trie

        # If no cache exists or use_cache is False, create new trie
        root = ET.parse(file_path).getroot()
        trie = XMLTrie.from_xml(root, coding_system)

        # Cache the result if caching is enabled
        if use_cache:
            trie.save_to_cache(cache_key)

        return trie

    @staticmethod
    def _parse_diag(trie: "XMLTrie", diag: ET.Element, parent_code: Optional[str] = None) -> None:
        # Extract code and description from the current diag element
        code = diag.findtext("name")
        desc = diag.findtext("desc", default="")

        if parent_code is None:
            parent_code = code

        diags = diag.findall("diag")
        if diags:
            max_code = diags[-1].findtext("name")
            min_code = diags[0].findtext("name")
            node = Category(id=code, code=code, desc=desc, min=min_code, max=max_code)
        else:
            node = ICD(id=code, code=code, desc=desc)

        inclusion_notes = [
            inclusion_note
            for inclusion_term in diag.findall("inclusionTerm")
            if (inclusion_note := inclusion_term.findtext("note")) is not None
        ]
        node.inclusion_notes = inclusion_notes

        trie.insert(node, root_char=parent_code)

        # Recursively process nested diag elements
        for sub_diag in diag.findall("diag"):
            XMLTrie._parse_diag(trie, sub_diag, parent_code=code)

    @staticmethod
    def _parse_pcs_row(trie: "XMLTrie", pcs_row: ET.Element, fixed_axes: list, parent_code: str) -> None:
        """Parse a pcsRow element and generate composite codes."""
        variable_axes = []

        for axis in sorted(pcs_row.findall("axis"), key=lambda x: int(x.get("pos", 0))):
            axis_codes = [(label.attrib["code"], label.text, axis.findtext("title")) for label in axis.findall("label")]
            variable_axes.append(axis_codes)

        nodes = []
        for i, axis in enumerate(variable_axes):
            if i == 0:
                for code, label, title in axis:
                    # Combine with fixed axes code
                    complete_code = "".join([x.find("label").attrib["code"] for x in fixed_axes] + [code])
                    desc_begin = " | ".join(f"{x.find('title').text}: {x.find('label').text}" for x in fixed_axes)
                    desc_end = f"{title}: {label}"
                    desc = f"{desc_begin} | {desc_end}"
                    node = ICD(id=complete_code, code=complete_code, desc=desc)
                    trie.insert(node, root_char=parent_code)
                    nodes.append(node)
            else:
                new_nodes = []
                for node in nodes:
                    for code, label, title in axis:
                        complete_code = node.code + code
                        desc_begin = node.desc
                        desc_end = f"{title}: {label}"
                        desc = f"{desc_begin} | {desc_end}"
                        new_node = ICD(id=complete_code, code=complete_code, desc=desc)
                        trie.insert(new_node, root_char=parent_code)
                        new_nodes.append(new_node)
                nodes = new_nodes


def get_hard_negatives_for_code(code: str, trie: Trie, num: int | None = None, seed: int = 42) -> List[str]:
    """Gets hard negatives for a given code by going one level up in the trie."""
    seen_codes = set([code])

    def _extract_negatives_from_node(code: str) -> List[str]:
        node = trie.all[code]
        children_for_parent = []
        if node.parent_id:
            parent = trie.all[node.parent_id]
            children_for_parent = trie.get_all_children(parent.id)
        children_for_node = trie.get_all_children(node.id)
        hard_negatives = []

        # Collect hard negatives from children and parent
        for node in children_for_node + children_for_parent:
            if node.code in seen_codes:
                continue
            hard_negatives.append(node.code)
            seen_codes.add(node.code)
        return hard_negatives

    hard_negatives = _extract_negatives_from_node(code)
    # If no hard negatives, attempt using truncated code
    if not hard_negatives:
        truncated_code = code[:3]
        hard_negatives = _extract_negatives_from_node(truncated_code)

    shuffle(hard_negatives, random_state=seed)
    return hard_negatives if num is None else hard_negatives[:num]


def get_hard_negatives_for_list_of_codes(codes: list[str], trie: Trie, num: int) -> List[str]:
    hard_negatives = set()  # Use a set to store unique hard negatives
    for code in codes:
        hard_negatives.update(get_hard_negatives_for_code(code, trie, num=num))
    hard_negatives.difference_update(codes)
    return list(hard_negatives)  # Convert back to a list


def add_hard_negatives_to_set(
    data: pl.DataFrame, trie: Trie, source_col: str, dest_col: str, num: int | None = None
) -> pl.DataFrame:
    data = data.with_columns(
        pl.col(source_col)
        .map_elements(
            lambda codes: get_hard_negatives_for_list_of_codes(codes, trie, num=num), return_dtype=pl.List(pl.Utf8)
        )
        .alias(dest_col)
    )
    # Ensure negatives column is not null but an empty list
    data = data.with_columns(pl.col(dest_col).fill_null([]).alias(dest_col))
    return data


def get_random_negatives_for_codes(codes: list[str], trie: Trie, num: int, seed: int = 42) -> List[str]:
    """
    Get soft negatives for a given code. These are codes that are not parents, children,
    or the code itself.

    Args:
        code (str): The code to find soft negatives for.
        trie (Trie): The trie containing all the codes and their relationships.
        num (int | None): The number of soft negatives to return. If None, return all.

    Returns:
        List[str]: A list of soft negative codes.
    """
    # Get all nodes that are parents, children, or the node itself
    excluded_codes = set(codes)

    # Find all codes in the trie, excluding the ones from the excluded set
    soft_negatives = [n.code for n in trie.all.values() if n.code not in excluded_codes and isinstance(n, Node)]

    shuffle(soft_negatives, random_state=seed)
    return soft_negatives[:num]


if __name__ == "__main__":
    xml_trie = XMLTrie.from_xml_file(
        Path(__file__).parent.parent.parent / "data/medical-coding-systems/icd" / "icd10cm_tabular_2025.xml",
        coding_system="icd10cm",
    )
    print(f"Number of nodes: {len(xml_trie.all)}")
    print(f"Number of leaves: {len(xml_trie.get_leaves())}")

    test_code = "A00"
    text_for_code = xml_trie[test_code].desc
    print(f"Code {test_code} corresponds to: {text_for_code}")

    test_code = "D00"
    text_for_code = xml_trie[test_code].desc
    print(f"Code {test_code} corresponds to: {text_for_code}")

    xml_trie = XMLTrie.from_xml_file(
        Path(__file__).parent.parent.parent / "data/medical-coding-systems/icd" / "icd10pcs_tables_2025.xml",
        coding_system="icd10pcs",
    )
    print(f"Number of nodes: {len(xml_trie.all)}")
    print(f"Number of leaves: {len(xml_trie.get_leaves())}")

    test_code = "0016"
    text_for_code = xml_trie[test_code].desc
    print(f"Code {test_code} corresponds to: {text_for_code}")
