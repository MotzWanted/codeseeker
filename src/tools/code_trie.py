import csv
from collections import namedtuple
import typing
import xml.etree.ElementTree as ET
import pydantic
from pathlib import Path
from random import shuffle
from typing import Dict, List, Optional

import dill
import polars as pl
from pydantic_settings import SettingsConfigDict
from rich.progress import track

pydantic.Field
SeventhCharacter = namedtuple("SeventhCharacter", ["character", "name", "parent_name"])


class Root(pydantic.BaseModel):
    id: str
    name: str
    min: str
    max: str
    assignable: bool = False
    parent_id: str = ""
    children_ids: List[str] = pydantic.Field(default_factory=list)

    model_config = SettingsConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="forbid"
    )


class Category(Root):
    description: str
    chapter_id: str = ""
    parent_id: str
    children_ids: List[str] = pydantic.Field(default_factory=list)
    assignable: bool = False
    notes: Optional[List[str]] = pydantic.Field(default_factory=lambda: [])
    excludes1: Optional[list[str]] = pydantic.Field(default_factory=lambda: [])
    excludes2: Optional[list[str]] = pydantic.Field(default_factory=lambda: [])
    use_additional_code: Optional[list[str]] = pydantic.Field(
        default_factory=lambda: []
    )
    includes: Optional[list[str]] = pydantic.Field(default_factory=lambda: [])
    code_first: Optional[list[str]] = pydantic.Field(default_factory=lambda: [])
    code_also: Optional[list[str]] = pydantic.Field(default_factory=lambda: [])
    inclusion_term: Optional[list[str]] = pydantic.Field(default_factory=lambda: [])
    etiology: bool = False
    manifestation: bool = False

    def __repr__(self) -> str:
        return f"{self.id} {self.min}-{self.max} {self.description}"

    def within(self, code: str) -> bool:
        return self.min[: len(code)] <= code[: len(self.max)] <= self.max[: len(code)]


class Node(Category):
    assignable: bool = True
    min: str = ""
    max: str = ""

    def within(self, code: str) -> bool:
        raise NotImplementedError("Method not implemented")

    def full_desc(self, nodes: Dict[str, "Node"]) -> str:
        if self.parent_id:
            parent = nodes[self.parent_id]
            return f"{parent.full_desc(nodes)} {self.description}".strip()
        else:
            return self.description

    def list_desc(self, nodes: Dict[str, "Node"]) -> List[str]:
        if self.parent_id:
            parent = nodes[self.parent_id]
            return parent.list_desc(nodes) + [self.description]
        else:
            return []

    def is_leaf(self) -> bool:
        return not self.children_ids

    def __repr__(self) -> str:
        return f"{self.name} {self.description}"


class ICD(Node):
    def __repr__(self) -> str:
        return f"{self.id} {self.name} {self.description}"

    def within(self, code: str) -> bool:
        return code[: len(self.name)] == self.name[: len(code)]


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
    def load_from_cache(cls, cache_key: str) -> Optional["XMLTrie"]:
        """Load a trie instance from cache if it exists, or return None if loading fails."""
        cache_path = cls._cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return dill.load(f)
            except ModuleNotFoundError as e:
                # Handle the missing module by logging the error and proceeding with re-parsing
                print(
                    f"Error loading cache {cache_key}: {e}. Re-parsing and re-saving cache."
                )
                cache_path.unlink()
                return None  # Trigger re-parsing
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

    def get_chapter_name(self, code: str) -> str:
        parents = self.get_all_parents(code)
        return parents[-1].name

    def insert(self, node: Node, root_char: str) -> None:
        root_id = None
        for root in self.roots:
            if self.all[root].name == root_char:
                root_id = root
                break

        is_root = False
        if not root_id:
            root_id = f"{root_char}"
            root_node = Category(
                id=root_id, name=root_char, description="", min=root_char, max=root_char
            )
            self.roots.append(root_id)
            self.all[root_id] = root_node
            is_root = True

        root = self.all[root_id]

        if node.name and node.name[0] != root.name:
            node.name = root.name + node.name

        if is_root:
            self.all[node.id] = node
            self.lookup[node.name] = node.id
            return

        # self._insert(root, node)
        self.all[node.id] = node
        self.lookup[node.name] = node.id

    # def _insert(self, candidate: Node, node: Node):
    #     if candidate.children_ids:
    #         for c_id in candidate.children_ids:
    #             c = self.all[c_id]
    #             if c.within(node.name):
    #                 return self._insert(c, node)

    #     node.parent_id = candidate.id
    #     candidate.children_ids.append(node.id)

    def get_leaves(self) -> list[ICD]:
        return [
            n for n in self.all.values() if not n.children_ids and isinstance(n, ICD)
        ]

    def get_guidelines(self, codes: list[str]) -> dict[str, typing.Any]:
        guideline_fields = [
            "notes",
            "includes",
            "excludes1",
            "excludes2",
            "use_additional_code",
            "code_first",
            "code_also",
            "inclusion_term",
        ]
        guidelines = []
        included_parents = set()
        for c in codes:
            node = self[c]
            for parent in reversed([node] + self.get_all_parents(node.id)):
                if "icd10cm" in parent.name or parent.id in included_parents:
                    continue
                guideline_data = {
                    k: v
                    for k, v in parent.model_dump(include=guideline_fields).items()
                    if v
                }
                if not guideline_data:
                    continue
                guideline_data["code"] = parent.name
                guideline_data["assignable"] = parent.assignable
                guidelines.append(
                    guideline_data
                )  # Prepend guideline_data to guidelines
                included_parents.add(parent.id)

        return guidelines

    def __getitem__(self, code: str) -> Node:
        node_id = self.lookup[code]
        return self.all[node_id]

    @staticmethod
    def _create_node(code: str, desc: str) -> Node:
        if code:
            return ICD(id=code, name=code, description=desc.strip())
        else:
            if "Kap." in desc:
                desc = " ".join(desc.split(":")[1:])
            min, max = desc.split("[")[-1].replace("]", "").split("-")
            desc = desc.split("[")[0]
            return Category(
                id=min.strip() + "-" + max.strip(),
                max=max.strip(),
                min=min.strip(),
                description=desc.strip(),
                name=min.strip(),
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
            if self.all[root].name == root_char:
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
            self.lookup[node.name] = node.id
            return

        # self._insert(root, node)
        self.all[node.id] = node
        self.lookup[node.name] = node.id

    def pad_code(code: str) -> str:
        """Pads a code with 'X' until it reaches 6 characters."""
        while len(code) < 7:
            if len(code) == 3:
                code += "."
            code += "X"
        return code

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
        code_root = Root(
            id=f"{root_node_id}",
            name=f"{root_node_id}",
            min="1",
            max=f"{num_tables}",
            assignable=False,
        )
        trie.roots.append(code_root.id)
        trie.all[code_root.id] = code_root
        trie.lookup[code_root.name] = code_root.id
        pcs_tables = root.findall("pcsTable")
        for table_index, pcs_table in track(
            enumerate(pcs_tables),
            description="Parsing PCS tables",
            total=len(pcs_tables),
        ):
            table_id = f"{root_node_id}_Table_{table_index + 1}"
            num_rows = len(pcs_table.findall("pcsRow"))
            table_node = Category(
                id=table_id,
                parent_id=root_node_id,
                name=table_id,
                description=f"PCS Table {table_index + 1}",
                min="1",
                max=f"{num_rows}",
                assignable=False,
            )
            trie.insert(table_node, root_char=root_node_id)

            fixed_axes = []
            for axis in sorted(
                pcs_table.findall("axis"), key=lambda x: int(x.get("pos", 0))
            ):
                fixed_axes.append(axis)

            for pcs_row in pcs_table.findall("pcsRow"):
                row_id = f"{table_id}_Row_{pcs_row.get('codes')}"
                axis_pos = [axis.attrib["pos"] for axis in pcs_row.findall("axis")]
                row_node = Category(
                    id=row_id,
                    name=row_id,
                    parent_id=table_id,
                    description=f"PCS Table {table_index + 1} PCS Row {pcs_row.get('codes')}",
                    min=min(axis_pos),
                    max=max(axis_pos),
                    assignable=False,
                )
                trie.insert(row_node, root_char=table_id)
                XMLTrie._parse_pcs_row(trie, pcs_row, fixed_axes, parent_name=row_id)

        return trie

    @staticmethod
    def parse_tabular(trie: Trie, root: ET.Element, root_node_id: str) -> "XMLTrie":
        num_chapters = len(root.findall("chapter"))
        code_root = Root(
            id=f"{root_node_id}", name=f"{root_node_id}", min="1", max=f"{num_chapters}"
        )
        trie.roots.append(code_root.id)
        trie.all[code_root.id] = code_root
        trie.lookup[code_root.name] = code_root.id
        for chapter in track(
            root.findall("chapter"), description="Parsing ICD-10-CM chapters"
        ):
            chapter_id = chapter.findtext("name", "")
            section_index = chapter.findall(".//sectionIndex")[0]
            first, last = (
                section_index.findall("sectionRef")[0].get("first"),
                section_index.findall("sectionRef")[-1].get("last"),
            )
            trie._parse_cm_category(
                trie,
                chapter,
                name=chapter_id,
                first=first,
                last=last,
                parent_name=root_node_id,
            )
            for section in chapter.findall("section"):
                section_id = section.get("id", "")
                first = section_id.split("-")[0]
                last = section_id.split("-")[-1]
                if first == last:
                    section_id = f"{first}-{last}"
                trie._parse_cm_category(
                    trie,
                    section,
                    name=section_id,
                    first=first,
                    last=last,
                    parent_name=chapter_id,
                )
                # For each diagnosis within the section
                for diag in section.findall("diag"):
                    trie._parse_cm_element(
                        trie,
                        diag,
                        chapter_id=chapter_id,
                        parent_name=section_id,
                        seventh_characters=[],
                    )
        return trie

    @staticmethod
    def from_xml_file(
        file_path: str, coding_system: str, use_cache: bool = True
    ) -> "XMLTrie":
        cache_key = f"xmltrie_{coding_system}_{Path(file_path).stem}"

        # Load from cache if enabled and valid cache exists
        if use_cache:
            cached_trie = XMLTrie.load_from_cache(cache_key)
            if cached_trie is not None:
                return cached_trie

        # If no valid cache or use_cache is False, parse the XML again
        root = ET.parse(file_path).getroot()
        trie = XMLTrie.from_xml(root, coding_system)

        # Cache the trie after parsing
        if use_cache:
            trie.save_to_cache(cache_key)

        return trie

    @staticmethod
    def _parse_cm_category(
        trie: "XMLTrie",
        chapter: ET.Element,
        name: str,
        first: str,
        last: str,
        parent_name: str,
    ) -> None:
        desc = chapter.findtext("desc", default="")
        notes = [note.text for note in chapter.findall("notes/note")]
        includes = [note.text for note in chapter.findall("includes/note")]
        excludes1 = [excl.text for excl in chapter.findall("excludes1/note")]
        excludes2 = [excl.text for excl in chapter.findall("excludes2/note")]
        use_additional_code = [
            note.text for note in chapter.findall("useAdditionalCode/note")
        ]
        node = Category(
            id=name,
            name=name,
            parent_id=parent_name,
            description=desc,
            min=first,
            max=last,
            notes=notes,
            includes=includes,
            excludes1=excludes1,
            excludes2=excludes2,
            use_additional_code=use_additional_code,
        )
        trie.insert(node, root_char=parent_name)

    @staticmethod
    def _parse_cm_element(
        trie: "XMLTrie",
        element: ET.Element,
        chapter_id: str,
        parent_name: str,
        seventh_characters: list[SeventhCharacter],
    ) -> None:
        # Extract code and description from the current diag element
        name = element.findtext("name")
        desc = element.findtext("desc", default="")
        node_data = {
            "notes": [note.text for note in element.findall("notes/note")],
            "includes": [note.text for note in element.findall("includes/note")],
            "inclusion_term": [
                term.text for term in element.findall("inclusionTerm/note")
            ],
            "excludes1": [excl.text for excl in element.findall("excludes1/note")],
            "excludes2": [excl.text for excl in element.findall("excludes2/note")],
            "code_first": [note.text for note in element.findall("codeFirst/note")],
            "use_additional_code": [
                note.text for note in element.findall("useAdditionalCode/note")
            ],
            "code_also": [note.text for note in element.findall("codeAlso/note")],
            "manifestation": len(element.findall("codeFirst/note")) > 0,
            "etiology": len(element.findall("useAdditionalCode/note")) > 0,
        }
        diags = element.findall("diag")

        if not seventh_characters and element.find("sevenChrDef") is not None:
            for child in element.findall("sevenChrDef/extension"):
                seventh_characters.append(
                    SeventhCharacter(
                        character=child.attrib["char"],
                        name=child.text,
                        parent_name=name,
                    )
                )

        if diags:
            max_code = diags[-1].findtext("name")
            min_code = diags[0].findtext("name")
            node = Category(
                id=name,
                chapter_id=chapter_id,
                parent_id=parent_name,
                name=name,
                description=desc,
                min=min_code,
                max=max_code,
                **node_data,
            )
        else:
            node = ICD(
                id=name,
                chapter_id=chapter_id,
                parent_id=parent_name,
                name=name,
                description=desc,
                **node_data,
            )

        trie.insert(node, root_char=parent_name)

        if (
            seventh_characters
            and not diags
            and seventh_characters[0].parent_name in name
        ):
            for sc in seventh_characters:
                padded_code = XMLTrie.pad_code(name)
                sc_name = padded_code + sc.character
                sc_desc = desc + " " + f"({sc.name})"
                sc_node = ICD(
                    id=sc_name,
                    chapter_id=chapter_id,
                    parent_id=name,
                    name=sc_name,
                    description=sc_desc,
                    **node_data,
                )
                trie.insert(sc_node, root_char=name)
            seventh_characters = []

        # Recursively process nested diag elements
        for sub_diag in element.findall("diag"):
            XMLTrie._parse_cm_element(
                trie,
                sub_diag,
                chapter_id=chapter_id,
                parent_name=name,
                seventh_characters=seventh_characters,
            )

    @staticmethod
    def _parse_pcs_row(
        trie: "XMLTrie", pcs_row: ET.Element, fixed_axes: list, parent_name: str
    ) -> None:
        """Parse a pcsRow element and generate composite codes."""
        variable_axes = []

        for axis in sorted(pcs_row.findall("axis"), key=lambda x: int(x.get("pos", 0))):
            axis_codes = [
                (label.attrib["code"], label.text, axis.findtext("title"))
                for label in axis.findall("label")
            ]
            variable_axes.append(axis_codes)

        nodes = []
        for i, axis in enumerate(variable_axes):
            if i == 0:
                for code, label, title in axis:
                    # Combine with fixed axes code
                    complete_code = "".join(
                        [x.find("label").attrib["code"] for x in fixed_axes] + [code]
                    )
                    desc_begin = " | ".join(
                        f"{x.find('title').text}: {x.find('label').text}"
                        for x in fixed_axes
                    )
                    desc_end = f"{title}: {label}"
                    desc = f"{desc_begin} | {desc_end}"
                    node = ICD(
                        id=complete_code,
                        name=complete_code,
                        parent_id=parent_name,
                        description=desc,
                    )
                    trie.insert(node, root_char=parent_name)
                    nodes.append(node)
            else:
                new_nodes = []
                for node in nodes:
                    for code, label, title in axis:
                        complete_code = node.name + code
                        desc_begin = node.description
                        desc_end = f"{title}: {label}"
                        desc = f"{desc_begin} | {desc_end}"
                        new_node = ICD(
                            id=complete_code,
                            name=complete_code,
                            parent_id=parent_name,
                            description=desc,
                        )
                        trie.insert(new_node, root_char=parent_name)
                        new_nodes.append(new_node)
                nodes = new_nodes


def get_hard_negatives_for_code(
    code: str, trie: Trie, num: int | None = None, seed: int = 42
) -> List[str]:
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
            if node.name in seen_codes:
                continue
            hard_negatives.append(node.name)
            seen_codes.add(node.name)
        return hard_negatives

    hard_negatives = _extract_negatives_from_node(code)
    # If no hard negatives, attempt using truncated code
    if not hard_negatives:
        truncated_code = code[:3]
        hard_negatives = _extract_negatives_from_node(truncated_code)

    shuffle(hard_negatives, random_state=seed)
    return hard_negatives if num is None else hard_negatives[:num]


def get_hard_negatives_for_list_of_codes(
    codes: list[str], trie: Trie, num: int
) -> List[str]:
    hard_negatives = set()  # Use a set to store unique hard negatives
    for code in codes:
        hard_negatives.update(get_hard_negatives_for_code(code, trie, num=num))
    hard_negatives.difference_update(codes)
    return list(hard_negatives)  # Convert back to a list


def add_hard_negatives_to_set(
    data: pl.DataFrame,
    trie: Trie,
    source_col: str,
    dest_col: str,
    num: int | None = None,
) -> pl.DataFrame:
    data = data.with_columns(
        pl.col(source_col)
        .map_elements(
            lambda codes: get_hard_negatives_for_list_of_codes(codes, trie, num=num),
            return_dtype=pl.List(pl.Utf8),
        )
        .alias(dest_col)
    )
    # Ensure negatives column is not null but an empty list
    data = data.with_columns(pl.col(dest_col).fill_null([]).alias(dest_col))
    return data


def _split_into_pcs_and_cm(
    cm_trie: XMLTrie, pcs_trie: XMLTrie, codes: list[str]
) -> tuple[list[str], list[str]]:
    """Split the codes into PCS and CM codes."""
    pcs_codes = []
    cm_codes = []
    for code in codes:
        if code in cm_trie.lookup:
            cm_codes.append(code)
        elif code in pcs_trie.lookup:
            pcs_codes.append(code)
        elif code[:6] in pcs_trie.lookup:
            pcs_codes.append(code[:6])
        elif code[:5] in cm_trie.lookup:
            cm_codes.append(code[:5])
        elif code[:4] in pcs_trie.lookup:
            pcs_codes.append(code[:4])
        elif code[:3] in cm_trie.lookup:
            cm_codes.append(code[:3])
        else:
            raise ValueError(f"Code {code} is not in the CM or PCS trie")
    return pcs_codes, cm_codes


def get_code_objects(
    cm_trie: XMLTrie, pcs_trie: XMLTrie, codes: list[str]
) -> list[Node]:
    """Get the code descriptions."""
    pcs_codes, cm_codes = _split_into_pcs_and_cm(cm_trie, pcs_trie, codes)
    nodes = []
    for code in pcs_codes:
        nodes.append(pcs_trie[code])
    for code in cm_codes:
        nodes.append(cm_trie[code])
    return sorted(nodes, key=lambda code: code.name)


def get_code_guidelines(
    cm_trie: XMLTrie, pcs_trie: XMLTrie, codes: list[str]
) -> list[tuple[str, dict[str, str]]]:
    """Get the code descriptions."""
    pcs_codes, cm_codes = _split_into_pcs_and_cm(cm_trie, pcs_trie, codes)
    guidelines = []
    guidelines.extend(pcs_trie.get_guidelines(pcs_codes))
    guidelines.extend(cm_trie.get_guidelines(cm_codes))
    # merged_guidelines = []
    # for code, guideline_data in guidelines:
    #     merged_guideline = defaultdict(list)
    #     for guideline in guideline_data:
    #         for k, v in guideline.items():
    #             merged_guideline[k].extend(v)
    #     merged_guidelines.append((code, merged_guideline))

    return guidelines


def get_random_negatives_for_codes(
    codes: list[str], trie: Trie, num: int, seed: int = 42
) -> List[str]:
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
    soft_negatives = [
        n.name
        for n in trie.all.values()
        if n.name not in excluded_codes and isinstance(n, Node)
    ]

    shuffle(soft_negatives, random_state=seed)
    return soft_negatives[:num]


if __name__ == "__main__":
    xml_trie = XMLTrie.from_xml_file(
        Path(__file__).parent.parent.parent
        / "data/medical-coding-systems/icd"
        / "icd10cm_tabular_2025.xml",
        coding_system="icd10cm",
        use_cache=False,
    )
    print(f"Number of nodes: {len(xml_trie.all)}")
    print(f"Number of leaves: {len(xml_trie.get_leaves())}")

    test_code = "Z66"
    text_for_code = xml_trie[test_code].description
    print(f"Code {test_code} corresponds to: {text_for_code}")
    # guidelines = xml_trie.get_guidelines(test_code)
    # print(f"Guidelines for code {test_code}: {guidelines}")

    test_code = "D00"
    text_for_code = xml_trie[test_code].description
    print(f"Code {test_code} corresponds to: {text_for_code}")

    xml_trie = XMLTrie.from_xml_file(
        Path(__file__).parent.parent.parent
        / "data/medical-coding-systems/icd"
        / "icd10pcs_tables_2025.xml",
        coding_system="icd10pcs",
        use_cache=False,
    )
    print(f"Number of nodes: {len(xml_trie.all)}")
    print(f"Number of leaves: {len(xml_trie.get_leaves())}")

    test_code = "0016"
    text_for_code = xml_trie[test_code].description
    print(f"Code {test_code} corresponds to: {text_for_code}")
