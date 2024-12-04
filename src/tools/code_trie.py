import csv
from dataclasses import dataclass

from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
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
                id=min.strip()+"-"+max.strip(),
                max=max.strip(), min=min.strip(), desc=desc.strip(), code=min.strip()
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
    def from_xml(root: ET.Element, coding_system) -> "XMLTrie":
        trie = XMLTrie()
        code_root_id = f"{coding_system}"
        if "icd10cm" in coding_system:
            trie = XMLTrie.parse_tabular(trie, root, code_root_id)
        elif "icd10pcs" in coding_system:
            trie = XMLTrie.parse_table(trie, root, code_root_id)
        return trie
    
    @staticmethod
    def parse_table(trie: Trie, root: ET.Element, root_node_id: str) -> "XMLTrie":
        """Parse PCS tables and insert them into the trie."""
        num_tables = len(root.findall("pcsTable"))
        code_root = Category(id=f"{root_node_id}", code=f"{root_node_id}", desc="", min="1", max=f"{num_tables}", assignable=False)
        trie.roots.append(code_root.id)
        trie.all[code_root.id] = code_root
        trie.lookup[code_root.code] = code_root.id
        for table_index, pcs_table in track(enumerate(root.findall("pcsTable")), description="Parsing PCS tables"):
            table_id = f"{root_node_id}_Table_{table_index + 1}"
            num_rows = len(pcs_table.findall("pcsRow"))
            table_node = Category(
                id=table_id, code=table_id, desc=f"PCS Table {table_index + 1}", min="1", max=f"{num_rows}", assignable=False
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
        code_root = Category(id=f"{root_node_id}", code=f"{root_node_id}", desc="", min="1", max=f"{num_chapters}", assignable=False)
        trie.roots.append(code_root.id)
        trie.all[code_root.id] = code_root
        trie.lookup[code_root.code] = code_root.id
        for chapter in track(root.findall("chapter"), description="Parsing ICD-10-CM chapters"):
            for section in chapter.findall("section"):
                section_id = section.get("id", "")
                section_desc = section.findtext("desc", "")
                node = Category(id=f"{section_id}", code=section_id, desc=section_desc, min=section_id.split("-")[0], max=section_id.split("-")[-1], assignable=False)
                trie.insert(node, root_char=root_node_id)
                for diag in section.findall("diag"):
                    XMLTrie._parse_diag(trie, diag, parent_code=section_id)
        return trie
            
    
    @staticmethod
    def from_xml_file(file_path: str, coding_system: str) -> "XMLTrie":
        root = ET.parse(file_path).getroot()
        return XMLTrie.from_xml(root, coding_system)
    
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
            axis_codes = [(label.attrib["code"], label.text, axis.find("title").text) for label in axis.findall("label")]
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


if __name__ == "__main__":
    xml_trie = XMLTrie.from_xml_file(
        "/nfs/nas/mlrd/datasets/medical_coding_systems/icd10cm_2023/icd10cm_tabular_2023.xml",
        coding_system=f"icd10cm",
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
        f"/nfs/nas/mlrd/datasets/medical_coding_systems/icd10pcs_2023/icd10pcs_tables_2023.xml",
        coding_system=f"icd10pcs",
    )
    print(f"Number of nodes: {len(xml_trie.all)}")
    print(f"Number of leaves: {len(xml_trie.get_leaves())}")

    test_code = "0016"
    text_for_code = xml_trie[test_code].desc
    print(f"Code {test_code} corresponds to: {text_for_code}")
