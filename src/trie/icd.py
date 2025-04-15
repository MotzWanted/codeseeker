from functools import reduce
import itertools
from operator import mul
import xml.etree.ElementTree as ET
from pathlib import Path


from trie import models, xml_utils
from trie.base import TRIE_CACHE_DIR, Trie
from trie.connectors.cms import download_cms_icd_version

from rich.progress import track
from rich.console import Console


class ICD10Trie(Trie):
    """ICD10Trie is a trie data structure for storing and retrieving ICD-10 codes."""

    CACHE_DIR = TRIE_CACHE_DIR / "icd"

    def __init__(self, path_to_files: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.files_map = models.ICDFileMap.from_directory(path_to_files)
        with Console() as console:
            files_to_parse = [file.name for file in self.files_map.model_dump(exclude_none=True).values()]
            console.print(f"[bold blue] Following files will be parsed: {files_to_parse} [/bold blue]")

    @classmethod
    def from_cms(cls, year: int, use_update: bool = False, *args, **kwargs) -> "ICD10Trie":
        """Download the ICD files from CMS for the specified year
        (preferring updated 'month-tagged' versions if use_update is True),
        then build and return the trie.
        """
        download_path = cls.CACHE_DIR / f"icd_{year}"
        download_path.mkdir(parents=True, exist_ok=True)

        # Download needed files if not present:
        download_cms_icd_version(download_path, year, use_update=use_update)

        return cls(download_path, *args, **kwargs)

    @classmethod
    def from_directory(cls, local_directory: Path, *args, **kwargs) -> "ICD10Trie":
        """Build the ICD trie from files that already exist locally."""
        if not local_directory.exists():
            raise FileNotFoundError(f"Directory {local_directory} does not exist.")
        return cls(local_directory, *args, **kwargs)

    def parse(self) -> "ICD10Trie":
        """Parse the downloaded ICD files."""
        self.parse_tabular_files()

        self.parse_alphabetic_indexes()

        # TODO: parse pdf guidelines
        # self.parse_guidelines(expected_files)

    def parse_tabular_files(self) -> None:
        """Parse the tabular files."""
        pcs_tabular_root: ET.Element = ET.parse(self.files_map.pcs_tabular).getroot()
        cm_tabular_root: ET.Element = ET.parse(self.files_map.cm_tabular).getroot()

        pcs_data: list[models.PcsTable] = xml_utils.parse_pcs_tables(pcs_tabular_root)
        cm_data: list[models.CmChapter] = xml_utils.parse_cm_table(cm_tabular_root)

        self.insert_pcs_tables_into_trie(pcs_data)
        self.insert_cm_chapters_into_trie(cm_data)

    def parse_alphabetic_indexes(self) -> None:
        """Parse the alphabetical indexes."""
        # TODO: figure out what to do with cross-references in indexes (e.g., "see also")
        root_indexes = []
        for index, file in self.files_map.model_dump(exclude_none=True).items():
            if index not in self.files_map.ALPHABETIC_INDEXES:
                continue
            root_indexes.append(ET.parse(file).getroot())

        index_terms = []
        for root in root_indexes:
            index_terms.extend(xml_utils.parse_icd10cm_index(root))

        # build alphabetic index
        self.insert_terms_into_trie(index_terms)

    def insert_terms_into_trie(self, terms: list[models.CmIndexTerm]) -> None:
        """Insert terms into the trie."""
        num_digits = len(str(len(terms)))
        id_format = f"{{:0{num_digits}d}}"
        for idx, term in track(
            enumerate(terms, start=1), description="Parsing ICD-10 Alphabetic Indexes", total=len(terms)
        ):
            parent_id = id_format.format(idx)
            if isinstance(term.code, list):
                tmp_node = models.Term(
                    id=parent_id,
                    assignable=term.assignable,
                    title=term.title,
                    code=None,
                    see=term.see,
                    see_also=term.see_also,
                    parent_id="",
                )
                self.handle_cell_terms(tmp_node, term.code)
            else:
                tmp_node = models.Term(
                    id=parent_id,
                    assignable=term.assignable,
                    title=term.title,
                    code=term.code,
                    see=term.see,
                    see_also=term.see_also,
                    parent_id="",
                )
                self.index[tmp_node.id] = tmp_node
            if term.sub_terms:
                self.insert_sub_terms_into_trie(term.sub_terms, parent_id)

    def insert_sub_terms_into_trie(self, sub_terms: list[models.CmIndexTerm | models.Term], parent_id: str) -> None:
        """Insert terms into the trie."""
        for idx, term in enumerate(sub_terms):
            current_id = f"{parent_id}.{idx}"
            if isinstance(term.code, list):
                tmp_node = models.Term(
                    id=current_id,
                    assignable=term.assignable,
                    title=term.title,
                    code=None,
                    see=term.see,
                    see_also=term.see_also,
                    parent_id=parent_id,
                )
                self.handle_cell_terms(tmp_node, term.code)
            else:
                tmp_node = models.Term(
                    id=current_id,
                    assignable=term.assignable,
                    title=term.title,
                    code=term.code,
                    see=term.see,
                    see_also=term.see_also,
                    parent_id=parent_id,
                )
                self.insert_to_index(tmp_node)
            if isinstance(term, models.CmIndexTerm) and term.sub_terms:
                self.insert_sub_terms_into_trie(term.sub_terms, current_id)

    def handle_cell_terms(self, term: models.Term, cells: list[models.CmCell]) -> list[models.CmCell]:
        """Handle multi-column CmCell list."""
        self.insert_to_index(term)
        cell_codes = [
            models.Term(
                id=term.id + "X" + str(cell.col),
                assignable=cell.assignable,
                title=f"{term.title} ({cell.heading})",
                code=cell.code,
                parent_id=term.id,
            )
            for cell in cells
        ]
        for cell in cell_codes:
            self.insert_to_index(cell)

    @staticmethod
    def pad_cm_code(code: str) -> str:
        """Pads a code with 'X' until it reaches 6 characters."""
        while len(code) < 7:
            if len(code) == 3:
                code += "."
            code += "X"
        return code

    def insert_pcs_tables_into_trie(self, tables: list[models.PcsTable]) -> None:
        """ "Insert PCS tables into the trie."""
        root = models.Root(id="pcs", name="pcs", min="1", max=str(len(tables)))
        self.roots.append(root.id)
        self.tabular[root.id] = root

        for table_index, table in track(
            enumerate(tables, start=1), description="Parsing ICD-10-PCS tables", total=len(tables)
        ):
            table_node = models.Category(
                id=table.table_id,
                parent_id=root.id,
                name=table.table_id,
                description=f"PCS Table {table_index}",
                min="1",
                max=str(len(table.rows)),
            )
            self.insert_to_tabular(table_node)

            for row_idx, pcs_row in enumerate(table.rows, start=1):
                row_node = models.Category(
                    id=f"{table_node.id}_{row_idx}",
                    parent_id=table_node.id,
                    name=f"PCS Row {row_idx}",
                    description=f"PCS Table {table_index} PCS Row {pcs_row.codes}",
                    min="1",
                    max=str(pcs_row.codes),
                )
                self.insert_to_tabular(row_node)

                # Combine axis labels to form codes
                self._insert_pcs_axes(table, pcs_row, parent_id=row_node.id)

    def _insert_pcs_axes(self, table: list[models.PcsTable], row: models.PcsRow, parent_id: str):
        """Insert all valid PCS code combinations generated from table + row axes.

        `table_axes` contains fixed axis values (e.g., Section, Body System, Operation) with one label each.
        `row_axes` contains variable axis values with multiple label options.
        """
        base_code = "".join([axis.code for axis in table.table_axes])
        base_desc = ". ".join(f"{axis.title}: {axis.label}" for axis in table.table_axes).strip()

        variable_axes = [[models.PcsCode(id="", name=base_code, parent_id="", description=base_desc, assignable=False)]]
        for axis in row.axes:
            variable_axes.append(
                [
                    models.PcsCode(
                        id="",
                        name=label.code,
                        description=f"{axis.title}: {label.label}",
                        parent_id="",
                        assignable=False,
                    )
                    for label in axis.labels
                ]
            )

        # Generate all combinations of codes (cartesian product)
        size = reduce(mul, (len(sublist) for sublist in variable_axes), 1)
        if row.codes != size:
            raise ValueError(f"Number of codes ({row.codes}) does not match the expected number ({len(size)}).")
        for combination in itertools.product(*variable_axes):
            combined_code = "".join([code.name for code in combination])
            combined_desc = ". ".join([code.description for code in combination]).strip()
            new_code = models.PcsCode(
                id=combined_code,
                name=combined_code,
                parent_id=parent_id,
                description=combined_desc,
                assignable=True,
            )
            self.insert_to_tabular(new_code)

    def insert_cm_chapters_into_trie(self, chapters: list[models.CmChapter]) -> None:
        """Insert CM chapters into the trie."""
        root = models.Root(id="cm", name="cm", min="1", max=str(len(chapters)))
        self.roots.append(root.id)
        self.tabular[root.id] = root
        self.lookup[root.name] = root.id

        for ch in track(chapters, description="Parsing ICD-10-CM chapters"):
            ch_node = models.Category(
                id=f"{root.id}_{ch.chapter_id}",
                name=ch.chapter_id,
                parent_id="cm",
                description=ch.chapter_desc,
                min=ch.first,
                max=ch.last,
            )
            self.insert_to_tabular(ch_node)
            for sec in ch.sections:
                sec_node = models.Category(
                    id=f"{ch_node.id}_{sec.section_id}",
                    name=sec.section_id,
                    parent_id=ch_node.id,
                    description=sec.description,
                    min=sec.first,
                    max=sec.last,
                )
                self.insert_to_tabular(sec_node)
                for diag in sec.diags:
                    self._insert_cm_diag(diag, parent_id=sec_node.id)

    def _insert_cm_diag(
        self, diag: models.CmDiag, parent_id: str, seven_chr: list[models.SeventhCharacter] = []
    ) -> None:
        """Insert a CM diagnosis into the trie."""
        if diag.children:
            node = models.CmCode(
                **diag.model_dump(),
                id=diag.name,
                parent_id=parent_id,
                description=diag.desc,
                min=diag.children[0].name,
                max=diag.children[-1].name,
                assignable=False,
            )
        else:
            node = models.CmCode(
                **diag.model_dump(),
                id=diag.name,
                parent_id=parent_id,
                description=diag.desc,
                assignable=False if seven_chr else True,
            )
        self.insert_to_tabular(node)

        if diag.seventh_characters:
            seven_chr = diag.seventh_characters

        # `seven_chr[0].parent_name in diag.name` handles a few weird edge cases
        if seven_chr and not diag.children and seven_chr[0].parent_name in diag.name:
            for sc in seven_chr:
                padded_code = self.pad_cm_code(code=diag.name)
                sc_name = padded_code + sc.character
                sc_desc = diag.desc + " " + f"({sc.name})"
                sc_node = models.CmCode(
                    **diag.model_dump(exclude={"name"}),
                    id=sc_name,
                    name=sc_name,
                    parent_id=diag.name,
                    description=sc_desc,
                    assignable=True,
                )
                self.insert_to_tabular(sc_node)
            seven_chr = []

        # Recursively insert children
        for child_diag in diag.children:
            self._insert_cm_diag(child_diag, parent_id=diag.name, seven_chr=seven_chr)


if __name__ == "__main__":
    xml_trie = ICD10Trie.from_cms(year=2022, use_update=False)
    xml_trie.parse()
    print(f"Number of nodes: {len(xml_trie.tabular)}")
    print(f"Number of leaves: {len(xml_trie.get_leaves())}")
    print(f"Number of index terms: {len(xml_trie.index)}")
    print(f"Number of roots: {len(xml_trie.roots)}")
    print(f"Number of PCS codes: {len(xml_trie.get_root_leaves('pcs'))}")
    print(f"Number of CM codes: {len(xml_trie.get_root_leaves('cm'))}")

    test_code = "Z66"
    text_for_code = xml_trie[test_code].description
    print(f"Code {test_code} corresponds to: {text_for_code}")

    test_code = "0016070"
    text_for_code = xml_trie[test_code].description
    print(f"Code {test_code} corresponds to: {text_for_code}")

    test_term = "00001"
    term_codes = xml_trie.get_all_term_codes(test_term)
    print(f"Term {test_term} has a total of {len(term_codes)} codes.")
