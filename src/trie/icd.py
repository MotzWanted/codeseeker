from functools import reduce
import itertools
from operator import mul
import re
import typing
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz


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
            files_to_parse = [
                file.name
                for file in self.files_map.model_dump(exclude_none=True).values()
            ]
            console.print(
                f"[bold blue] Following files will be parsed: {files_to_parse} [/bold blue]"
            )

    @classmethod
    def from_cms(
        cls, year: int, use_update: bool = False, *args, **kwargs
    ) -> "ICD10Trie":
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
    def from_dir(cls, path: Path, *args, **kwargs) -> "ICD10Trie":
        """Build the ICD trie from files that already exist locally."""
        if not path.exists():
            raise FileNotFoundError(f"Directory {path} does not exist.")
        return cls(path, *args, **kwargs)

    def parse(self) -> None:
        """Parse the downloaded ICD files."""
        self.parse_tabular_files()
        self.parse_alphabetic_indexes()
        self.parse_guidelines()

    @staticmethod
    def _page_text(page: "fitz.Page") -> str:  # noqa: D401
        """Return text from *page* using whichever API is available.

        Handles the *get_text* vs *getText* naming difference between PyMuPDF
        1.23+ and older versions.  The `# type: ignore[attr-defined]` suppresses
        Pylance complaints when it cannot see the attribute in the active stub.
        """
        if hasattr(page, "get_text"):
            return page.get_text()  # type: ignore[attr-defined]
        return page.getText()  # type: ignore[attr-defined]

    @staticmethod
    def flex_title_pat(title: str) -> str:
        """Build a whitespace-flexible regex for a heading."""
        parts = [re.escape(part) for part in re.split(r"\s+", title.strip()) if part]
        return r"\s+".join(parts)

    def parse_guidelines(self) -> None:
        doc = fitz.open(self.files_map.cm_guidelines)

        # 1) Fast path: many PDFs ship their own outline
        if hasattr(doc, "get_toc"):
            toc = doc.get_toc(simple=True)  # type: ignore[attr-defined]
        elif hasattr(doc, "getToC"):
            toc = doc.getToC(simple=True)  # type: ignore[attr-defined]
        else:
            toc = []
        chapters: list[dict[str, typing.Any]] = []
        sect_rx = re.compile(r"Section\s+([I]+)\b", re.I)  # NEW – Roman‑numeral capture
        chap_rx = re.compile(r"Chapter\s+(\d+)", re.I)
        for level, title, page in track(
            toc, total=doc.page_count, description="Parsing Guidelines"
        ):
            if chap_rx.search(title):
                num = chap_rx.search(title).group(1)  # type: ignore
                key = f"chapter_{num}"
            elif sect_rx.search(title):
                num = sect_rx.search(title).group(1)  # type: ignore
                key = f"section_{num}"
            else:
                continue  # skip non-chapter/section entries
            clean_title = (
                title.split(":", 1)[1].strip() if ":" in title else title.strip()
            )
            chapters.append(
                {"key": key, "num": num, "title": clean_title, "start_page": int(page)}
            )

        if not chapters:
            Console().print(
                "[yellow]No chapter entries found in outline – skipping.[/yellow]"
            )
            return

        chapters.sort(key=lambda c: c["start_page"])
        for idx, chap in enumerate(chapters):
            start_pg = chap["start_page"]
            end_pg = (
                chapters[idx + 1]["start_page"]
                if idx < len(chapters) - 1
                else doc.page_count
            )

            # gather the raw pages once
            pages_text = "\n".join(
                self._page_text(doc[p]) for p in range(start_pg - 1, end_pg)
            )

            # 1) find where the current chapter title begins
            if chap["key"].startswith("chapter_"):
                hdr_rx = rf"^.*?Chapter\s+{chap['num']}\b.*$"
            else:  # section_I, section_II, …
                hdr_rx = self.flex_title_pat(chap["title"])
            start_match = re.search(hdr_rx, pages_text, re.M | re.I)
            if not start_match:
                raise ValueError(
                    f"Could not find chapter title {chap['title']} in pages {start_pg}–{end_pg}."
                )
            char_start = start_match.start()

            # 2) find where the NEXT chapter title begins (if any)
            next_ch = chapters[idx + 1] if idx < len(chapters) - 1 else None
            if next_ch:
                if chap["key"].startswith("chapter_"):
                    hdr_rx = rf"^.*?Chapter\s+{next_ch['num']}\b.*$"
                else:  # section_I, section_II, …
                    hdr_rx = self.flex_title_pat(next_ch["title"])
                match = re.search(hdr_rx, pages_text, re.M | re.I)
                char_end = match.start() if match else len(pages_text)
            else:
                char_end = len(pages_text)

            content = pages_text[char_start:char_end].strip()

            if chap["key"] == "section_I":
                section_re = re.compile(r"^\s*[ABC]\.\s", re.M)
                section_matches = section_re.findall(content)
                if len(section_matches) == 3:
                    part_a = content[
                        content.find(section_matches[0]) : content.find(
                            section_matches[1]
                        )
                    ].strip()
                    part_b = content[
                        content.find(section_matches[1]) : content.find(
                            section_matches[2]
                        )
                    ].strip()

                    self.guidelines["IA"] = models.Guideline(
                        id="section_I_A",
                        number="IA",
                        title="A. Conventions for the ICD‑10‑CM",
                        content=part_a,
                    )
                    self.guidelines["IB"] = models.Guideline(
                        id="section_I_B",
                        number="IB",
                        title="B. General Coding Guidelines",
                        content=part_b,
                    )
                    continue

            self.guidelines[chap["num"]] = models.Guideline(
                id=chap["key"],
                number=chap["num"],
                title=chap["title"],
                content=content,
            )

    def parse_tabular_files(self) -> None:
        """Parse the tabular files."""
        pcs_tabular_root: ET.Element = ET.parse(self.files_map.pcs_tabular).getroot()  # type: ignore
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
            enumerate(terms, start=1),
            description="Parsing ICD-10 Alphabetic Indexes",
            total=len(terms),
        ):
            parent_id = id_format.format(idx)
            if isinstance(term.code, list):
                tmp_node = models.Term(
                    id=parent_id,
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
                    title=term.title,
                    code=term.code,
                    manifestation_code=term.manifestation_code,
                    see=term.see,
                    see_also=term.see_also,
                    parent_id="",
                )
                self.index[tmp_node.id] = tmp_node
            if term.sub_terms:
                self.insert_sub_terms_into_trie(term.sub_terms, parent_id)

    def insert_sub_terms_into_trie(
        self, sub_terms: list[models.CmIndexTerm], parent_id: str
    ) -> None:
        """Insert terms into the trie."""
        for idx, term in enumerate(sub_terms):
            current_id = f"{parent_id}.{idx}"
            if isinstance(term.code, list):
                tmp_node = models.Term(
                    id=current_id,
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
                    title=term.title,
                    code=term.code,
                    manifestation_code=term.manifestation_code,
                    see=term.see,
                    see_also=term.see_also,
                    parent_id=parent_id,
                )
                self.insert_to_index(tmp_node)
            if isinstance(term, models.CmIndexTerm) and term.sub_terms:
                self.insert_sub_terms_into_trie(term.sub_terms, current_id)

    def handle_cell_terms(self, term: models.Term, cells: list[models.CmCell]) -> None:
        """Handle multi-column CmCell list."""
        self.insert_to_index(term)
        cell_codes = [
            models.Term(
                id=term.id + "X" + str(cell.col),
                assignable=cell.assignable,
                title=f"{cell.heading}",
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
        self.tabular[root.id] = root  # type: ignore

        for table_index, table in track(
            enumerate(tables, start=1),
            description="Parsing ICD-10-PCS tables",
            total=len(tables),
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

    def _insert_pcs_axes(
        self, table: models.PcsTable, row: models.PcsRow, parent_id: str
    ):
        """Insert all valid PCS code combinations generated from table + row axes.

        `table_axes` contains fixed axis values (e.g., Section, Body System, Operation) with one label each.
        `row_axes` contains variable axis values with multiple label options.
        """
        base_code = "".join([axis.code for axis in table.table_axes])
        base_desc = ". ".join(
            f"{axis.title}: {axis.label}" for axis in table.table_axes
        ).strip()

        variable_axes = [
            [
                models.PcsCode(
                    id="",
                    name=base_code,
                    parent_id="",
                    description=base_desc,
                    assignable=False,
                )
            ]
        ]
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
            raise ValueError(
                f"Number of codes ({row.codes}) does not match the expected number ({len(size)})."
            )
        for combination in itertools.product(*variable_axes):
            combined_code = "".join([code.name for code in combination])
            combined_desc = ". ".join(
                [code.description for code in combination]
            ).strip()
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
        self.tabular[root.id] = root  # type: ignore
        self.lookup[root.name] = root.id

        for ch in track(chapters, description="Parsing ICD-10-CM chapters"):
            ch_node = models.CmCategory(
                **ch.model_dump(),
                id=f"{root.id}_{ch.chapter_id}",
                name=ch.chapter_id,
                parent_id="cm",
                description=ch.chapter_desc,
                min=ch.first,
                max=ch.last,
            )
            self.insert_to_tabular(ch_node)
            for sec in ch.sections:
                sec_node = models.CmCategory(
                    **sec.model_dump(),
                    id=f"{ch_node.id}_{sec.section_id}",
                    name=sec.section_id,
                    parent_id=ch_node.id,
                    min=sec.first,
                    max=sec.last,
                )
                self.insert_to_tabular(sec_node)
                for diag in sec.diags:
                    self._insert_cm_diag(diag, parent_id=sec_node.id)

    def _insert_cm_diag(
        self,
        diag: models.CmDiag,
        parent_id: str,
        seven_chr: list[models.SeventhCharacter] = [],
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

    xml_trie.group_by_category(["A00.1", "A01.00", "A17.82", "T63.123", "T63.42"])

    test_code = "0016070"
    text_for_code = xml_trie[test_code].description
    print(f"Code {test_code} corresponds to: {text_for_code}")

    test_term = "00001"
    term_codes = xml_trie.get_all_term_codes(test_term)
    print(f"Term {test_term} has a total of {len(term_codes)} codes.")
    sub_terms = xml_trie.get_all_term_children(test_term)
    print(f"Term {test_term} has a total of {len(sub_terms)} sub-terms.")

    queries = ["neoplasm", "hypertension", "croissant"]
    search_results = xml_trie.find_terms(queries, main_terms=False)
    print(f"Search results {search_results}:")
