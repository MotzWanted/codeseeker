from trie import models
import xml.etree.ElementTree as ET


def parse_cm_table(root: ET.Element) -> list[models.CmChapter]:
    chapters_data = []
    for ch_elem in root.findall("chapter"):
        ch_name = ch_elem.findtext("name", "")
        ch_desc = ch_elem.findtext("desc", "")
        sec_index = ch_elem.findall(".//sectionIndex")[0]
        first_in_ch = sec_index.findall("sectionRef")[0].get("first")
        last_ind_ch = sec_index.findall("sectionRef")[-1].get("last")
        sections_data = []
        for sec in ch_elem.findall("section"):
            sec_id = sec.get("id", "")
            desc = sec.findtext("desc", "")
            first = sec_id.split("-")[0]  # sec_id might not contain "-"
            last = sec_id.split("-")[-1]
            diag_data = _parse_diag_elements(sec.findall("diag"))
            sections_data.append(
                models.CmSection(section_id=sec_id, description=desc, diags=diag_data, first=first, last=last)
            )

        chapters_data.append(
            models.CmChapter(
                chapter_id=ch_name, chapter_desc=ch_desc, sections=sections_data, first=first_in_ch, last=last_ind_ch
            )
        )
    return chapters_data


def parse_pcs_tables(root: ET.Element) -> list[models.PcsTable]:
    pcs_tables_data = []
    pcs_tables = root.findall("pcsTable")

    for i, pcs_table in enumerate(pcs_tables, start=1):
        table_axes = []
        for axis in pcs_table.findall("axis"):
            pos = int(axis.get("pos"))
            title = axis.findtext("title")
            label = axis.find("label")
            table_axes.append(models.PcsTableAxis(pos=pos, code=label.get("code"), label=label.text, title=title))
        rows_data = []
        for pcs_row in pcs_table.findall("pcsRow"):
            row_codes = int(pcs_row.get("codes"))
            row_axes = []
            for axis in pcs_row.findall("axis"):
                axis_codes = int(axis.get("values"))
                pos = int(axis.get("pos"))
                title = axis.findtext("title")
                labels = [
                    models.PcsAxisLabel(code=label.get("code"), label=label.text or "", title=title)
                    for label in axis.findall("label")
                ]
                row_axes.append(models.PcsAxis(codes=axis_codes, pos=pos, title=title, labels=labels))

            rows_data.append(models.PcsRow(codes=row_codes, axes=row_axes))

        table_model = models.PcsTable(table_id=f"pcs_table_{i}", table_axes=table_axes, rows=rows_data)
        pcs_tables_data.append(table_model)

    return pcs_tables_data


def _parse_diag_elements(diag_elems: list[ET.Element]) -> list[models.CmDiag]:
    diags = []
    for diag in diag_elems:
        sub_diags = diag.findall("diag")
        diag_model = models.CmDiag(
            name=diag.findtext("name"),
            desc=diag.findtext("desc"),
            notes=[n.text for n in diag.findall("notes/note")],
            includes=[n.text for n in diag.findall("includes/note")],
            inclusion_term=[n.text for n in diag.findall("inclusionTerm/note")],
            excludes1=[n.text for n in diag.findall("excludes1/note")],
            excludes2=[n.text for n in diag.findall("excludes2/note")],
            use_additional_code=[n.text for n in diag.findall("useAdditionalCode/note")],
            code_first=[n.text for n in diag.findall("codeFirst/note")],
            code_also=[n.text for n in diag.findall("codeAlso/note")],
            seventh_characters=[
                models.SeventhCharacter(
                    character=child.attrib["char"], name=child.text, parent_name=diag.findtext("name")
                )
                for child in diag.findall("sevenChrDef/extension")
            ],
            children=_parse_diag_elements(sub_diags) if sub_diags else [],
        )
        diags.append(diag_model)
    return diags


def parse_index_heading(root: ET.Element) -> dict[int, str]:
    """
    Parses the <indexHeading> block, if present, into a col → heading label map.
    """
    heading_elem = root.find("indexHeading")
    if heading_elem is None:
        return {}

    return {int(head.attrib["col"]): head.text.strip() for head in heading_elem.findall("head") if head.text}


def parse_index_cells(element: ET.Element, index_heading: dict[int, str]) -> list[models.CmCell]:
    """Parse the <indexCells> block, if present, into a list of CmCell models."""
    if not index_heading:
        return []
    cells = []
    for cell in element.findall("cell"):
        col = int(cell.attrib["col"])
        code_text = cell.text.strip() if cell.text else ""
        if code_text and code_text != "-":
            cells.append(models.CmCell(col=col, heading=index_heading.get(col, f"Col {col}"), code=code_text))
    return cells


def parse_icd10_index_term(element: ET.Element, index_heading: dict[int, str]) -> models.CmIndexTerm:
    """
    Parse a <mainTerm> or <term> element into a CmIndexTerm model.
    Handles both standard <code> and multi-column <cell col="X"> format.
    """

    # Extract full title (including <nemod>)
    title_elem = element.find("title")
    title = " ".join(title_elem.itertext()).strip() if title_elem is not None else ""

    # Extract <cell> codes if present
    cells = parse_index_cells(element, index_heading)
    # Fallback to <code> tag if no usable <cell> tags found
    code: str | list[models.CmCell] | None = cells if cells else (element.findtext("code") or None)

    see = element.findtext("see")
    see_also = element.findtext("seeAlso")

    # Recursively parse sub-terms
    sub_terms = [parse_icd10_index_term(sub_term, index_heading) for sub_term in element.findall("term")]

    return models.CmIndexTerm(title=title, code=code, see=see, see_also=see_also, sub_terms=sub_terms)


# def parse_icd10_index_term(element: ET.Element) -> models.CmIndexTerm:
#     """
#     Parse a <mainTerm> or <term> element into an ICD10CMIndexTerm model, recursively handling sub-<term>.
#     """
#     title_elem = element.find("title")
#     title = " ".join(title_elem.itertext()).strip()

#     # Pull code/see/seeAlso
#     code = element.findtext("code")
#     see = element.findtext("see") or element.findtext("seeAlso")
#     see_also = element.findtext("seeAlso")

#     # Parse nested <term> children
#     sub_terms = []
#     for sub_term in element.findall("term"):
#         sub_terms.append(parse_icd10_index_term(sub_term))

#     # Create the model instance
#     return models.CmIndexTerm(
#         title=title,
#         code=code if code else None,
#         see=see if see else None,
#         seeAlso=see_also if see_also else None,
#         sub_terms=sub_terms,
#     )


def parse_icd10cm_index(root: ET.Element) -> list[models.CmIndexTerm]:
    """
    Parse the full ICD-10-CM index file into a list of CmLetter entries.
    Supports both standard indexes and multi-column ones (like Neoplasm).
    """
    index_heading = parse_index_heading(root)
    index_heading = index_heading if index_heading else None

    letters_data = []
    for letter_elem in root.findall("letter"):
        letters_data.extend([parse_icd10_index_term(mt, index_heading) for mt in letter_elem.findall("mainTerm")])
    return letters_data
