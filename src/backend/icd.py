from fastapi import APIRouter
from typing import Optional

router = APIRouter(prefix="/icd", tags=["ICD-10"])


@router.get("/guidelines")
def get_guidelines(section: Optional[str] = None):
    """Return ICD-10 guidelines. Optionally filter by section."""
    return {"guidelines": [], "section": section}


@router.get("/alphabetical-index")
def get_alphabetical_index(term: Optional[str] = None):
    """Return the ICD-10 alphabetical index. Optionally filter by term."""
    return {"index": [], "term": term}


@router.get("/tabular-list")
def get_tabular_list(chapter: Optional[str] = None):
    """Return the ICD-10 tabular list. Optionally filter by chapter."""
    return {"tabular": [], "chapter": chapter}
