from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/qdrant", tags=["Qdrant"])


class QdrantSearchRequest(BaseModel):
    query: str
    limit: int = 10


class QdrantSearchResult(BaseModel):
    id: str
    title: str
    code: str
    score: float


class QdrantSearchResponse(BaseModel):
    results: List[QdrantSearchResult]


@router.post("/search", response_model=QdrantSearchResponse)
def search_terms(request: QdrantSearchRequest):
    """Search for terms in the alphabetical index using Qdrant."""
    # Stub: Replace with real Qdrant search logic
    return QdrantSearchResponse(results=[])
