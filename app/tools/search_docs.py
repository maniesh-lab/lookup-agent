"""Tool 1: search the shop's policy documents."""

import json

from pydantic import BaseModel, Field

from app.core.config import settings
from app.retrieval.search import search

DESCRIPTION = (
    "Search Lumen Lane's policy documents (returns, refunds, shipping, premium membership, "
    "warranty, damaged items, order changes, payments, support). Use it for any question "
    "about rules, time frames, fees or what a policy covers. Returns the best-matching "
    "sections with their source file names."
)


class SearchDocsArgs(BaseModel):
    query: str = Field(description="What to look for, in plain words, e.g. 'return shipping cost wrong item'")


def search_docs(query: str, k: int | None = None) -> str:
    hits = search(query, mode=settings.search_mode, k=k or settings.search_top_k)
    if not hits:
        return json.dumps({"results": [], "note": "No matching policy sections found."})
    return json.dumps({"results": [h.as_dict() for h in hits]})
