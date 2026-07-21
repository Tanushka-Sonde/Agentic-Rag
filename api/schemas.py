"""
api/schemas.py   [CHANGE 2 + CHANGE 4 — modified]
───────────────
Change 2: ChatResponse gains agent_name, code, image_b64, chart_type.
Change 4: New StopResponse schema.
All existing schemas unchanged.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question:   str            = Field(..., min_length=1, max_length=2000)
    session_id: str | None     = Field(None)
    filters:    dict[str, Any] | None = Field(None)


class IngestRequest(BaseModel):
    file_path: str
    metadata:  dict[str, Any] | None = Field(None)


class Citation(BaseModel):
    label:         str
    source_file:   str
    page_label:    str
    page:          Any
    section:       str
    engagement_id: str
    doc_type:      str
    kind:          str = "text"


class ChatResponse(BaseModel):
    session_id:       str
    answer:           str
    citations:        list[Citation]
    tables:           list[str]
    queries_used:     list[str]
    chunks_retrieved: int
    latency_ms:       int
    # Change 2 — multi-agent fields
    agent_name:       str        = "retrieval"
    code:             str | None = None
    image_b64:        str | None = None
    chart_type:       str | None = None


# Change 4
class StopResponse(BaseModel):
    session_id: str
    cancelled:  bool


class ResetResponse(BaseModel):
    session_id: str
    cleared:    bool


class HealthResponse(BaseModel):
    status:  str
    version: str


class StatusResponse(BaseModel):
    total_chunks:    int
    total_documents: int
    namespaces:      dict[str, int]


class DocumentListResponse(BaseModel):
    documents: list[dict[str, Any]]


class IngestResponse(BaseModel):
    file_path:      str
    chunks_indexed: int
    success:        bool