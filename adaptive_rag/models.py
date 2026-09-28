"""Shared graph state and validated model decisions."""
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class Document(TypedDict):
    id: str
    title: str
    content: str
    source: str
    kind: Literal["local", "web"]


class State(TypedDict):
    question: str
    documents: list[Document]
    route: Literal["local", "web"]
    answer: str
    draft: str
    relevant: bool
    supported: bool
    feedback: str
    search_query: str
    retry_count: int
    max_retries: int
    generation_count: int
    status: Literal["running", "verified", "unverified"]
    trace: list[dict]
    errors: list[str]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RouteDecision(Decision):
    route: Literal["local", "web"]
    reason: str


class GradeDecision(Decision):
    relevant_ids: list[str]
    reason: str


class CheckDecision(Decision):
    supported: bool
    complete: bool
    reason: str
    search_query: str = Field(min_length=1, max_length=500)
