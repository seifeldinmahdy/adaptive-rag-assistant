"""Deterministic branch tests run the REAL compiled LangGraph with fake I/O.

They deliberately inject grading/checking outcomes; they are not live LLM results.
"""
import json
from pathlib import Path

import pytest

from adaptive_rag.graph import build_graph, initial_state, normalize_citations, run_question
from adaptive_rag.models import CheckDecision, Document, GradeDecision, RouteDecision
from adaptive_rag.providers import ProviderError
from adaptive_rag.retrieval import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = json.loads((ROOT / "examples/questions.json").read_text())
LOCAL = Document(id="local-test-1", title="Fixture", content="Relevant test evidence", source="fixture.md", kind="local")
WEB = Document(id="web-test", title="Web fixture", content="Supplemental test evidence", source="https://example.com/evidence", kind="web")


class FakeKB:
    catalog = ["RAG", "LangGraph"]

    def __init__(self, empty=False):
        self.empty = empty

    def retrieve(self, question):
        return [] if self.empty else [LOCAL]


class FakeProvider:
    def __init__(self, route="local", relevant=True, failures=0, search_error=False, empty_search=False, invalid_citation=False):
        self.route, self.relevant, self.failures = route, relevant, failures
        self.search_error, self.empty_search, self.invalid_citation = search_error, empty_search, invalid_citation
        self.checks = 0
        self.searches = 0
        self.payloads = []

    def decide(self, task, instructions, payload, schema):
        if task == "route":
            return RouteDecision(route=self.route, reason="Fixture route")
        if task == "grade":
            return GradeDecision(relevant_ids=[LOCAL["id"]] if self.relevant else [], reason="Fixture grade")
        self.checks += 1
        passed = self.checks > self.failures
        return CheckDecision(supported=passed, complete=passed, reason="Verified fixture" if passed else "Missing fact", search_query="targeted missing fact")

    def generate(self, payload):
        self.payloads.append(payload)
        citation = "web-unknown" if self.invalid_citation else payload["documents"][0]["id"]
        return f"A fixture answer [{citation}]."

    def search(self, query):
        self.searches += 1
        if self.search_error:
            raise ProviderError("Simulated unavailable search")
        return [] if self.empty_search else [WEB]


SCENARIOS = [
    ({}, "verified", 1, 0),
    ({}, "verified", 1, 0),
    ({"relevant": False}, "verified", 1, 1),
    ({"failures": 1}, "verified", 2, 1),
    ({"route": "web"}, "verified", 1, 1),
    ({"route": "web", "failures": 1}, "verified", 2, 2),
    ({"relevant": False, "failures": 99}, "unverified", 3, 3),
    ({"failures": 99}, "unverified", 3, 2),
]


@pytest.mark.parametrize("case,scenario", list(zip(QUESTIONS, SCENARIOS)), ids=[f"question-{i}" for i in range(1, 9)])
def test_eight_questions(case, scenario):
    options, status, generations, searches = scenario
    provider = FakeProvider(**options)
    state = run_question(build_graph(provider, FakeKB()), case["question"])
    path = [e["node"] for e in state["trace"]]
    expected = ["route_question"]
    if provider.route == "local":
        expected += ["retrieve_documents", "grade_documents"]
    if provider.route == "web" or not provider.relevant:
        expected += ["web_search"]
    expected += ["generate_answer", "check_answer"]
    expected += ["web_search", "generate_answer", "check_answer"] * (generations - 1)
    assert path == expected
    assert state["status"] == status
    assert state["generation_count"] == generations
    assert state["retry_count"] == generations - 1
    assert provider.searches == searches
    if status == "unverified":
        assert "could not verify" in state["answer"]
        assert state["draft"] != state["answer"]
    if generations > 1:
        assert provider.payloads[-1]["feedback"] == "Missing fact"
    assert len({d["id"] for d in state["documents"]}) == len(state["documents"])
    if not provider.relevant:
        assert all(d["kind"] == "web" for d in state["documents"])


@pytest.mark.parametrize("search_error", [True, False])
def test_unavailable_or_empty_web_terminates(search_error):
    provider = FakeProvider(route="web", search_error=search_error, empty_search=True)
    state = run_question(build_graph(provider, FakeKB()), "Unknown question")
    assert state["status"] == "unverified"
    assert state["generation_count"] == 3
    assert provider.searches == 3
    assert bool(state["errors"]) == search_error


def test_empty_retrieval_falls_back_to_web():
    state = run_question(build_graph(FakeProvider(), FakeKB(empty=True)), "Missing subject")
    assert state["status"] == "verified"
    assert "web_search" in [e["node"] for e in state["trace"]]


def test_zero_retries_and_unknown_citation():
    provider = FakeProvider(invalid_citation=True)
    state = run_question(build_graph(provider, FakeKB()), "Question", max_retries=0)
    assert state["status"] == "unverified"
    assert state["generation_count"] == 1
    assert provider.searches == 0
    assert provider.checks == 0


def test_typographic_and_grouped_citations():
    text = "State uses a reducer [local‑langgraph‑1, web–abc]. Ordinary prose — unchanged."
    assert normalize_citations(text) == "State uses a reducer [local-langgraph-1] [web-abc]. Ordinary prose — unchanged."
    assert normalize_citations("[unrelated text]") == "[unrelated text]"
    assert normalize_citations("Evidence【web‑abc】［local‑rag‑1］") == "Evidence[web-abc][local-rag-1]"


def test_typographic_citations_pass_through_real_graph():
    provider = FakeProvider()
    provider.generate = lambda payload: "Supported answer [local‑test‑1]."
    state = run_question(build_graph(provider, FakeKB()), "Question", max_retries=0)
    assert state["status"] == "verified"
    assert state["answer"] == "Supported answer [local-test-1]."


def test_checkpoint_and_fresh_state():
    graph = build_graph(FakeProvider(), FakeKB())
    config = {"configurable": {"thread_id": "test-thread"}}
    graph.invoke(initial_state("First question"), config)
    snapshot = graph.get_state(config)
    assert snapshot.values["status"] == "verified"
    assert snapshot.next == ()
    graph.invoke(initial_state("Second question"), config)
    assert graph.get_state(config).values["generation_count"] == 1
    assert len(graph.get_state(config).values["trace"]) == 5


def test_real_local_retrieval():
    kb = KnowledgeBase(ROOT / "knowledge_base")
    docs = kb.retrieve("BM25 dense embeddings lexical retrieval")
    assert docs and any("BM25" in d["content"] for d in docs)
    assert kb.retrieve("zzzzzzzz") == []


@pytest.mark.parametrize("question,retries", [("", 2), ("q", -1), ("q", 6)])
def test_invalid_input(question, retries):
    with pytest.raises(ValueError):
        initial_state(question, retries)
