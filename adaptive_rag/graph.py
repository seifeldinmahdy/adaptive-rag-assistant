"""Six named nodes, ordinary/conditional edges, and a bounded feedback cycle."""
import re
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .models import CheckDecision, GradeDecision, RouteDecision, State
from .providers import Provider, ProviderError
from .retrieval import KnowledgeBase


def initial_state(question: str, max_retries: int = 2) -> State:
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if not 0 <= max_retries <= 5:
        raise ValueError("max_retries must be between 0 and 5.")
    return State(
        question=question.strip(), documents=[], route="local", answer="", draft="",
        relevant=False, supported=False, feedback="", search_query=question.strip(),
        retry_count=0, max_retries=max_retries, generation_count=0,
        status="running", trace=[], errors=[],
    )


def event(state: State, node: str, detail: str, **updates) -> dict:
    return {**updates, "trace": [*state["trace"], {"node": node, "detail": detail}]}


def normalize_citations(answer: str) -> str:
    """Accept typographic dashes and grouped citations without changing prose."""
    dashes = str.maketrans({char: "-" for char in "‐‑‒–—−﹘﹣－"})

    def replace(match):
        parts = re.split(r"[,;]\s*", match.group(1).translate(dashes).strip())
        if all(re.fullmatch(r"(?:local|web)-[^\s,;\[\]]+", part) for part in parts):
            return " ".join(f"[{part}]" for part in parts)
        return match.group(0)

    return re.sub(r"[\[【［]([^\[\]【】［］\n]+)[\]】］]", replace, answer)


def build_graph(provider: Provider, kb: KnowledgeBase, checkpointer=None):
    def route_question(state: State):
        try:
            result = provider.decide("route", (
                "Route to local if the question is about a subject in the knowledge-base "
                "catalog, even if its precise answer might be missing. Route unrelated "
                "questions and questions requiring current/latest facts to web."
            ), {"question": state["question"], "catalog": kb.catalog}, RouteDecision)
            route, reason, errors = result.route, result.reason, state["errors"]
        except ProviderError as exc:
            route, reason = "web", "Router unavailable; fall back to web."
            errors = [*state["errors"], str(exc)]
        return event(state, "route_question", reason, route=route, errors=errors)

    def retrieve_documents(state: State):
        documents = kb.retrieve(state["question"])
        return event(state, "retrieve_documents", f"Retrieved {len(documents)} local passages.", documents=documents)

    def grade_documents(state: State):
        errors = state["errors"]
        if not state["documents"]:
            documents, reason = [], "No local passages found."
        else:
            try:
                result = provider.decide("grade", (
                    "Select IDs of documents containing information that helps answer the "
                    "actual question. Keyword overlap alone is insufficient. Reject general "
                    "descriptions when the question needs an absent specific fact."
                ), {"question": state["question"], "documents": state["documents"]}, GradeDecision)
                documents = [d for d in state["documents"] if d["id"] in result.relevant_ids]
                reason = result.reason
            except ProviderError as exc:
                documents, reason = [], "Grading failed; fall back to web."
                errors = [*errors, str(exc)]
        return event(state, "grade_documents", reason, documents=documents, relevant=bool(documents), errors=errors)

    def web_search(state: State):
        # Only a search after a failed generation consumes an answer retry.
        retries = state["retry_count"] + int(state["generation_count"] > 0)
        query = state["search_query"] or state["question"]
        errors = state["errors"]
        try:
            results = provider.search(query)
            detail = f"Search: {query} — {len(results)} results; retry {retries}."
        except ProviderError as exc:
            results, detail = [], f"Web search failed; retry {retries}."
            errors = [*errors, str(exc)]
        # Stable IDs deduplicate URLs; refreshed snippets replace old versions.
        merged = {d["id"]: d for d in state["documents"]}
        merged.update({d["id"]: d for d in results})
        return event(state, "web_search", detail, documents=list(merged.values()), retry_count=retries, errors=errors)

    def generate_answer(state: State):
        errors = state["errors"]
        try:
            draft = provider.generate({
                "question": state["question"], "documents": state["documents"],
                "feedback": state["feedback"],
            }) if state["documents"] else "Insufficient evidence to answer this question."
        except ProviderError as exc:
            draft = "Answer generation failed."
            errors = [*errors, str(exc)]
        return event(state, "generate_answer", f"Generated attempt {state['generation_count'] + 1}.",
                     draft=normalize_citations(draft), generation_count=state["generation_count"] + 1, errors=errors)

    def check_answer(state: State):
        errors = state["errors"]
        known = {d["id"] for d in state["documents"]}
        citations = set(re.findall(r"\[((?:local|web)-[^\]\s]+)\]", state["draft"]))
        query = state["question"]
        if not known:
            passed, reason = False, "No evidence available."
        elif not citations or not citations <= known:
            passed, reason = False, "Answer has missing or unknown document citations."
        else:
            try:
                result = provider.decide("check", (
                    "Check the draft strictly against the supplied evidence, not your own "
                    "knowledge. supported=true only if all material claims are supported "
                    "by their cited documents. complete=true only if the question is "
                    "sufficiently answered. An admission of missing evidence is incomplete. "
                    "Explain gaps and propose a targeted web search query."
                ), {"question": state["question"], "documents": state["documents"], "answer": state["draft"]}, CheckDecision)
                passed = result.supported and result.complete
                reason, query = result.reason, result.search_query
            except ProviderError as exc:
                passed, reason = False, "Answer check failed; answer is not verified."
                errors = [*errors, str(exc)]
        exhausted = state["retry_count"] >= state["max_retries"]
        status = "verified" if passed else "unverified" if exhausted else "running"
        answer = state["draft"] if passed else ""
        if not passed and exhausted:
            answer = "I could not verify a sufficiently supported answer within the retry limit. " + reason
        return event(state, "check_answer", reason, supported=passed, feedback=reason,
                     search_query=query, status=status, answer=answer, errors=errors)

    builder = StateGraph(State)
    for node in (route_question, retrieve_documents, grade_documents, web_search, generate_answer, check_answer):
        builder.add_node(node.__name__, node)
    builder.add_edge(START, "route_question")
    builder.add_conditional_edges("route_question", lambda s: s["route"], {"local": "retrieve_documents", "web": "web_search"})
    builder.add_edge("retrieve_documents", "grade_documents")
    builder.add_conditional_edges("grade_documents", lambda s: s["relevant"], {True: "generate_answer", False: "web_search"})
    builder.add_edge("web_search", "generate_answer")
    builder.add_edge("generate_answer", "check_answer")
    builder.add_conditional_edges("check_answer", lambda s: "retry" if s["status"] == "running" else "verified_or_exhausted", {
        "retry": "web_search", "verified_or_exhausted": END,
    })
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def run_question(graph, question: str, max_retries: int = 2) -> State:
    # Each independent question has a fresh thread, so counters/evidence cannot leak.
    state = initial_state(question, max_retries)
    return graph.invoke(state, {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 12 + 3 * max_retries})
