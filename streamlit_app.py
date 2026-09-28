"""Simple browser UI for the existing adaptive RAG graph."""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from adaptive_rag.graph import build_graph, run_question
from adaptive_rag.providers import OllamaCloud
from adaptive_rag.retrieval import KnowledgeBase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Adaptive RAG Assistant", page_icon="📚")
st.title("📚 Adaptive RAG Assistant")
st.caption("Ask a question. Get an answer grounded in local documents or web research.")

with st.sidebar:
    st.header("Settings")
    st.caption(f"Ollama Cloud · {os.getenv('OLLAMA_MODEL', 'gpt-oss:20b')}")
    max_retries = st.slider("Maximum retries", 0, 5, 2)
    st.caption("Extra attempts when an answer needs better evidence.")
    st.divider()
    st.markdown("**Local knowledge base**")
    st.write("RAG, LangGraph, retrieval, and checkpointing.")
    st.caption("Each question starts a fresh research run.")

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder="How do conditional edges work in LangGraph?",
        height=100,
    )
    submitted = st.form_submit_button("Research", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    elif not os.getenv("OLLAMA_API_KEY"):
        st.error("Set OLLAMA_API_KEY in the project's .env file, then restart the app.")
    else:
        st.session_state.pop("result", None)
        provider = None
        try:
            with st.spinner("Researching and checking the answer…"):
                provider = OllamaCloud(
                    os.environ["OLLAMA_API_KEY"],
                    os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
                )
                graph = build_graph(provider, KnowledgeBase(ROOT / "knowledge_base"))
                st.session_state["result"] = run_question(graph, question, max_retries)
        except Exception:
            st.error("The research run failed. Check your configuration and try again.")
        finally:
            if provider is not None:
                provider.close()

if result := st.session_state.get("result"):
    st.divider()
    st.subheader(result["question"])
    if result["status"] == "verified":
        st.success("Answer passed the evidence check.")
    else:
        st.warning("The answer could not be verified within the retry limit.")
    st.markdown(result["answer"])
    st.caption(
        f"{result['generation_count']} answer attempt(s) · "
        f"{result['retry_count']} retries · {len(result['documents'])} sources"
    )

    with st.expander("Sources"):
        if not result["documents"]:
            st.write("No sources found.")
        for document in result["documents"]:
            st.markdown(f"**{document['title'] or 'Untitled source'}**")
            st.caption(f"[{document['id']}]")
            if document["kind"] == "web":
                st.markdown(f"[Open source]({document['source']})")
            else:
                st.caption(Path(document["source"]).name)
            st.text(document["content"][:700])
            st.divider()

    with st.expander("Execution path"):
        st.code("START → " + " → ".join(step["node"] for step in result["trace"]) + " → END", language=None)
        for step in result["trace"]:
            st.markdown(f"**{step['node']}**")
            st.write(step["detail"])

    if result["errors"]:
        with st.expander("Service errors", expanded=True):
            for error in result["errors"]:
                st.error(error)
