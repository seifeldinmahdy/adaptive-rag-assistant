# Assignment evaluation

**23 automated tests passed. Eight live questions completed using Ollama Cloud (`gpt-oss:20b`) with no recorded service errors; all eight final answers were accepted by the answer checker.**

The final live results combine the original run with reruns of questions 2 and 5 after fixing typographic source IDs/brackets and clarifying the citation prompt. Each JSON result records whether it came from the initial evaluation or a citation-fix rerun. The consolidated report retains the final result for each question. These are actual API responses, not scripted answers. Checker acceptance is not an independent guarantee of factual accuracy.

## Actual live execution paths

R = route_question; D = retrieve_documents; G = grade_documents; W = web_search; A = generate_answer; C = check_answer. Every path begins at START and ends at END.

| # | Question | Actual path | Generations | Retries | Checker result |
| --- | --- | --- | --- | --- | --- |
| 1 | What is retrieval-augmented generation and why are citations useful? | R → D → G → A → C → W → A → C | 2 | 1 | verified |
| 2 | How do State, Nodes, and Conditional Edges work in LangGraph? | R → D → G → A → C | 1 | 0 | verified |
| 3 | How does BM25 differ from dense retrieval? | R → D → G → A → C | 1 | 0 | verified |
| 4 | How does a retry limit prevent infinite loops in adaptive RAG? | R → D → G → A → C | 1 | 0 | verified |
| 5 | Who won the 2026 FIFA World Cup? | R → W → A → C | 1 | 0 | verified |
| 6 | What is the James Webb Space Telescope designed to observe? | R → W → A → C | 1 | 0 | verified |
| 7 | What exact date was LangGraph version 0.2 released? | R → D → G → W → A → C | 1 | 0 | verified |
| 8 | Explain BM25 and give the exact title and publication year of the original Okapi TREC-3 paper. | R → W → A → C → W → A → C | 2 | 1 | verified |

The live runs demonstrate **5 distinct paths**: local success, local generation followed by web retry, direct web success, irrelevant local documents followed by web search, and direct web followed by another search. Question 8 routed directly to web despite the intended local-first hypothesis; the actual trace is reported above.

## Live answers

### 1. What is retrieval-augmented generation and why are citations useful?

Retrieval‑augmented generation (RAG) is a pipeline that grounds a language model’s answer in external documents. The system first indexes a corpus, then retrieves passages that are relevant to a user’s question (using lexical matching such as BM25, dense vectors, or a hybrid) and supplies those passages as context to the generator. The goal is to reduce unsupported claims, though it does not guarantee correctness [local-rag-1].

Citations are useful because they turn the model’s output into verifiable evidence. By linking each claim back to a specific retrieved unit—whether a chunk, sentence, or span—users can see exactly where the information came from, which helps detect hallucinations and debug errors [web-949ee5f3aa3d]. Citations also enable users to quickly confirm the truthfulness of a statement without reading the entire retrieved passage, improving verification efficiency [web-b3053bf09758]. In production, claim‑level numbering or inline markers are common patterns that expose provenance and separate the model’s generation from the evidence it cites [web-949ee5f3aa3d].

Evidence referenced in this answer:

- `local-rag-1`: [Retrieval-augmented generation (RAG)](../knowledge_base/rag.md)
- `web-b3053bf09758`: [Concise and Sufficient Sub-Sentence Citations for Retrieval-Augmented Generation](https://www.arxiv.org/pdf/2509.20859)
- `web-949ee5f3aa3d`: [Citing Sources in a RAG Answer](https://ragexplainedbetter.com/generation/citations/)

### 2. How do State, Nodes, and Conditional Edges work in LangGraph?

LangGraph represents an application as a directed graph where **State**, **Nodes**, and **Edges** coordinate the flow of data and control.

**State**  
- A shared container (often a `TypedDict`) holds values such as the user’s question, retrieved documents, the current answer, and a retry counter.  
- Nodes read from and write to this state.  
- By default, a node’s returned dictionary overwrites existing keys; an annotated reducer can be used to merge or accumulate updates instead.

**Nodes**  
- Python callables that receive the current state, perform logic (e.g., querying a model or database), and return a partial state update.  
- They are the computational units of the graph.

**Edges**  
- **Normal edges** deterministically transfer control from one node to the next.  
- **Conditional edges** use a routing function that inspects the current state and selects the next node dynamically.  
- The graph starts at a special `START` node and ends at an `END` node.  
- Cycles are allowed, enabling the system to revise an answer rather than following a strictly linear path.

Before execution, the graph must be compiled. This structure lets LangGraph orchestrate complex, stateful workflows while keeping the logic modular and reusable. [local-langgraph-1]

Evidence referenced in this answer:

- `local-langgraph-1`: [LangGraph state, nodes, and edges](../knowledge_base/langgraph.md)

### 3. How does BM25 differ from dense retrieval?

BM25 is a lexical ranking method that scores passages by the frequency of query terms, their inverse document frequency, and a length‑normalization factor. It relies on exact term matches and term statistics in the corpus. Dense retrieval, in contrast, represents both queries and passages as continuous embeddings and ranks them by comparing these vectors (e.g., cosine similarity). Thus, BM25 uses discrete term‑based signals, while dense retrieval uses semantic embeddings to capture meaning beyond exact term overlap. Hybrid approaches can combine both signals, and a reranker may reorder candidates from either method. [local-retrieval-1]

Evidence referenced in this answer:

- `local-retrieval-1`: [Chunking and retrieval evaluation](../knowledge_base/retrieval.md)

### 4. How does a retry limit prevent infinite loops in adaptive RAG?

In an adaptive RAG workflow the assistant may regenerate an answer if the checker flags it as unsupported or incomplete.  
To avoid looping forever, the graph tracks a **retry_count** in its state and compares it to a configured **max_retries**.  
When a new web search or regeneration is triggered, the graph first checks `retry_count < max_retries`. If the limit is reached, the loop stops and the graph reports that the answer could not be verified instead of continuing to generate drafts. This explicit stop condition, together with LangGraph’s `recursion_limit` guard, ensures the retry loop is bounded and prevents infinite execution. [local-persistence-1] [local-adaptive_rag-1]

Evidence referenced in this answer:

- `local-persistence-1`: [Checkpointing and bounded loops](../knowledge_base/persistence.md)
- `local-adaptive_rag-1`: [Adaptive RAG routing and grading](../knowledge_base/adaptive_rag.md)

### 5. Who won the 2026 FIFA World Cup?

Spain won the 2026 FIFA World Cup, defeating Argentina 1‑0 in the final after extra time[web-0240dfe9b865][web-c5ea2ad23109].

Evidence referenced in this answer:

- `web-0240dfe9b865`: [Spain crowned FIFA World Cup 2026™ Champions in New York New Jersey](https://inside.fifa.com/organisation/news/spain-crowned-world-cup-2026-champions-new-york-new-jersey)
- `web-c5ea2ad23109`: [Spain beats Argentina 1-0 to win World Cup on Ferran Torres' goal | AP News](https://apnews.com/article/argentina-spain-world-cup-final-score-messi-fccc26aa12d9226e63d06b601b770617)

### 6. What is the James Webb Space Telescope designed to observe?

The James Webb Space Telescope is built as a **near‑ and mid‑infrared observatory** that will:

* **Peer back to the earliest galaxies and stars** that formed after the Big Bang, allowing astronomers to study the first luminous objects in the Universe [web-a3e66fff555c][web-96f9b713d9fc][web-bed126b9d1d5].
* **Look through dust‑enshrouded regions** to investigate the birth of stars and planetary systems, both in our own Galaxy and in other galaxies [web-a3e66fff555c][web-96f9b713d9fc][web-bed126b9d1d5].
* **Characterize exoplanets and other worlds**, including their atmospheres and potential habitability, by observing them in the infrared [web-3c89b5f66cc7][web-bed126b9d1d5].
* **Study the evolution of galaxies and the role of dark matter/energy** over cosmic time, using its unprecedented sensitivity and wavelength coverage (0.6–28.5 µm) [web-96f9b713d9fc].

In short, Webb is designed to observe the Universe’s earliest structures, the processes that form stars and planets, and the properties of exoplanets, all through infrared light that can penetrate dust and reach back over 13 billion years.

Evidence referenced in this answer:

- `web-a3e66fff555c`: [Webb FAQs - NASA Science](https://science.nasa.gov/mission/webb/faqs-full/)
- `web-96f9b713d9fc`: [Fact Sheet - NASA Science](https://science.nasa.gov/mission/webb/fact-sheet/)
- `web-bed126b9d1d5`: [Overview - NASA Science](https://science.nasa.gov/mission/webb/about-overview/)
- `web-3c89b5f66cc7`: [ESA Science & Technology - Objectives](https://sci.esa.int/web/jwst/-/33080-objectives)

### 7. What exact date was LangGraph version 0.2 released?

LangGraph v0.2 (tag 0.2.0) was released on **2024‑08‑07** [web-ef0113179a4f] (the GitHub release shows a published timestamp of 2024‑08‑07T03:34:20Z, and the blog post lists “August 7, 2024” as the release date)【web-ef0113179a4f】.

Evidence referenced in this answer:

- `web-ef0113179a4f`: [0.2.0](https://github.com/langchain-ai/langgraph/releases/tag/0.2.0)

### 8. Explain BM25 and give the exact title and publication year of the original Okapi TREC-3 paper.

**BM25 (Okapi BM25)**  
BM25 is a probabilistic ranking function that estimates how well a document \(D\) matches a query \(Q\).  
It extends the Robertson–Sparck Jones model by adding:  

* **Term frequency (TF)** – how often a query term appears in \(D\).  
* **Inverse document frequency (IDF)** – how rare the term is in the collection.  
* **Document‑length normalization** – a factor that reduces the score of very long documents.  

The classic BM25 score is  

\[
\text{score}(D,Q)=\sum_{i=1}^{n}\text{IDF}(q_i)\;
\frac{f(q_i,D)\,(k_1+1)}
{f(q_i,D)+k_1\!\left(1-b+b\frac{|D|}{\text{avgdl}}\right)}
\]

where \(f(q_i,D)\) is the term frequency, \(|D|\) the document length, \(\text{avgdl}\) the average collection length, and \(k_1\) and \(b\) are tuning parameters [web-9a1194cbfbab].

**Original Okapi TREC‑3 paper**  
The paper that introduced BM25 is titled **“Okapi at TREC‑3”** and was published in **1994** (presented at TREC‑3, November 1994, and released in NIST Special Publication 500‑225) [web-305b1d6e479a].

Evidence referenced in this answer:

- `web-9a1194cbfbab`: [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- `web-305b1d6e479a`: [Okapi at TREC-3 | Papers We Love](https://paperswelove.org/papers/okapi-at-trec-3-03e7eb52/)

## Deterministic control-flow coverage

These eight additional runs use the real compiled graph with scripted provider decisions. Fixture answers test branching and termination, not model answer quality. The scenarios deliberately differ from the live routes.

| # | Actual scripted path | Generations | Result |
| --- | --- | --- | --- |
| 1 | R → D → G → A → C | 1 | verified |
| 2 | R → D → G → A → C | 1 | verified |
| 3 | R → D → G → W → A → C | 1 | verified |
| 4 | R → D → G → A → C → W → A → C | 2 | verified |
| 5 | R → W → A → C | 1 | verified |
| 6 | R → W → A → C → W → A → C | 2 | verified |
| 7 | R → D → G → W → A → C → W → A → C → W → A → C | 3 | unverified |
| 8 | R → D → G → A → C → W → A → C → W → A → C | 3 | unverified |

The repeated-failure scenarios stop after three generations (two retries) and return an explicit unverified message. Other tests cover empty retrieval, unavailable/empty search, unknown citations, typographic/grouped citations, zero retries, checkpoints and fresh state, input validation, cloud JSON repair, sanitized API errors, and search parsing.

## Reproduce

```bash
python -m pytest -q
python -m streamlit run streamlit_app.py
```

To repeat the live scenarios, submit the questions in `examples/questions.json`
through the UI. The saved graph diagram is in `architecture.mmd`.

Complete states, evidence, checker feedback, and node traces are in [live_evaluation.json](live_evaluation.json) and [offline_evaluation.json](offline_evaluation.json). See [README](../README.md) for architecture and the explanation of State, Nodes, Edges, Conditional Edges, Routing, and Loops.
