# Retrieval-augmented generation (RAG)

Retrieval-augmented generation grounds a language model's answer in external documents.
The pipeline indexes documents, retrieves passages for a question, and supplies those
passages as context to a generator. Retrieval can use lexical matching (such as BM25),
dense vector similarity, or a hybrid of both. An embedding maps text into a numeric
vector, allowing semantic similarity search. This project uses local BM25 retrieval;
it does not need an embedding API or a vector database.

Grounding can reduce unsupported claims but does not guarantee correctness. A useful
answer cites the supplied evidence, and explicitly acknowledges gaps. Retrieval
quality, document freshness, and generation quality all affect the final result.

Source: https://arxiv.org/abs/2005.11401 (introductory teaching notes, not a quotation).
