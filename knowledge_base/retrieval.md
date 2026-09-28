# Chunking and retrieval evaluation

Chunking splits long documents into smaller passages. Smaller chunks can improve
precision but lose context; larger chunks preserve context but can include irrelevant
material. Overlap can preserve information across chunk boundaries. A good chunk size
depends on the corpus and task and should be evaluated on representative questions.

BM25 ranks passages using query-term frequency, inverse document frequency, and
document-length normalization. Dense retrieval instead compares embeddings. Hybrid
retrieval combines lexical and semantic signals. A reranker can reorder candidates.

Retrieval precision measures the fraction of retrieved passages that are relevant.
Retrieval recall measures the fraction of relevant passages that were retrieved.
Answer evaluation should also consider factual support, completeness, and citations.
