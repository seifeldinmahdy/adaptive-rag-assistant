"""Small local BM25 index with overlapping word chunks."""
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .models import Document

STOPWORDS = set("a an the is are was were in on of to for and or how what does do with it by".split())


def tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


class KnowledgeBase:
    def __init__(self, directory: Path):
        self.documents: list[Document] = []
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ") if text.strip() else path.stem
            words = text.split()
            for i, start in enumerate(range(0, len(words), 180), 1):
                self.documents.append(Document(
                    id=f"local-{path.stem}-{i}", title=title,
                    content=" ".join(words[start:start + 220]),
                    source=str(path), kind="local",
                ))
        self.index = BM25Okapi([tokenize(d["content"]) for d in self.documents]) if self.documents else None

    @property
    def catalog(self) -> list[str]:
        return sorted({document["title"] for document in self.documents})

    def retrieve(self, question: str, k: int = 3) -> list[Document]:
        if self.index is None:
            return []
        scores = self.index.get_scores(tokenize(question))
        ranked = sorted(zip(scores, self.documents), key=lambda pair: pair[0], reverse=True)
        return [dict(document) for score, document in ranked[:k] if score > 0]
