from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    score: float
    document: dict[str, Any]


class PokemonRetriever:
    """Simple TF-IDF retriever for starter RAG experiments."""

    def __init__(self, documents: list[dict[str, Any]]):
        if not documents:
            raise ValueError("The retriever requires at least one document.")

        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.document_matrix = self.vectorizer.fit_transform(doc["text"] for doc in documents)

    def _extract_generation_filter(self, query: str) -> str | None:
        query_lower = query.lower()
        patterns = {
            "generation i": [r"generation\s*1\b", r"gen\s*1\b", r"generation\s*i\b", r"gen\s*i\b"],
            "generation ii": [r"generation\s*2\b", r"gen\s*2\b", r"generation\s*ii\b", r"gen\s*ii\b"],
            "generation iii": [r"generation\s*3\b", r"gen\s*3\b", r"generation\s*iii\b", r"gen\s*iii\b"],
            "generation iv": [r"generation\s*4\b", r"gen\s*4\b", r"generation\s*iv\b", r"gen\s*iv\b"],
            "generation v": [r"generation\s*5\b", r"gen\s*5\b", r"generation\s*v\b", r"gen\s*v\b"],
            "generation vi": [r"generation\s*6\b", r"gen\s*6\b", r"generation\s*vi\b", r"gen\s*vi\b"],
            "generation vii": [r"generation\s*7\b", r"gen\s*7\b", r"generation\s*vii\b", r"gen\s*vii\b"],
            "generation viii": [r"generation\s*8\b", r"gen\s*8\b", r"generation\s*viii\b", r"gen\s*viii\b"],
            "generation ix": [r"generation\s*9\b", r"gen\s*9\b", r"generation\s*ix\b", r"gen\s*ix\b"],
        }

        for generation, generation_patterns in patterns.items():
            if any(re.search(pattern, query_lower) for pattern in generation_patterns):
                return generation.title()
        return None

    def _apply_metadata_filters(
        self, results: list[RetrievalResult], query: str
    ) -> list[RetrievalResult]:
        generation_filter = self._extract_generation_filter(query)
        if generation_filter:
            filtered = [
                result
                for result in results
                if str(result.document.get("metadata", {}).get("generation", "")).lower()
                == generation_filter.lower()
            ]
            if filtered:
                return filtered
        return results

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_matrix).flatten()
        ranked_indexes = scores.argsort()[::-1][: max(top_k, 25)]

        results = [
            RetrievalResult(score=float(scores[idx]), document=self.documents[idx])
            for idx in ranked_indexes
            if scores[idx] > 0
        ]

        results = self._apply_metadata_filters(results, query)
        return results[:top_k]


def format_context(results: list[RetrievalResult]) -> str:
    if not results:
        return "No relevant Pokemon context was found."

    chunks = []
    for i, result in enumerate(results, start=1):
        chunks.append(f"[{i}] score={result.score:.3f}\n{result.document['text']}")
    return "\n\n".join(chunks)