from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi


class SparseIndex:
    """BM25 sparse index for text retrieval."""

    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_ids = []

    def build(self, documents: List[str], doc_ids: List[str]):
        """Build BM25 index from documents."""
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc.lower().split() for doc in documents]

        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
        self.doc_ids = doc_ids

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "id": self.doc_ids[idx],
                        "score": float(scores[idx]),
                        "text": self.documents[idx],
                    }
                )

        return results
