from typing import List, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..core.config import ITRConfig
from ..core.types import InstructionFragment, SearchResult
from ..indexing.corpus import InstructionCorpus, ToolCorpus
from ..indexing.embedder import Embedder


class DenseRetriever:
    """Dense vector retrieval using embeddings."""

    def __init__(self, config: ITRConfig):
        self.config = config
        self.embedder = Embedder(config.embedding_model)

    def search(
        self, query: str, corpus: Union[InstructionCorpus, ToolCorpus], top_m: int
    ) -> List[SearchResult]:
        """Search using dense embeddings."""
        # Get query embedding
        query_embedding = self.embedder.embed(query)[0]

        # Get all items from corpus
        if isinstance(corpus, InstructionCorpus):
            items = corpus.get_all()
        else:
            items = corpus.get_all()

        if not items:
            return []

        # Generate embeddings if not present
        for item in items:
            if item.embedding is None:
                if isinstance(item, InstructionFragment):
                    item.embedding = self.embedder.embed(item.content)[0]
                else:  # Tool
                    text = f"{item.name}\n{item.description}"
                    item.embedding = self.embedder.embed(text)[0]

        # Calculate similarities
        embeddings = np.array([item.embedding for item in items])
        similarities = cosine_similarity([query_embedding], embeddings)[0]

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_m]

        results = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    id=items[idx].id, score=float(similarities[idx]), item=items[idx]
                )
            )

        return results
