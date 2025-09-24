from typing import Dict, List, Union

from ..core.config import ITRConfig
from ..core.types import InstructionFragment, Tool
from ..indexing.corpus import InstructionCorpus, ToolCorpus
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods."""

    def __init__(self, config: ITRConfig):
        self.config = config
        self.dense_retriever = DenseRetriever(config)
        self.sparse_retriever = SparseRetriever(config)

    def retrieve(
        self, query: str, corpus: Union[InstructionCorpus, ToolCorpus], top_m: int
    ) -> List[Union[InstructionFragment, Tool]]:
        """Hybrid retrieval with score fusion."""
        # Get dense results
        dense_results = self.dense_retriever.search(query, corpus, top_m * 2)
        dense_scores = {r.id: r.score for r in dense_results}

        # Get sparse results
        sparse_results = self.sparse_retriever.search(query, corpus, top_m * 2)
        sparse_scores = {r.id: r.score for r in sparse_results}

        # Normalize scores to [0, 1]
        if dense_scores:
            max_dense = max(dense_scores.values())
            if max_dense > 0:
                dense_scores = {k: v / max_dense for k, v in dense_scores.items()}

        if sparse_scores:
            max_sparse = max(sparse_scores.values())
            if max_sparse > 0:
                sparse_scores = {k: v / max_sparse for k, v in sparse_scores.items()}

        # Combine scores
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        hybrid_scores: Dict[str, float] = {}

        for id in all_ids:
            d_score = dense_scores.get(id, 0) * self.config.dense_weight
            s_score = sparse_scores.get(id, 0) * self.config.sparse_weight
            hybrid_scores[id] = d_score + s_score

        # Get top_m candidates
        top_candidates = sorted(
            hybrid_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_m]

        # Get actual items
        results = []
        if isinstance(corpus, InstructionCorpus):
            for id, _ in top_candidates:
                item = corpus.get(id)
                if item:
                    results.append(item)
        else:
            for id, _ in top_candidates:
                item = corpus.get(id)
                if item:
                    results.append(item)

        return results
