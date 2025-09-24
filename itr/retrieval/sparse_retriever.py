from typing import List, Union

from ..core.config import ITRConfig
from ..core.types import SearchResult
from ..indexing.corpus import InstructionCorpus, ToolCorpus
from ..indexing.sparse_index import SparseIndex


class SparseRetriever:
    """BM25 sparse retrieval."""

    def __init__(self, config: ITRConfig):
        self.config = config
        self.instruction_index = SparseIndex()
        self.tool_index = SparseIndex()

    def search(
        self, query: str, corpus: Union[InstructionCorpus, ToolCorpus], top_m: int
    ) -> List[SearchResult]:
        """Search using BM25."""
        if isinstance(corpus, InstructionCorpus):
            items = corpus.get_all()
            if not items:
                return []

            # Build index if needed
            documents = [item.content for item in items]
            doc_ids = [item.id for item in items]
            self.instruction_index.build(documents, doc_ids)

            # Search
            search_results = self.instruction_index.search(query, top_m)
        else:  # ToolCorpus
            items = corpus.get_all()
            if not items:
                return []

            # Build index if needed
            documents = [f"{item.name} {item.description}" for item in items]
            doc_ids = [item.id for item in items]
            self.tool_index.build(documents, doc_ids)

            # Search
            search_results = self.tool_index.search(query, top_m)

        # Convert to SearchResult format
        results = []
        items_dict = {item.id: item for item in items}

        for res in search_results:
            if res["id"] in items_dict:
                results.append(
                    SearchResult(
                        id=res["id"], score=res["score"], item=items_dict[res["id"]]
                    )
                )

        return results
