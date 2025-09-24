import hashlib
import logging
from typing import List, Union

import numpy as np


class Embedder:
    """Generate embeddings for text using sentence transformers or fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default dimension

        # Try to load sentence transformers, but fallback if not available
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.use_transformer = True
            logging.info(
                f"Successfully loaded sentence transformer model: {model_name}"
            )
        except (ImportError, OSError, RuntimeError) as e:
            # Fallback to simple hash-based embeddings
            self.use_transformer = False
            logging.warning(
                f"Could not load sentence transformer model {model_name}: {e}. Using fallback embeddings."
            )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]

        if self.use_transformer and self.model:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings
            except (RuntimeError, ValueError, OSError) as e:
                logging.warning(
                    f"Sentence transformer encoding failed: {e}. Using fallback embeddings."
                )
                # Continue to fallback method

        # Fallback: Simple hash-based embeddings
        embeddings = []
        for text in texts:
            # Create a deterministic embedding from text
            embedding = self._simple_embedding(text)
            embeddings.append(embedding)

        return np.array(embeddings)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts in batches."""
        return self.embed(texts)

    def _simple_embedding(self, text: str) -> np.ndarray:
        """Create a simple deterministic embedding from text."""
        # Use hash to create deterministic values
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to float array
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8)[: self.embedding_dim]

        # Pad if necessary
        if len(embedding) < self.embedding_dim:
            padding = np.zeros(self.embedding_dim - len(embedding))
            embedding = np.concatenate([embedding, padding])

        # Normalize to [-1, 1]
        embedding = (embedding.astype(np.float32) / 128.0) - 1.0

        # Add some text features
        text_features = np.array(
            [
                len(text) / 1000.0,  # Length feature
                text.count(" ") / 100.0,  # Word count approximation
                len(set(text.lower().split())) / 100.0,  # Unique words
            ]
        )

        # Mix in text features
        embedding[: len(text_features)] = (
            embedding[: len(text_features)] + text_features
        ) / 2

        return embedding
