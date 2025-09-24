from dataclasses import dataclass
from typing import List, Tuple

import tiktoken


@dataclass
class TextChunk:
    """Represents a chunk of text."""

    text: str
    token_count: int
    start_idx: int
    end_idx: int


class TextChunker:
    """Simple text chunker that splits text into token-sized chunks."""

    def __init__(self, chunk_size_range: Tuple[int, int] = (200, 600)):
        self.min_size, self.max_size = chunk_size_range
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
        except Exception:
            # Fallback to simple word-based tokenization
            self.tokenizer = None
            self.use_tiktoken = False

    def chunk(self, text: str) -> List[TextChunk]:
        """Split text into chunks based on token count."""
        chunks = []

        # Split by paragraphs first
        paragraphs = text.split("\n\n")

        current_chunk = []
        current_tokens = 0
        start_idx = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if current_tokens + para_tokens > self.max_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        token_count=current_tokens,
                        start_idx=start_idx,
                        end_idx=start_idx + len(chunk_text),
                    )
                )

                # Start new chunk
                current_chunk = [para]
                current_tokens = para_tokens
                start_idx += len(chunk_text) + 2
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    token_count=current_tokens,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                )
            )

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken or fallback to word count."""
        if self.use_tiktoken:
            return len(self.tokenizer.encode(text))
        else:
            # Simple approximation: words * 1.3 (average tokens per word)
            return int(len(text.split()) * 1.3)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self._count_tokens(text)
