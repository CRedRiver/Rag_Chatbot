from typing import List

class RecursiveCharacterTextSplitter:
    """
    Splits on separators in order, merges small pieces, and preserves overlap.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    # ------------------------------------------------------------------
    def _merge(self, splits: List[str], sep: str) -> List[str]:
        """Merge small splits into chunks, rolling back to maintain overlap."""
        chunks, current, total = [], [], 0
        sep_len = len(sep)

        for piece in splits:
            piece_len = len(piece)
            joined_len = total + piece_len + (sep_len if current else 0)

            if joined_len > self.chunk_size and current:
                chunks.append(sep.join(current))
                # roll back until within overlap budget
                while current and total > self.chunk_overlap:
                    removed = current.pop(0)
                    total -= len(removed) + sep_len
            current.append(piece)
            total += piece_len + (sep_len if len(current) > 1 else 0)

        if current:
            chunks.append(sep.join(current))
        return [c for c in chunks if c.strip()]

    def _split(self, text: str, separators: List[str]) -> List[str]:
        # Pick the first separator that actually appears in the text
        sep, remaining = separators[-1], []
        for i, s in enumerate(separators):
            if s == "" or s in text:
                sep, remaining = s, separators[i + 1:]
                break

        raw = text.split(sep) if sep else list(text)
        good, final = [], []

        for piece in raw:
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                good.append(piece)
            else:
                if good:
                    final.extend(self._merge(good, sep))
                    good = []
                # recurse into the oversized piece with finer separators
                final.extend(self._split(piece, remaining) if remaining else [piece])

        if good:
            final.extend(self._merge(good, sep))

        return [c for c in final if c.strip()]