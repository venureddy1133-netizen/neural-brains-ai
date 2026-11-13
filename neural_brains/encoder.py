"""Deterministic Unicode-aware text-to-vector encoder using hashed words and bigrams.

The encoder keeps the existing signed feature-hashing design and output size,
so it remains compatible with the project's 64-input neural network. Unicode
letters, combining marks, and numbers are treated as token characters.
"""

import hashlib
import math
import unicodedata


class TextEncoder:
    """Convert text into a deterministic, L2-normalized feature vector."""

    def __init__(self, dimensions=64):
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("dimensions must be an integer")
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    @staticmethod
    def _is_token_character(char):
        """Return True for Unicode letters, marks, and numbers.

        Including Unicode marks (Mn, Mc, Me) is important for scripts such as
        Telugu and Devanagari, where vowel signs and other combining marks
        contribute to a written syllable.
        """
        return unicodedata.category(char)[0] in {"L", "M", "N"}

    @classmethod
    def tokenize(cls, text):
        """Split text into Unicode-aware tokens, preserving internal apostrophes."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # Canonicalize equivalent Unicode spellings and make Latin casing stable.
        normalized = unicodedata.normalize("NFC", text)
        normalized = unicodedata.normalize("NFC", normalized.casefold())

        tokens = []
        current = []
        length = len(normalized)

        for index, char in enumerate(normalized):
            if cls._is_token_character(char):
                current.append(char)
                continue

            # Treat straight and curly apostrophes as an internal token joiner
            # only when they occur between token characters (e.g. "don't").
            if (
                char in {"'", "\u2019"}
                and current
                and index + 1 < length
                and cls._is_token_character(normalized[index + 1])
            ):
                current.append("'")
                continue

            if current:
                tokens.append("".join(current))
                current = []

        if current:
            tokens.append("".join(current))

        return tokens

    def encode(self, text):
        """Encode words and adjacent-word bigrams into a signed hash vector."""
        tokens = self.tokenize(text)
        terms = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]

        features = [0.0] * self.dimensions

        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self.dimensions
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            features[bucket] += sign

        # L2 normalization keeps vector magnitude from growing with text length.
        norm = math.sqrt(sum(value * value for value in features))
        if norm == 0.0:
            return features
        return [value / norm for value in features]
