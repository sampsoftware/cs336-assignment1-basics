import logging
import os
from collections.abc import Iterator, Iterable

import builtins

if not hasattr(builtins, "profile"):

    def profile(func):
        return func


logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Class comment

    """

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None

        Returns:
            None
        """
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens

        logger.debug("Initialized Tokenizer")

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        same format that your BPE training code output) and (optionally) a list of special tokens.

        Args:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None

        Returns:
            None
        """
        pass

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        Args:
            text: the text to encode

        Returns:
            list of encoded token ids


        """
        encoded_text = text.encode("utf-8")
        encoded_tokens: list[int] = []
        for b in encoded_text:
            encoded_tokens.append(b)
        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of  strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into
        memory.

        Args:
            iterable: Iterable list of strings to encode

        Returns:
            Iterable list of token ids.

        """
        pass

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text

        Args:
            ids: Token ids

        Returns:
            decoded string
        """
        decoded = "TEST"

        return decoded
