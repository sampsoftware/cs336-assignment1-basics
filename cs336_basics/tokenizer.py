import logging
import os
from collections.abc import Iterator, Iterable
from collections import Counter
import regex as re

import builtins
import base64
from cs336_basics import bpe

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
            vocab: Token map
            merges: List of merges in order.
            special_tokens: List of n special tokens.

        Returns:
            None
        """
        self.vocab = dict(vocab or {})
        self.merges = list(merges or [])
        self.special_tokens = sorted(set((special_tokens or [])), key=len, reverse=True)
        self._special_token_bytes = set((t.encode("utf-8") for t in (special_tokens or [])))
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._token_to_id = {tok: i for i, tok in self.vocab.items()}

        for special_tb in self._special_token_bytes:
            if special_tb not in self._token_to_id:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = special_tb
                self._token_to_id[special_tb] = new_id

        assert len(self.vocab) == len(self._token_to_id), "Vocab map does not have unique values"

        logger.debug(
            "Initialized Tokenizer; %d original vocab items, %d merges and %d special tokens; %d total vocabs",
            len(vocab or {}),
            len(self.merges),
            len(self.special_tokens),
            len(self.vocab),
        )

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Args:
            vocab: Serialized token map, integer id to Base64-encoded byte string. Space delimits id to string,
                newline terminates. Spaces may (do) occur at the beginning of the line before the id.
            merges: List of merges in order. Pair of byte strings, base64-encoded, space-delimited.
            special_tokens: list[str] | None = None

        Returns: A Tokenizer initialized on the files.
        """

        vocab = {}
        merges = []

        with open(vocab_filepath, "rb") as f:
            for line in f:
                parts = line.split()
                vocab[int(parts[0])] = base64.b64decode(parts[1])

        with open(merges_filepath, "rb") as f:
            for line in f:
                parts = line.split()
                merges.append((base64.b64decode(parts[0]), (base64.b64decode(parts[1]))))

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.

        Args:
            text: the text to encode

        Returns:
            list of encoded token ids


        """
        if self.special_tokens:
            docs_and_sts = re.split(
                "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")",
                text,
            )
        else:
            docs_and_sts = [text]

        encoded_list = []
        for doc_or_st in docs_and_sts:
            if doc_or_st in self.special_tokens:
                encoded_list.append(self._token_to_id[doc_or_st.encode('utf-8')])
            else:
                pretokens = []
                pretoken_matches = re.finditer(bpe.PAT, doc_or_st)
                for pretoken_match in pretoken_matches:
                    pretoken_string = pretoken_match.group(0)
                    pretoken_bytes = []
                    for b in pretoken_string.encode("utf-8"):
                        pretoken_bytes.append(bytes([b]))
                    pretokens.append(tuple(pretoken_bytes))

                for pretoken in pretokens:
                    in_process_pretoken = pretoken
                    while True:
                        possible_merges = [p for p in zip(in_process_pretoken[:-1], in_process_pretoken[1:]) if p in self._merge_ranks]
                        if not possible_merges:
                            break
                        best = min(possible_merges, key=self._merge_ranks.__getitem__)

                        in_process_pretoken = bpe.apply_merged_token(in_process_pretoken, best)

                    for token in in_process_pretoken:
                        encoded_list.append(self._token_to_id[token])

        return encoded_list



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

        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


