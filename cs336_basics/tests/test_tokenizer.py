import pytest

from cs336_basics.tokenizer import Tokenizer


def test_initialize():
    t = Tokenizer(None, None)
    assert isinstance(t, Tokenizer)


def test_encode():
    t = Tokenizer(None, None)
    encoded = t.encode("TEST")
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert isinstance(encoded[0], int)


def test_decode():
    t = Tokenizer(None, None)
    decoded = t.decode(None)
    assert isinstance(decoded, str)


def test_from_files():
    vocab_filepath = "data/verysmall_tiny.txt_merge.txt"
    merges_filepath = "data/verysmall_tiny.txt_merge.txt"
    special_tokens = ["<|endoftext|>"]

    t = Tokenizer(None, None)
    t.from_files(vocab_filepath, merges_filepath, special_tokens)

    print(f"vocab length={len(t.vocab)} merges length={len(t.merges)} special_tokens length={len(t.special_tokens)}")
