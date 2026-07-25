import pytest

from cs336_basics.tokenizer import Tokenizer


def test_initialize():
    t = Tokenizer(None, None)
    assert isinstance(t, Tokenizer)


def test_from_files_basic():
    vocab_filepath = "data/verysmall_tiny.txt_vocab.txt"
    merges_filepath = "data/verysmall_tiny.txt_merges.txt"
    special_tokens = ["<|endoftext|>"]

    t = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    assert len(t.vocab) == 1000
    assert len(t.merges) == 743
    assert len(t.special_tokens) == 1


def test_from_files_added_special_tokens():
    vocab_filepath = "data/verysmall_tiny.txt_vocab.txt"
    merges_filepath = "data/verysmall_tiny.txt_merges.txt"
    special_tokens = ["<|endoftext|>", "<|anotherspecialtoken|>", "tokenwithnodelimiter"]

    t = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    assert len(t.vocab) == 1002
    assert len(t.merges) == 743
    assert len(t.special_tokens) == 3


def test_encode():
    text = 'the cat ate'
    vocab = {
        0: b' ', 
        1: b'a', 
        2: b'c', 
        3: b'e', 
        4: b'h', 
        5: b't', 
        6: b'th', 
        7: b' c', 
        8: b' a', 
        9: b'the', 
        10: b' at',
    }
    merges = [
        (b't', b'h'), 
        (b' ', b'c'), 
        (b' ', b'a'), 
        (b'th', b'e'), 
        (b' a', b't'),
    ]
    encoded = [9, 7, 1, 5, 10, 3]

    t = Tokenizer(vocab, merges)
    assert t.encode(text) == encoded


def test_decode():
    text = 'the cat ate'
    vocab = {
        0: b' ', 
        1: b'a', 
        2: b'c', 
        3: b'e', 
        4: b'h', 
        5: b't', 
        6: b'th', 
        7: b' c', 
        8: b' a', 
        9: b'the', 
        10: b' at',
    }
    encoded = [9, 7, 1, 5, 10, 3]

    t = Tokenizer(vocab, None)

    assert t.decode(encoded) == text


