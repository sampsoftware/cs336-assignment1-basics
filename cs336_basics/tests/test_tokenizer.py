import pytest

from cs336_basics.tokenizer import Tokenizer

def test_initialize():
    t = Tokenizer(None, None)
    assert isinstance(t, Tokenizer)

def test_encode():
    t = Tokenizer(None,None)
    encoded = t.encode("TEST")
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert isinstance(encoded[0], int)

def test_decode():
    t = Tokenizer(None,None)
    decoded = t.decode(None)
    assert isinstance(decoded,str)