from cs336_basics.bpe_tokenizer_trainer import find_token_pair_to_merge
from cs336_basics.bpe_tokenizer_trainer import apply_merged_token
import pytest

@pytest.mark.parametrize("value, expected",[
	({(b'a',b'b'): 5, (b'c',b'd'): 2}, (b'a', b'b')),
	({(b'a',b'b'): 3, (b'c',b'd'): 3}, (b'c', b'd')),
])
def test_find_token_pair_to_merge(value, expected):
	assert find_token_pair_to_merge(value) == expected

@pytest.mark.parametrize("pretoken, merged, expected", [
	((b'a', b'b', b'c'),                 (b'a',b'b'),  (b'ab', b'c')),
	((b'a', b'b', b'c'),                 (b'b',b'c'),  (b'a', b'bc')),
	((b'a', b'b', b'c'),                 (b'x',b'y'),  (b'a', b'b', b'c')),
	((b'x', b'a', b'b', b'y', b'a', b'b'), (b'a',b'b'), (b'x', b'ab', b'y', b'ab')),
	((b'a', b'b'),                       (b'a',b'b'),  (b'ab',)),
	((b'a',),                            (b'a',b'b'),  (b'a',)),
	((b' t', b'h', b'e'),                (b' t',b'h'), (b' th', b'e')),
])
def test_apply_merged_token(pretoken, merged, expected):
	assert apply_merged_token(pretoken, merged) == expected



