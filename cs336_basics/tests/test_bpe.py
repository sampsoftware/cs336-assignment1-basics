from cs336_basics.bpe_tokenizer_trainer import apply_merged_token
from cs336_basics.bpe_tokenizer_trainer import merge_and_update_counts
import pytest


@pytest.mark.parametrize("pretoken, merged, expected", [
	((b'a', b'b', b'c'),                 (b'a',b'b'),  (b'ab', b'c')),
	((b'a', b'b', b'c'),                 (b'b',b'c'),  (b'a', b'bc')),
	((b'a', b'b', b'c'),                 (b'x',b'y'),  (b'a', b'b', b'c')),
	((b'x', b'a', b'b', b'y', b'a', b'b'), (b'a',b'b'), (b'x', b'ab', b'y', b'ab')),
	((b'a', b'b'),                       (b'a',b'b'),  (b'ab',)),
	((b'a',),                            (b'a',b'b'),  (b'a',)),
	((b' t', b'h', b'e'),                (b' t',b'h'), (b' th', b'e')),
	((b'a', b'a', b'a'),                (b'a',b'a'), (b'aa', b'a')),
])
def test_apply_merged_token(pretoken, merged, expected):
	assert apply_merged_token(pretoken, merged) == expected

@pytest.mark.parametrize(
	"pretokens, selected_token_pair, counts_in, expected_pretokens, expected_counts", [
  # 1. Plain merge in the middle; (a,b) hits zero and must be deleted.
  ({(b'x',b'a',b'b',b'y'):1}, (b'a',b'b'),
   {(b'x',b'a'):1, (b'a',b'b'):1, (b'b',b'y'):1},
   {(b'x',b'ab',b'y'):1}, {(b'x',b'ab'):1, (b'ab',b'y'):1}),

  # 2. Merge at the start.
  ({(b'a',b'b',b'y'):2}, (b'a',b'b'),
   {(b'a',b'b'):2, (b'b',b'y'):2},
   {(b'ab',b'y'):2}, {(b'ab',b'y'):2}),
 
  # 3. Merge at the end.
  ({(b'x',b'a',b'b'):3}, (b'a',b'b'),
   {(b'x',b'a'):3, (b'a',b'b'):3},
   {(b'x',b'ab'):3}, {(b'x',b'ab'):3}),
  
  # 4. Whole pretoken collapses to one token -> contributes no pairs at all.
  ({(b'a',b'b'):4}, (b'a',b'b'),
   {(b'a',b'b'):4},
   {(b'ab',):4}, {}),
 
  # 5. Adjacent merge sites, no separator. The middle (b,a) is the double-subtract trap.
  ({(b'a',b'b',b'a',b'b'):1}, (b'a',b'b'),
   {(b'a',b'b'):2, (b'b',b'a'):1},
   {(b'ab',b'ab'):1}, {(b'ab',b'ab'):1}),

  # 6. Self-overlap: (a,a) counts 2, only ONE merge happens, count still goes to 0.
  ({(b'a',b'a',b'a'):1}, (b'a',b'a'),
   {(b'a',b'a'):2},
   {(b'aa',b'a'):1}, {(b'aa',b'a'):1}),
 
  # 7. Other pretokens also contribute; subtract only this one's share, don't delete.
  ({(b'x',b'a',b'b',b'y'):1}, (b'a',b'b'),
   {(b'x',b'a'):10, (b'a',b'b'):10, (b'b',b'y'):10},
   {(b'x',b'ab',b'y'):1},
   {(b'x',b'a'):9, (b'a',b'b'):9, (b'b',b'y'):9, (b'x',b'ab'):1, (b'ab',b'y'):1}),

  # 8. Multibyte tokens.
  ({(b' t',b'h',b'e'):2}, (b' t',b'h'),
   {(b' t',b'h'):2, (b'h',b'e'):2},
   {(b' th',b'e'):2}, {(b' th',b'e'):2}),

  # 9. Two pretokens sharing the same pair. Tests cross-pretoken accumulation
  #    and that repeated decrements of one key land exactly on zero.
  ({(b'x',b'a',b'b'):2, (b'a',b'b',b'y'):3}, (b'a',b'b'),
   {(b'x',b'a'):2, (b'a',b'b'):5, (b'b',b'y'):3},
   {(b'x',b'ab'):2, (b'ab',b'y'):3},
   {(b'x',b'ab'):2, (b'ab',b'y'):3}),

  # 10. An UNCHANGED pretoken alongside a changed one. Its counts must be left alone.
  ({(b'a',b'b'):1, (b'c',b'd'):7}, (b'a',b'b'),
   {(b'a',b'b'):1, (b'c',b'd'):7},
   {(b'ab',):1, (b'c',b'd'):7},
   {(b'c',b'd'):7}),
])
def test_merge_and_update_counts(
	pretokens, selected_token_pair, counts_in, expected_pretokens, expected_counts
):
	assert merge_and_update_counts(
		pretokens, selected_token_pair, counts_in
	) == (expected_pretokens, expected_counts)
