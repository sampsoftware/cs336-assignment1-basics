import builtins
import regex as re

if not hasattr(builtins, "profile"):
    def profile(func):
        return func


## GPT-2 Pretokenizer regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_INITIAL_TOKENS = 256

def extend_pretoken_map(pretokens, doc):
    pretoken_matches = re.finditer(PAT, doc)
    for pretoken_match in pretoken_matches:
        pretoken_string = pretoken_match.group(0)
        pretoken_bytes = []
        for b in pretoken_string.encode("utf-8"):
            pretoken_bytes.append(bytes([b]))
        pretoken = tuple(pretoken_bytes)
        pretokens[pretoken] += 1


@profile
def apply_merged_token(pretoken: tuple[bytes, ...], selected_bpe_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """
    Given a token to merge, looks for adjacent tokens and replaces them with the merged token.

    Args:
        pretoken: The single pretoken, an array of BPE tokens
        selected_bpe_pair: The token pair to merge

    Returns:
        A newly merged pretoken
    """
    if len(pretoken) == 1:
        return pretoken

    new_pretoken = []
    new_bpe_token = selected_bpe_pair[0] + selected_bpe_pair[1]
    just_merged = False
    for k1, k2 in zip(pretoken[:-1], pretoken[1:]):
        if k1 == selected_bpe_pair[0] and k2 == selected_bpe_pair[1]:
            if just_merged:
                just_merged = False
            else:
                new_pretoken.append(new_bpe_token)
                just_merged = True
        else:
            if not just_merged:
                new_pretoken.append(k1)
            just_merged = False
    if not just_merged:
        new_pretoken.append(k2)

    return tuple(new_pretoken)
