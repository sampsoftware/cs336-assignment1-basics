import regex as re
import logging
from collections import Counter
import os
from multiprocessing import Pool
from functools import partial
from collections import defaultdict
import builtins

if not hasattr(builtins, "profile"):

    def profile(func):
        return func


logger = logging.getLogger(__name__)

MAX_CPU_ALLOCATION_PERCENT = 85
NUM_INITIAL_TOKENS = 256

## Read ahead this many bytes
MINI_CHUNK_SIZE = 4096

## GPT-2 Pretokenizer regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    input_path: str | os.PathLike,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[tuple[int, int]]:
    """
    Chunk the file into parts that can be counted independently. May return fewer chunks if the boundaries
    end up overlapping. Takes care to not split across defined documents. Modified from example to accept
    pathname string and internalize file management.

    Args:
        input_path: path to the input file
        desired_num_chunks: target number of chunks. May return fewer. Generally set to number of available processors.
        split_special_token: The token indicating document boundaries where split should occur.

    Returns:
        Pairs of integers marking the borders of chunks to be split.

    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    logger.debug("Chunking %s into %d parts on %s", input_path, desired_num_chunks, split_special_token)

    with open(input_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(MINI_CHUNK_SIZE)

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += MINI_CHUNK_SIZE

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    sorted_boundaries = sorted(set(chunk_boundaries))

    bounds = []
    for start, end in zip(sorted_boundaries[:-1], sorted_boundaries[1:]):
        bounds.append((start, end))

    return bounds


def pretokenize_chunk(input_path: str | os.PathLike, special_tokens: set, bounds: tuple[int, int]) -> dict:
    """
    Creates larger word-like tokens ("pretokens") and counts them.

    Args:
        input_path: location of the input file
        special_tokens: tokens to be ignored in the process
        bounds: byte count positions indicating beginning and end of a chunk

    Returns:
        A dict that maps a pretoken tuple to a count

    This function generally threaded; the caller is responsible for assembling pretokenized
    chunks into a coherent whole.
    """

    docs = []
    start, end = bounds
    logger.debug("Thread %d pretokenizing on range %d - %d", os.getpid(), start, end)

    ## Each thread gets its own file handler - safe because it is read-only
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Remove special_tokens - add them back in later. Docs becomes a list of text
    # segments that do not have any special tokens.
    docs.extend(list(filter(None, re.split("|".join(re.escape(t) for t in special_tokens), chunk))))
    logger.debug("Thread %d has %d docs", os.getpid(), len(docs))

    # Pre-tokenize the documents to begin finding token patterns. Pretokens are composed
    # of tuples of byte arrays.
    pretokens = Counter()
    for doc in docs:
        pretoken_matches = re.finditer(PAT, doc)
        for pretoken_match in pretoken_matches:
            pretoken_string = pretoken_match.group(0)
            pretoken_bytes = []
            for b in pretoken_string.encode("utf-8"):
                pretoken_bytes.append(bytes([b]))
            pretoken = tuple(pretoken_bytes)

            pretokens[pretoken] += 1

    logger.debug("Thread %d found %d pretokens", os.getpid(), len(pretokens))
    return pretokens


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


@profile
def merge_and_update_counts(
    pretokens: Counter[tuple[bytes, ...]],
    selected_token_pair: tuple[bytes, bytes],
    bpe_token_pair_counts: dict[tuple[bytes, bytes], int],
    bpe_token_pair_to_pretoken_index: dict[tuple[bytes, bytes], set[tuple[bytes, bytes]]],
):
    """
    Go through all of the pretokens and do any merges necessary. For pretokens that have merges,
    look through them and create new pairs of the new token plus the next token in the pretoken.
    Count them appropriately.

    Args:
        pretokens: All of the pretokens and their counts
        selected_token_pair: The pair of tokens that get merged this round
        bpe_token_pair_counts: The counts of all bpe token pairs. Maintain this with the effects of the merges.
        bpe_token_pair_to_ptk_index: A map of which bpe token pairs are in which pretokens, so the process can
            update only those pretokens. Also maintain this index when merging

    Returns:
        None. The relevant maps are mutated directly.

    """

    for pretoken in list(bpe_token_pair_to_pretoken_index[selected_token_pair]):
        n = pretokens[pretoken]
        new_pretoken = apply_merged_token(pretoken, selected_token_pair)

        if pretoken != new_pretoken:
            # Decrement/remove all counts from the old pretoken
            for t1, t2 in zip(pretoken[:-1], pretoken[1:]):
                new_bpe_token_pair = (bytes(t1), bytes(t2))
                new_count = bpe_token_pair_counts[new_bpe_token_pair] - n
                if new_count > 0:
                    bpe_token_pair_counts[new_bpe_token_pair] = new_count
                else:
                    bpe_token_pair_counts.pop(new_bpe_token_pair, 0)
            # Add all counts from the new pretoken
            for t1, t2 in zip(new_pretoken[:-1], new_pretoken[1:]):
                new_bpe_token_pair = (bytes(t1), bytes(t2))
                bpe_token_pair_counts[new_bpe_token_pair] = bpe_token_pair_counts[new_bpe_token_pair] + n

            # Remove the old pair-to-pretoken map
            for t1, t2 in zip(pretoken[:-1], pretoken[1:]):
                merged_bpe_pretoken_pair = (bytes(t1), bytes(t2))
                bpe_token_pair_to_pretoken_index[merged_bpe_pretoken_pair].discard(pretoken)

            # Add the new pair-to-pretoken map
            for t1, t2 in zip(new_pretoken[:-1], new_pretoken[1:]):
                new_bpe_token_pair = (bytes(t1), bytes(t2))
                bpe_token_pair_to_pretoken_index[new_bpe_token_pair].add(new_pretoken)

            pretokens.pop(pretoken, 0)
            assert new_pretoken not in pretokens
            pretokens[new_pretoken] = n


def determine_num_processes(input_path: str | os.PathLike) -> int:
    """
    Use as many processes as are available, but not more than the number of possible chunks to avoid
    excess process spawning overhead.

    Args:
        Input path of the file

    Returns:
        Optimum number of processes

    """
    num_available_cpus = int(os.cpu_count() or 1) * MAX_CPU_ALLOCATION_PERCENT // 100
    with open(input_path, "rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
    num_possible_chunks = file_size // MINI_CHUNK_SIZE
    num_processes = max(1, min(num_possible_chunks, num_available_cpus))

    logger.info(
        "Total CPUS %d, available CPUs %d, possible chunks %d, optimal processes %d",
        os.cpu_count(),
        num_available_cpus,
        num_possible_chunks,
        num_processes,
    )
    return num_processes


@profile
def build_merges(
    pretokens: dict[tuple[bytes, ...], int],
    num_merges: int,
) -> list[tuple[bytes, bytes]]:
    """
    Given the pretoken list, generate the list of merges

    Args:
        pretokens: The pretoken list
        num_merges: The number of merges to find

    Returns:
        A list of BPE tokens that are merged, in creation order
    """

    bpe_token_pair_counts = {}
    bpe_token_pair_to_pretoken_index = defaultdict(set)
    for pretoken, n in pretokens.items():
        for t1, t2 in zip(pretoken[:-1], pretoken[1:]):
            new_bpe_token_pair = (bytes(t1), bytes(t2))
            bpe_token_pair_counts[new_bpe_token_pair] = bpe_token_pair_counts.get(new_bpe_token_pair, 0) + n
            bpe_token_pair_to_pretoken_index[new_bpe_token_pair].add(pretoken)

    logger.debug("Found %d BPE pairs", len(bpe_token_pair_counts))

    # Generate one merged bpe token per loop
    merge_list = []
    for i in range(num_merges):
        # Select a token. It will be the one with the highest count and the greatest lexical value.
        max_count = max(bpe_token_pair_counts.values())
        most_frequent = [k for k, v in bpe_token_pair_counts.items() if v == max_count]
        selected_token_pair = sorted(most_frequent)[len(most_frequent) - 1]

        merge_list.append(selected_token_pair)

        # Get rid of the pair we just selected
        bpe_token_pair_counts.pop(selected_token_pair, 0)

        ### Merge the bpe tokens in each pretoken and update the bpe_pair counts
        merge_and_update_counts(pretokens, selected_token_pair, bpe_token_pair_counts, bpe_token_pair_to_pretoken_index)

        if i % 100 == 0:
            logger.debug("Vocab %d of %d", i, num_merges)

    return merge_list


def build_token_map(
    merge_list: list[tuple[bytes, bytes]], special_tokens: list[str]
) -> dict[int, bytes]:
    """
    Concatenate the base BPE tokens, the special tokens, and the discovered merge pairs into one token map

    Assumption: Initial BPE tokens are the utf-8 bytes 0..255

    Args:
        merge_list: List of pairs of BPE tokens that are merged
        special_tokens: List of byte arrays that are our special tokens

    Returns:
        BPE Token Map, fully built
    """

    bpe_token_map = {}
    num_special_tokens = len(special_tokens)

    for i in range(NUM_INITIAL_TOKENS):
        bpe_token_map[i] = bytes([i])

    for i, token in enumerate(special_tokens):
        bpe_token_map[i + NUM_INITIAL_TOKENS] = token.encode("utf-8")

    for i, token_pair in enumerate(merge_list):
        bpe_token_map[i + NUM_INITIAL_TOKENS + num_special_tokens] = token_pair[0] + token_pair[1]

    return bpe_token_map


@profile
def train_tokenizer(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], outfile=None
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE Tokenizer.

    Args:
        input_path: Where to get the file of text to train the tokenizer.
        vocab_size: Size of the ending vocabulary after all of the byte-pair merges. Includes 256 UTF-8
            bytes, the special tokens, and then all merges to bring up to vocab_size.
        special_tokens: A list of special tokens.

    Returns:
        The BPE Token map, mapping integer tokens to byte arrays.
        A list of all merges, in order.
    """

    logger.info("STARTING RUN")

    num_processes = determine_num_processes(input_path)
    bounds = find_chunk_boundaries(input_path, num_processes, b"<|endoftext|>")

    # Distribute bound pairs among an optimal number of threads
    pretokens = Counter()
    with Pool(num_processes) as pool:
        work = partial(pretokenize_chunk, input_path, special_tokens)
        pretoken_iter = pool.imap_unordered(work, bounds)
        for ptk in pretoken_iter:
            pretokens.update(ptk)
    logger.debug("Found %d pretokens", len(pretokens))

    merge_list = build_merges(pretokens, vocab_size - NUM_INITIAL_TOKENS - len(special_tokens))
    bpe_token_map = build_token_map(merge_list, special_tokens)

    assert len(bpe_token_map) == vocab_size, f"vocab_size={vocab_size} len(bpm)={len(bpe_token_map)}"
    logger.debug(
        "Vocab size %d, token map size %d, merges %d, num special tokens %d, num initial tokens %d",
        vocab_size,
        len(bpe_token_map),
        len(merge_list),
        len(special_tokens),
        NUM_INITIAL_TOKENS,
    )

    if outfile:
        with open(outfile, "w") as f:
            for i, bpe_token in bpe_token_map.items():
                f.write(f"{i:>8} {str(bpe_token)}\n")

    return bpe_token_map, merge_list
