import os
import regex as re
import logging
from collections import Counter
from multiprocessing import Pool
from functools import partial
from cs336_basics import config

logger = logging.getLogger(__name__)

## GPT-2 Pretokenizer regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

## Capping CPU use at 85% of cpu's to leave some capacity for other tasks
num_processes = os.cpu_count() * 85 // 100


def find_chunk_boundaries(
    input_path: str,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[set]:
    """
    Chunk the file into parts that can be counted independently. May return fewer chunks if the boundaries 
    end up overlapping. Takes care to not split across defined documents. Modified from example to accept 
    pathname string and internalize file management.

    Args:
        pathname: path to the input file
        desired_num_chunks: target number of chunks. May return fewer. Generally set to number of available processors.
        split_special_token: The token indicating document boundaries where split should occur.

    Returns:
        Pairs of integers marking the borders of chunks to be split.

    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    logger.debug("Chunking %s into %d parts on %s",input_path, desired_num_chunks, split_special_token)

    with open(input_path,"rb") as file:
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def pretokenize_chunk(
    input_path: str,
    special_tokens: set, 
    bounds: ()
) -> list[dict]:
    """
    Creates larger word-like tokens ("pretokens") and counts them.

    Args:
        input_path: location of the input file
        special_tokens: tokens to be ignored in the process
        bounds: byte count positions indicating beginning and end of a chunk

    This function generally threaded; the caller is responsible for assembling pretokenized
    chunks into a coherent whole.
    """

    docs = []
    start, end = bounds
    logger.debug("Thread %d pretokenizing on range %d - %d",os.getpid(), start, end)


    ## Each thread gets its own file handler - safe because it is read-only 
    with open(input_path,"rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    ### 
    # REMOVE special_tokens - add them back in later. Docs becomes a list of text
    # segments that do not have any special tokens.
    logger.debug("Thread %d removing special tokens",os.getpid())
    docs.extend(
        list(
            filter(
                None,
                re.split(
                    "|".join(
                        re.escape(t) for t in special_tokens
                        )
                    , chunk
                    )
                )
            )
        )
    logger.debug("Thread %d has %d docs",os.getpid(), len(docs))

    ###
    # Pre-tokenize the documents to begin finding token patterns. Here we discard
    # the text and keep only the pretokens and thier counts. Pretokens are composed
    # of tuples of byte arrays. In this step they are all len()=1 but that will change
    # as we find adjacent tokens to merge.
    pretokens = {}
    for doc in docs:
        token_matches = re.finditer(PAT, doc)
        for tm in token_matches:
            tks = tm.group(0)
            tkba = []
            for b in tks.encode('utf-8'):
                tkba.append(bytes([b]))
            tkt = tuple(tkba)

            pretokens[tkt] = pretokens.get(tkt, 0) + 1

    logger.debug("Thread %d found %d pretokens",os.getpid(),len(pretokens))

    return pretokens


def apply_merged_token(
    ptk: tuple[bytes, ...],
    stp: tuple[bytes,bytes]
) -> dict[tuple[bytes,bytes], int]:
    """
    Given a token to merge, looks for adjacent tokens and replaces them with the merged token.

    Args:
        ptk: The single pretoken, an array of BPE tokens
        stp: selected_token_pair, The token pair to merge

    Return:
        A newly merged pretoken
    """
    if len(ptk) == 1:
        return ptk

    ntk = []
    new_bpe_token = stp[0]+stp[1]
    just_merged = False
    for k1, k2 in zip(ptk[:-1],ptk[1:]):
        test_token = (k1,k2)
        if test_token == stp:
            if not just_merged:
                ntk.append(new_bpe_token)
            just_merged = True
        else:
            if not just_merged:
                ntk.append(k1)
            just_merged = False
    if not just_merged:
        ntk.append(k2)


    # k1 = b''                            # Use k1 = b'' as a flag to know that a token was not just merged
    # ntk = []                            # This is the new (pre)token key
    # merged_token = stp[0]+stp[1]        # This is the new bpe token as opposed to the token pair
    # i = 1                               # The loop index. Since I increment it monotonically, I can replace with a for loop?
    # while i < len(ptk):                 # Can I use a for loop?
    #     k2 = ptk[i]                     # Starting at the second element, so the second key is the nth element
    #     if k1 == b'':                   # If we merged last time, this will be the merged pretoken and we skip over
    #         k1 = ptk[i-1]               # If our flag indicates we did not merge, new first token is the actual token
    #     test_token = (k1,k2)            # This is a paired token not a merged token

    #     if test_token == stp:           # If we found a merge...
    #         k1 = merged_token           # ...compare it next time
    #         ntk.append(k1)              # And that's the BPE token to put in the pretoken
    #     else:
    #         if k1 != merged_token:      # Since we used k1 first as a bpe token and then as a flag, we skip counting it
    #             ntk.append(k1)          # the second time through.
    #         k1 = b''                    # And reset the flag
    #     i += 1                          # And iterate

    # if k1 == b'':                       # if the very last iteration was not a merge
    #     ntk.append(k2)                  # Keep k2

    new_ptk = tuple(ntk)

    return new_ptk


def train_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[
        dict[int, bytes], 
        list[tuple[bytes, bytes]]
    ]:

    logger.info("STARTING RUN")
    logger.info("CPU count=%d, using %d processes", os.cpu_count(), num_processes)
    input_path = config.get_data_dir(1) + input_path

    pretokens = Counter()

    ###############
    # Review the input file and find safe boundaries upon which to chunk the text.
    boundaries = find_chunk_boundaries(input_path, num_processes, b'<|endoftext|>')
    bounds = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        bounds.append((start,end))
    ## Now we have the boundaries
    ###############

    ###############
    # Split the chunks among CPU threads
    with Pool(num_processes) as pool:
        work = partial(pretokenize_chunk, input_path, special_tokens)
        pretoken_iter = pool.imap_unordered(work, bounds)
        for ptk in pretoken_iter:
            pretokens.update(ptk)
    logger.debug("Found %d pretokens",len(pretokens))
    # Now we have all of the wordlike pretokens
    ##############

    ##############
    # Assemble an index of paired BPE tokens and their frequencies
    bpe_token_pair_counts = {}
    for ptk, n in pretokens.items():
        for t1, t2 in zip(ptk[:-1], ptk[1:]):
            paired_token = (bytes(t1),bytes(t2))
            bpe_token_pair_counts[paired_token] = bpe_token_pair_counts.get(paired_token,0) + n
    # Now we have the bpe pairs and their initial frequencies

    ##############
    # Loop to generate merges, one merged bpe token per loop
    selected_tokens = []
    for i in range(20):
        ###
        # Select a token. It will be the one with the highest count and the greatest lexical value.        
        max_count = max(bpe_token_pair_counts.values())
        most_frequent = [k for k,v in bpe_token_pair_counts.items() if v == max_count]
        selected_token_pair = sorted(most_frequent)[len(most_frequent)-1]
        selected_tokens.append(selected_token_pair)
        bpe_token_pair_counts.pop(selected_token_pair,0)

        logger.debug("Max count %d with %d most frequent, selected %s, %d bpe pairs exist.",
            max_count, len(most_frequent), selected_token_pair, len(bpe_token_pair_counts)
        )
        ## Now we have the token to merge this round. And, it is permanently gone from the count list.

        ###
        # Go through all of the pretokens and do any merges necessary. For pretokens that have merges,
        # look through them and create new pairs of the new token plus the next token in the pretoken.
        # Count them appropriately.
        new_pretokens = {}
        for ptk, n in pretokens.items():
            nptk = apply_merged_token(ptk, selected_token_pair)

            if (ptk != nptk):
                ## Decrement/remove all counts from the old pretoken
                for t1, t2 in zip(ptk[:-1], ptk[1:]):
                    new_pair = (bytes(t1),bytes(t2))
                    new_count = bpe_token_pair_counts.get(new_pair,0) - n
                    if new_count > 0:
                        bpe_token_pair_counts[new_pair] = new_count
                    else:
                        bpe_token_pair_counts.pop(new_pair,0)
                ## Add all counts from the new pretoken
                for t1, t2 in zip(nptk[:-1], nptk[1:]):
                    new_pair = (bytes(t1),bytes(t2))
                    bpe_token_pair_counts[new_pair] = bpe_token_pair_counts.get(new_pair,0) + n
            if nptk in new_pretokens:
                logger.debug("Hmm")
            assert nptk not in new_pretokens
            new_pretokens[nptk] = n

        ## ..and swap the new list into the current list's place
        pretokens = new_pretokens
