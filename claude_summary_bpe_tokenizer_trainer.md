# BPE Tokenizer Trainer — Work Summary & Notes

_CS336 Assignment 1. A reflection on the process, the bugs, the lessons, and the
working patterns worth remembering. Intentionally contains **no solution code** —
it's a study aid, not an answer key._

---

## 1. What was built

A BPE (byte-pair encoding) tokenizer **trainer**: reads a text corpus, learns a
vocabulary of a target size, and emits `(vocab: dict[int, bytes], merges: list[pair])`.
The pipeline, end to end:

1. **Size the work** — pick a process count from file size and CPU budget.
2. **Chunk** the file on document boundaries (special-token delimited) so chunks
   can be counted independently.
3. **Pretokenize in parallel** — GPT-2 regex splits text into "pretokens"
   (`tuple[bytes, ...]`), counted by frequency, across a process `Pool`.
4. **Build merges** — repeatedly select the most frequent adjacent BPE-token pair
   (lexicographic tie-break), merge it everywhere, and maintain the counts
   incrementally.
5. **Build the vocab map** — 256 base bytes + special tokens + one entry per merge.

It passes the course correctness tests and the 1.5s speed budget, and is clean under
`ruff` and `ty`.

---

## 2. The technical arc (what actually happened)

Roughly in order, because the order is the lesson:

- **Got correctness first** with a straightforward but O(n²)-shaped approach: each
  merge round scanned *every* pretoken.
- **Fixed the incremental pair-count maintenance** — the subtle part. Settled on the
  robust framing: for any pretoken that changes, **subtract all its old adjacent
  pairs × freq, add all its new adjacent pairs × freq.** Avoided fragile
  per-merge-site delta arithmetic.
- **Handled the merge overlap edge case** in `apply_merged_token` (the `(a,a,a)`
  problem — see §4).
- **Extracted a testable `merge_and_update_counts`** and wrote a parametrized test
  suite (~10 rows) covering the trap cases. This is what made the later
  optimization safe.
- **Profiled** with `cProfile` → `snakeviz` → `line_profiler`, and discovered the
  real cost: `apply_merged_token` was called **3.5M times**, ~4,760 per round, almost
  all **no-ops** (the selected pair wasn't even in that pretoken).
- **Added a reverse index** (`pair → set of pretokens containing it`) so each round
  touches only affected pretokens. This is the big win. It also introduced a cluster
  of sync bugs (§4).
- **Micro-optimized** the inner loop (removed a throwaway tuple allocation, guarded a
  per-round debug log) after `line_profiler` pointed at the exact lines.
- **Refactored** into named functions (`determine_num_processes`, `build_merges`,
  `build_token_map`, …) so `train_tokenizer` reads as an orchestrator.
- **Cleaned up** with `ruff` (lint + format) and `ty` (types), fixing a pile of
  annotations that had drifted out of sync with the code — plus two *real* latent
  bugs `ty` surfaced (§4).

---

## 3. What was learned (concepts worth keeping)

**BPE mechanics**
- Pretokens are `tuple[bytes, ...]`; counts are weighted by pretoken frequency.
- Merge selection = highest count, tie broken by taking the lexicographically
  greatest pair. Ordering of merges is part of the graded output.
- `num_merges = vocab_size − 256 − len(special_tokens)`. Total vocab must be exactly
  `vocab_size`. (Off-by-this-formula = wrong vocab size; see §4.)
- Special tokens are encoded whole and never split or merged.

**Incremental state maintenance (the recurring hard part)**
- "Subtract-all-old / add-all-new" beats clever per-site deltas — easier to prove
  correct, harder to get subtly wrong.
- **Self-overlap** matters: `(a,a,a)` merging `(a,a)` produces `(aa,a)` with only ONE
  merge, even though the pair `(a,a)` counted 2. Any "decrement by number of pair
  occurrences" scheme dies here.
- When you keep a derived index alongside primary state, **it's a second source of
  truth that must stay in lockstep** — every place you update counts, you update the
  index, over the same pairs.

**The reverse-index invariants (learned the hard way)**
- The index tells you *which* items to process; it must never become the *only* items
  that exist. (Bug: rebuilding the pretoken dict from only the affected set dropped
  every untouched pretoken.)
- **Snapshot before mutating** what you iterate (`list(...)`), or you get
  `RuntimeError: Set changed size during iteration`.
- **`dict.copy()` is shallow** — nested sets are shared references. "new_" naming is a
  lie if the inner containers are the same objects.
- Store what you can't cheaply recompute (`pair → pretokens`); recompute what you can
  (a pretoken already encodes its own pairs via `zip`). Don't build a second index for
  data you already hold.

**Python idioms / footguns collected**
- `Counter` returns `0` for missing keys and `.update` merges counts — but it *keeps*
  zero/negative entries, so it's wrong when you need count-and-**prune** (use a plain
  dict there). Count-and-keep → `Counter`; count-and-prune → `dict`.
- Tuple elements that are already `bytes` don't need `bytes(x)` — that's a redundant
  copy in a hot loop.
- `x = list[bytes]` binds the *type object*, not an empty list. `:` declares a type,
  `=` binds a value.
- `assert expr, msg` — the comma makes the second operand the **message**, evaluated
  lazily only on failure. Great for "log detail on failed assert" without an `if`.
  Also the footgun that made a whole test file "fail" once (compared a tuple against a
  message).
- `defaultdict(default_factory)` auto-creates on **access** — merely reading a missing
  key inserts it. Can silently re-leak keys you meant to prune.
- `open(path, 'w')` truncates on *open*, before any write. `'a'` appends, `'x'` fails
  if exists.
- f-string field width right-aligns numbers by default: `f"{n:>{w}}"`.

**Performance mental model**
- **Profile before optimizing.** The 3.5M-calls finding was invisible until measured;
  it reframed the whole problem from "make the function faster" to "stop calling it."
- In pure-Python hot loops, **object allocation is the tax** — building throwaway
  tuples/slices to compare or iterate dominates over the "real" work.
- The real order-of-magnitude lever is usually a **representation change** (here:
  integer token IDs + array/`numpy` ops), not "same algorithm in C." That's also the
  gateway to Cython/Rust if wanted.
- **Know the floor.** After overhead is stripped, what remains is the necessary work,
  and no amount of cleverness gets blood from a stone. Stopping there is judgment, not
  giving up.

**Tooling**
- `cProfile` (function-granular; blind to worker processes and to inline code inside
  one function) → `snakeviz` (icicle: width = cumtime, siblings partition the parent,
  gaps under a box = self time) → `line_profiler`/`kernprof` (per-line; the right tool
  once the hotspot is inside a single function).
- `ruff check` / `ruff format` for lint+style; `ty check` for types. They're the
  *static* analog of the profiler — they found the drifted annotations mechanically
  and caught two real latent bugs for free.

---

## 4. Areas that were troublesome (and how they resolved)

Ranked by how much time they ate:

1. **Incremental pair-count correctness.** Stale leaked loop variables and
   element-vs-pair key confusion produced impossible counts. Fixed by committing to
   the subtract-old/add-new framing and unit-testing it in isolation.
2. **The reverse-index sync bugs — four distinct ones, in sequence:**
   - Maintained the index under **single-token keys** while building/reading it under
     **pair keys** → the read keys were never updated → stale → wrong merges.
   - A **copy-paste** left a `.discard` where an `.add` belonged (and referenced the
     wrong pretoken).
   - **Iterated a set while mutating it** → `RuntimeError`. Needed a snapshot.
   - **Dropped all unaffected pretokens** by rebuilding the dict from only the
     affected set → counts drifted; merges 0–2 right, diverged at merge 3.
   Lesson: derived indexes are where the bugs live; each fix exposed the next.
3. **`apply_merged_token` overlap flag.** A `just_merged` flag rewrite dropped a token
   on `(a,a,a)`. The original guard was doing *two* jobs (suppress double-emit AND
   prevent overlapping re-merge); I'd wrongly called it dead weight.
4. **Annotations that lie.** A recurring theme: `-> list[set]` returning `list[int]`,
   `-> tuple[...]` on a function that returns `None`, an index typed
   `set[bytes]`/`tuple[bytes,bytes]` when it held `set[tuple[bytes, ...]]`. Each wrong
   annotation caused a *downstream* `ty` error far from the actual mistake. Wrong
   annotations are worse than none.
5. **Refactor-without-retest regression.** Extracting `build_merges` silently dropped
   the `− 256 − len(special_tokens)` from the merge count, over-generating the vocab —
   and the assertion had been "fixed" to match the bug. `ruff` was green; only
   re-running pytest would have caught it.
6. **Two real bugs `ty` found (not cosmetic):** `int(os.cpu_count())` crashes when
   `cpu_count()` returns `None`; `argparse type=list[bytes]` misuses the `type=`
   callable contract.

---

## 5. Notes-to-self — how I prefer to work

- **Profiler-first, then structure, then lint/types.** Measure, optimize the real
  hotspot, refactor for clarity, finish with `ruff`/`ty`. This order worked well.
- **Let me puzzle it out.** I do better with a guiding question and a pointer to the
  suspect line than with a handed answer. I pushed back (correctly) when over-lectured
  or over-flagged; keep feedback terse and specific.
- **"Review my code and the log"** iterative cycles were productive — small change,
  fresh log, re-review.
- **I inline deliberately for speed** and accept the modularity cost. When I do, leave
  a one-line note saying it's profiler-driven, so it reads as intent not sloppiness.
- **Re-run pytest after any refactor**, even a "pure" extraction. `ruff` green ≠
  correct. This is my personal failure mode to guard.
- **Fix annotations to match tested behavior**, not the other way round — the runtime
  was correct and tested; the labels were the liars.
- I have real perf-engineering instincts (PHP→LLVM background): I recognize the floor
  and stop. Trust that.

---

## 6. For the educator — patterns to watch for recurrence

These are the transferable threads most likely to resurface in later CS336 work
(transformers, optimizers, training loops, Triton kernels, distributed training,
scaling laws). Worth flagging if they show up again:

- **Incremental / cached state maintenance.** The single richest bug source here.
  Recurs anywhere state is updated in place rather than recomputed: KV caches,
  optimizer moments (Adam m/v), running statistics, gradient accumulation,
  distributed all-reduce bookkeeping. Same failure shape: a derived structure drifts
  out of sync with primary state. Watch for "update it in the same place you update
  the thing it mirrors."

- **Iterate-vs-mutate and aliasing.** Snapshotting before mutation, and shallow-copy
  sharing nested containers, will bite again with tensor views, in-place ops
  (`x += ...` vs `x = x + ...`), and anything touching `.data`/`.detach()`.

- **Representation as the performance lever.** The "integer IDs + array ops beats
  Python loops" insight is the *same* idea as "vectorize with tensors instead of
  Python loops" — foreshadows the whole PyTorch/Triple-nested-loop-to-kernel arc.
  If the student reaches for a Python loop over elements in a later assignment, this
  is the moment to recall.

- **Profile-first discipline.** They internalized "measure before optimizing" and
  "know when you've hit the floor." Check that this holds under time pressure in
  later, larger problems (it's easy to abandon).

- **Annotations/types as documentation that must not lie.** The "wrong annotation
  misleads the reader" theme will recur with tensor shapes and dtypes — the ML analog
  of this exact bug is a comment/annotation claiming a shape the tensor doesn't have.
  `jaxtyping`-style shape annotations are the same discipline.

- **Refactor-without-retest.** A concrete, observed regression. Worth watching whether
  the habit of re-running the full test suite after restructuring sticks.

- **The inline-for-speed vs. modularity tension.** Will return sharply in kernel/Triple
  work where fusing operations (the inlining analog) trades readability for speed. The
  student is comfortable making that trade consciously — good, but it's worth
  confirming they still *document* the intent.

- **Working style:** responds best to Socratic guidance and pointed code review, not
  handed solutions; experienced enough to self-diagnose given the right nudge. Terse,
  specific feedback lands; over-explaining and over-flagging do not.

---

## 7. Session 2 — Streaming trainer + Tokenizer (encode/decode): RESUME STATUS

_Picking the work back up starts here. This section is the live to-do, not a reflection._

### What got done this session

**Trainer — streaming pretokenizer rewrite (done, verified).**
- Rewrote `pretokenize_chunk` to **stream** the file in blocks (`MINI_CHUNK_SIZE * 16`)
  instead of `f.read(end - start)` whole-chunk. Motivation: OWT training OOM'd at
  ~61 GiB against the old 62 GiB WSL2 ceiling.
- Core loop: `data = carry + f.read(min(block, end - cursor))`, split on special-token
  **bytes** pattern (`b"|".join(re.escape(t) ...)`), emit all-but-last segment, **carry
  the last** segment forward (handles both partial-token and partial-doc at block seams).
  Decode **per segment** (`errors="ignore"`), not per block — keeps UTF-8 chars whole
  across seams. `else` branch (no special tokens) carries `data` whole.
- Special tokens now stored as **`list[bytes]`**, sorted length-desc (longest-match).
- **Verified behavior-preserving:** new OWT vocab/merges are **byte-identical** to the
  pre-streaming `.bk` backups (`diff` clean). Peak memory dropped from ~61 → ~24 GiB.

**Infra fixes (all real, all landed):**
- WSL2 memory: was 62 GiB (default 50% of 128 GiB host). Raised to 94 via
  `%UserProfile%\.wslconfig` (`memory=96GB`, `swap=32GB`, `[experimental]
  autoMemoryReclaim=gradual`). `wsl --shutdown` to apply (kills container).
- cgroup memory logging: env is **cgroup v2** now (was v1 pre-reboot). `util.py`
  resolves `/proc/self/cgroup` → `.../memory.current`, wrapped fail-soft (a telemetry
  read must NEVER crash the run — it did twice: EROFS write, then v1-path FileNotFound).
- `util.py` bug fixed: `FileHandler("train.log")` was created but never `addHandler`ed —
  logs weren't being written. Also filter belongs on **handlers**, not the root logger.
- Trainer bug fixed: `merge_and_update_counts` double-removed the selected pair (explicit
  `pop` + subtract-loop). Removed the pop; added a **post-merge assert** that the pair is
  gone (correct as post-condition, was mis-placed as pre-condition first).

**Training runs complete & saved (with `--outfile` this time):**
- OWT 32k and TinyStories 10k tokenizers trained, base64-serialized, verified.
- Stats generated. **Longest OWT token = 64 B of mojibake** (`ÃÂÃÂ…`, double-encoded
  UTF-8); other long tokens are ASCII-art rules (`----`, `====`). TinyStories longest =
  15 B real words (` accomplishment`). Histogram saved to scratchpad. This is the answer
  to `train_bpe_expts_owt` 2a/2b + `tokenizer_experiments` 2b.

**Tokenizer class (`tokenizer.py`) — encode/decode:**
- `__init__` — DONE. Copies vocab/merges (`dict(...)`/`list(...)`), builds `_special_token_bytes`
  (sorted len-desc), `_merge_ranks` (`pair→rank`, O(1)), `_token_to_id` (inverse of vocab).
  Appends special tokens to vocab if absent (`max(self.vocab)+1`). Uniqueness assert.
- `from_files` — DONE. classmethod, parses base64 vocab/merges, `return cls(...)`.
- `decode` — DONE: `b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")`.
- `encode` — **DONE and hand-traced** against §2.6 oracle → `[9,7,1,5,10,3]`. Structure:
  capturing-group `re.split` on specials (keeps them) → walk in order → special piece =
  `_token_to_id[piece.encode("utf-8")]`; text piece = pretokenize (ordered list, dupes
  kept — NOT a Counter) → per-pretoken rank-merge loop (`min(possible_merges,
  key=self._merge_ranks.__getitem__)` + `bpe.apply_merged_token`) → per-token
  `_token_to_id` lookup.

### IMMEDIATE next steps (in order)

1. **Add the empty-special-tokens guard in `encode`.** If `self.special_tokens` is empty,
   the split pattern becomes `"()"` and `re.split` shreds text into single chars →
   corrupts pretokenization. Guard: no specials → `docs_and_sts = [text]` (skip split).
   **`test_encode` uses no special tokens, so it FAILS without this guard.**
2. **Uncomment `assert t.encode(text) == encoded`** in `test_encode` (§2.6 oracle). Run it.
3. **`encode_iterable`** — still a stub (`pass`). It's the lazy wrapper:
   `for chunk in iterable: yield from self.encode(chunk)`. No byte-chunking needed —
   the iterable (file handle) already yields lines.
4. **Extract `bpe.py`** — shared module for `PAT` + `apply_merged_token` (currently
   `encode` calls `bpe.apply_merged_token`/`bpe.PAT` but they still live in the trainer).
   Decided naming: **`bpe.py`** (primitives, leaf) / **`tokenizer.py`** (class) /
   rename `bpe_tokenizer_trainer.py` → **`tokenizer_trainer.py`**. Update `cli.py`,
   `pyproject` `[project.scripts]`, imports. Both trainer and tokenizer import from `bpe`
   (acyclic). `bpe.py` should expose an encode-appropriate `pretokenize` that returns an
   **ordered list** (trainer `Counter`s it; encoder uses it directly).
5. **Two high-value tests:** (a) special-token-in-text (`"a<|endoftext|>b"` → special id
   at the right position — currently the special branch is untested), (b) **round-trip**
   `decode(encode(s)) == s` for strings incl. special token, non-ASCII/emoji, whitespace
   runs.

### Landmines / notes for resumption
- Recurring typo: `__get_item__` → **`__getitem__`** (bit us 3×). Consider
  `key=lambda p: self._merge_ranks[p]` to sidestep it (possible_merges is pre-filtered).
- `encode` needs **ordered pretokens with duplicates** — do NOT reuse the trainer's
  Counter-based `extend_pretoken_map` (dedups, loses order). Pretokenize inline / via a
  list-returning helper.
- `test_bpe.py` `test_merge_and_update_counts` was updated for the 4-arg mutate-in-place
  signature (builds a `defaultdict(set)` index, asserts mutated dicts). Passing 18/18.
- Then the assignment moves to **Section 3: Transformer LM** (linear, embedding, rmsnorm,
  rope, softmax, attention, block, LM). Did a lot of conceptual groundwork already:
  input `(batch, seq)` → output `(batch, seq, vocab)`; per-position next-token prediction;
  causal mask enables parallel training; cross-entropy = -log(p_true) = bits-to-encode
  reality; LM = compressor.
