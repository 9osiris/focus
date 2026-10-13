from .chunk import words
from .tokens import Tokenizer

SHINGLE_K = 3
DEDUP_JACCARD = 0.8


def shingles(text, k=SHINGLE_K):
    ws = words(text)
    if len(ws) < k:
        return {tuple(ws)} if ws else set()
    return {tuple(ws[i:i + k]) for i in range(len(ws) - k + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def chunk_sim(a, b):
    return jaccard(shingles(a.text), shingles(b.text))


def dedup(scored, threshold=DEDUP_JACCARD):
    # scored: list of (chunk, score, factors); drops near-duplicates,
    # keeping the higher score. returns (kept, dropped_with_reason)
    ordered = sorted(scored, key=lambda s: s[1], reverse=True)
    kept = []
    dropped = []
    for chunk, score, factors in ordered:
        dup_of = None
        for kchunk, _, _ in kept:
            if chunk_sim(chunk, kchunk) >= threshold:
                dup_of = kchunk
                break
        if dup_of is None:
            kept.append((chunk, score, factors))
        else:
            dropped.append((chunk, score, factors,
                            "dup of %s (jaccard %.2f)" %
                            (dup_of.id, chunk_sim(chunk, dup_of))))
    return kept, dropped


def select(scored, budget_tokens, mmr_lambda=0.5):
    # scored: list of (chunk, score, factors)
    # returns (selected, excluded) where excluded items carry a reason.
    # pins go first no matter what; the rest are picked greedily by
    # score-per-token with an mmr penalty for looking like something
    # already selected.
    selected = []
    excluded = []
    used = 0

    pins = [(c, s, f) for c, s, f in scored if c.pinned]
    rest = [(c, s, f) for c, s, f in scored if not c.pinned]

    for chunk, score, factors in pins:
        selected.append((chunk, score, factors, "pinned"))
        used += chunk.tokens

    def density(item):
        chunk, score, _ = item
        return score / max(1, chunk.tokens)

    rest = sorted(rest, key=density, reverse=True)
    picked = [c for c, _, _, _ in selected]

    while rest:
        best_idx, best_eff = None, None
        for i, (chunk, score, factors) in enumerate(rest):
            overlap = 0.0
            for p in picked:
                overlap = max(overlap, chunk_sim(chunk, p))
            eff = density((chunk, score, factors)) - mmr_lambda * overlap
            if best_eff is None or eff > best_eff:
                best_eff, best_idx = eff, i
        chunk, score, factors = rest.pop(best_idx)
        if used + chunk.tokens <= budget_tokens:
            selected.append((chunk, score, factors,
                             "score %.3f" % score))
            picked.append(chunk)
            used += chunk.tokens
        else:
            excluded.append((chunk, score, factors,
                             "over budget (%d tokens left)" %
                             (budget_tokens - used)))

    return selected, excluded


def compress_tail(selected, budget_tokens, tokenizer=None,
                  snippet_lines=3):
    # if the picks blow the budget, shrink the lowest-scoring non-pinned
    # picks to snippets until it fits. pins are sacred, but if pins alone
    # exceed the budget they get truncated too, last resort, never dropped.
    # snippet token counts use the same tokenizer as the chunks.
    tok = tokenizer if tokenizer is not None else Tokenizer()

    def shrink(i):
        chunk, score, factors, reason = selected[i]
        short = chunk.snippet(snippet_lines)
        chunk.text = short
        chunk.tokens = tok.count(short)
        return "truncated %s to snippet (%d tokens)" % (chunk.id,
                                                         chunk.tokens)

    used = sum(c.tokens for c, _, _, _ in selected)
    notes = []
    if used <= budget_tokens:
        return selected, notes
    order = sorted(range(len(selected)),
                   key=lambda i: selected[i][1])
    for i in order:
        if selected[i][0].pinned:
            continue
        notes.append(shrink(i))
        used = sum(c.tokens for c, _, _, _ in selected)
        if used <= budget_tokens:
            return selected, notes
    for i in order:
        if not selected[i][0].pinned:
            continue
        notes.append(shrink(i) + " (pinned, last resort)")
        used = sum(c.tokens for c, _, _, _ in selected)
        if used <= budget_tokens:
            break
    return selected, notes
