import math
import time

from .chunk import words

DEFAULT_WEIGHTS = {
    "recency": 0.30,
    "relevance": 0.40,
    "novelty": 0.15,
    "frequency": 0.15,
}


def parse_weights(spec):
    # "recency=0.2,relevance=0.5" -> dict, unknown keys ignored
    out = dict(DEFAULT_WEIGHTS)
    if not spec:
        return out
    for part in spec.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip().lower()
        if key in out:
            try:
                out[key] = float(val)
            except ValueError:
                pass
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


class Scorer:
    def __init__(self, query, store=None, weights=None,
                 half_life_h=24.0, now=None):
        self.query_terms = words(query)
        self.store = store
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self.half_life = max(1e-6, half_life_h)
        self.now = now if now is not None else time.time()
        # idf falls back to the candidate set when no store stats exist
        self._cand_df = {}
        self._cand_n = 0

    def learn_candidates(self, chunks):
        df = {}
        for c in chunks:
            for t in set(words(c.text)):
                df[t] = df.get(t, 0) + 1
        self._cand_df = df
        self._cand_n = len(chunks)

    def idf(self, term):
        # store stats win when present, else candidate-local df
        if self.store is not None:
            df, n = self.store.df.get(term, 0), self.store.docs
            if n > 0 and df > 0:
                return math.log(1 + n / df)
        df, n = self._cand_df.get(term, 0), self._cand_n
        if n > 0 and df > 0:
            return math.log(1 + n / df)
        return 1.0

    def recency(self, chunk):
        age_h = max(0.0, (self.now - chunk.mtime) / 3600.0)
        return 0.5 ** (age_h / self.half_life)

    def relevance(self, chunk):
        if not self.query_terms:
            return 0.0
        text_terms = words(chunk.text)
        if not text_terms:
            return 0.0
        tf = {}
        for t in text_terms:
            tf[t] = tf.get(t, 0) + 1
        total = 0.0
        for t in set(self.query_terms):
            if t in tf:
                total += self.idf(t) * (tf[t] / len(text_terms))
        # normalize by the best possible hit for this query
        best = sum(self.idf(t) for t in set(self.query_terms))
        return min(1.0, total / best) if best > 0 else 0.0

    def frequency(self, chunk):
        if self.store is None:
            return 0.0
        count = self.store.seen_count(chunk.id)
        if count <= 0:
            return 0.0
        cap = self.store.max_seen()
        if cap <= 1:
            return 1.0
        return math.log1p(count) / math.log1p(cap)

    def factors(self, chunk):
        return {
            "recency": round(self.recency(chunk), 4),
            "relevance": round(self.relevance(chunk), 4),
            "frequency": round(self.frequency(chunk), 4),
            # novelty is computed against the selected set during
            # budget allocation (mmr), so it is not a static factor
            "novelty": None,
        }

    def static_score(self, chunk):
        f = self.factors(chunk)
        return sum(self.weights[k] * f[k]
                   for k in ("recency", "relevance", "frequency"))
