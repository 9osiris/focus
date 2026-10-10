import json
import os

from .chunk import words


def home_dir():
    return os.environ.get("FOCUS_HOME",
                          os.path.join(os.path.expanduser("~"), ".focus"))


class Store:
    # tiny json store: seen counts per chunk fingerprint, term document
    # frequencies for idf, and the pinned set. all best-effort, a missing
    # or corrupt store just behaves like a fresh one.
    def __init__(self, path=None):
        self.path = path or os.path.join(home_dir(), "store.json")
        self.seen = {}
        self.df = {}
        self.docs = 0
        self.pins = []
        self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self.seen = data.get("seen", {}) or {}
        self.df = data.get("df", {}) or {}
        self.docs = data.get("docs", 0) or 0
        self.pins = data.get("pins", []) or []

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"seen": self.seen, "df": self.df,
                           "docs": self.docs, "pins": self.pins}, f)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def seen_count(self, fp):
        rec = self.seen.get(fp)
        return rec.get("count", 0) if isinstance(rec, dict) else 0

    def max_seen(self):
        best = 0
        for rec in self.seen.values():
            if isinstance(rec, dict):
                best = max(best, rec.get("count", 0))
        return best

    def observe(self, chunks, now):
        for c in chunks:
            rec = self.seen.get(c.id)
            if isinstance(rec, dict):
                rec["count"] = rec.get("count", 0) + 1
                rec["last"] = now
                # df/docs only count first sightings, re-observing the
                # same chunk should not inflate the corpus stats
                if c.id in self.pins:
                    c.pinned = True
                continue
            self.seen[c.id] = {"count": 1, "first": now, "last": now}
            for t in set(words(c.text)):
                self.df[t] = self.df.get(t, 0) + 1
            self.docs += 1
            if c.id in self.pins:
                c.pinned = True

    def pin(self, fp):
        if fp not in self.pins:
            self.pins.append(fp)
            return True
        return False

    def unpin(self, fp):
        if fp in self.pins:
            self.pins.remove(fp)
            return True
        return False

    def stats(self):
        return {"chunks_seen": len(self.seen), "docs": self.docs,
                "terms": len(self.df), "pins": len(self.pins),
                "path": self.path}
