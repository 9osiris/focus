import hashlib
import json
import os
import re
import sys

# words are lowercase alphanumerics, good enough for overlap math
WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return WORD.findall(text.lower())


def estimate_tokens(text):
    # rough 4 chars per token, same rule ctxpack uses
    return max(1, len(text) // 4)


def fingerprint(source, offset, text):
    h = hashlib.sha1()
    h.update(source.encode("utf-8"))
    h.update(b"\0")
    h.update(str(offset).encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


class Chunk:
    __slots__ = ("id", "source", "text", "tokens", "mtime", "pinned", "meta")

    def __init__(self, source, text, mtime=0.0, offset=0, pinned=False, meta=None):
        self.source = source
        self.text = text
        self.tokens = estimate_tokens(text)
        self.mtime = mtime
        self.pinned = pinned
        self.meta = meta or {}
        self.id = fingerprint(source, offset, text)

    def snippet(self, max_lines=3):
        lines = self.text.splitlines()
        if len(lines) <= max_lines:
            return self.text
        return "\n".join(lines[:max_lines]) + "\n[...]"


def chunk_text(source, text, mtime=0.0, max_lines=40, overlap=8):
    lines = text.splitlines()
    if not lines:
        return []
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        piece = "\n".join(lines[start:end]).strip()
        if piece:
            chunks.append(Chunk(source, piece, mtime=mtime, offset=start))
        if end == len(lines):
            break
        start = end - overlap
    return chunks


def looks_binary(path):
    try:
        with open(path, "rb") as f:
            head = f.read(1024)
    except OSError:
        return True
    return b"\0" in head


def chunk_file(path, max_lines=40, overlap=8):
    try:
        st = os.stat(path)
    except OSError:
        return []
    if looks_binary(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return []
    return chunk_text(path, text, mtime=st.st_mtime,
                      max_lines=max_lines, overlap=overlap)


def chunk_jsonl(path, text_field="text"):
    chunks = []
    try:
        st = os.stat(path)
    except OSError:
        return []
    try:
        f = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return []
    with f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                text = rec.get(text_field, "")
            else:
                text = str(rec)
            if not isinstance(text, str) or not text.strip():
                continue
            meta = {"line": i}
            if isinstance(rec, dict):
                for k in ("ts", "timestamp", "time", "role", "kind"):
                    if k in rec:
                        meta[k] = rec[k]
            c = Chunk(path, text.strip(), mtime=st.st_mtime, offset=i, meta=meta)
            ts = meta.get("ts") or meta.get("timestamp") or meta.get("time")
            if isinstance(ts, (int, float)):
                c.mtime = float(ts)
            chunks.append(c)
    return chunks


SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".focus"}


def collect(sources, max_lines=40, overlap=8):
    chunks = []
    for src in sources:
        if os.path.isdir(src):
            for root, dirs, files in os.walk(src):
                dirs[:] = [d for d in dirs
                           if d not in SKIP_DIRS and not d.startswith(".")]
                for name in sorted(files):
                    if name.startswith("."):
                        continue
                    path = os.path.join(root, name)
                    if path.endswith(".jsonl"):
                        chunks.extend(chunk_jsonl(path))
                    else:
                        chunks.extend(chunk_file(path, max_lines, overlap))
        elif os.path.isfile(src):
            if src.endswith(".jsonl"):
                chunks.extend(chunk_jsonl(src))
            else:
                chunks.extend(chunk_file(src, max_lines, overlap))
        elif src == "-":
            text = sys.stdin.read()
            chunks.extend(chunk_text("stdin", text, max_lines=max_lines,
                                     overlap=overlap))
    return chunks
