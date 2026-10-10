import argparse
import os
import sys
import time

from .budget import compress_tail, dedup, select
from .chunk import collect
from .render import render, render_trace
from .score import Scorer, parse_weights
from .store import Store

VERSION = "1.0.0"


def cmd_build(args):
    store = Store()
    chunks = collect(args.sources, max_lines=args.max_lines,
                     overlap=args.overlap)
    if not chunks:
        print("focus: no chunks found in sources", file=sys.stderr)
        return 1
    now = time.time()
    store.observe(chunks, now)
    store.save()

    scorer = Scorer(args.query, store=store,
                    weights=parse_weights(args.weights),
                    half_life_h=args.half_life, now=now)
    scorer.learn_candidates(chunks)

    scored = []
    for c in chunks:
        factors = scorer.factors(c)
        scored.append((c, scorer.static_score(c), factors))

    kept, dups = dedup(scored)
    selected, excluded = select(kept, args.budget, mmr_lambda=args.mmr)
    selected, notes = compress_tail(selected, args.budget)

    out = render(selected)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print("wrote %s (%d chunks, ~%d tokens)" %
              (args.out, len(selected),
               sum(c.tokens for c, _, _, _ in selected)))
    else:
        sys.stdout.write(out)

    if args.trace:
        sel_ids = {c.id for c, _, _, _ in selected}
        all_scored = kept + [(c, s, f) for c, s, f, _ in dups]
        sys.stderr.write("\n" + render_trace(all_scored, sel_ids,
                                             excluded) + "\n")
        for n in notes:
            sys.stderr.write("note: %s\n" % n)
        for c, s, f, reason in dups:
            sys.stderr.write("note: dup %s: %s\n" % (c.id, reason))
    return 0


def cmd_index(args):
    store = Store()
    chunks = collect(args.sources)
    now = time.time()
    store.observe(chunks, now)
    store.save()
    s = store.stats()
    print("indexed %d chunks from %d sources" % (len(chunks),
                                                 len(args.sources)))
    print("store: %(chunks_seen)d chunks, %(docs)d docs, "
          "%(terms)d terms, %(pins)d pins" % s)
    return 0


def cmd_pin(args):
    store = Store()
    if store.pin(args.id):
        store.save()
        print("pinned %s" % args.id)
    else:
        print("already pinned: %s" % args.id)
    return 0


def cmd_unpin(args):
    store = Store()
    if store.unpin(args.id):
        store.save()
        print("unpinned %s" % args.id)
    else:
        print("not pinned: %s" % args.id)
    return 0


def cmd_stats(args):
    store = Store()
    s = store.stats()
    print("store: %s" % s["path"])
    print("chunks seen: %(chunks_seen)d" % s)
    print("docs: %(docs)d  terms: %(terms)d  pins: %(pins)d" % s)
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="focus",
        description="salience-based context assembly for agent prompts")
    p.add_argument("--version", action="version", version="focus " + VERSION)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="assemble a context block for a query")
    b.add_argument("query", help="what the context is for")
    b.add_argument("-s", "--sources", nargs="+", required=True,
                   help="files, dirs, or .jsonl logs (use - for stdin)")
    b.add_argument("-b", "--budget", type=int, default=4000,
                   help="token budget for the block (default 4000)")
    b.add_argument("--out", help="write the block to a file")
    b.add_argument("--trace", action="store_true",
                   help="print the scoring audit trail to stderr")
    b.add_argument("--weights",
                   help="e.g. recency=0.2,relevance=0.5,novelty=0.2,"
                        "frequency=0.1")
    b.add_argument("--half-life", type=float, default=24.0,
                   help="recency half-life in hours (default 24)")
    b.add_argument("--mmr", type=float, default=0.5,
                   help="diversity penalty 0-1 (default 0.5)")
    b.add_argument("--max-lines", type=int, default=40)
    b.add_argument("--overlap", type=int, default=8)
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("index", help="feed sources into the store")
    i.add_argument("-s", "--sources", nargs="+", required=True)
    i.set_defaults(func=cmd_index)

    pn = sub.add_parser("pin", help="always include a chunk id")
    pn.add_argument("id", help="chunk id from --trace output")
    pn.set_defaults(func=cmd_pin)

    up = sub.add_parser("unpin", help="remove a pin")
    up.add_argument("id")
    up.set_defaults(func=cmd_unpin)

    st = sub.add_parser("stats", help="show store stats")
    st.set_defaults(func=cmd_stats)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
