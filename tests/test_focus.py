import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from focus import chunk as chunkmod
from focus import score as scoremod
from focus import budget as budgetmod
from focus.render import render, render_trace
from focus import store as storemod
from focus.cli import main as cli_main

NOW = 1_760_000_000.0


def mkchunk(text, mtime=NOW, source="s", pinned=False):
    return chunkmod.Chunk(source, text, mtime=mtime, pinned=pinned)


class ChunkTests(unittest.TestCase):
    def test_windows_with_overlap(self):
        text = "\n".join("line %d" % i for i in range(100))
        chunks = chunkmod.chunk_text("s", text, max_lines=40, overlap=8)
        self.assertEqual(len(chunks), 3)
        self.assertIn("line 39", chunks[0].text)
        self.assertIn("line 32", chunks[1].text)  # overlap region
        self.assertIn("line 99", chunks[2].text)

    def test_empty_text_no_chunks(self):
        self.assertEqual(chunkmod.chunk_text("s", "   \n  "), [])

    def test_fingerprint_stable(self):
        a = mkchunk("hello")
        b = mkchunk("hello")
        self.assertEqual(a.id, b.id)
        c = chunkmod.Chunk("other", "hello")
        self.assertNotEqual(a.id, c.id)

    def test_tokens_rough(self):
        self.assertEqual(chunkmod.estimate_tokens("a" * 400), 100)

    def test_snippet(self):
        c = mkchunk("\n".join("l%d" % i for i in range(10)))
        self.assertIn("[...]", c.snippet(3))
        self.assertNotIn("[...]", mkchunk("a\nb").snippet(3))

    def test_jsonl_parsing(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                         delete=False) as f:
            f.write('{"text": "first", "ts": 100}\n')
            f.write('not json\n')
            f.write('{"text": "   "}\n')
            f.write('{"text": "second"}\n')
            path = f.name
        try:
            chunks = chunkmod.chunk_jsonl(path)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0].text, "first")
            self.assertEqual(chunks[0].mtime, 100)
            self.assertEqual(chunks[0].meta["line"], 0)
        finally:
            os.unlink(path)

    def test_collect_skips_binary_and_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "ok.txt"), "w") as f:
                f.write("hello world\n")
            with open(os.path.join(d, "bin.dat"), "wb") as f:
                f.write(b"\x00\x01\x02binary")
            with open(os.path.join(d, ".hidden"), "w") as f:
                f.write("secret\n")
            os.mkdir(os.path.join(d, ".git"))
            with open(os.path.join(d, ".git", "x"), "w") as f:
                f.write("nope\n")
            chunks = chunkmod.collect([d])
            sources = {c.source for c in chunks}
            self.assertTrue(any(s.endswith("ok.txt") for s in sources))
            self.assertFalse(any("bin.dat" in s for s in sources))
            self.assertFalse(any(".hidden" in s for s in sources))
            self.assertFalse(any(".git" in s for s in sources))


class ScoreTests(unittest.TestCase):
    def test_recency_newer_wins(self):
        s = scoremod.Scorer("nothing matching", now=NOW)
        old = mkchunk("zzz qqq", mtime=NOW - 48 * 3600)
        new = mkchunk("zzz qqq", mtime=NOW - 1 * 3600)
        self.assertGreater(s.recency(new), s.recency(old))

    def test_relevance_query_match_wins(self):
        s = scoremod.Scorer("circuit breaker trips")
        chunks = [mkchunk("the circuit breaker trips after failures"),
                  mkchunk("deploy checklist for release day")]
        s.learn_candidates(chunks)
        self.assertGreater(s.relevance(chunks[0]), s.relevance(chunks[1]))

    def test_relevance_empty_query_zero(self):
        s = scoremod.Scorer("")
        self.assertEqual(s.relevance(mkchunk("anything")), 0.0)

    def test_idf_rare_term_heavier(self):
        s = scoremod.Scorer("q")
        s.learn_candidates([mkchunk("alpha beta"), mkchunk("alpha gamma"),
                            mkchunk("alpha delta")])
        self.assertGreater(s.idf("beta"), s.idf("alpha"))

    def test_parse_weights(self):
        w = scoremod.parse_weights("recency=1,relevance=3")
        # unspecified keys keep their defaults, then it normalizes
        self.assertAlmostEqual(w["recency"], 1 / 4.3)
        self.assertAlmostEqual(w["relevance"], 3 / 4.3)
        w2 = scoremod.parse_weights("bogus=9")
        self.assertEqual(w2, scoremod.DEFAULT_WEIGHTS)

    def test_frequency_from_store(self):
        with tempfile.TemporaryDirectory() as d:
            st = storemod.Store(os.path.join(d, "s.json"))
            c = mkchunk("hello world")
            st.observe([c], NOW)
            st.observe([c], NOW)
            s = scoremod.Scorer("q", store=st, now=NOW)
            self.assertGreater(s.frequency(c), 0.0)
            fresh = mkchunk("never seen")
            self.assertEqual(s.frequency(fresh), 0.0)

    def test_static_score_bounded(self):
        s = scoremod.Scorer("hello", now=NOW)
        c = mkchunk("hello world", mtime=NOW)
        v = s.static_score(c)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)


class BudgetTests(unittest.TestCase):
    def scored(self, texts, scores=None):
        out = []
        for i, t in enumerate(texts):
            c = mkchunk(t)
            sc = scores[i] if scores else 0.5
            out.append((c, sc, {"recency": 0, "relevance": 0,
                                "frequency": 0, "novelty": None}))
        return out

    def test_dedup_drops_lower_score_dup(self):
        text = "the gateway retries with backoff and then gives up"
        kept, dropped = budgetmod.dedup(
            self.scored([text, text + " indeed"], [0.9, 0.4]))
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(kept[0][1], 0.9)

    def test_dedup_keeps_different(self):
        kept, dropped = budgetmod.dedup(
            self.scored(["apples are red fruit", "servers run hot at night"]))
        self.assertEqual(len(kept), 2)
        self.assertEqual(len(dropped), 0)

    def test_select_respects_budget(self):
        texts = ["chunk number %d with some filler words" % i
                 for i in range(10)]
        scored = self.scored(texts, [0.1 * i for i in range(10)])
        selected, excluded = budgetmod.select(scored, 60)
        used = sum(c.tokens for c, _, _, _ in selected)
        self.assertLessEqual(used, 60)
        self.assertTrue(len(excluded) > 0)

    def test_select_prefers_density(self):
        short = mkchunk("x" * 40)     # 10 tokens
        long_ = mkchunk("x" * 4000)   # 1000 tokens
        scored = [(short, 0.5, {}), (long_, 0.9, {})]
        selected, _ = budgetmod.select(scored, 100)
        ids = [c.id for c, _, _, _ in selected]
        self.assertIn(short.id, ids)
        self.assertNotIn(long_.id, ids)

    def test_pins_go_first(self):
        texts = ["filler words here " * 20 for _ in range(5)]
        scored = self.scored(texts, [0.9, 0.8, 0.7, 0.6, 0.5])
        scored[4][0].pinned = True
        selected, _ = budgetmod.select(scored, 10_000)
        self.assertEqual(selected[0][3], "pinned")
        self.assertTrue(scored[4][0].id == selected[0][0].id)

    def test_mmr_penalizes_similar(self):
        a = "the cat sat on the mat today"
        b = "the cat sat on the mat today indeed"
        c = "quantum routers need jittered backoff logic"
        scored = self.scored([a, b, c], [0.9, 0.89, 0.5])
        selected, _ = budgetmod.select(scored, 10_000, mmr_lambda=0.9)
        texts = [ch.text for ch, _, _, _ in selected]
        self.assertIn(c, texts)
        self.assertIn(a, texts)

    def test_compress_tail_shrinks(self):
        big = mkchunk("\n".join("line %d" % i for i in range(50)))
        scored = [(big, 0.5, {})]
        selected, excluded = budgetmod.select(scored, 10_000)
        selected[0][0].pinned = True
        selected, notes = budgetmod.compress_tail(selected, 10)
        self.assertLessEqual(selected[0][0].tokens, 10)
        self.assertTrue(any("truncated" in n for n in notes))
        self.assertTrue(any("pinned" in n for n in notes))


class RenderTests(unittest.TestCase):
    def test_render_has_headers(self):
        c = mkchunk("hello world", source="notes.md")
        out = render([(c, 0.8, {}, "score 0.800")])
        self.assertIn("## notes.md", out)
        self.assertIn("hello world", out)
        self.assertIn("0.800", out)

    def test_trace_marks_in_out(self):
        a = mkchunk("aaa", source="a")
        b = mkchunk("bbb", source="b")
        scored = [(a, 0.9, {"recency": 1, "relevance": 1, "frequency": 0,
                            "novelty": None}),
                  (b, 0.1, {"recency": 0, "relevance": 0, "frequency": 0,
                            "novelty": None})]
        out = render_trace(scored, {a.id}, [(b, 0.1, {}, "nope")])
        self.assertIn("IN", out)
        self.assertIn("OUT", out)
        self.assertIn("nope", out)


class StoreTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            st = storemod.Store(p)
            c = mkchunk("hello world")
            st.observe([c], NOW)
            st.pin(c.id)
            st.save()
            st2 = storemod.Store(p)
            self.assertEqual(st2.seen_count(c.id), 1)
            self.assertIn(c.id, st2.pins)
            self.assertGreater(st2.df.get("hello", 0), 0)

    def test_reobserve_no_df_inflation(self):
        with tempfile.TemporaryDirectory() as d:
            st = storemod.Store(os.path.join(d, "s.json"))
            c = mkchunk("hello world")
            st.observe([c], NOW)
            st.observe([c], NOW)
            st.observe([c], NOW)
            self.assertEqual(st.docs, 1)
            self.assertEqual(st.df.get("hello"), 1)
            self.assertEqual(st.seen_count(c.id), 3)

    def test_corrupt_store_tolerated(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            with open(p, "w") as f:
                f.write("{not json")
            st = storemod.Store(p)
            self.assertEqual(st.stats()["chunks_seen"], 0)

    def test_unpin(self):
        with tempfile.TemporaryDirectory() as d:
            st = storemod.Store(os.path.join(d, "s.json"))
            self.assertTrue(st.pin("abc"))
            self.assertFalse(st.pin("abc"))
            self.assertTrue(st.unpin("abc"))
            self.assertFalse(st.unpin("abc"))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = os.path.join(self.tmp.name, "focushome")
        self.old = os.environ.get("FOCUS_HOME")
        os.environ["FOCUS_HOME"] = self.home
        self.corpus = os.path.join(self.tmp.name, "corpus")
        os.mkdir(self.corpus)
        with open(os.path.join(self.corpus, "a.md"), "w") as f:
            f.write("# retry logic\n\nthe gateway retries 429s with "
                    "jittered backoff, three attempts max.\n")
        with open(os.path.join(self.corpus, "b.md"), "w") as f:
            f.write("# lunch\n\n" + "had a sandwich. it was fine. " * 60)

    def tearDown(self):
        if self.old is None:
            os.environ.pop("FOCUS_HOME", None)
        else:
            os.environ["FOCUS_HOME"] = self.old
        self.tmp.cleanup()

    def test_build_end_to_end(self):
        rc = cli_main(["build", "retry backoff gateway",
                       "-s", self.corpus, "-b", "500"])
        self.assertEqual(rc, 0)
        st = storemod.Store(os.path.join(self.home, "store.json"))
        self.assertGreater(st.stats()["chunks_seen"], 0)

    def test_build_out_file(self):
        out = os.path.join(self.tmp.name, "ctx.md")
        rc = cli_main(["build", "retry", "-s", self.corpus,
                       "--out", out, "-b", "100"])
        self.assertEqual(rc, 0)
        with open(out) as f:
            body = f.read()
        self.assertIn("gateway", body)
        self.assertNotIn("sandwich", body)

    def test_build_no_sources_found(self):
        rc = cli_main(["build", "q", "-s",
                       os.path.join(self.tmp.name, "nope")])
        self.assertEqual(rc, 1)

    def test_index_and_stats(self):
        self.assertEqual(cli_main(["index", "-s", self.corpus]), 0)
        self.assertEqual(cli_main(["stats"]), 0)

    def test_pin_unpin(self):
        self.assertEqual(cli_main(["pin", "abc123"]), 0)
        self.assertEqual(cli_main(["unpin", "abc123"]), 0)

    def test_pinned_chunk_always_included(self):
        cli_main(["index", "-s", self.corpus])
        st = storemod.Store(os.path.join(self.home, "store.json"))
        target = None
        for fp in st.seen:
            target = fp
            break
        cli_main(["pin", target])
        out = os.path.join(self.tmp.name, "ctx2.md")
        # tiny budget that would normally drop everything
        cli_main(["build", "zzzzqqqq", "-s", self.corpus,
                  "--out", out, "-b", "40"])
        with open(out) as f:
            body = f.read()
        self.assertIn("##", body)  # pinned chunk forced its way in


if __name__ == "__main__":
    unittest.main()
