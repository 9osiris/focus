focus - salience-based context assembly for agents [experimental project]

disclaimer: this is a scratch tool, not a serious product. it does
rough 4-chars-per-token math, not real tokenization, and the scoring is
heuristics, not science. it works for my own agent experiments and
probably nobody else's.

why i made this
i kept pasting whole memory dumps and doc folders into prompts and
watching the actually relevant stuff drown. embeddings search finds
similar things but it has no opinion about what matters right now. i
wanted the layer between retrieval and the prompt that acts like
attention: score every candidate on recency, relevance to the query,
how often it gets referenced, and how surprising it is, then pack a
fixed token budget with the winners and show its work. the trace output
is the whole point, if i cant see why a chunk made the cut i dont trust
the block. i avoided real tokenizers and any ml deps on purpose, this
thing should run anywhere with just python. the store is a json file in
~/.focus, nothing fancy, because i wanted to read it by hand when the
scores look wrong.

the guide
needs python 3.9+, no dependencies.

```
pip install .
focus index -s ./notes ./chat.jsonl
focus build "how do retries work" -s ./notes ./chat.jsonl -b 2000 --trace
```

build prints a markdown context block to stdout. --out writes it to a
file. --trace dumps the scoring audit trail to stderr so you can see
every candidate, its factor scores, and why it got in or cut. pins
always survive:

```
focus build "q" -s ./notes --trace   # find the chunk id you want
focus pin 4bcf07d63984af13           # it now goes in every build
focus unpin 4bcf07d63984af13
```

tune it with --weights recency=0.2,relevance=0.5 --half-life 48
--mmr 0.7. mmr is the diversity penalty, higher means fewer near-dupes
in the block. budget is a hard cap, tail chunks get truncated to
snippets before anything gets dropped, pins get truncated last and are
never dropped.

how it scores
each chunk gets four factors, weighted and summed. recency decays with
a 24h half life by default. relevance is idf-weighted query term
overlap, idf comes from the store once you have indexed anything, else
from the candidate set. frequency is log-scaled reference count from
the store. novelty is not a static factor, it is applied during
selection as an mmr penalty against chunks already picked, so the block
does not fill up with five versions of the same paragraph. near-dupes
above 0.8 shingle jaccard get dropped before selection even starts.

examples/corpus has a tiny demo set, try:
focus build "weighted upstream selection" -s examples/corpus -b 400 --trace
