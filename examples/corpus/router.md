# router notes

the gateway retries 429s with jittered backoff, max 3 attempts.
circuit breaker trips after 3 consecutive failures, half-open probe
after 30 seconds. streaming responses pass through as sse and we
capture usage from the final chunk.

open question: weighted upstream selection. the weight field is parsed
and stored but routing still walks the try list in order. need smooth
weighted round robin for the primary pick, weight 0 means fallback only.

budget accounting lives in cost.py. per-1k prompt/completion prices,
daily caps per api key, 402 when exceeded.
