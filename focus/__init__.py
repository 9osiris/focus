"""focus: salience-based context assembly for agent prompts."""

from .budget import compress_tail, dedup, select
from .chunk import Chunk, chunk_file, chunk_jsonl, chunk_text, collect
from .cli import VERSION, main
from .render import render, render_trace
from .score import Scorer, parse_source_weights, parse_weights
from .store import Store
from .tokens import Tokenizer, estimate_tokens

__version__ = VERSION
__all__ = ["Chunk", "Store", "Scorer", "main", "collect", "chunk_text",
           "chunk_file", "chunk_jsonl", "dedup", "select", "compress_tail",
           "render", "render_trace", "parse_weights",
           "parse_source_weights", "Tokenizer",
           "estimate_tokens", "__version__"]
