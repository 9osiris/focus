import sys


def estimate_tokens(text):
    # rough 4 chars per token, same rule ctxpack uses
    return max(1, len(text) // 4)


class Tokenizer:
    # counts tokens. default is the 4-char estimate, no deps.
    # pass a tiktoken encoding name (cl100k_base, o200k_base) or a
    # model name (gpt-4o) for exact counts when tiktoken is installed.
    # no hard dependency: missing tiktoken or a bad name just warns
    # and falls back to the estimate.
    def __init__(self, name=None):
        self.name = name
        self._encode = None
        self.exact = False
        if name is None:
            return
        try:
            import tiktoken
        except ImportError:
            self._warn("tiktoken not installed, falling back to the "
                       "4-char estimate (pip install tiktoken)")
            return
        enc = None
        try:
            enc = tiktoken.get_encoding(name)
        except ValueError:
            try:
                enc = tiktoken.encoding_for_model(name)
            except (ValueError, KeyError):
                pass
        if enc is None:
            self._warn("unknown tiktoken encoding %r, falling back "
                       "to the 4-char estimate" % name)
            return
        self._encode = enc.encode
        self.exact = True

    def _warn(self, msg):
        sys.stderr.write("focus: warning: %s\n" % msg)

    def count(self, text):
        if self._encode is not None:
            return max(1, len(self._encode(text)))
        return estimate_tokens(text)

    def describe(self):
        if self.exact:
            return "tiktoken %s" % self.name
        return "estimate (4 chars/token)"
