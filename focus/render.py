def render(selected):
    # selected: list of (chunk, score, factors, reason)
    parts = []
    total = sum(c.tokens for c, _, _, _ in selected)
    parts.append("<!-- focus context block: %d chunks, ~%d tokens -->" %
                 (len(selected), total))
    for chunk, score, factors, reason in selected:
        parts.append("")
        parts.append("## %s" % chunk.source)
        parts.append("_salience %.3f, ~%d tokens, %s_" %
                     (score, chunk.tokens, reason))
        parts.append("")
        parts.append(chunk.text)
    return "\n".join(parts).strip() + "\n"


def render_trace(scored, selected_ids, excluded):
    # scored: list of (chunk, score, factors); a full audit trail of
    # why each candidate made the cut or not.
    lines = []
    lines.append("id        score   rec    rel    freq   verdict")
    lines.append("-" * 60)
    by_id = {c.id: (c, s, f) for c, s, f in scored}
    for chunk, score, factors in sorted(scored, key=lambda s: s[1],
                                        reverse=True):
        if chunk.id in selected_ids:
            verdict = "IN"
        else:
            verdict = "OUT"
        lines.append("%s  %.3f   %.3f  %.3f  %.3f  %s  %s" %
                     (chunk.id, score,
                      factors["recency"], factors["relevance"],
                      factors["frequency"], verdict, chunk.source))
    if excluded:
        lines.append("")
        lines.append("excluded:")
        for chunk, score, factors, reason in excluded:
            lines.append("  %s (%s): %s" % (chunk.id, chunk.source, reason))
    return "\n".join(lines)
