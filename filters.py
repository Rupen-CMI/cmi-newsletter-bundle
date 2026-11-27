"""Layer 1 of the evidence filter: drops only what is always junk, judged from the
source name and headline alone. Everything subtle is left to layer 2 (score.py)."""
import re

import config


def _src_rx(words):
    return re.compile(r"(?<![\w])(?:" + "|".join(re.escape(w) for w in words) + r")(?![\w])",
                      re.I)


_REFERENCE_RX = _src_rx(config.REFERENCE_SOURCES)
_SELF_PUBLISHED_RX = _src_rx(config.SELF_PUBLISHED_SOURCES)
_CONSUMER_RE = re.compile(
    r"^\W*how to\b|what'?s open and closed|\bopen and closed\b|\bhoroscopes?\b|\brecipes?\b"
    r"|\bquiz\b|\bgiveaway\b|\bobituary\b|\bwebinar\b", re.I)
# Funding-call digests on Substack are what R&D-funding advisers sell, not personal posts.
_FUNDING_LIST_RE = re.compile(
    r"\bfunding opportunit|\bcalls? for (proposals|applications)\b|\bgrant (calls?|programmes?|programs?)\b", re.I)


def _latin(t):
    letters = [c for c in (t or "") if c.isalpha()]
    if not letters:
        return True
    return sum(1 for c in letters if ord(c) > 0x24F) <= len(letters) / 2


def l1_reason(a) -> str:
    """"" to keep, else the first matching junk reason."""
    import collect
    import score
    src = a.source or ""
    title = a.title or ""
    if collect._blocked(a):
        return "blocked source"
    if any(w in src.lower() for w in score._LOW_VALUE_SOURCES):
        return "social/video"
    if _REFERENCE_RX.search(src):
        return "reference page"
    if _SELF_PUBLISHED_RX.search(src) and not _FUNDING_LIST_RE.search(title):
        return "self-published"
    if score._is_ticker_noise(a):
        return "stock noise"
    if config.is_research_pr(a.source, a.title):
        return "research PR"
    if score._SEO_LISTICLE.search(title):
        return "SEO listicle"
    if _latin(title) and len(title.split()) <= 2:
        return "bare title"
    if _CONSUMER_RE.search(title):
        return "consumer page"
    return ""


def apply_l1(articles) -> tuple[list, list[dict]]:
    kept, dropped = [], []
    for a in articles:
        try:
            reason = l1_reason(a)
        except Exception:  # noqa: BLE001 -- never fail the brief
            reason = ""
        if reason:
            dropped.append({"title": a.title, "source": a.source,
                            "bucket": getattr(a, "bucket", None), "reason": reason})
        else:
            kept.append(a)
    return kept, dropped
