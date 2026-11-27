"""Keyword-free relevance of a news article to the SELLER, via a small multilingual
embedding model run locally on CPU (fastembed / ONNX -- no torch, no GPU, no API).

Why: score.relevancy() is a keyword lexicon plus bucket floors, so any article that a
seller-scoped query returned counts as "relevant", and fresh national headlines (stock
market, sports, defence) outrank on-category stories. Cosine similarity between the
article text and a seller description is language-independent (the model is multilingual,
so a Polish headline and an English seller text live in the same space) and needs no
word list.

Everything here is optional: if fastembed or the model is unavailable every function
returns None and callers keep their existing behaviour. Nothing here raises.
"""
from __future__ import annotations

import os
import re
import threading

MODEL_NAME = os.environ.get("CMI_EMBED_MODEL",
                            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

_lock = threading.Lock()
_model = None            # loaded encoder, or False once loading failed (don't retry per call)
# One inference at a time, on a fixed thread count. With 30-40 brief workers each calling
# embed() at once, every ONNX call grabbed every core: 1,700% CPU on a 28-core VM, briefs
# 5x slower, and the same oversubscription hung an 8-16 core laptop outright.
_embed_lock = threading.Lock()
_THREADS = int(os.environ.get("CMI_EMBED_THREADS") or max(1, min(8, (os.cpu_count() or 2) // 2)))
# text -> vector. score, evidence_score and competitor_score embed the same article texts,
# and a lead's signal and report briefs share most of their pool: embed each text once.
_cache: dict[str, object] = {}
_CACHE_MAX = 200_000


def _load():
    global _model
    if _model is not None:
        return _model or None
    with _lock:
        if _model is not None:
            return _model or None
        try:
            from fastembed import TextEmbedding
            cache = os.environ.get("CMI_EMBED_CACHE") or os.path.join(
                os.path.expanduser("~"), ".cache", "cmi_fastembed")
            _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=cache, threads=_THREADS)
        except Exception:  # noqa: BLE001 -- missing package / no network for first download
            _model = False
    return _model or None


def set_encoder(enc) -> None:
    """Install a custom encoder (tests / alternative backends). `enc` is any object with
    .embed(list[str]) -> iterable of vectors, or None/False to force 'unavailable'."""
    global _model
    _model = enc if enc is not None else False
    _cache.clear()


def reset() -> None:
    """Forget the cached model so the next call tries to load it again."""
    global _model
    _model = None
    _cache.clear()


def available() -> bool:
    return _load() is not None


def embed(texts: list[str]):
    """L2-normalised float32 matrix (n, d) for `texts`, or None when no model."""
    if not texts:
        return None
    m = _load()
    if m is None:
        return None
    try:
        import numpy as np
        keys = [t or " " for t in texts]
        todo = list(dict.fromkeys(k for k in keys if k not in _cache))
        if todo:
            with _embed_lock:
                todo = [k for k in todo if k not in _cache]   # another worker may have done them
                if todo:
                    new = np.asarray(list(m.embed(todo)), dtype="float32")
                    if new.ndim != 2 or len(new) != len(todo):
                        return None
                    if len(_cache) + len(todo) > _CACHE_MAX:
                        _cache.clear()
                    _cache.update(zip(todo, new))
        vecs = np.asarray([_cache[k] for k in keys], dtype="float32")
        if vecs.ndim != 2 or len(vecs) != len(texts):
            return None
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
    except Exception:  # noqa: BLE001
        return None


def seller_text(profile) -> str:
    """value_prop + product names/applications + buyer industries (what the seller sells
    and to whom). Capped so the encoder's 128-token window is spent on the useful part."""
    parts = [(getattr(profile, "value_prop", "") or "").strip()]
    prods = []
    for p in (getattr(profile, "products", None) or []):
        if isinstance(p, dict):
            n, app = (p.get("name") or "").strip(), (p.get("application") or "").strip()
            prods.append(f"{n} ({app})" if app else n)
        elif p:
            prods.append(str(p).strip())
    if prods:
        parts.append("Products: " + "; ".join(x for x in prods if x)[:300])
    inds = [str(i).strip() for i in (getattr(profile, "industries", None) or []) if str(i).strip()]
    if inds:
        parts.append("Buyers: " + ", ".join(inds))
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


def article_text(a) -> str:
    title = (getattr(a, "title", "") or "").strip()
    summ = (getattr(a, "summary", "") or "").strip()
    # Google News summaries are usually the headline again; don't double it.
    if summ and summ.lower().startswith(title.lower()[:40]):
        summ = ""
    return f"{title}. {summ}"[:500].strip()


def seller_relevance(articles: list, profile) -> list[float] | None:
    """Cosine(article, seller) per article, in input order; None if no model/seller text."""
    if not articles:
        return []
    st = seller_text(profile)
    if not st:
        return None
    mat = embed([st] + [article_text(a) for a in articles])
    if mat is None:
        return None
    sims = mat[1:] @ mat[0]
    return [float(x) for x in sims]


def title_similarity_groups(titles: list[str], threshold: float = 0.85) -> list[int] | None:
    """Cluster id per title (single-link over cosine >= threshold), or None if no model."""
    if len(titles) < 2:
        return list(range(len(titles)))
    mat = embed(titles)
    if mat is None:
        return None
    sim = mat @ mat.T
    parent = list(range(len(titles)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    n = len(titles)
    import numpy as np
    ii, jj = np.where(np.triu(sim >= threshold, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return [find(i) for i in range(n)]


# Business-news check: an article is compared with short descriptions of business events and of
# non-news pages; margin = best business cosine - best non-news cosine. Measured on 58 hand-labelled
# Test15 headlines: AUC 0.89. English descriptions; non-English news scores somewhat lower, so callers
# use it mainly for ranking and only hard-drop far below zero.
NEWS_POS = [
    "Company invests in a new factory to expand production capacity",
    "Firm acquires a rival business in a multi-million deal",
    "Startup raises funding round led by investors to scale its platform",
    "Company wins a contract to supply equipment to a customer",
    "Manufacturer launches a new product line for industrial customers",
    "Government sets new regulation that companies in the sector must comply with",
    "Demand for the product rises as buyers increase orders, prices change",
    "Two companies form a joint venture to build a new plant",
]
NEWS_NEG = [
    "Encyclopedia article defining a concept and its history",
    "Celebrity wedding story and personal lifestyle anecdote",
    "How-to guide and tips for consumers at home",
    "Opinion column about a social issue",
    "Product review of a gadget for personal use",
    "What is open and closed on the holiday weekend",
    "Crime story about a person arrested by police",
    "Academic study finds health risks for patients",
]


def news_margin(articles: list) -> list[float] | None:
    """Per article: best cosine to a business-event description minus best cosine to a
    non-news description. None if no model; [] for empty input."""
    if not articles:
        return []
    mat = embed(NEWS_POS + NEWS_NEG + [article_text(a) for a in articles])
    if mat is None:
        return None
    npos, nneg = len(NEWS_POS), len(NEWS_NEG)
    sims = mat[npos + nneg:] @ mat[:npos + nneg].T
    return [round(float(r[:npos].max() - r[npos:].max()), 3) for r in sims]


def topic_similarity(articles: list, text: str) -> list[float] | None:
    """Cosine(article, text) per article; None if no text or no model; [] for empty input."""
    if not articles:
        return []
    if not text or not text.strip():
        return None
    mat = embed([text] + [article_text(a) for a in articles])
    if mat is None:
        return None
    return [round(float(x), 3) for x in mat[1:] @ mat[0]]
