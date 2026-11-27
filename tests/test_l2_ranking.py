"""Layer-2 ranking: business-news check, topic fit, rank score."""
from datetime import datetime, timezone

import pytest

import brief_builder
import relevance
import score
from collect import Article

NOW = datetime.now(timezone.utc)


def _art(title="Some headline", **kw):
    a = Article(title=title, url="https://x.example/" + title[:8].replace(" ", ""),
                source="Wire", published=NOW, summary="", bucket="trigger")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_is_topical_by_similarity():
    assert score.is_topical(_art(topic_rel=0.0, topic_sim=0.5))
    assert not score.is_topical(_art(topic_rel=0.3, topic_sim=0.3))


def test_topical_evidence_news_margin():
    assert not score.topical_evidence(_art(topic_rel=0.0, topic_sim=0.5, news_margin=-0.2))
    assert score.topical_evidence(_art(topic_rel=0.0, topic_sim=0.5, news_margin=0.1))
    assert score.topical_evidence(_art(topic_rel=0.0, topic_sim=0.5, news_margin=None))


def test_l2_score_monotonic():
    lo = _art(seller_rel=0.15, ev_relevancy=0.5, ev_recency=0.5)
    hi = _art(seller_rel=0.35, ev_relevancy=0.5, ev_recency=0.5)
    assert score.l2_score(hi) > score.l2_score(lo)
    a = _art(news_margin=-0.1, ev_relevancy=0.5, ev_recency=0.5)
    b = _art(news_margin=0.2, ev_relevancy=0.5, ev_recency=0.5)
    assert score.l2_score(b) > score.l2_score(a)


def test_l2_score_topical_uses_topic_sim():
    lo = _art(topic_rel=0.0, topic_sim=0.40, seller_rel=0.9)
    hi = _art(topic_rel=0.0, topic_sim=0.70, seller_rel=0.0)
    assert score.l2_score(hi, topical=True) > score.l2_score(lo, topical=True)


def test_eligible_drops_news_junk():
    good = _art("Good", composite=0.9, news_margin=0.1)
    junk = _art("Junk", composite=0.9, news_margin=-0.4)
    out = score.eligible([good, junk])
    assert good in out and junk not in out


def test_rank_by_evidence_uses_l2_score():
    a = _art("A", l2_score=0.2, ev_relevancy=0.9, ev_recency=0.9, composite=0.5)
    b = _art("B", l2_score=0.8, ev_relevancy=0.5, ev_recency=0.5, composite=0.5)
    assert brief_builder._rank_by_evidence([a, b]) == [b, a]


def test_news_margin_model():
    if not relevance.available():
        pytest.skip("embedding model not available")
    arts = [_art("Firm acquires rival in $2bn deal"),
            _art("Here's what is open and closed on Thanksgiving")]
    m = relevance.news_margin(arts)
    assert m[0] > m[1]


def test_drop_repeats_same_story():
    a = _art("British Growers and JCM&Co partner to shape the future of UK horticulture")
    b = _art("British Growers and JCM&Co join forces to champion UK horticulture")
    c = _art("芙蓉総合リースとTOKIUM、資本業務提携を開始")
    d = _art("法人向けリース・ファイナンスサービスを提供する芙蓉総合リース株式会社と経理AIエージェントを提供する株式会社TOKIUMが資本業務提携を開始")
    e = _art("Spain approves new energy, digital rules for data centers")
    f = _art("Canada Invests $635 Million in EV Charging Infrastructure")
    g = _art("Canada discourages foreign EV charging standards ahead of Chinese EV imports")
    assert brief_builder._drop_repeats([a, b, c, d, e, f, g]) == [a, c, e, f, g]


def test_rank_by_evidence_drops_repeats():
    a = _art("GCC puts regional collaboration at heart of tourism recovery", l2_score=0.8,
             ev_relevancy=0.9, ev_recency=0.9, composite=0.5)
    b = _art("GCC puts regional collaboration at heart of tourism recovery", l2_score=0.6,
             ev_relevancy=0.9, ev_recency=0.9, composite=0.5)
    assert brief_builder._rank_by_evidence([b, a]) == [a]
