from datetime import datetime, timezone

import pytest

import collect
import filters


def art(title, source):
    return collect.Article(title=title, url="https://news.google.com/x", source=source,
                           published=datetime.now(timezone.utc))


@pytest.mark.parametrize("title,source,reason", [
    ("Cloud seeding", "Britannica", "reference page"),
    ("Some long enough headline here", "Medium", "self-published"),
    ("Some long enough headline here", "YouTube", "social/video"),
    ("Publicis Groupe", "Forbes", "bare title"),
    ("宇宙ビジネス", "日経", ""),
    ("Honda'dan fabrika yatırımı", "Milliyet", ""),
    ("Here's what is open and closed on Thanksgiving 2026", "Local News", "consumer page"),
    ("How to Get a Banking Licence in Switzerland", "Finextra", "consumer page"),
    ("Gamindo raises €1.4 million to teach employees how to use AI", "EU-Startups", ""),
    ("IGNIS and Acciona form joint venture to build a data center in Spain", "Reuters", ""),
    ("Biocon Q1 revenue rises 10% to ₹4,300 cr on steady biosimilar sales", "ETPharma", ""),
    ("British berry harvest rises 6% despite heatwaves", "Hortidaily", ""),
    ("Novartis India Share Price Today: 6.19% Fall", "Business Standard", "stock noise"),
    ("Biocon shares jump 6% on reports of block deal", "Fortune India", ""),
    ("Canada’s EV Market Share Climbs From 7% to 13% in 18 Months", "Drive Tesla Canada", ""),
    ("Sun Pharma Q2 results: Net profit falls 36% to Rs 1,989 cr on one-time hit, revenue up 12%; stock jumps", "Moneycontrol.com", ""),
    ("Netflix ad revenue tracks to $3B as stock drops 8% on Q2 growth slowdown", "PPC Land", ""),
    ("Why Did BHP Group (NYSE:BHP) Gain 2.2% as Copper Recovered", "Kalkine", "stock noise"),
    ("TSMC plans 3nm semiconductor manufacturing at second Japan factory, investment to exceed USD 23 billion", "The Times of India", ""),
    ("British berry sector discusses market trends and automation", "FreshPlaza", ""),
    ("Shimano Europe President on competition, supply chains and market outlook", "Bike Europe", ""),
    ("EV Market Trends: H1 2026 Review and Forecast", "Investing News Network", "research PR"),
    ("The New Dawn of Retail | Italy Market Outlook 2026", "Colliers", "research PR"),
    ("Innovation, Research & Smart Cities: Mid-November 2026 Funding Opportunities (28 new opportunities)", "Substack", ""),
])
def test_reason(title, source, reason):
    assert filters.l1_reason(art(title, source)) == reason


def test_apply_l1():
    good = art("IGNIS and Acciona form joint venture to build a data center in Spain", "Reuters")
    bad = art("Cloud seeding", "Britannica")
    kept, dropped = filters.apply_l1([good, bad])
    assert kept == [good]
    assert len(dropped) == 1
    assert set(dropped[0]) == {"title", "source", "bucket", "reason"}
