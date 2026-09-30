"""Free TF-IDF trend clustering — the ZERO-COST path of the mother repo's clusterer.

The mother `clusterer.py` runs KMeans on OpenAI EMBEDDINGS (paid) hybridized with TF-IDF+LDA.
Here we keep ONLY the free half: TF-IDF vectors + KMeans + silhouette, with top-term labels
(`label_clusters_from_tfidf`). No embeddings, no API cost — pure scikit-learn on CPU.

Used to group the 'industry' signals into coherent themes so the Industry News section reads
like a trend, not a random pile of headlines.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from collect import Article

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    _SKLEARN = True
except Exception:
    _SKLEARN = False


@dataclass
class Cluster:
    label: str
    top_terms: list[str]
    articles: list[Article] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.articles)


def cluster_articles(articles: list[Article], k: int | None = None) -> list[Cluster]:
    """TF-IDF + KMeans clustering with top-term labels. Degrades to one cluster if
    sklearn is missing or there are too few articles to cluster meaningfully."""
    arts = [a for a in articles if (a.title or a.summary)]
    if not _SKLEARN or len(arts) < 4:
        return [Cluster(label="", top_terms=[], articles=arts)] if arts else []

    texts = [f"{a.title} {a.summary}" for a in arts]
    vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2), min_df=1)
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return [Cluster(label="", top_terms=[], articles=arts)]

    k = k or max(2, min(5, len(arts) // 3))
    k = min(k, len(arts))
    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
    terms = vec.get_feature_names_out()

    clusters: list[Cluster] = []
    for cid in range(k):
        members = [arts[i] for i in range(len(arts)) if km.labels_[i] == cid]
        if not members:
            continue
        # top TF-IDF terms for this centroid → human label
        centroid = km.cluster_centers_[cid]
        top_idx = centroid.argsort()[::-1][:3]
        top_terms = [terms[j] for j in top_idx if centroid[j] > 0]
        label = " / ".join(t.title() for t in top_terms)
        clusters.append(Cluster(label=label, top_terms=top_terms, articles=members))

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def dominant_theme(articles: list[Article]) -> tuple[str, list[Article]]:
    """Return (theme_label, articles_of_the_biggest_cluster). Empty label if not clustered."""
    clusters = cluster_articles(articles)
    if not clusters:
        return "", []
    top = clusters[0]
    return top.label, top.articles
