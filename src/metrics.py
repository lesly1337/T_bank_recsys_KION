"""Метрики top-K рекомендаций: Recall@K, MAP@K, NDCG@K, HitRate@K, coverage.

Все функции принимают:
    recs : dict[user_id -> list[item_id]]  (упорядоченный список рекомендаций)
    gt   : dict[user_id -> set[item_id]]   (что пользователь посмотрел в окне таргета)
Оцениваются ВСЕ пользователи из gt; если для пользователя нет рекомендаций, метрика = 0.
"""
from __future__ import annotations

import numpy as np


def per_user(recs: dict, gt: dict, k: int = 10) -> dict[str, np.ndarray]:
    """Метрики по каждому пользователю (нужно для бутстрепа)."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    out = {"recall": [], "map": [], "ndcg": [], "hit": []}
    for u, rel in gt.items():
        rec = list(recs.get(u, []))[:k]
        hits = np.array([i in rel for i in rec], dtype=np.float64)
        denom = min(len(rel), k)
        n_hit = hits.sum()
        out["recall"].append(n_hit / denom)
        out["hit"].append(float(n_hit > 0))
        if n_hit:
            prec_at = np.cumsum(hits) / np.arange(1, len(hits) + 1)
            out["map"].append((prec_at * hits).sum() / denom)
            dcg = (hits * discounts[: len(hits)]).sum()
            out["ndcg"].append(dcg / discounts[:denom].sum())
        else:
            out["map"].append(0.0)
            out["ndcg"].append(0.0)
    return {m: np.asarray(v) for m, v in out.items()}


def evaluate(recs: dict, gt: dict, k: int = 10, n_items_catalog: int | None = None) -> dict[str, float]:
    pu = per_user(recs, gt, k)
    res = {f"{m}@{k}": float(v.mean()) for m, v in pu.items()}
    if n_items_catalog:
        rec_items = {i for u in gt for i in list(recs.get(u, []))[:k]}
        res["coverage"] = len(rec_items) / n_items_catalog
    return res


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Разность средних (b - a) и 95% ДИ по парному бутстрепу пользователей."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
