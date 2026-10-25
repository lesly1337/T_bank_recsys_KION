"""Временной сплит train / val / test и сэмплинг пользователей.

Срез с порогом T:
    история  = interactions с dt <  T          (только она доступна признакам)
    таргет   = interactions с T <= dt < T + 7  (что предсказываем)
Оцениваются тёплые пользователи: есть и в таргете, и в истории.

Для строк истории total_dur / watched_pct окончательные (последний просмотр был до T),
поэтому брать их в признаки можно без утечки.

Пользователи сэмплируются (по N_USERS на срез), выборки срезов не пересекаются.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from src.data import load_interactions

CUTOFFS = {
    "train": date(2021, 8, 2),
    "val": date(2021, 8, 9),
    "test": date(2021, 8, 16),
}
N_USERS = 5000
SEED = 42


@dataclass
class Split:
    name: str
    T: date
    hist: pl.DataFrame      # вся история < T (все пользователи: нужна для популярности/KNN)
    users: list[int]        # сэмпл тёплых пользователей
    gt: dict[int, set[int]] # user -> множество объектов в окне таргета


def make_splits(n_users: int = N_USERS, seed: int = SEED) -> dict[str, Split]:
    inter = load_interactions()
    rng = np.random.default_rng(seed)
    taken: set[int] = set()
    out = {}
    for name, T in CUTOFFS.items():
        hist = inter.filter(pl.col("dt") < T)
        target = inter.filter((pl.col("dt") >= T) & (pl.col("dt") < T + timedelta(days=7)))
        warm = (
            target.select("user_id").unique()
            .join(hist.select("user_id").unique(), on="user_id", how="semi")
            .sort("user_id")["user_id"].to_numpy()
        )
        warm = np.array([u for u in warm if u not in taken])
        users = sorted(rng.choice(warm, size=n_users, replace=False).tolist())
        taken.update(users)
        gt_df = target.filter(pl.col("user_id").is_in(users)).group_by("user_id").agg("item_id")
        gt = {u: set(items) for u, items in gt_df.iter_rows()}
        out[name] = Split(name, T, hist, users, gt)
        print(f"[split] {name}: T={T} hist={hist.height:,} warm={len(warm):,} "
              f"sampled={len(users)} gt_pairs={sum(map(len, gt.values())):,}")
    return out
