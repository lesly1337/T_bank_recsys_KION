"""Генерация кандидатов: популярное за 7 дней + ItemKNN (cosine, scipy.sparse).

Всё строится только на истории (dt < T). Уже просмотренное исключается.
Возвращает кадр кандидатов user_id, item_id + «ядровые» признаки генераторов:
    knn_score, knn_rank, pop7_rank, in_knn, in_pop
и таблицу соседей item_sim(item_a, item_b, sim) — её используют SQL-признаки.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
import scipy.sparse as sp

from src.split import Split

KNN_TOP = 50        # кандидатов от ItemKNN на пользователя
POP_TOP = 30        # кандидатов от популярного на пользователя
USER_CAP = 50       # для KNN берём последние USER_CAP просмотров пользователя
NEIGHBORS = 100     # соседей на объект в матрице сходства
SHRINK = 10.0


def _item_sim(hist: pl.DataFrame) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    capped = (
        hist.sort(["user_id", "dt"], descending=[False, True])
        .group_by("user_id", maintain_order=True).head(USER_CAP)
    )
    item_ids = np.sort(capped["item_id"].unique().to_numpy())
    iidx = {int(i): k for k, i in enumerate(item_ids)}
    uid, u_inv = np.unique(capped["user_id"].to_numpy(), return_inverse=True)
    cols = np.searchsorted(item_ids, capped["item_id"].to_numpy())
    X = sp.csr_matrix((np.ones(len(cols), np.float32), (u_inv, cols)),
                      shape=(len(uid), len(item_ids)))
    G = (X.T @ X).tocsr()
    G.setdiag(0)
    G.eliminate_zeros()
    d = np.sqrt(np.asarray(X.sum(axis=0)).ravel())
    G = G.tocoo()
    vals = G.data / (d[G.row] * d[G.col] + SHRINK)
    S = sp.csr_matrix((vals.astype(np.float32), (G.row, G.col)), shape=G.shape)
    # оставляем NEIGHBORS сильнейших соседей в каждой строке
    rows, cols_, data = [], [], []
    for r in range(S.shape[0]):
        s, e = S.indptr[r], S.indptr[r + 1]
        if e == s:
            continue
        v, c = S.data[s:e], S.indices[s:e]
        if len(v) > NEIGHBORS:
            top = np.argpartition(-v, NEIGHBORS)[:NEIGHBORS]
            v, c = v[top], c[top]
        rows.append(np.full(len(v), r)); cols_.append(c); data.append(v)
    S = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols_))), shape=S.shape)
    return S, item_ids, iidx


def build_candidates(split: Split) -> tuple[pl.DataFrame, pl.DataFrame]:
    hist, T, users = split.hist, split.T, split.users
    S, item_ids, iidx = _item_sim(hist)

    # --- популярное за 7 дней (глобальный ранг)
    pop = (
        hist.filter(pl.col("dt") >= T - timedelta(days=7))
        .group_by("item_id").len().sort(["len", "item_id"], descending=[True, False])
        .with_row_index("pop7_rank", offset=1)
        .select("item_id", pl.col("pop7_rank").cast(pl.Int32))
    )
    pop_rank = dict(zip(pop["item_id"].to_list(), pop["pop7_rank"].to_list()))
    pop_list = pop["item_id"].head(POP_TOP + 300).to_list()

    uh = hist.filter(pl.col("user_id").is_in(users)).sort(["user_id", "dt"], descending=[False, True])
    seen = {u: items for u, items in uh.group_by("user_id", maintain_order=True).agg("item_id").iter_rows()}

    rows = []
    batch = 500
    for b in range(0, len(users), batch):
        ub = users[b:b + batch]
        r, c = [], []
        for k, u in enumerate(ub):
            its = [iidx[i] for i in seen[u][:USER_CAP] if i in iidx]
            r += [k] * len(its); c += its
        Xu = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(ub), len(item_ids)))
        scores = np.asarray((Xu @ S).todense())
        for k, u in enumerate(ub):
            sc = scores[k]
            seen_u = set(seen[u])
            for i in seen_u:
                if i in iidx:
                    sc[iidx[i]] = -np.inf
            top = np.argpartition(-sc, KNN_TOP)[:KNN_TOP]
            top = top[np.argsort(-sc[top])]
            knn_items = [int(item_ids[t]) for t in top if sc[t] > 0]
            pop_items = [i for i in pop_list if i not in seen_u][:POP_TOP]
            knn_rank = {i: n + 1 for n, i in enumerate(knn_items)}
            for i in dict.fromkeys(knn_items + pop_items):
                s = float(sc[iidx[i]]) if i in iidx else 0.0
                rows.append((u, i, max(s, 0.0), knn_rank.get(i, KNN_TOP + 1),
                             pop_rank.get(i, 10_000), i in knn_rank, i in pop_items))
    cand = pl.DataFrame(rows, schema={
        "user_id": pl.Int32, "item_id": pl.Int32, "knn_score": pl.Float32, "knn_rank": pl.Int32,
        "pop7_rank": pl.Int32, "in_knn": pl.Int8, "in_pop": pl.Int8}, orient="row")
    gt_pairs = pl.DataFrame([(u, i) for u, its in split.gt.items() for i in its],
                            schema={"user_id": pl.Int32, "item_id": pl.Int32}, orient="row")
    cand = cand.join(gt_pairs.with_columns(pl.lit(1, pl.Int8).alias("target")),
                     on=["user_id", "item_id"], how="left").with_columns(pl.col("target").fill_null(0))

    coo = S.tocoo()
    item_sim = pl.DataFrame({
        "item_a": item_ids[coo.row].astype(np.int32), "item_b": item_ids[coo.col].astype(np.int32),
        "sim": coo.data.astype(np.float32)})
    return cand, item_sim


def candidate_recall(cand: pl.DataFrame, split: Split) -> dict[str, float]:
    n_gt = sum(min(len(v), 10) for v in split.gt.values())
    n_gt_all = sum(len(v) for v in split.gt.values())
    pos = cand.filter(pl.col("target") == 1)
    return {
        "cand_per_user": cand.height / len(split.users),
        "recall_all": pos.height / n_gt_all,
        "recall_knn": pos.filter(pl.col("in_knn") == 1).height / n_gt_all,
        "recall_pop": pos.filter(pl.col("in_pop") == 1).height / n_gt_all,
        "users_with_pos": pos["user_id"].n_unique() / len(split.users),
        "_n_gt_cap10": n_gt,
    }
