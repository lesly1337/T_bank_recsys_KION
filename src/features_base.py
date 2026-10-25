"""Базовые признаки для ранжировщика LightGBM.

Признаки (~25 штук):
1. Сигналы кандидатов:
   - knn_score, knn_rank, pop7_rank, in_knn, in_pop
2. Пользователь (поведение в истории dt < T):
   - u_cnt_all, u_cnt_7d, u_cnt_30d
   - u_dur_sum, u_dur_mean, u_pct_mean
   - u_days_since_last
3. Демография пользователя:
   - age, income, sex, kids_flg (категориальные)
4. Контент (поведение в истории dt < T):
   - i_cnt_all, i_cnt_7d, i_cnt_30d
   - i_pop_trend_7_30
   - i_dur_mean, i_pct_mean
   - i_days_since_last
5. Метаданные контента:
   - content_type, release_year, age_rating
   - n_genres, n_countries, n_actors
6. Косвенные связи (user x item):
   - u_genre_match_cnt, u_genre_match_share
   - u_director_match, u_actor_match_cnt
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.data import load_items, load_users


def build_base_features(
    cand: pl.DataFrame,
    hist: pl.DataFrame,
    T: date,
    users_df: pl.DataFrame | None = None,
    items_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if users_df is None:
        users_df = load_users()
    if items_df is None:
        items_df = load_items()

    user_ids = cand.select("user_id").unique()
    uhist = hist.join(user_ids, on="user_id", how="semi")

    # ---------------- 1. User history features
    d_7 = T - timedelta(days=7)
    d_30 = T - timedelta(days=30)

    u_feats = (
        uhist.group_by("user_id")
        .agg(
            pl.len().alias("u_cnt_all"),
            (pl.col("dt") >= d_7).sum().alias("u_cnt_7d"),
            (pl.col("dt") >= d_30).sum().alias("u_cnt_30d"),
            pl.col("total_dur").sum().alias("u_dur_sum"),
            pl.col("total_dur").mean().alias("u_dur_mean"),
            pl.col("watched_pct").mean().alias("u_pct_mean"),
            (pl.lit(T) - pl.col("dt").max()).dt.total_days().alias("u_days_since_last"),
        )
    )

    # Демография
    u_demo = users_df.select(
        pl.col("user_id"),
        pl.col("age").fill_null("unknown").cast(pl.Categorical),
        pl.col("income").fill_null("unknown").cast(pl.Categorical),
        pl.col("sex").fill_null("unknown").cast(pl.Categorical),
        pl.col("kids_flg").fill_null(-1).cast(pl.Int32),
    )
    u_all = u_feats.join(u_demo, on="user_id", how="left").with_columns(
        pl.col("age").fill_null("unknown"),
        pl.col("income").fill_null("unknown"),
        pl.col("sex").fill_null("unknown"),
        pl.col("kids_flg").fill_null(-1),
    )

    # ---------------- 2. Item history features
    cand_items = cand.select("item_id").unique()
    i_feats = (
        hist.group_by("item_id")
        .agg(
            pl.len().alias("i_cnt_all"),
            (pl.col("dt") >= d_7).sum().alias("i_cnt_7d"),
            (pl.col("dt") >= d_30).sum().alias("i_cnt_30d"),
            pl.col("total_dur").mean().alias("i_dur_mean"),
            pl.col("watched_pct").mean().alias("i_pct_mean"),
            (pl.lit(T) - pl.col("dt").max()).dt.total_days().alias("i_days_since_last"),
        )
        .join(cand_items, on="item_id", how="semi")
        .with_columns(
            (
                (pl.col("i_cnt_7d") + 1.0)
                / (pl.col("i_cnt_30d") / 4.0 + 1.0)
            ).alias("i_pop_trend_7_30")
        )
    )

    # Метаданные item
    i_meta = items_df.select(
        pl.col("item_id"),
        pl.col("content_type").fill_null("unknown").cast(pl.Categorical),
        pl.col("release_year").fill_null(2000.0).cast(pl.Float32),
        pl.col("age_rating").fill_null(-1.0).cast(pl.Float32),
        pl.col("genres").list.len().fill_null(0).cast(pl.Int32).alias("n_genres"),
        pl.col("countries").list.len().fill_null(0).cast(pl.Int32).alias("n_countries"),
        pl.col("actors").list.len().fill_null(0).cast(pl.Int32).alias("n_actors"),
    )
    i_all = cand_items.join(i_feats, on="item_id", how="left").join(i_meta, on="item_id", how="left").with_columns(
        pl.col("i_cnt_all").fill_null(0),
        pl.col("i_cnt_7d").fill_null(0),
        pl.col("i_cnt_30d").fill_null(0),
        pl.col("i_dur_mean").fill_null(0.0),
        pl.col("i_pct_mean").fill_null(0.0),
        pl.col("i_days_since_last").fill_null(999),
        pl.col("i_pop_trend_7_30").fill_null(1.0),
        pl.col("content_type").fill_null("unknown"),
        pl.col("release_year").fill_null(2000.0),
        pl.col("age_rating").fill_null(-1.0),
        pl.col("n_genres").fill_null(0),
        pl.col("n_countries").fill_null(0),
        pl.col("n_actors").fill_null(0),
    )

    # ---------------- 3. Pair affinity features (User x Item)
    # Жанры: user -> set of watched genres
    ug_hist = (
        uhist.select("user_id", "item_id")
        .join(items_df.select("item_id", "genres").explode("genres").drop_nulls(), on="item_id")
        .select("user_id", "genres")
        .unique()
    )
    cand_ig = (
        cand.select("user_id", "item_id")
        .join(items_df.select("item_id", "genres").explode("genres").drop_nulls(), on="item_id")
    )
    genre_match = (
        cand_ig.join(ug_hist, on=["user_id", "genres"], how="semi")
        .group_by(["user_id", "item_id"])
        .agg(pl.len().alias("u_genre_match_cnt"))
    )

    # Режиссеры
    ud_hist = (
        uhist.select("user_id", "item_id")
        .join(items_df.select("item_id", "directors").explode("directors").drop_nulls(), on="item_id")
        .select("user_id", "directors")
        .unique()
    )
    cand_id = (
        cand.select("user_id", "item_id")
        .join(items_df.select("item_id", "directors").explode("directors").drop_nulls(), on="item_id")
    )
    dir_match = (
        cand_id.join(ud_hist, on=["user_id", "directors"], how="semi")
        .select("user_id", "item_id")
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("u_director_match"))
    )

    # Актеры
    ua_hist = (
        uhist.select("user_id", "item_id")
        .join(items_df.select("item_id", "actors").explode("actors").drop_nulls(), on="item_id")
        .select("user_id", "actors")
        .unique()
    )
    cand_ia = (
        cand.select("user_id", "item_id")
        .join(items_df.select("item_id", "actors").explode("actors").drop_nulls(), on="item_id")
    )
    act_match = (
        cand_ia.join(ua_hist, on=["user_id", "actors"], how="semi")
        .group_by(["user_id", "item_id"])
        .agg(pl.len().alias("u_actor_match_cnt"))
    )

    # Собираем всё вместе в cand
    df = (
        cand
        .join(u_all, on="user_id", how="left")
        .join(i_all, on="item_id", how="left")
        .join(genre_match, on=["user_id", "item_id"], how="left")
        .join(dir_match, on=["user_id", "item_id"], how="left")
        .join(act_match, on=["user_id", "item_id"], how="left")
        .with_columns(
            pl.col("u_genre_match_cnt").fill_null(0).cast(pl.Int32),
            pl.col("u_director_match").fill_null(0).cast(pl.Int8),
            pl.col("u_actor_match_cnt").fill_null(0).cast(pl.Int32),
        )
        .with_columns(
            (pl.col("u_genre_match_cnt") / (pl.col("n_genres") + 1e-4)).alias("u_genre_match_share")
        )
    )

    return df
