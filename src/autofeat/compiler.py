"""Векторизованный Polars и DuckDB компилятор признаков.

Выполняет расчет признаков на истории строго до даты отсечки T (dt < T).
Гарантирует 100% отсутствие утечек данных (no future data leakage).
Поддерживает пакетное вычисление признаков по группам сущностей:
- user_id
- item_id
- демография x item (age x item, sex x item)
- user x item метаданные (жанры, актеры, тип контента)
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

import duckdb
import numpy as np
import polars as pl

from src.autofeat.grammar import Aggregation, Author, Entity, FeatureDef, Filter, Measure, Window


class FeatureCompiler:
    """Вычисляет признаки FeatureDef над историей взаимодействия."""

    def __init__(
        self,
        hist: pl.DataFrame,
        T: date,
        users_df: pl.DataFrame,
        items_df: pl.DataFrame,
    ):
        self.T = T
        # Строгая фильтрация истории без утечек
        self.hist = hist.filter(pl.col("dt") < T)
        self.users_df = users_df
        self.items_df = items_df

    def compile(self, cand: pl.DataFrame, features: Sequence[FeatureDef]) -> pl.DataFrame:
        """Вычисляет все переданные признаки и джойнит их к cand."""
        if not features:
            return cand

        # Разделяем признаки по сущностям для эффективной агрегации
        user_feats = [f for f in features if f.entity == Entity.USER]
        item_feats = [f for f in features if f.entity == Entity.ITEM]
        age_item_feats = [f for f in features if f.entity == Entity.AGE_ITEM]
        sex_item_feats = [f for f in features if f.entity == Entity.SEX_ITEM]
        affinity_feats = [f for f in features if f.entity == Entity.USER_ITEM_AFFINITY]

        res = cand

        if user_feats:
            u_df = self._compile_user_features(cand.select("user_id").unique(), user_feats)
            res = res.join(u_df, on="user_id", how="left")

        if item_feats:
            i_df = self._compile_item_features(cand.select("item_id").unique(), item_feats)
            res = res.join(i_df, on="item_id", how="left")

        if age_item_feats:
            res = self._compile_age_item_features(res, age_item_feats)

        if sex_item_feats:
            res = self._compile_sex_item_features(res, sex_item_feats)

        if affinity_feats:
            res = self._compile_affinity_features(res, affinity_feats)

        # Заполняем пропуски: для recency значение по умолчанию 999.0, для остальных 0.0
        fill_dict = {
            f.name: (999.0 if f.agg == Aggregation.RECENCY else 0.0)
            for f in features
            if f.name in res.columns
        }
        res = res.with_columns([pl.col(col).fill_null(val) for col, val in fill_dict.items()])

        return res

    def _compile_user_features(self, target_users: pl.DataFrame, features: list[FeatureDef]) -> pl.DataFrame:
        """Пакетный расчет признаков пользователя."""
        uhist = self.hist.join(target_users, on="user_id", how="semi")
        # Джойним тип контента для фильтров
        uhist = uhist.join(self.items_df.select("item_id", "content_type"), on="item_id", how="left")

        agg_exprs = []
        for f in features:
            expr = self._build_agg_expression(f)
            if expr is not None:
                agg_exprs.append(expr.alias(f.name))

        if not agg_exprs:
            return target_users

        out = uhist.group_by("user_id").agg(agg_exprs)
        return target_users.join(out, on="user_id", how="left")

    def _compile_item_features(self, target_items: pl.DataFrame, features: list[FeatureDef]) -> pl.DataFrame:
        """Пакетный расчет признаков объекта контента."""
        ihist = self.hist.join(target_items, on="item_id", how="semi")
        ihist = ihist.join(self.items_df.select("item_id", "content_type"), on="item_id", how="left")

        agg_exprs = []
        for f in features:
            expr = self._build_agg_expression(f)
            if expr is not None:
                agg_exprs.append(expr.alias(f.name))

        if not agg_exprs:
            return target_items

        out = ihist.group_by("item_id").agg(agg_exprs)
        return target_items.join(out, on="item_id", how="left")

    def _compile_age_item_features(self, cand: pl.DataFrame, features: list[FeatureDef]) -> pl.DataFrame:
        """Расчет признаков в разрезе Возраст пользователя x Контент."""
        hist_with_age = self.hist.join(
            self.users_df.select("user_id", pl.col("age").fill_null("unknown").alias("u_age")),
            on="user_id",
            how="left",
        )
        agg_exprs = []
        for f in features:
            expr = self._build_agg_expression(f)
            if expr is not None:
                agg_exprs.append(expr.alias(f.name))

        if not agg_exprs:
            return cand

        grouped = hist_with_age.group_by(["u_age", "item_id"]).agg(agg_exprs)

        cand_with_age = cand.join(
            self.users_df.select("user_id", pl.col("age").fill_null("unknown").alias("u_age")),
            on="user_id",
            how="left",
        )
        joined = cand_with_age.join(grouped, on=["u_age", "item_id"], how="left")
        return joined.drop("u_age")

    def _compile_sex_item_features(self, cand: pl.DataFrame, features: list[FeatureDef]) -> pl.DataFrame:
        """Расчет признаков в разрезе Пол пользователя x Контент."""
        hist_with_sex = self.hist.join(
            self.users_df.select("user_id", pl.col("sex").fill_null("unknown").alias("u_sex")),
            on="user_id",
            how="left",
        )
        agg_exprs = []
        for f in features:
            expr = self._build_agg_expression(f)
            if expr is not None:
                agg_exprs.append(expr.alias(f.name))

        if not agg_exprs:
            return cand

        grouped = hist_with_sex.group_by(["u_sex", "item_id"]).agg(agg_exprs)

        cand_with_sex = cand.join(
            self.users_df.select("user_id", pl.col("sex").fill_null("unknown").alias("u_sex")),
            on="user_id",
            how="left",
        )
        joined = cand_with_sex.join(grouped, on=["u_sex", "item_id"], how="left")
        return joined.drop("u_sex")

    def _compile_affinity_features(self, cand: pl.DataFrame, features: list[FeatureDef]) -> pl.DataFrame:
        """Расчет аффинити-признаков (user x item по атрибутам)."""
        # Жанровые и контентные предпочтения
        user_ids = cand.select("user_id").unique()
        uhist = self.hist.join(user_ids, on="user_id", how="semi")

        # 1. Доля сериалов в истории пользователя
        u_content_stats = (
            uhist.join(self.items_df.select("item_id", "content_type"), on="item_id", how="left")
            .group_by("user_id")
            .agg(
                (pl.col("content_type") == "series").mean().alias("u_series_pref_ratio"),
                (pl.col("watched_pct") >= 50).mean().alias("u_completion_rate"),
                (pl.col("total_dur") >= 7200).sum().alias("u_binge_watch_cnt"),
            )
        )

        had_content_type = "content_type" in cand.columns
        res = cand.join(u_content_stats, on="user_id", how="left")
        if not had_content_type:
            res = res.join(
                self.items_df.select("item_id", "content_type"), on="item_id", how="left"
            )

        # Вычисляем соответствие типа контента
        if "u_content_type_match" in [f.name for f in features]:
            res = res.with_columns(
                pl.when(
                    ((pl.col("content_type") == "series") & (pl.col("u_series_pref_ratio") > 0.5))
                    | ((pl.col("content_type") == "movie") & (pl.col("u_series_pref_ratio") <= 0.5))
                )
                .then(1.0)
                .otherwise(0.0)
                .alias("u_content_type_match")
            )

        if not had_content_type:
            res = res.drop("content_type", strict=False)

        # Удаляем промежуточные колонки статистики, если их не было в списке целевых фичей
        temp_cols = ["u_series_pref_ratio", "u_completion_rate", "u_binge_watch_cnt"]
        feature_names = {f.name for f in features}
        to_drop = [c for c in temp_cols if c not in feature_names and c in res.columns]
        if to_drop:
            res = res.drop(to_drop)

        return res

    def _build_agg_expression(self, f: FeatureDef) -> pl.Expr | None:
        """Строит выражение Polars для вычисления признака с учетом окна и фильтров."""
        cond = pl.lit(True)
        if f.window.days is not None:
            cutoff = self.T - timedelta(days=f.window.days)
            cond = cond & (pl.col("dt") >= cutoff)

        if f.filter_type == Filter.MOVIES:
            cond = cond & (pl.col("content_type") == "movie")
        elif f.filter_type == Filter.SERIES:
            cond = cond & (pl.col("content_type") == "series")
        elif f.filter_type == Filter.COMPLETED:
            cond = cond & (pl.col("watched_pct") >= 50)

        # Measure column
        if f.measure == Measure.COUNT:
            val_col = pl.lit(1)
        elif f.measure == Measure.TOTAL_DUR:
            val_col = pl.col("total_dur")
        elif f.measure == Measure.WATCHED_PCT:
            val_col = pl.col("watched_pct")
        elif f.measure == Measure.COMPLETED:
            val_col = (pl.col("watched_pct") >= 50).cast(pl.Float32)
        elif f.measure == Measure.LONG_WATCH:
            val_col = (pl.col("total_dur") >= 3600).cast(pl.Float32)
        else:
            val_col = pl.lit(1)

        # Применяем фильтр к значению
        filtered_val = pl.when(cond).then(val_col).otherwise(None)

        if f.agg == Aggregation.COUNT:
            return pl.when(cond).then(1).otherwise(None).count()
        elif f.agg == Aggregation.SUM:
            return filtered_val.sum()
        elif f.agg == Aggregation.MEAN:
            return filtered_val.mean()
        elif f.agg == Aggregation.MAX:
            return filtered_val.max()
        elif f.agg == Aggregation.STD:
            return filtered_val.std()
        elif f.agg == Aggregation.RECENCY:
            dt_filtered = pl.when(cond).then(pl.col("dt")).otherwise(None)
            return (pl.lit(self.T) - dt_filtered.max()).dt.total_days()
        elif f.agg == Aggregation.SHARE:
            # Share of sum over total sum
            part_sum = filtered_val.sum()
            total_sum = val_col.sum()
            return part_sum / (total_sum + 1e-4)

        return None
