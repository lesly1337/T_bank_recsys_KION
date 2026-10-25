"""Загрузка датасета KION (МТС) с кэшированием в parquet.

Исходные файлы (data/*.csv):
    interactions.csv : user_id, item_id, last_watch_dt, total_dur, watched_pct
    users.csv        : user_id, age, income, sex, kids_flg
    items.csv        : item_id, content_type, title, title_orig, release_year, genres,
                       countries, for_kids, age_rating, studios, directors, actors,
                       description, keywords

ВАЖНО про природу interactions: одна строка = одна пара (user, item) за весь период.
`last_watch_dt` — дата ПОСЛЕДНЕГО просмотра, `total_dur` и `watched_pct` — накопленные
значения на конец периода. Это влияет на постановку задачи и на утечки (см. EDA).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "processed"

# Поля items, которые содержат списки через ", "
ITEM_LIST_COLS = ["genres", "countries", "studios", "directors", "actors"]


def _cached(name: str, builder) -> pl.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    df = builder()
    df.write_parquet(path)
    return df


def load_interactions() -> pl.DataFrame:
    def build() -> pl.DataFrame:
        return (
            pl.read_csv(
                DATA_DIR / "interactions.csv",
                schema_overrides={
                    "user_id": pl.Int32,
                    "item_id": pl.Int32,
                    "last_watch_dt": pl.Utf8,
                    "total_dur": pl.Int64,
                    "watched_pct": pl.Float32,
                },
            )
            .with_columns(pl.col("last_watch_dt").str.to_date("%Y-%m-%d"))
            .rename({"last_watch_dt": "dt"})
        )

    return _cached("interactions", build)


def load_users() -> pl.DataFrame:
    def build() -> pl.DataFrame:
        return pl.read_csv(
            DATA_DIR / "users.csv",
            schema_overrides={"user_id": pl.Int32, "kids_flg": pl.Int8},
        )

    return _cached("users", build)


def load_items() -> pl.DataFrame:
    def build() -> pl.DataFrame:
        df = pl.read_csv(
            DATA_DIR / "items.csv",
            schema_overrides={"item_id": pl.Int32},
            infer_schema_length=20000,
        )
        # списковые поля: "a, b, c" -> ["a", "b", "c"] (нормализуем регистр/пробелы)
        return df.with_columns(
            [
                pl.col(c)
                .str.to_lowercase()
                .str.split(",")
                .list.eval(pl.element().str.strip_chars())
                .list.filter(pl.element() != "")
                .alias(c)
                for c in ITEM_LIST_COLS
            ]
        )

    return _cached("items", build)
