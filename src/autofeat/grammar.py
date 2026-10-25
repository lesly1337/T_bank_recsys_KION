"""Декларативная грамматика признаков и SQL-шаблоны для AutoFeat.

Каждый признак описывается структурой FeatureDef, представляющей точку в пространстве:
    feature = AGG( MEASURE )
              FROM interactions [JOIN relations]
              WHERE dt in WINDOW [AND FILTER]
              GROUP BY ENTITY
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Entity(str, Enum):
    USER = "user"                       # GROUP BY user_id
    ITEM = "item"                       # GROUP BY item_id
    USER_ITEM_AFFINITY = "user_item"    # пересечение интересов user и метаданных item
    AGE_ITEM = "age_item"               # демография: age x item_id
    SEX_ITEM = "sex_item"               # демография: sex x item_id


class Window(str, Enum):
    W1 = "1d"
    W3 = "3d"
    W7 = "7d"
    W14 = "14d"
    W30 = "30d"
    W60 = "60d"
    ALL = "all"

    @property
    def days(self) -> Optional[int]:
        mapping = {"1d": 1, "3d": 3, "7d": 7, "14d": 14, "30d": 30, "60d": 60, "all": None}
        return mapping[self.value]


class Measure(str, Enum):
    COUNT = "count"
    TOTAL_DUR = "total_dur"
    WATCHED_PCT = "watched_pct"
    COMPLETED = "completed"        # watched_pct >= 50
    LONG_WATCH = "long_watch"      # total_dur >= 3600 (более 1 часа)


class Aggregation(str, Enum):
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    STD = "std"
    SHARE = "share"                # доля от общего по пользователю/объекту
    RECENCY = "recency"            # дней с последнего события до T


class Filter(str, Enum):
    ALL = "all"
    MOVIES = "movies"              # только фильмы
    SERIES = "series"              # только сериалы
    COMPLETED = "completed"        # только досмотренные (watched_pct >= 50)


class Author(str, Enum):
    HUMAN = "human"
    BRUTE_FORCE = "brute_force"
    AI_AGENT = "ai_agent"


@dataclass(frozen=True)
class FeatureDef:
    name: str
    author: Author
    entity: Entity
    window: Window
    measure: Measure
    agg: Aggregation
    filter_type: Filter = Filter.ALL
    rationale: str = ""

    def to_sql(self, table_name: str = "interactions") -> str:
        """Компилирует декларативное определение в читаемый SQL-запрос."""
        # 1. Measure SQL expr
        if self.measure == Measure.COUNT:
            col_expr = "*"
        elif self.measure == Measure.TOTAL_DUR:
            col_expr = "total_dur"
        elif self.measure == Measure.WATCHED_PCT:
            col_expr = "watched_pct"
        elif self.measure == Measure.COMPLETED:
            col_expr = "CASE WHEN watched_pct >= 50 THEN 1 ELSE 0 END"
        elif self.measure == Measure.LONG_WATCH:
            col_expr = "CASE WHEN total_dur >= 3600 THEN 1 ELSE 0 END"
        else:
            col_expr = "*"

        # 2. Aggregation SQL expr
        if self.agg == Aggregation.COUNT:
            agg_expr = f"COUNT({col_expr})"
        elif self.agg == Aggregation.SUM:
            agg_expr = f"SUM({col_expr})"
        elif self.agg == Aggregation.MEAN:
            agg_expr = f"AVG({col_expr})"
        elif self.agg == Aggregation.MAX:
            agg_expr = f"MAX({col_expr})"
        elif self.agg == Aggregation.STD:
            agg_expr = f"STDDEV({col_expr})"
        elif self.agg == Aggregation.SHARE:
            agg_expr = f"SUM({col_expr}) / NULLIF(SUM(SUM({col_expr})) OVER (PARTITION BY {self.entity.value}_id), 0)"
        elif self.agg == Aggregation.RECENCY:
            agg_expr = f"DATE_DIFF('day', MAX(dt), T)"
        else:
            agg_expr = f"COUNT(*)"

        # 3. Where clause
        conditions = ["dt < T"]
        if self.window.days is not None:
            conditions.append(f"dt >= T - INTERVAL '{self.window.days} day'")
        if self.filter_type == Filter.MOVIES:
            conditions.append("content_type = 'movie'")
        elif self.filter_type == Filter.SERIES:
            conditions.append("content_type = 'series'")
        elif self.filter_type == Filter.COMPLETED:
            conditions.append("watched_pct >= 50")

        where_clause = " AND ".join(conditions)
        group_col = f"{self.entity.value}_id" if self.entity in (Entity.USER, Entity.ITEM) else "entity_key"

        return (
            f"SELECT {group_col}, {agg_expr} AS {self.name}\n"
            f"FROM {table_name}\n"
            f"WHERE {where_clause}\n"
            f"GROUP BY {group_col}"
        )
