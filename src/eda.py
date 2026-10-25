"""Первичный анализ датасета KION.

Запуск:  .venv\\Scripts\\python -m src.eda
Результат: текстовый отчёт в stdout + графики в reports/eda/*.png
"""
from __future__ import annotations

import sys
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.data import ITEM_LIST_COLS, ROOT, load_interactions, load_items, load_users

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)

OUT = ROOT / "reports" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def h(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)


def main() -> None:
    inter = load_interactions()
    users = load_users()
    items = load_items()

    # ------------------------------------------------------------------ 1. Общее
    h("1. Размеры и схемы")
    for name, df in [("interactions", inter), ("users", users), ("items", items)]:
        print(f"{name:13s} shape={df.shape}  mem={df.estimated_size('mb'):.1f} MB")
        print("   nulls:", {c: n for c, n in zip(df.columns, df.null_count().row(0)) if n})
    print(inter.head())

    # ------------------------------------------------------------------ 2. Interactions
    h("2. Interactions: ключи, даты, дубли")
    n_pairs = inter.select(pl.struct("user_id", "item_id").n_unique()).item()
    print(f"rows={inter.height:,}  unique (user,item)={n_pairs:,}  -> дублей пар: {inter.height - n_pairs:,}")
    print(f"users={inter['user_id'].n_unique():,}  items={inter['item_id'].n_unique():,}")
    dmin, dmax = inter["dt"].min(), inter["dt"].max()
    print(f"даты: {dmin} .. {dmax}  ({(dmax - dmin).days + 1} дней)")
    density = inter.height / (inter["user_id"].n_unique() * inter["item_id"].n_unique())
    print(f"плотность матрицы: {density:.2e}")

    daily = inter.group_by("dt").agg(
        pl.len().alias("events"), pl.col("user_id").n_unique().alias("users")
    ).sort("dt")
    weekly = (
        inter.with_columns(pl.col("dt").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.len().alias("events"), pl.col("user_id").n_unique().alias("users"),
             pl.col("item_id").n_unique().alias("items"))
        .sort("week")
    )
    print("\nПо неделям (неделя = понедельник):")
    print(weekly)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(daily["dt"], daily["events"], label="events")
    ax.plot(daily["dt"], daily["users"], label="active users")
    ax.set_title("Активность по дням (по last_watch_dt)")
    ax.legend()
    save(fig, "daily_activity.png")

    dow = inter.group_by(pl.col("dt").dt.weekday().alias("weekday")).len().sort("weekday")
    print("\nПо дням недели (1=пн):", dict(zip(dow["weekday"], dow["len"])))

    # ------------------------------------------------------------------ 3. total_dur / watched_pct
    h("3. total_dur и watched_pct")
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    print(inter.select(
        [pl.col("total_dur").quantile(x).alias(f"dur_q{x}") for x in q]
    ))
    print(inter.select(
        [pl.col("watched_pct").quantile(x).alias(f"pct_q{x}") for x in q]
    ))
    print("watched_pct null:", inter["watched_pct"].null_count(),
          "| >100:", inter.filter(pl.col("watched_pct") > 100).height)
    for thr_d, thr_p in [(300, None), (None, 10), (None, 50), (None, 90)]:
        cond = pl.col("total_dur") >= thr_d if thr_d else pl.col("watched_pct") >= thr_p
        share = inter.select(cond.mean()).item()
        label = f"total_dur>={thr_d}s" if thr_d else f"watched_pct>={thr_p}"
        print(f"  доля строк с {label:18s}: {share:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(np.log10(inter["total_dur"].to_numpy().clip(1)), bins=80)
    axes[0].set_title("log10(total_dur, сек)")
    axes[1].hist(inter["watched_pct"].drop_nulls().to_numpy(), bins=101)
    axes[1].set_title("watched_pct")
    save(fig, "dur_pct_hist.png")

    # ------------------------------------------------------------------ 4. Активность пользователей / популярность объектов
    h("4. Длинные хвосты: пользователи и объекты")
    u_cnt = inter.group_by("user_id").len()
    i_cnt = inter.group_by("item_id").len().sort("len", descending=True)
    print("взаимодействий на пользователя:",
          u_cnt.select([pl.col("len").quantile(x).alias(f"q{x}") for x in q] + [pl.col("len").max().alias("max")]))
    print("доля пользователей с 1 взаимодействием:", round((u_cnt["len"] == 1).mean(), 3),
          "| с <=3:", round((u_cnt["len"] <= 3).mean(), 3))
    print("взаимодействий на объект:",
          i_cnt.select([pl.col("len").quantile(x).alias(f"q{x}") for x in q] + [pl.col("len").max().alias("max")]))
    cum = i_cnt["len"].cum_sum() / i_cnt["len"].sum()
    for k in [10, 100, 1000]:
        print(f"  топ-{k} объектов покрывают {cum[k - 1]:.1%} взаимодействий")
    print(f"  Gini популярности объектов: {gini(i_cnt['len'].to_numpy()):.3f}")
    top = i_cnt.head(15).join(items.select("item_id", "title", "content_type"), on="item_id", how="left")
    print(top)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].loglog(np.sort(u_cnt["len"].to_numpy())[::-1])
    axes[0].set_title("Активность пользователей (rank-freq)")
    axes[1].loglog(i_cnt["len"].to_numpy())
    axes[1].set_title("Популярность объектов (rank-freq)")
    save(fig, "long_tail.png")

    # ------------------------------------------------------------------ 5. Временная динамика: холодный старт, дрейф
    h("5. Холодный старт и дрейф по неделям (важно для сплита)")
    test_start = dmax - timedelta(days=6)
    rows = []
    for k in range(4):
        ws = test_start - timedelta(days=7 * k)
        we = ws + timedelta(days=7)
        past = inter.filter(pl.col("dt") < ws)
        cur = inter.filter((pl.col("dt") >= ws) & (pl.col("dt") < we))
        past_u = set(past["user_id"].unique().to_list())
        past_i = set(past["item_id"].unique().to_list())
        cur_u = cur["user_id"].unique().to_list()
        cur_i = cur["item_id"].unique().to_list()
        warm_rows = cur.filter(pl.col("user_id").is_in(list(past_u)) & pl.col("item_id").is_in(list(past_i)))
        # популярность: топ-10 за предыдущие 7 дней vs. топ-10 текущей недели
        prev7 = inter.filter((pl.col("dt") >= ws - timedelta(days=7)) & (pl.col("dt") < ws))
        top_prev = set(prev7.group_by("item_id").len().sort("len", descending=True).head(10)["item_id"].to_list())
        top_cur = set(cur.group_by("item_id").len().sort("len", descending=True).head(10)["item_id"].to_list())
        rows.append(dict(
            week_start=ws, events=cur.height, users=len(cur_u), items=len(cur_i),
            cold_users=round(1 - np.mean([u in past_u for u in cur_u]), 3),
            cold_items=round(1 - np.mean([i in past_i for i in cur_i]), 3),
            warm_rows_share=round(warm_rows.height / cur.height, 3),
            top10_overlap_prev_week=len(top_prev & top_cur),
        ))
    print(pl.DataFrame(rows).sort("week_start"))

    # ------------------------------------------------------------------ 6. Бейзлайн популярности на последней неделе
    h("6. Sanity-бейзлайн: популярное за последние N дней -> Recall@10 / MAP-ish на тестовой неделе")
    hist = inter.filter(pl.col("dt") < test_start)
    test = inter.filter(pl.col("dt") >= test_start)
    warm_test = test.filter(pl.col("user_id").is_in(hist["user_id"].unique().implode()))
    seen = hist.select("user_id", "item_id")
    gt = warm_test.group_by("user_id").agg(pl.col("item_id").alias("gt"))
    print(f"тестовая неделя: {test_start}..{dmax}, users={test['user_id'].n_unique():,}, "
          f"тёплых={gt.height:,}")
    for days in [1, 3, 7, 14, 30, 60, None]:
        h_win = hist if days is None else hist.filter(pl.col("dt") >= test_start - timedelta(days=days))
        pop = h_win.group_by("item_id").len().sort("len", descending=True).head(200)["item_id"].to_list()
        # рекомендации без уже просмотренного (повторы в данных невозможны — пара уникальна)
        recs = (
            gt.select("user_id")
            .join(pl.DataFrame({"item_id": pop, "rank": list(range(len(pop)))},
                               schema={"item_id": pl.Int32, "rank": pl.Int64}), how="cross")
            .join(seen, on=["user_id", "item_id"], how="anti")
            .sort(["user_id", "rank"])
            .group_by("user_id", maintain_order=True)
            .head(10)
            .group_by("user_id").agg(pl.col("item_id").alias("rec"))
        )
        m = gt.join(recs, on="user_id", how="left").with_columns(
            pl.col("rec").fill_null(pl.lit([], dtype=pl.List(pl.Int32)))
        ).with_columns(
            pl.col("gt").list.set_intersection("rec").list.len().alias("hits"),
            pl.col("gt").list.len().alias("n_gt"),
        ).with_columns(
            (pl.col("hits") / pl.min_horizontal(pl.col("n_gt"), pl.lit(10))).alias("recall10")
        )
        print(f"  окно={str(days):>4} дн: Recall@10={m['recall10'].mean():.4f}  HitRate@10={(m['hits'] > 0).mean():.4f}")

    # ------------------------------------------------------------------ 7. Users
    h("7. Users")
    inter_users = inter.select("user_id").unique()
    cov = inter_users.join(users, on="user_id", how="semi").height / inter_users.height
    print(f"users.csv: {users.height:,} строк; покрытие пользователей из interactions: {cov:.1%}")
    for c in ["age", "income", "sex", "kids_flg"]:
        vc = users[c].value_counts(sort=True).with_columns((pl.col("count") / users.height).round(3).alias("share"))
        print(f"\n{c}:\n{vc}")

    # ------------------------------------------------------------------ 8. Items
    h("8. Items")
    inter_items = inter.select("item_id").unique()
    cov = inter_items.join(items, on="item_id", how="semi").height / inter_items.height
    print(f"items.csv: {items.height:,} строк; покрытие объектов из interactions: {cov:.1%}")
    for c in ["content_type", "for_kids", "age_rating"]:
        print(f"\n{c}:\n{items[c].value_counts(sort=True)}")
    print("\nrelease_year:", items.select(
        [pl.col("release_year").quantile(x).alias(f"q{x}") for x in [0.05, 0.25, 0.5, 0.75, 0.95]]))
    print("\nМногозначные поля (кардинальность / ср. длина списка / топ-5):")
    for c in ITEM_LIST_COLS:
        ex = items.select(c).explode(c).drop_nulls()
        topv = ex[c].value_counts(sort=True).head(5)
        print(f"  {c:10s} uniq={ex[c].n_unique():6d}  mean_len={items[c].list.len().mean():.2f}  "
              f"top={list(zip(topv[c], topv['count']))}")

    # тип контента vs поведение
    ct = inter.join(items.select("item_id", "content_type"), on="item_id").group_by("content_type").agg(
        pl.len().alias("rows"),
        pl.col("total_dur").median().alias("dur_med"),
        pl.col("watched_pct").median().alias("pct_med"),
    )
    print("\nПоведение по content_type:\n", ct)

    # ------------------------------------------------------------------ 9. Сигналы для pair-признаков
    h("9. Насколько полезны косвенные связи (user x genre / director / ...)")
    # доля тестовых взаимодействий, где жанр объекта уже встречался в истории пользователя
    it_g = items.select("item_id", "genres").explode("genres").drop_nulls()
    hist_ug = hist.select("user_id", "item_id").join(it_g, on="item_id").select("user_id", "genres").unique()
    test_ug = warm_test.select("user_id", "item_id").join(it_g, on="item_id")
    hit = test_ug.join(hist_ug, on=["user_id", "genres"], how="semi").select("user_id", "item_id").unique().height
    print(f"доля тёплых тестовых пар, у которых хотя бы один жанр уже смотрелся пользователем: "
          f"{hit / warm_test.height:.3f}")
    for col in ["directors", "actors", "studios", "countries"]:
        it_x = items.select("item_id", col).explode(col).drop_nulls()
        hist_ux = hist.select("user_id", "item_id").join(it_x, on="item_id").select("user_id", col).unique()
        test_ux = warm_test.select("user_id", "item_id").join(it_x, on="item_id")
        hit = test_ux.join(hist_ux, on=["user_id", col], how="semi").select("user_id", "item_id").unique().height
        print(f"  ... по {col:10s}: {hit / warm_test.height:.3f}")

    print(f"\nГрафики сохранены в {OUT}")


if __name__ == "__main__":
    main()
