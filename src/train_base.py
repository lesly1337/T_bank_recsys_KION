"""Обучение и валидация базового ранжировщика LightGBM lambdarank.

Сравнение с бейзлайнами:
1. Популярное за 7 дней (бейзлайн из EDA)
2. ItemKNN (косинусное сходство)
3. Потолок кандидатов (Candidate Recall)
4. LightGBM lambdarank на 28 базовых признаках

Метрики: Recall@10, MAP@10, NDCG@10, HitRate@10, Coverage.
Парный бутстреп с 95% доверительным интервалом.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.candidates import build_candidates, candidate_recall
from src.data import load_items
from src.features_base import build_base_features
from src.metrics import evaluate, paired_bootstrap, per_user
from src.split import Split, make_splits

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"

IGNORE_COLS = {"user_id", "item_id", "target"}


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in IGNORE_COLS]


def prepare_split_data(
    split: Split,
    items_df: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Строит кандидатов и базовые признаки для среза."""
    cand, item_sim = build_candidates(split)
    feats = build_base_features(cand, split.hist, split.T, items_df=items_df)
    return cand, feats


def extract_topk_recs(
    df: pl.DataFrame,
    sort_cols: list[str],
    descending: list[bool],
    k: int = 10,
) -> dict[int, list[int]]:
    """Извлекает top-K рекомендаций для каждого пользователя по заданным колонкам сортировки."""
    top = (
        df.sort(sort_cols, descending=descending)
        .group_by("user_id", maintain_order=True)
        .head(k)
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id"))
    )
    return {u: items for u, items in top.iter_rows()}


def train_ranker(
    train_feats: pl.DataFrame,
    val_feats: pl.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
) -> lgb.Booster:
    """Обучает LightGBM lambdarank."""
    # LightGBM требует сортировки по группам
    tr_sorted = train_feats.sort("user_id")
    va_sorted = val_feats.sort("user_id")

    pdf_tr = tr_sorted.to_pandas()
    pdf_va = va_sorted.to_pandas()

    groups_tr = pdf_tr.groupby("user_id", sort=False).size().to_numpy()
    groups_va = pdf_va.groupby("user_id", sort=False).size().to_numpy()

    dtrain = lgb.Dataset(
        pdf_tr[feature_cols],
        label=pdf_tr["target"],
        group=groups_tr,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        pdf_va[feature_cols],
        label=pdf_va["target"],
        group=groups_va,
        reference=dtrain,
        free_raw_data=False,
    )

    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "verbose": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    callbacks = []
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    bst = lgb.train(
        default_params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return bst


def evaluate_all(
    feats_df: pl.DataFrame,
    split: Split,
    model: lgb.Booster | None,
    feature_cols: list[str],
    n_catalog: int,
    k: int = 10,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Вычисляет метрики для Pop7, ItemKNN и LightGBM, а также сырые метрики по пользователям."""
    # 1. Популярное за 7 дней (сортировка по pop7_rank по возрастанию)
    recs_pop = extract_topk_recs(feats_df, ["user_id", "pop7_rank"], [False, False], k=k)
    metrics_pop = evaluate(recs_pop, split.gt, k=k, n_items_catalog=n_catalog)
    pu_pop = per_user(recs_pop, split.gt, k=k)

    # 2. ItemKNN (сортировка по knn_score убыванию, затем knn_rank возрастанию)
    recs_knn = extract_topk_recs(feats_df, ["user_id", "knn_score"], [False, True], k=k)
    metrics_knn = evaluate(recs_knn, split.gt, k=k, n_items_catalog=n_catalog)
    pu_knn = per_user(recs_knn, split.gt, k=k)

    results = {
        "pop7": metrics_pop,
        "item_knn": metrics_knn,
    }
    pu_dict = {
        "pop7_ndcg": pu_pop["ndcg"],
        "pop7_recall": pu_pop["recall"],
        "knn_ndcg": pu_knn["ndcg"],
        "knn_recall": pu_knn["recall"],
    }

    # 3. LightGBM
    if model is not None:
        pdf = feats_df.to_pandas()
        scores = model.predict(pdf[feature_cols])
        df_with_score = feats_df.with_columns(pl.Series("score", scores))
        recs_lgb = extract_topk_recs(df_with_score, ["user_id", "score"], [False, True], k=k)
        metrics_lgb = evaluate(recs_lgb, split.gt, k=k, n_items_catalog=n_catalog)
        pu_lgb = per_user(recs_lgb, split.gt, k=k)
        results["lightgbm"] = metrics_lgb
        pu_dict["lgb_ndcg"] = pu_lgb["ndcg"]
        pu_dict["lgb_recall"] = pu_lgb["recall"]

    return results, pu_dict


def run_base_benchmark(n_users: int = 5000, seed: int = 42) -> str:
    print(f"=== Запуск бенчмарка базового ранжировщика (n_users={n_users}, seed={seed}) ===")
    t0 = datetime.now()

    items_df = load_items()
    n_catalog = items_df.height

    splits = make_splits(n_users=n_users, seed=seed)

    print("\n--- 1. Генерация кандидатов и базовых признаков ---")
    c_tr, f_tr = prepare_split_data(splits["train"], items_df=items_df)
    c_va, f_val = prepare_split_data(splits["val"], items_df=items_df)
    c_te, f_te = prepare_split_data(splits["test"], items_df=items_df)

    cand_rec_tr = candidate_recall(c_tr, splits["train"])
    cand_rec_va = candidate_recall(c_va, splits["val"])
    cand_rec_te = candidate_recall(c_te, splits["test"])

    print(f"Train: кандидатов/user={cand_rec_tr['cand_per_user']:.1f}, recall потолок={cand_rec_tr['recall_all']:.4f}")
    print(f"Val:   кандидатов/user={cand_rec_va['cand_per_user']:.1f}, recall потолок={cand_rec_va['recall_all']:.4f}")
    print(f"Test:  кандидатов/user={cand_rec_te['cand_per_user']:.1f}, recall потолок={cand_rec_te['recall_all']:.4f}")

    feature_cols = get_feature_columns(f_tr)
    print(f"\nБазовых признаков: {len(feature_cols)}")

    print("\n--- 2. Обучение LightGBM lambdarank ---")
    model = train_ranker(f_tr, f_val, feature_cols)
    print(f"Обучение завершено. Лучшая итерация: {model.best_iteration}")

    # Важность признаков
    gains = model.feature_importance(importance_type="gain")
    splits_cnt = model.feature_importance(importance_type="split")
    imp_df = (
        pl.DataFrame({
            "feature": feature_cols,
            "gain": gains,
            "split": splits_cnt,
        })
        .with_columns((pl.col("gain") / pl.col("gain").sum() * 100.0).alias("gain_pct"))
        .sort("gain", descending=True)
    )

    print("\nТоп-10 признаков по Gain:")
    for row in imp_df.head(10).iter_rows(named=True):
        print(f"  {row['feature']:<24} {row['gain_pct']:6.2f}% (split: {row['split']})")

    print("\n--- 3. Оценка качества на Validation и Test ---")
    val_res, val_pu = evaluate_all(f_val, splits["val"], model, feature_cols, n_catalog, k=10)
    test_res, test_pu = evaluate_all(f_te, splits["test"], model, feature_cols, n_catalog, k=10)

    # Бутстреп разницы LightGBM vs Pop7 и vs ItemKNN
    diff_val_pop, v_pop_lo, v_pop_hi = paired_bootstrap(val_pu["pop7_ndcg"], val_pu["lgb_ndcg"])
    diff_val_knn, v_knn_lo, v_knn_hi = paired_bootstrap(val_pu["knn_ndcg"], val_pu["lgb_ndcg"])

    diff_te_pop, t_pop_lo, t_pop_hi = paired_bootstrap(test_pu["pop7_ndcg"], test_pu["lgb_ndcg"])
    diff_te_knn, t_knn_lo, t_knn_hi = paired_bootstrap(test_pu["knn_ndcg"], test_pu["lgb_ndcg"])

    # Формируем итоговый текстовый отчет
    report_lines = [
        "# Отчёт: Базовый ранжировщик LightGBM vs Бейзлайны",
        f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Сэмпл пользователей: {n_users} на срез | Всего признаков: {len(feature_cols)}",
        "",
        "## 1. Потолок генерации кандидатов (Candidate Recall)",
        "| Срез | Кандидатов/user | Recall (all) | Recall (KNN) | Recall (Pop7) | Доля user c pos |",
        "|---|---|---|---|---|---|",
        f"| Train | {cand_rec_tr['cand_per_user']:.1f} | {cand_rec_tr['recall_all']:.4f} | {cand_rec_tr['recall_knn']:.4f} | {cand_rec_tr['recall_pop']:.4f} | {cand_rec_tr['users_with_pos']:.3f} |",
        f"| Val   | {cand_rec_va['cand_per_user']:.1f} | {cand_rec_va['recall_all']:.4f} | {cand_rec_va['recall_knn']:.4f} | {cand_rec_va['recall_pop']:.4f} | {cand_rec_va['users_with_pos']:.3f} |",
        f"| Test  | {cand_rec_te['cand_per_user']:.1f} | {cand_rec_te['recall_all']:.4f} | {cand_rec_te['recall_knn']:.4f} | {cand_rec_te['recall_pop']:.4f} | {cand_rec_te['users_with_pos']:.3f} |",
        "",
        "## 2. Метрики ранжирования Top-10 (Validation)",
        "| Модель | Recall@10 | MAP@10 | NDCG@10 | HitRate@10 | Coverage |",
        "|---|---|---|---|---|---|",
        f"| Популярное 7д | {val_res['pop7']['recall@10']:.4f} | {val_res['pop7']['map@10']:.4f} | {val_res['pop7']['ndcg@10']:.4f} | {val_res['pop7']['hit@10']:.4f} | {val_res['pop7']['coverage']:.4f} |",
        f"| ItemKNN       | {val_res['item_knn']['recall@10']:.4f} | {val_res['item_knn']['map@10']:.4f} | {val_res['item_knn']['ndcg@10']:.4f} | {val_res['item_knn']['hit@10']:.4f} | {val_res['item_knn']['coverage']:.4f} |",
        f"| **LightGBM**  | **{val_res['lightgbm']['recall@10']:.4f}** | **{val_res['lightgbm']['map@10']:.4f}** | **{val_res['lightgbm']['ndcg@10']:.4f}** | **{val_res['lightgbm']['hit@10']:.4f}** | **{val_res['lightgbm']['coverage']:.4f}** |",
        "",
        f"Статзначимость на Val (Δ NDCG@10, 95% ДИ бутстрепа):",
        f"- LightGBM vs Pop7:   {diff_val_pop:+.4f} [{v_pop_lo:+.4f}, {v_pop_hi:+.4f}]",
        f"- LightGBM vs ItemKNN: {diff_val_knn:+.4f} [{v_knn_lo:+.4f}, {v_knn_hi:+.4f}]",
        "",
        "## 3. Метрики ранжирования Top-10 (Test)",
        "| Модель | Recall@10 | MAP@10 | NDCG@10 | HitRate@10 | Coverage |",
        "|---|---|---|---|---|---|",
        f"| Популярное 7д | {test_res['pop7']['recall@10']:.4f} | {test_res['pop7']['map@10']:.4f} | {test_res['pop7']['ndcg@10']:.4f} | {test_res['pop7']['hit@10']:.4f} | {test_res['pop7']['coverage']:.4f} |",
        f"| ItemKNN       | {test_res['item_knn']['recall@10']:.4f} | {test_res['item_knn']['map@10']:.4f} | {test_res['item_knn']['ndcg@10']:.4f} | {test_res['item_knn']['hit@10']:.4f} | {test_res['item_knn']['coverage']:.4f} |",
        f"| **LightGBM**  | **{test_res['lightgbm']['recall@10']:.4f}** | **{test_res['lightgbm']['map@10']:.4f}** | **{test_res['lightgbm']['ndcg@10']:.4f}** | **{test_res['lightgbm']['hit@10']:.4f}** | **{test_res['lightgbm']['coverage']:.4f}** |",
        "",
        f"Статзначимость на Test (Δ NDCG@10, 95% ДИ бутстрепа):",
        f"- LightGBM vs Pop7:   {diff_te_pop:+.4f} [{t_pop_lo:+.4f}, {t_pop_hi:+.4f}]",
        f"- LightGBM vs ItemKNN: {diff_te_knn:+.4f} [{t_knn_lo:+.4f}, {t_knn_hi:+.4f}]",
        "",
        "## 4. Топ-15 важности базовых признаков (LightGBM)",
        "| Ранг | Признак | Доля Gain | Число сплитов |",
        "|---|---|---|---|",
    ]
    for idx, row in enumerate(imp_df.head(15).iter_rows(named=True), 1):
        report_lines.append(f"| {idx} | `{row['feature']}` | {row['gain_pct']:.2f}% | {row['split']} |")

    report_lines.append(f"\nВремя работы пайплайна: {(datetime.now() - t0).total_seconds():.1f} с")
    report_text = "\n".join(report_lines)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "base_ranker.md"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"\nОтчёт успешно сохранён в {report_file}")
    print("\n" + report_text)
    return report_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение базового ранжировщика LightGBM")
    parser.add_argument("--users", type=int, default=5000, help="Количество пользователей на срез (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно (default: 42)")
    args = parser.parse_args()

    run_base_benchmark(n_users=args.users, seed=args.seed)
