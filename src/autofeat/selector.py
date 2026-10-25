"""3-этапный протокол отбора признаков (строго по validation, тест не трогается):
1. Дешевые фильтры: константность, пропуски, корреляция дубликатов (|r| > 0.98).
2. Null Importance: проверка против шумового распределения при перестановке (permutation) таргета.
3. Жадный forward-отбор: добавление признаков по приросту NDCG@10 на валидации.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np
import polars as pl
import lightgbm as lgb

from src.train_base import train_ranker, evaluate_all
from src.split import Split


class FeatureSelector:
    """Оркестратор трехэтапного отбора сгенерированных признаков."""

    def __init__(
        self,
        base_feature_cols: List[str],
        n_catalog: int,
    ):
        self.base_cols = base_feature_cols
        self.n_catalog = n_catalog

    def cheap_filter(
        self,
        df: pl.DataFrame,
        candidate_cols: List[str],
        max_missing: float = 0.90,
        min_std: float = 1e-4,
        max_corr: float = 0.98,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Этап 1: Быстрые фильтры по статистикам признаков."""
        passed: List[str] = []
        rejected: Dict[str, str] = {}

        # Проверяем существующие базовые признаки для дедупликации (кэшируем массивы)
        all_numeric = [c for c in self.base_cols if df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
        base_cache = {}
        for base_c in all_numeric:
            b_arr = df[base_c].fill_null(0.0).to_numpy()
            b_std = float(np.std(b_arr))
            if b_std > 1e-5:
                base_cache[base_c] = (b_arr, b_std)

        passed_cache = {}

        for col in candidate_cols:
            if col not in df.columns:
                rejected[col] = "not_found"
                continue

            series = df[col]
            # 1. Пропуски
            null_rate = series.is_null().sum() / len(series)
            if null_rate > max_missing:
                rejected[col] = f"high_null_rate_{null_rate:.2f}"
                continue

            arr = series.fill_null(0.0).to_numpy()

            # 2. Дисперсия
            std = float(np.std(arr))
            if std < min_std:
                rejected[col] = f"low_variance_std_{std:.6f}"
                continue

            # 3. Корреляция с базовыми признаками
            is_duplicate = False
            for base_c, (base_arr, b_std) in base_cache.items():
                corr = float(np.corrcoef(arr, base_arr)[0, 1])
                if not np.isnan(corr) and abs(corr) >= max_corr:
                    rejected[col] = f"duplicate_with_base_{base_c}_corr_{corr:.2f}"
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # 4. Корреляция с уже отобранными кандидатами
            for prev_col, (prev_arr, p_std) in passed_cache.items():
                corr = float(np.corrcoef(arr, prev_arr)[0, 1])
                if not np.isnan(corr) and abs(corr) >= max_corr:
                    rejected[col] = f"duplicate_with_{prev_col}_corr_{corr:.2f}"
                    is_duplicate = True
                    break

            if not is_duplicate:
                passed.append(col)
                passed_cache[col] = (arr, std)

        return passed, rejected

    def null_importance_test(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        candidate_cols: List[str],
        num_boost_round: int = 50,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Этап 2: Null Importance тест (сравнение с важностью при перестановке таргета)."""
        if not candidate_cols:
            return [], {}

        features_to_check = self.base_cols + candidate_cols

        # 1. Обучаем на реальном таргете
        model_real = train_ranker(
            train_df,
            val_df,
            features_to_check,
            num_boost_round=num_boost_round,
            early_stopping_rounds=15,
        )
        real_gains = dict(zip(features_to_check, model_real.feature_importance(importance_type="gain")))

        # 2. Обучаем на перемешанном таргете (null distribution без ранней остановки на валидации)
        train_perm = train_df.with_columns(
            pl.col("target").sample(fraction=1.0, shuffle=True, seed=123).alias("target")
        )
        model_null = train_ranker(
            train_perm,
            val_df,
            features_to_check,
            num_boost_round=min(num_boost_round, 30),
            early_stopping_rounds=None,
        )
        null_gains = dict(zip(features_to_check, model_null.feature_importance(importance_type="gain")))

        passed = []
        gain_ratios = {}
        for c in candidate_cols:
            r_gain = real_gains.get(c, 0.0)
            n_gain = null_gains.get(c, 0.0)
            ratio = r_gain / (n_gain + 1e-4)
            gain_ratios[c] = ratio
            # Оставляем те признаки, у которых реальная важность превышает шум
            if r_gain > 0 and (ratio >= 1.2 or (r_gain > 10.0 and ratio >= 1.0)):
                passed.append(c)

        # Сортируем прошедшие по gain ratio
        passed.sort(key=lambda col: gain_ratios.get(col, 0.0), reverse=True)
        return passed, gain_ratios

    def forward_selection(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        val_split: Split,
        candidate_cols: List[str],
        k: int = 10,
        min_delta: float = 0.0001,
        max_steps: int = 10,
        verbose: bool = True,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Этап 3: Жадный forward-отбор по NDCG@10 на валидации."""
        current_features = list(self.base_cols)

        # Базовая модель
        base_model = train_ranker(train_df, val_df, current_features, num_boost_round=100)
        base_res, _ = evaluate_all(val_df, val_split, base_model, current_features, self.n_catalog, k=k)
        best_ndcg = base_res["lightgbm"]["ndcg@10"]

        if verbose:
            print(f"  [Forward-отбор] Старт с бейзлайна: NDCG@10 = {best_ndcg:.4f} (признаков: {len(current_features)})")

        history = [{
            "step": 0,
            "feature_added": "BASELINE",
            "val_ndcg": best_ndcg,
            "delta": 0.0,
            "total_features": len(current_features),
        }]

        selected: List[str] = []
        remaining = list(candidate_cols)

        step = 1
        while remaining and step <= max_steps:
            best_cand = None
            best_cand_ndcg = best_ndcg

            for cand in remaining:
                trial_features = current_features + [cand]
                model = train_ranker(train_df, val_df, trial_features, num_boost_round=100)
                res, _ = evaluate_all(val_df, val_split, model, trial_features, self.n_catalog, k=k)
                cand_ndcg = res["lightgbm"]["ndcg@10"]

                if cand_ndcg > best_cand_ndcg:
                    best_cand_ndcg = cand_ndcg
                    best_cand = cand

            delta = best_cand_ndcg - best_ndcg
            if best_cand is not None and delta >= min_delta:
                selected.append(best_cand)
                remaining.remove(best_cand)
                current_features.append(best_cand)
                best_ndcg = best_cand_ndcg

                if verbose:
                    print(f"  [Step {step}] Добавлен: {best_cand:<30} -> NDCG@10 = {best_ndcg:.4f} (Δ={delta:+.4f})")

                history.append({
                    "step": step,
                    "feature_added": best_cand,
                    "val_ndcg": best_ndcg,
                    "delta": delta,
                    "total_features": len(current_features),
                })
                step += 1
            else:
                if verbose:
                    print(f"  [Forward-отбор] Остановка на шаге {step}: нет признаков с приростом >= {min_delta}")
                break

        return selected, history
