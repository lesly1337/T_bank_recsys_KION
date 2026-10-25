"""Генераторы признаков трех типов:
1. HumanFeatureGenerator: доменная экспертиза (соревнования KION/MTS).
2. BruteForceFeatureGenerator: систематический перебор грамматики признаков.
3. AIAgentFeatureGenerator: гипотезы на основе EDA и рефлексии (дрейф популярности, специфика сериалов, длинный хвост).
"""
from __future__ import annotations

from typing import List

from src.autofeat.grammar import Aggregation, Author, Entity, FeatureDef, Filter, Measure, Window


class HumanFeatureGenerator:
    """Генератор признаков на основе человеческой экспертизы (доменные эвристики)."""

    def generate(self) -> List[FeatureDef]:
        features = [
            # 1. Активность пользователя в разных окнах
            FeatureDef(
                name="h_u_dur_sum_7d",
                author=Author.HUMAN,
                entity=Entity.USER,
                window=Window.W7,
                measure=Measure.TOTAL_DUR,
                agg=Aggregation.SUM,
                rationale="Суммарное время просмотра за последнюю неделю (краткосрочная активность)",
            ),
            FeatureDef(
                name="h_u_dur_sum_30d",
                author=Author.HUMAN,
                entity=Entity.USER,
                window=Window.W30,
                measure=Measure.TOTAL_DUR,
                agg=Aggregation.SUM,
                rationale="Суммарное время просмотра за месяц (базовый уровень вовлеченности)",
            ),
            FeatureDef(
                name="h_u_cnt_completed_30d",
                author=Author.HUMAN,
                entity=Entity.USER,
                window=Window.W30,
                measure=Measure.COMPLETED,
                agg=Aggregation.COUNT,
                filter_type=Filter.COMPLETED,
                rationale="Количество досмотренных до конца фильмов/сериалов за 30 дней",
            ),
            FeatureDef(
                name="h_u_mean_pct_14d",
                author=Author.HUMAN,
                entity=Entity.USER,
                window=Window.W14,
                measure=Measure.WATCHED_PCT,
                agg=Aggregation.MEAN,
                rationale="Средний процент досмотра контента за 2 недели",
            ),
            # 2. Популярность и качество контента
            FeatureDef(
                name="h_i_cnt_14d",
                author=Author.HUMAN,
                entity=Entity.ITEM,
                window=Window.W14,
                measure=Measure.COUNT,
                agg=Aggregation.COUNT,
                rationale="Просмотры контента за 14 дней (промежуточное окно популярности)",
            ),
            FeatureDef(
                name="h_i_dur_sum_7d",
                author=Author.HUMAN,
                entity=Entity.ITEM,
                window=Window.W7,
                measure=Measure.TOTAL_DUR,
                agg=Aggregation.SUM,
                rationale="Суммарные минуты просмотра контента за 7 дней (сила вовлечения)",
            ),
            FeatureDef(
                name="h_i_completed_cnt_30d",
                author=Author.HUMAN,
                entity=Entity.ITEM,
                window=Window.W30,
                measure=Measure.COMPLETED,
                agg=Aggregation.COUNT,
                filter_type=Filter.COMPLETED,
                rationale="Количество успешных досмотров фильма за месяц",
            ),
            FeatureDef(
                name="h_i_mean_pct_7d",
                author=Author.HUMAN,
                entity=Entity.ITEM,
                window=Window.W7,
                measure=Measure.WATCHED_PCT,
                agg=Aggregation.MEAN,
                rationale="Средний процент досмотра новинок за неделю",
            ),
            # 3. Сегментная популярность (Демография x Контент)
            FeatureDef(
                name="h_age_item_cnt_30d",
                author=Author.HUMAN,
                entity=Entity.AGE_ITEM,
                window=Window.W30,
                measure=Measure.COUNT,
                agg=Aggregation.COUNT,
                rationale="Популярность контента внутри возрастной группы пользователя за 30 дней",
            ),
            FeatureDef(
                name="h_sex_item_cnt_30d",
                author=Author.HUMAN,
                entity=Entity.SEX_ITEM,
                window=Window.W30,
                measure=Measure.COUNT,
                agg=Aggregation.COUNT,
                rationale="Популярность контента внутри гендерной группы пользователя за 30 дней",
            ),
            # 4. Аффинити и контентные предпочтения
            FeatureDef(
                name="u_content_type_match",
                author=Author.HUMAN,
                entity=Entity.USER_ITEM_AFFINITY,
                window=Window.ALL,
                measure=Measure.COUNT,
                agg=Aggregation.MEAN,
                rationale="Соответствие типа контента (сериал/фильм) привычке пользователя",
            ),
        ]
        return features


class BruteForceFeatureGenerator:
    """Алгоритмический генератор признаков: комбинаторный перебор сетки грамматики."""

    def generate(self) -> List[FeatureDef]:
        features: List[FeatureDef] = []

        # Сетка параметров перебора
        entities = [Entity.USER, Entity.ITEM]
        windows = [Window.W3, Window.W7, Window.W14, Window.W30]
        measures = [Measure.COUNT, Measure.TOTAL_DUR, Measure.WATCHED_PCT, Measure.COMPLETED]
        aggregations = [Aggregation.SUM, Aggregation.MEAN, Aggregation.MAX, Aggregation.STD]

        for ent in entities:
            for win in windows:
                for meas in measures:
                    for agg in aggregations:
                        # Фильтруем бессмысленные комбинации
                        if meas == Measure.COUNT:
                            if agg in (Aggregation.SUM, Aggregation.STD, Aggregation.MAX):
                                continue
                        if meas in (Measure.WATCHED_PCT, Measure.COMPLETED) and agg == Aggregation.SUM:
                            continue

                        name = f"bf_{ent.value[:1]}_{meas.value[:3]}_{agg.value[:3]}_{win.value}"
                        features.append(
                            FeatureDef(
                                name=name,
                                author=Author.BRUTE_FORCE,
                                entity=ent,
                                window=win,
                                measure=meas,
                                agg=agg,
                                filter_type=Filter.ALL,
                                rationale=f"Комбинаторная генерация: {ent.value} x {win.value} x {meas.value} x {agg.value}",
                            )
                        )

        # Ограничиваем разумным размером пула (например, 36 признаков)
        return features[:36]


class AIAgentFeatureGenerator:
    """ИИ-агент генератор признаков: гипотезы, вдохновленные находками EDA и спецификой данных."""

    def generate(self) -> List[FeatureDef]:
        features = [
            # Гипотеза 1 (EDA: watched_pct сериалов 13% vs фильмы 45%, но время 80 мин vs 42 мин)
            # Для сериалов watched_pct обманчив, критично суммарное время просмотра и признак длительного просмотра.
            FeatureDef(
                name="ai_i_series_long_watch_cnt_30d",
                author=Author.AI_AGENT,
                entity=Entity.ITEM,
                window=Window.W30,
                measure=Measure.LONG_WATCH,
                agg=Aggregation.COUNT,
                filter_type=Filter.SERIES,
                rationale="EDA гипотеза: сила сериала определяется числом зрителей с длительным сеансом (>=1 ч), а не процентом",
            ),
            FeatureDef(
                name="ai_u_series_dur_sum_14d",
                author=Author.AI_AGENT,
                entity=Entity.USER,
                window=Window.W14,
                measure=Measure.TOTAL_DUR,
                agg=Aggregation.SUM,
                filter_type=Filter.SERIES,
                rationale="EDA гипотеза: фанаты сериалов смотрят их запоем; суммарный хронометраж сериалов за 2 недели",
            ),
            # Гипотеза 2 (EDA: быстрый дрейф популярности, 1-дневное окно дает лучший результат в EDA)
            # Нужны сверхкороткие окна (1d и 3d) для фиксации хайпа и виральных новинок.
            FeatureDef(
                name="ai_i_cnt_1d",
                author=Author.AI_AGENT,
                entity=Entity.ITEM,
                window=Window.W1,
                measure=Measure.COUNT,
                agg=Aggregation.COUNT,
                rationale="EDA гипотеза: однодневный всплеск популярности новинок отражает вирусный интерес",
            ),
            FeatureDef(
                name="ai_i_cnt_3d",
                author=Author.AI_AGENT,
                entity=Entity.ITEM,
                window=Window.W3,
                measure=Measure.COUNT,
                agg=Aggregation.COUNT,
                rationale="EDA гипотеза: 3-дневное окно популярности сглаживает шум, но реагирует быстрее 7d",
            ),
            FeatureDef(
                name="ai_i_recency_days",
                author=Author.AI_AGENT,
                entity=Entity.ITEM,
                window=Window.ALL,
                measure=Measure.COUNT,
                agg=Aggregation.RECENCY,
                rationale="EDA гипотеза: свежесть последнего просмотра контента в платформе",
            ),
            # Гипотеза 3 (EDA: распределение watched_pct бимодальное с пиками на 0 и 100)
            # Отсев случайных кликов (клиент бросил сразу) и оценка только качественных просмотров (досмотренные фильмы).
            FeatureDef(
                name="ai_i_completed_mean_pct_30d",
                author=Author.AI_AGENT,
                entity=Entity.ITEM,
                window=Window.W30,
                measure=Measure.WATCHED_PCT,
                agg=Aggregation.MEAN,
                filter_type=Filter.COMPLETED,
                rationale="EDA гипотеза: средний процент просмотра среди тех, кто реально стал смотреть (отсечение кликбейта)",
            ),
            FeatureDef(
                name="ai_u_completed_cnt_7d",
                author=Author.AI_AGENT,
                entity=Entity.USER,
                window=Window.W7,
                measure=Measure.COMPLETED,
                agg=Aggregation.COUNT,
                filter_type=Filter.COMPLETED,
                rationale="EDA гипотеза: текущий темп завершения контента пользователем за неделю",
            ),
            # Гипотеза 4 (EDA: демографический таргетинг для разогрева)
            FeatureDef(
                name="ai_age_item_mean_pct_30d",
                author=Author.AI_AGENT,
                entity=Entity.AGE_ITEM,
                window=Window.W30,
                measure=Measure.WATCHED_PCT,
                agg=Aggregation.MEAN,
                rationale="EDA гипотеза: насколько качественно этот фильм смотрят ровесники пользователя",
            ),
            # Гипотеза 5 (EDA: тяжелые зрители выходных дней и запойный просмотр)
            FeatureDef(
                name="ai_u_long_watch_cnt_30d",
                author=Author.AI_AGENT,
                entity=Entity.USER,
                window=Window.W30,
                measure=Measure.LONG_WATCH,
                agg=Aggregation.COUNT,
                rationale="EDA гипотеза: частота длительных сессий пользователя (>1 ч) за месяц",
            ),
            FeatureDef(
                name="ai_i_long_watch_mean_30d",
                author=Author.AI_AGENT,
                entity=Entity.ITEM,
                window=Window.W30,
                measure=Measure.LONG_WATCH,
                agg=Aggregation.MEAN,
                rationale="EDA гипотеза: доля длинных сеансов среди всех просмотров этого фильма",
            ),
        ]
        return features
