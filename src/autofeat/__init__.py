"""Модуль автогенерации и отбора признаков (AutoFeat).

Содержит:
- grammar: декларативная грамматика описания признаков и SQL-шаблоны
- compiler: векторизованный Polars/DuckDB компилятор признаков
- generators: авторы признаков (Human, Brute-force, AI-Agent)
- selector: 3-этапный протокол отбора (фильтры, null importance, forward-отбор)
- run_search: оркестратор бенчмарка и сравнения с бейзлайном
"""
