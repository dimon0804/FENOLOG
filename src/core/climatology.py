"""Климатическая норма вегетационного индекса и z-оценка отклонения от неё.

Норма считается по дню года на основе истории того же полигона. В выданном наборе
у 19 полигонов из 78 есть история 2010-2025, у остальных 59 — только сезон 2025.
Для полей без истории норма берётся по типу культуры (запасной вариант), а если
и его нет — климатология не рассчитывается, и z-оценка не определена.

Формула нормы восстановлена по обучающему набору (reports/eda_train.md, раздел 6.2):
организаторы усредняют наблюдения того же полигона в окне ±8 дней по дню года
по всем годам, КРОМЕ текущего. MAE воспроизведения 0.0016 при 0.005-0.006 у окон
±7 и ±9, то есть ширина окна определена однозначно.

Исключение текущего года (leave-one-out) — не косметика, а необходимость.
Если аномальный сезон входит в собственную норму, он её же и сдвигает вниз,
z-оценка занижается по модулю, и детектор слепнет ровно там, где аномалия есть.
Эффект тем сильнее, чем короче история: при 5 сезонах один провальный год
двигает норму на пятую часть своей глубины.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Половина ширины окна по дню года, в днях: норма усредняется по соседним датам.
# Значение 8 — не подобрано на глаз, а восстановлено по эталонной климатологии
# организаторов на 39 полигонах обучающего набора.
DOY_WINDOW = 8
# Минимум сезонов, при котором норме по полигону можно доверять
MIN_YEARS = 3
# Нижняя граница разброса: при std около нуля z-оценка улетает в сотни
# и любая мелкая рябь превращается в «критическую аномалию».
MIN_STD = 0.02


def _circular_doy_mask(doy_grid: np.ndarray, target: int, window: int) -> np.ndarray:
    """Маска окна по дню года с учётом перехода через Новый год."""
    diff = np.abs(doy_grid - target)
    diff = np.minimum(diff, 366 - diff)
    return diff <= window


def fit_climatology(
    dates: pd.Series,
    values: pd.Series,
    window: int = DOY_WINDOW,
) -> pd.DataFrame:
    """Строит норму на каждый день года: среднее, разброс и число опорных лет.

    Возвращает таблицу с индексом doy от 1 до 366.
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values}).dropna()
    if df.empty:
        return pd.DataFrame(index=range(1, 367), columns=["mean", "std", "n_years", "n_obs"], dtype=float)

    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    doy_grid = df["doy"].to_numpy()

    rows = []
    for doy in range(1, 367):
        mask = _circular_doy_mask(doy_grid, doy, window)
        sub = df.loc[mask]
        if len(sub) < 3:
            rows.append((doy, np.nan, np.nan, sub["year"].nunique(), len(sub)))
            continue
        rows.append((doy, sub["value"].mean(), sub["value"].std(ddof=1), sub["year"].nunique(), len(sub)))

    out = pd.DataFrame(rows, columns=["doy", "mean", "std", "n_years", "n_obs"]).set_index("doy")
    # Разброс не может быть нулевым — иначе z-оценка уходит в бесконечность
    out["std"] = out["std"].fillna(np.nan).clip(lower=MIN_STD)
    return out


def fit_climatology_loo(
    dates: pd.Series,
    values: pd.Series,
    window: int = DOY_WINDOW,
) -> pd.DataFrame:
    """Норма по дню года с исключением текущего года — эталонная формула организаторов.

    В отличие от `fit_climatology`, норма здесь своя для каждого сезона: точка
    2019 года сравнивается с нормой, посчитанной по 2010-2018 и 2020-2024.
    Возвращает таблицу с двухуровневым индексом (year, doy) и колонками
    mean, std, n_years, n_obs.

    Годы, для которых после исключения не осталось хотя бы двух других сезонов,
    получают NaN: строить норму по одному году бессмысленно, разброс не определён.
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values}).dropna()
    empty = pd.DataFrame(columns=["mean", "std", "n_years", "n_obs"], dtype=float)
    if df.empty:
        return empty

    df["doy"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    doy_grid = df["doy"].to_numpy()
    year_grid = df["year"].to_numpy()
    val_grid = df["value"].to_numpy()

    rows = []
    for doy in range(1, 367):
        win = _circular_doy_mask(doy_grid, doy, window)
        if not win.any():
            continue
        y_in = year_grid[win]
        v_in = val_grid[win]
        for year in years:
            keep = y_in != year          # leave-one-out: текущий сезон в свою норму не входит
            v = v_in[keep]
            if len(v) < 3:
                rows.append((year, doy, np.nan, np.nan, 0, len(v)))
                continue
            rows.append(
                (year, doy, float(v.mean()), float(v.std(ddof=1)),
                 int(len(np.unique(y_in[keep]))), int(len(v)))
            )

    if not rows:
        return empty
    out = pd.DataFrame(rows, columns=["year", "doy", "mean", "std", "n_years", "n_obs"])
    out["std"] = out["std"].clip(lower=MIN_STD)
    return out.set_index(["year", "doy"])


def lookup_norm(
    clim: pd.DataFrame,
    years: np.ndarray,
    doys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Достаёт (среднее, разброс) нормы на пары (год, день года).

    Работает и с таблицей `fit_climatology` (индекс doy), и с таблицей
    `fit_climatology_loo` (индекс year+doy) — вызывающему коду не нужно знать,
    какая из них построена.
    """
    if clim is None or len(clim) == 0:
        nan = np.full(len(doys), np.nan)
        return nan, nan.copy()

    if isinstance(clim.index, pd.MultiIndex):
        key = pd.MultiIndex.from_arrays([np.asarray(years), np.asarray(doys)])
        # Для года вне истории (например, свежий сезон без прошлого) берём среднее
        # по всем годам: leave-one-out тут вырождается в обычную норму.
        fallback = clim.groupby(level="doy")[["mean", "std"]].mean()
        got = clim.reindex(key)
        mean = got["mean"].to_numpy()
        std = got["std"].to_numpy()
        miss = ~np.isfinite(mean)
        if miss.any():
            fb = fallback.reindex(np.asarray(doys)[miss])
            mean[miss] = fb["mean"].to_numpy()
            std[miss] = fb["std"].to_numpy()
        return mean, std

    got = clim.reindex(np.asarray(doys))
    return got["mean"].to_numpy(), got["std"].to_numpy()


def zscore(values: np.ndarray, clim_mean: np.ndarray, clim_std: np.ndarray) -> np.ndarray:
    """Стандартизованное отклонение от нормы: z = (x - mean) / std."""
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (values - clim_mean) / clim_std
    return np.where(np.isfinite(z), z, np.nan)


def has_enough_history(dates: pd.Series, min_years: int = MIN_YEARS) -> bool:
    """Достаточно ли у полигона сезонов, чтобы считать норму по нему самому."""
    return pd.to_datetime(pd.Series(dates)).dt.year.nunique() >= min_years
