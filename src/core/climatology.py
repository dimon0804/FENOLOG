"""Климатическая норма вегетационного индекса и z-оценка отклонения от неё.

Норма считается по дню года на основе истории того же полигона. В выданном наборе
у 19 полигонов из 78 есть история 2010-2025, у остальных 59 — только сезон 2025.
Для полей без истории норма берётся по типу культуры (запасной вариант), а если
и его нет — климатология не рассчитывается, и z-оценка не определена.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Половина ширины окна по дню года, в днях: норма усредняется по соседним датам
DOY_WINDOW = 10
# Минимум сезонов, при котором норме по полигону можно доверять
MIN_YEARS = 3


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
    out["std"] = out["std"].fillna(np.nan).clip(lower=0.02)
    return out


def zscore(values: np.ndarray, clim_mean: np.ndarray, clim_std: np.ndarray) -> np.ndarray:
    """Стандартизованное отклонение от нормы: z = (x - mean) / std."""
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (values - clim_mean) / clim_std
    return np.where(np.isfinite(z), z, np.nan)


def has_enough_history(dates: pd.Series, min_years: int = MIN_YEARS) -> bool:
    """Достаточно ли у полигона сезонов, чтобы считать норму по нему самому."""
    return pd.to_datetime(pd.Series(dates)).dt.year.nunique() >= min_years
