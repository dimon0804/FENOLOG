"""Восстановление пропусков во временном ряду вегетационного индекса.

Главный вывод разведки данных: основная ошибка — это не «дыры», а собственный шум
наблюдения (std разности соседних дней 0.093 при медианном разрыве в 3 дня).
Поэтому выигрывает сглаживание по окну, а не точная интерполяция между двумя точками.

Замеры на локальной валидации (3495 спрятанных точек):
    среднее двух соседей   RMSE 0.0907
    линейная интерполяция  RMSE 0.0894
    Уиттекер, lam=100      RMSE 0.0875
    смесь 50/50            RMSE 0.0849  <- текущая рабочая конфигурация
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Значения вне этого диапазона считаем браком съёмки
NDVI_MIN, NDVI_MAX = -0.2, 1.0


def clip_outliers(y: np.ndarray) -> np.ndarray:
    """Убирает физически невозможные значения (в данных есть -0.64 и 1.84)."""
    y = y.astype(float).copy()
    y[(y < NDVI_MIN) | (y > NDVI_MAX)] = np.nan
    return y


def whittaker_smooth(y: np.ndarray, w: np.ndarray, lam: float = 100.0, order: int = 2) -> np.ndarray:
    """Сглаживание Уиттекера: баланс между близостью к данным и гладкостью.

    y   — значения на регулярной сетке (в пропусках любое число, вес 0)
    w   — веса: 1 там, где наблюдение есть, 0 там, где его нет
    lam — сила сглаживания: больше значение — более гладкая кривая
    """
    n = len(y)
    D = sparse.diags(
        diagonals=[1.0, -2.0, 1.0],
        offsets=[0, 1, 2],
        shape=(n - order, n),
        dtype=float,
    )
    W = sparse.diags(w.astype(float))
    A = (W + lam * (D.T @ D)).tocsc()
    return spsolve(A, w * y)


def restore_on_grid(
    t_days: np.ndarray,
    y: np.ndarray,
    lam: float = 100.0,
    mix: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Восстанавливает ряд на сплошной посуточной сетке.

    t_days — целочисленные дни наблюдений (например, ordinal даты)
    y      — значения, NaN в пропусках
    mix    — доля Уиттекера в смеси; остальное берётся из линейной интерполяции

    Возвращает (сетка дней, восстановленные значения на всей сетке).
    """
    y = clip_outliers(np.asarray(y, dtype=float))
    t_days = np.asarray(t_days, dtype=np.int64)

    grid = np.arange(t_days.min(), t_days.max() + 1, dtype=np.int64)
    values = np.full(grid.shape, np.nan)
    values[t_days - grid[0]] = y

    known = ~np.isnan(values)
    if known.sum() == 0:
        return grid, values
    if known.sum() < 4:
        # Слишком мало точек для сглаживания — только линейная интерполяция
        return grid, np.interp(grid, grid[known], values[known])

    weights = known.astype(float)
    smooth = whittaker_smooth(np.nan_to_num(values), weights, lam=lam)
    linear = np.interp(grid, grid[known], values[known])
    return grid, mix * smooth + (1.0 - mix) * linear


def predict_at(grid: np.ndarray, restored: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Снимает восстановленные значения на нужных датах (в тех же единицах, что grid)."""
    idx = np.clip(np.asarray(targets, dtype=np.int64) - grid[0], 0, len(grid) - 1)
    out = restored[idx]
    return np.clip(out, 0.0, 1.0)
