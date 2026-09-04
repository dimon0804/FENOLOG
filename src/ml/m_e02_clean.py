"""E02 — робастная очистка ряда перед сглаживанием Уиттекера.

Все варианты собраны на одном скелете: очистка ряда → Уиттекер λ = 1000 без
примеси линейной. Скелет намеренно повторяет `restore_on_grid` из
`src/core/restore.py`, а не зовёт её: там очистка зашита внутрь (обрезание по
коридору) и веса наблюдений жёстко равны единице, а эксперимент меряет ровно
эти две вещи. Как только победивший вариант будет принят, его место — внутри
`restore_on_grid`, здесь останется только история замера.

Контроль эксперимента — `e02_ctl`: тот же скелет с отключённой очисткой. Он
обязан совпасть с `whit1000` из E01 до четвёртого знака; расхождение означало бы
ошибку в скелете, и все дельты пришлось бы считать заново.
"""
from __future__ import annotations

import numpy as np

from src.core.clean import (
    NDVI_HARD_MAX,
    NDVI_HARD_MIN,
    clamp_physical,
    soft_median_filter,
)
from src.core.restore import whittaker_smooth
from src.ml.dataset import PolygonView
from src.ml.registry import BaseMethod, register

FALLBACK_NDVI = 0.31
LAM = 1000.0

# Веса наблюдений по сенсору для варианта со взвешиванием. Sentinel-2 — 10 м и
# точная геометрия, MODIS — 250 м, то есть один пиксель заведомо шире поля и
# значение размазано по соседним угодьям. Landsat посередине. Абсолютный уровень
# весов эквивалентен изменению λ, поэтому максимум зафиксирован на единице и
# сравнение с контролем остаётся честным.
SENSOR_WEIGHTS = {
    "mid":  {"s2": 1.0, "landsat": 0.6, "modis": 0.3, "unknown": 0.6},
    "soft": {"s2": 1.0, "landsat": 0.8, "modis": 0.5, "unknown": 0.8},
    "hard": {"s2": 1.0, "landsat": 0.4, "modis": 0.12, "unknown": 0.4},
    "s2only": {"s2": 1.0, "landsat": 0.25, "modis": 0.02, "unknown": 0.25},
}


def _sensor_of_known(view: PolygonView) -> np.ndarray:
    """Восстанавливает сенсор каждого известного наблюдения по колонкам-источникам.

    `primary_ndvi` — склейка по приоритету s2 → landsat → modis, значение
    копируется без изменений. Поэтому источник определяется точным сравнением:
    какая из сенсорных колонок совпала со склейкой, тот сенсор и дал точку.
    Порядок проверки повторяет приоритет склейки, иначе совпавшие сразу два
    сенсора (это бывает у landsat и modis) были бы приписаны не тому.
    """
    frame = view.frame
    mask = frame["primary_ndvi"].notna().to_numpy()
    primary = frame.loc[mask, "primary_ndvi"].to_numpy(dtype=float)
    out = np.full(len(primary), "unknown", dtype=object)
    for tag, col in (("s2", "s2_ndvi"), ("landsat", "landsat_ndvi"), ("modis", "modis_ndvi")):
        if col not in frame.columns:
            continue
        cand = frame.loc[mask, col].to_numpy(dtype=float)
        hit = (out == "unknown") & np.isclose(cand, primary, rtol=0.0, atol=1e-12)
        out[hit] = tag
    return out


def _restore(ords: np.ndarray, values: np.ndarray, weights: np.ndarray | None, lam: float):
    """Уиттекер на посуточной сетке с произвольными весами наблюдений.

    Веса живут на сетке: 0 в днях без наблюдения, вес сенсора — в днях с ним.
    Если на один день пришлись два наблюдения (такое бывает при склейке
    сенсоров), выигрывает последнее — их значения по построению совпадают.
    """
    grid = np.arange(ords.min(), ords.max() + 1, dtype=np.int64)
    y = np.zeros(grid.shape, dtype=float)
    w = np.zeros(grid.shape, dtype=float)
    pos = ords - grid[0]
    y[pos] = values
    w[pos] = 1.0 if weights is None else weights
    if (w > 0).sum() < 4:
        # Сглаживать нечего — линейная интерполяция по тому, что есть.
        return grid, np.interp(grid, ords, values)
    return grid, whittaker_smooth(y, w, lam=lam)


class _CleanWhittaker(BaseMethod):
    """Скелет всех вариантов E02.

    clip      — 'drop' выбросить брак, 'clamp' придавить к границе, False не трогать
    k         — ширина медианного окна в наблюдениях (0 отключает фильтр)
    mode      — 'by_index' (окно по подряд идущим наблюдениям) или 'by_time'
    threshold — порог мягкой медианы, 0 = жёсткая замена всегда
    direction — 'both' / 'down' / 'up'
    weighted  — веса наблюдений по сенсору: False или ключ набора весов
    """

    def __init__(self, clip="drop", k=0, mode="by_index", half_window=8,
                 threshold=0.0, direction="both", weighted=False, lam=LAM):
        self.clip = clip
        self.k = k
        self.mode = mode
        self.half_window = half_window
        self.threshold = threshold
        self.direction = direction
        self.weighted = weighted
        self.lam = lam

    def predict(self, view: PolygonView, target_ords: np.ndarray) -> np.ndarray:
        ords = view.known_ord.astype(np.int64)
        values = view.known_values.astype(float)
        if len(ords) < 2:
            return np.full(len(target_ords), FALLBACK_NDVI)

        weights = None
        if self.weighted:
            table = SENSOR_WEIGHTS[self.weighted]
            sensors = _sensor_of_known(view)
            weights = np.array([table[s] for s in sensors], dtype=float)

        # Маску отсечения считаем здесь, а не в clean.clip_physical, потому что
        # тем же самым срезом надо укоротить и массив весов.
        if self.clip == "clamp":
            values = clamp_physical(values)
        elif self.clip:
            keep = np.isfinite(values) & (values >= NDVI_HARD_MIN) & (values <= NDVI_HARD_MAX)
            ords, values = ords[keep], values[keep]
            if weights is not None:
                weights = weights[keep]
        if len(ords) < 2:
            return np.full(len(target_ords), FALLBACK_NDVI)

        if self.k >= 3 and len(values) >= 3:
            values = soft_median_filter(
                ords, values, threshold=self.threshold, k=self.k,
                mode=self.mode, half_window=self.half_window, direction=self.direction,
            )

        grid, restored = _restore(ords, values, weights, self.lam)
        idx = np.clip(np.asarray(target_ords, dtype=np.int64) - grid[0], 0, len(grid) - 1)
        return np.clip(restored[idx], 0.0, 1.0)


def _variant(key: str, title: str, **kwargs):
    """Регистрирует вариант эксперимента: одна строка на один замер."""

    @register(key, title, experiment="E02")
    class _V(_CleanWhittaker):
        def __init__(self, _kw=kwargs):
            super().__init__(**_kw)

    return _V


# --- контроль и два шага очистки по отдельности -----------------------------
_variant("e02_ctl", "E02 контроль: Уиттекер λ=1000 без очистки", clip=False)
_variant("e02_clip", "E02 только отсечение вне [-0.2, 1.0]", clip="drop")
_variant("e02_clamp", "E02 придавливание к границам [-0.2, 1.0]", clip="clamp")

# --- жёсткая медиана: ширина окна и способ выбора соседей --------------------
_variant("e02_med3", "E02 медиана-3 по наблюдениям", k=3)
_variant("e02_med5", "E02 медиана-5 по наблюдениям", k=5)
_variant("e02_med3t", "E02 медиана-3 по календарю, ±8 дней", k=3, mode="by_time", half_window=8)
_variant("e02_med5t", "E02 медиана-5 по календарю, ±8 дней", k=5, mode="by_time", half_window=8)

# --- мягкая медиана: порог отделяет выброс от собственного шума --------------
_variant("e02_soft10", "E02 мягкая медиана-3, порог 0.10", k=3, threshold=0.10)
_variant("e02_soft15", "E02 мягкая медиана-3, порог 0.15", k=3, threshold=0.15)
_variant("e02_soft20", "E02 мягкая медиана-3, порог 0.20", k=3, threshold=0.20)
_variant("e02_soft25", "E02 мягкая медиана-3, порог 0.25", k=3, threshold=0.25)
_variant("e02_soft35", "E02 мягкая медиана-3, порог 0.35", k=3, threshold=0.35)

# --- асимметрия: только провалы вниз против зеркальной проверки вверх --------
_variant("e02_down15", "E02 мягкая медиана-3 вниз, порог 0.15", k=3, threshold=0.15, direction="down")
_variant("e02_down25", "E02 мягкая медиана-3 вниз, порог 0.25", k=3, threshold=0.25, direction="down")
_variant("e02_up25", "E02 мягкая медиана-3 вверх, порог 0.25 (зеркало)", k=3, threshold=0.25, direction="up")

# --- взвешивание наблюдений по сенсору --------------------------------------
_variant("e02_wsens", "E02 веса по сенсору s2/landsat/modis = 1/0.6/0.3", weighted=True)
