"""Базовые методы восстановления — эксперименты E01.

Здесь живут только методы без обучения: они опираются на сам ряд полигона и
ничего не знают про другие поля. Это точка отсчёта, относительно которой
меряется всё остальное.
"""
from __future__ import annotations

import numpy as np

from src.ml.dataset import PolygonView
from src.ml.registry import BaseMethod, register
from src.core.restore import predict_at, restore_on_grid

# Заглушка на случай полигона вообще без наблюдений — медиана primary_ndvi по набору
FALLBACK_NDVI = 0.31


def _neighbours(view: PolygonView, target_ords: np.ndarray):
    """Индексы ближайших известных наблюдений слева и справа от каждой цели."""
    pos = np.searchsorted(view.known_ord, target_ords, side="left")
    left = np.clip(pos - 1, 0, len(view.known_ord) - 1)
    right = np.clip(pos, 0, len(view.known_ord) - 1)
    return left, right


@register("mean2", "Среднее двух ближайших известных", experiment="E01")
def mean_of_two(view: PolygonView, target_ords: np.ndarray) -> np.ndarray:
    """Baseline организаторов: полусумма соседей без учёта расстояния до них."""
    if len(view.known_ord) == 0:
        return np.full(len(target_ords), FALLBACK_NDVI)
    li, ri = _neighbours(view, target_ords)
    return np.clip(0.5 * (view.known_values[li] + view.known_values[ri]), 0.0, 1.0)


@register("linear", "Линейная интерполяция по времени", experiment="E01")
def linear_interp(view: PolygonView, target_ords: np.ndarray) -> np.ndarray:
    """Учитывает расстояние до соседей — тем и выигрывает у полусуммы."""
    if len(view.known_ord) == 0:
        return np.full(len(target_ords), FALLBACK_NDVI)
    if len(view.known_ord) == 1:
        return np.full(len(target_ords), float(view.known_values[0]))
    return np.clip(np.interp(target_ords, view.known_ord, view.known_values), 0.0, 1.0)


@register("movavg21", "Скользящее среднее, окно 21 день", experiment="E01")
def moving_average(view: PolygonView, target_ords: np.ndarray) -> np.ndarray:
    """Проигрывает остальным: усредняет по окну, игнорируя расстояние по времени."""
    if len(view.known_ord) == 0:
        return np.full(len(target_ords), FALLBACK_NDVI)
    out = np.empty(len(target_ords), dtype=float)
    for k, t in enumerate(target_ords):
        m = np.abs(view.known_ord - t) <= 10
        out[k] = view.known_values[m].mean() if m.any() else FALLBACK_NDVI
    return np.clip(out, 0.0, 1.0)


class _Whittaker(BaseMethod):
    """Сглаживание Уиттекера с настраиваемой силой и долей примеси линейной.

    Флаг only_test отбрасывает наблюдения из train_dataset и оставляет методу
    только тестовый файл. Нужен ровно для одного замера: сколько стоит
    подмешивание обучающего набора в сам ряд.
    """

    def __init__(self, lam: float, mix: float, only_test: bool = False):
        self.lam = lam
        self.mix = mix
        self.only_test = only_test

    def _series(self, view: PolygonView):
        if not self.only_test or view.known_source is None:
            return view.known_ord, view.known_values
        keep = view.known_source == "test"
        return view.known_ord[keep], view.known_values[keep]

    def predict(self, view: PolygonView, target_ords: np.ndarray) -> np.ndarray:
        ords, values = self._series(view)
        if len(ords) < 2:
            return np.full(len(target_ords), FALLBACK_NDVI)
        grid, restored = restore_on_grid(ords, values, lam=self.lam, mix=self.mix)
        return predict_at(grid, restored, target_ords)


@register("whit10", "Уиттекер, λ = 10", experiment="E01")
class Whittaker10(_Whittaker):
    def __init__(self):
        super().__init__(lam=10.0, mix=1.0)


@register("whit100", "Уиттекер, λ = 100", experiment="E01")
class Whittaker100(_Whittaker):
    def __init__(self):
        super().__init__(lam=100.0, mix=1.0)


@register("whit1000", "Уиттекер, λ = 1000", experiment="E01")
class Whittaker1000(_Whittaker):
    def __init__(self):
        super().__init__(lam=1000.0, mix=1.0)


@register("blend50", "Смесь Уиттекера λ = 100 и линейной, 50/50", experiment="E01",
          tags=("current",))
class Blend50(_Whittaker):
    """Текущая рабочая конфигурация src/core/restore.py."""

    def __init__(self):
        super().__init__(lam=100.0, mix=0.5)


@register("whit1000_notrain", "Уиттекер λ = 1000 без наблюдений train", experiment="E01b")
class Whittaker1000NoTrain(_Whittaker):
    """Замер вклада train_dataset: тот же метод, но ряд собран только из теста."""

    def __init__(self):
        super().__init__(lam=1000.0, mix=1.0, only_test=True)
