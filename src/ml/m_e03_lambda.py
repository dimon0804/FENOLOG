"""E03 — сила сглаживания, подобранная под длину разрыва.

Гипотеза эксперимента. Единая λ на весь ряд — компромисс между двумя режимами:
на коротком разрыве соседи стоят вплотную и сильное сглаживание срезает реальную
динамику, на длинном опираться приходится на далёкие и зашумлённые точки, и
сглаживать надо жёстче. Плюс наблюдение из таблицы E01: на разрывах 62-181 дня
линейная интерполяция обгоняла Уиттекера λ = 1000, то есть там оптимальна другая
доля примеси mix.

Почему выбор параметра по длине разрыва вообще законен. Длина разрыва — это
расстояние до ближайших известных наблюдений, она видна методу в момент
инференса и в замаскированной строке ничего не подглядывает. Утечки нет.
Переобучение здесь другого рода: сама таблица «бин -> параметр» подобрана по
контрольному набору, поэтому ниже она зафиксирована как консенсус трёх зёрен
(42, 7, 2026), а не как оптимум одного разбиения.

Реализация. Уиттекер считается на весь ряд сразу, поэтому «своя λ для каждой
точки» получается так: для полигона строится несколько сглаженных версий ряда —
по одной на каждую λ из таблицы, — и для каждой цели значение снимается с той
версии, которая отвечает её бину. Локальное окно вокруг цели было бы дороже и
дало бы краевые эффекты на границах окна.

ИТОГ ЭКСПЕРИМЕНТА — отрицательный, подробности в reports/exp_e03.md.
Выигрыш побинного подбора не переносится между зёрнами: см. отчёт. Методы
оставлены зарегистрированными как воспроизводимое подтверждение замера.
"""
from __future__ import annotations

import numpy as np

from src.core.restore import predict_at, restore_on_grid
from src.ml.dataset import PolygonView
from src.ml.holdout import gap_bin
from src.ml.registry import BaseMethod, register

# Медиана primary_ndvi по набору — на случай полигона без наблюдений
FALLBACK_NDVI = 0.31

# Параметры для бинов, которые не встретились в подборе (182+ и любые будущие):
# берём текущий общий оптимум, чтобы метод никогда не падал на неизвестном бине.
DEFAULT = (1000.0, 1.0)

# Консенсус трёх зёрен: для каждого бина пара (λ, доля Уиттекера), суммарно
# минимизирующая квадрат ошибки на зёрнах 42, 7 и 2026 сразу. Выбор по одному
# зерну заметно другой — это и есть признак того, что сигнал слабый.
LAM_BY_BIN: dict[str, tuple[float, float]] = {
    "1-2": (3000.0, 0.8),
    "3-4": (500.0, 0.8),
    "5-7": (1000.0, 1.0),
    "8-13": (1000.0, 1.0),
    "14-29": (1000.0, 1.0),
    "30-61": (1000.0, 0.8),
    "62-181": (3000.0, 0.6),
}

# Та же процедура, но λ заморожена на 1000 и подбирается только доля примеси.
# Отдельный метод нужен, чтобы отделить вклад λ от вклада mix: в совместном
# подборе они компенсируют друг друга и по итоговой цифре не разобрать, что сработало.
MIX_BY_BIN: dict[str, tuple[float, float]] = {
    "1-2": (1000.0, 1.0),
    "3-4": (1000.0, 0.8),
    "5-7": (1000.0, 1.0),
    "8-13": (1000.0, 1.0),
    "14-29": (1000.0, 1.0),
    "30-61": (1000.0, 0.8),
    "62-181": (1000.0, 0.6),
}


def _gap_bins_for(known_ord: np.ndarray, target_ords: np.ndarray) -> np.ndarray:
    """Бин длины разрыва для каждой цели по видимым методу наблюдениям.

    Расстояния считаются ровно так же, как в протоколе валидации: -1 означает,
    что соседа с этой стороны нет, и gap_bin удваивает противоположное плечо.
    """
    pos = np.searchsorted(known_ord, target_ords, side="left")
    left = np.full(len(target_ords), -1, dtype=np.int64)
    right = np.full(len(target_ords), -1, dtype=np.int64)
    has_left = pos > 0
    left[has_left] = target_ords[has_left] - known_ord[pos[has_left] - 1]
    has_right = pos < len(known_ord)
    right[has_right] = known_ord[pos[has_right]] - target_ords[has_right]
    return gap_bin(left, right)


class _GapAdaptive(BaseMethod):
    """Уиттекер, у которого (λ, mix) выбираются по длине разрыва целевой точки."""

    def __init__(self, table: dict[str, tuple[float, float]]):
        self.table = table

    def predict(self, view: PolygonView, target_ords: np.ndarray) -> np.ndarray:
        ords, values = view.known_ord, view.known_values
        if len(ords) < 2:
            return np.full(len(target_ords), FALLBACK_NDVI)

        target_ords = np.asarray(target_ords, dtype=np.int64)
        bins = _gap_bins_for(ords, target_ords)
        params = [self.table.get(b, DEFAULT) for b in bins]

        out = np.empty(len(target_ords), dtype=float)
        # Группируем цели по паре параметров: сглаживание ряда стоит одного
        # разреженного решения, и повторять его на каждую точку незачем.
        for pair in set(params):
            mask = np.array([p == pair for p in params])
            lam, mix = pair
            grid, restored = restore_on_grid(ords, values, lam=lam, mix=mix)
            out[mask] = predict_at(grid, restored, target_ords[mask])
        return out


@register("e03_gap_lam", "Уиттекер, λ и mix по длине разрыва", experiment="E03")
class GapAdaptiveLambda(_GapAdaptive):
    def __init__(self):
        super().__init__(LAM_BY_BIN)


@register("e03_gap_mix", "Уиттекер λ = 1000, mix по длине разрыва", experiment="E03")
class GapAdaptiveMix(_GapAdaptive):
    def __init__(self):
        super().__init__(MIX_BY_BIN)
