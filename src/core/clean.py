"""Робастная очистка ряда NDVI перед сглаживанием — эксперимент E02.

Мотив. Сглаживание Уиттекера минимизирует сумму квадратов невязок, а квадрат —
функция неробастная: одна точка, сбитая облаком на 0.3 вниз, тянет за собой всю
окрестность. То есть выброс не выкидывается, а размазывается по соседним дням,
и ошибка появляется там, где исходные наблюдения были нормальными. Отсюда идея:
отсеять грубый брак и одиночные провалы ДО сглаживания, а не надеяться, что
сглаживание с ними справится само.

Три уровня очистки, по нарастанию агрессивности:

1. `clip_physical` — отсечение физически невозможных значений. NDVI по построению
   лежит в [-1, 1], а у растительности в сезон — заметно выше нуля. Значения
   вроде -2.13 и 1.84 в наборе есть, это брак съёмки, и лечится он только
   выбрасыванием точки целиком (не подтягиванием к границе: подтянутая к нулю
   точка всё равно останется провалом и всё равно потянет сглаживание вниз).

2. `median_filter` — классическая медиана по окну. Убивает одиночный выброс
   начисто и при этом не срезает фронт роста, в отличие от среднего.

3. `soft_median_filter` — медиана «по требованию»: точка заменяется, только если
   отклонение от медианы окна превысило порог. Смысл в том, что собственный шум
   наблюдения около 0.07, и жёсткая медиана вместе с выбросами гасит и обычные
   отклонения, то есть отчасти дублирует работу Уиттекера и теряет реальный
   сигнал. Порог отделяет «выброс» от «шум».

Отдельная развилка — как понимать «окно из трёх точек». Наблюдения идут по
нерегулярной сетке: медианный шаг 3 дня, но встречаются перерывы в месяцы.
Медиана по трём подряд идущим наблюдениям (`by_index`) в таком месте смешивает
июнь с августом и портит значение, которое было в порядке. Медиана по
календарному окну (`by_time`) берёт в соседи только то, что действительно рядом
по времени, а изолированную точку оставляет как есть. Оба варианта реализованы и
замеряются, потому что предсказать, что важнее — полнота фильтрации или
сохранность одиноких точек, — заранее нельзя.

Асимметричный режим `direction="down"` опирается на физику помехи: облако,
дымка и тень занижают отражение в красном диапазоне слабее, чем в ближнем ИК,
и NDVI падает; механизма, который бы систематически завышал NDVI на одну дату,
почти нет. Замер на данных (см. reports/exp_e02.md) эту асимметрию подтверждает
только в далёком хвосте, поэтому режим оформлен опцией, а не поведением по
умолчанию.
"""
from __future__ import annotations

import numpy as np

# Физически допустимый коридор. Нижняя граница мягче нуля: голая почва и стерня
# дают небольшой положительный NDVI, но снег и вода — уверенно отрицательный,
# и такие даты в межсезонье законны. Всё, что ниже -0.2, — уже брак.
NDVI_HARD_MIN, NDVI_HARD_MAX = -0.2, 1.0


def clip_physical(
    ords: np.ndarray,
    values: np.ndarray,
    lo: float = NDVI_HARD_MIN,
    hi: float = NDVI_HARD_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Выбрасывает наблюдения вне физического коридора вместе с их датами.

    Возвращает укороченную пару (дни, значения). Именно выбрасывает, а не
    заменяет на NaN: дальше по конвейеру ряд идёт как список наблюдений, и
    отсутствие точки для сглаживания честнее, чем точка с выдуманным значением.
    """
    values = np.asarray(values, dtype=float)
    ords = np.asarray(ords, dtype=np.int64)
    keep = np.isfinite(values) & (values >= lo) & (values <= hi)
    return ords[keep], values[keep]


def clamp_physical(
    values: np.ndarray,
    lo: float = NDVI_HARD_MIN,
    hi: float = NDVI_HARD_MAX,
) -> np.ndarray:
    """Придавливает бракованные значения к границам коридора, не удаляя точку.

    Альтернатива `clip_physical` для краёв сезона. Замер показал, почему она
    нужна: выброшенная точка в апреле или октябре — это единственная опора на
    краю ряда, за ней зимний перерыв в полгода, и без неё Уиттекер уходит в
    экстраполяцию на длинном разрыве. Придавленное к нулю значение по величине
    ошибочно, но по положению верно и удерживает кривую от разлёта.
    """
    return np.clip(np.asarray(values, dtype=float), lo, hi)


def _window_median_by_index(values: np.ndarray, k: int) -> np.ndarray:
    """Медиана по k подряд идущим наблюдениям, календарь игнорируется.

    Края ряда обрабатываются усечённым окном, а не отражением: отражение
    придумывает наблюдения, которых не было, и на коротких рядах это заметно.
    """
    n = len(values)
    half = k // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def _window_median_by_time(values: np.ndarray, ords: np.ndarray, half_window: int, k: int) -> np.ndarray:
    """Медиана по соседям, отстоящим не дальше half_window дней.

    Дополнительно окно ограничено k ближайшими наблюдениями — иначе в плотном
    участке ряда (когда Sentinel-2 и MODIS дают по снимку почти каждый день)
    окно раздувается до двух десятков точек и фильтр превращается в грубое
    сглаживание, дублирующее Уиттекера.
    """
    n = len(values)
    half = k // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        idx = np.arange(lo, hi)
        near = idx[np.abs(ords[idx] - ords[i]) <= half_window]
        # Сама точка всегда входит в окно, поэтому near непусто по построению.
        out[i] = np.median(values[near])
    return out


def window_median(
    ords: np.ndarray,
    values: np.ndarray,
    k: int = 3,
    mode: str = "by_index",
    half_window: int = 8,
) -> np.ndarray:
    """Медиана скользящего окна. mode: 'by_index' или 'by_time'."""
    values = np.asarray(values, dtype=float)
    ords = np.asarray(ords, dtype=np.int64)
    if len(values) < 3 or k < 3:
        return values.copy()
    if mode == "by_time":
        return _window_median_by_time(values, ords, half_window=half_window, k=k)
    return _window_median_by_index(values, k)


def soft_median_filter(
    ords: np.ndarray,
    values: np.ndarray,
    threshold: float,
    k: int = 3,
    mode: str = "by_index",
    half_window: int = 8,
    direction: str = "both",
) -> np.ndarray:
    """Медианный фильтр с порогом: точка правится, только если сильно выбилась.

    threshold — граница между «шумом» и «выбросом» в единицах NDVI. Ниже порога
    точка остаётся как есть, и подавление шума целиком остаётся за Уиттекером,
    который делает это оптимально (он видит весь ряд, а не три точки).

    direction:
        'both' — правим отклонения в обе стороны,
        'down' — только провалы вниз (гипотеза об облачной природе помехи),
        'up'   — только всплески вверх, нужен как контрольная зеркальная проверка:
                 если 'up' даёт тот же выигрыш, что и 'down', объяснение через
                 облачность не работает и дело просто в тяжёлых хвостах.

    threshold <= 0 означает жёсткий фильтр (заменяем всегда).
    """
    values = np.asarray(values, dtype=float)
    med = window_median(ords, values, k=k, mode=mode, half_window=half_window)
    if threshold <= 0:
        replace = np.ones(len(values), dtype=bool)
    else:
        dev = values - med
        if direction == "down":
            replace = dev < -threshold
        elif direction == "up":
            replace = dev > threshold
        else:
            replace = np.abs(dev) > threshold
    out = values.copy()
    out[replace] = med[replace]
    return out


def clean_series(
    ords: np.ndarray,
    values: np.ndarray,
    *,
    clip: str | bool = "drop",
    k: int = 3,
    mode: str = "by_index",
    half_window: int = 8,
    threshold: float = 0.0,
    direction: str = "both",
) -> tuple[np.ndarray, np.ndarray]:
    """Полный конвейер очистки: отсечение брака, затем медианный фильтр.

    Порядок важен. Сначала отсечение: значение -2.13 внутри окна сдвинет медиану
    и испортит соседей, если сначала фильтровать, а потом отсекать. k < 3
    отключает медианный этап и оставляет только отсечение — это нужно, чтобы
    отдельно замерить вклад каждого из двух шагов.

    clip: 'drop' — выбросить бракованную точку, 'clamp' — придавить к границе,
    False — не трогать вовсе.
    """
    ords = np.asarray(ords, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if clip == "clamp":
        values = clamp_physical(values)
    elif clip:
        ords, values = clip_physical(ords, values)
    if k >= 3 and len(values) >= 3:
        values = soft_median_filter(
            ords, values, threshold=threshold, k=k,
            mode=mode, half_window=half_window, direction=direction,
        )
    return ords, values
