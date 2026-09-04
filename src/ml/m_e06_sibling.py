"""E06. Остаток соседних полей на ту же дату.

Идея, которой нет в исходном плане, и она бьёт по самому дорогому месту задачи.

Считалось, что 0,066 собственного шума наблюдения — неустранимый потолок. Это
неверно. Шум наблюдения общий для полей, снятых одним пролётом: дымка, тонкое
облако, угол Солнца, поправки атмосферной коррекции — всё это сдвигает NDVI сразу
у группы полигонов в одну сторону. А наблюдения соседних полей на дату контрольной
точки **не замаскированы**: организаторы прячут только целевой полигон.

Проверка на данных (см. журнал, E06):
    стандартное отклонение остатка от сглаженного ряда   0,0643
    медиана парной корреляции остатков, все пары         0,246
    медиана корреляции остатка поля со средним по группе 0,521

Корреляция 0,52 означает, что около четверти дисперсии «шума» объяснимо и его
можно вычесть. Отсюда метод: восстановить ряд как обычно, а затем прибавить долю
от среднего остатка соседей на ту же дату.

    прогноз = сглаженный ряд поля(t) + beta * агрегат остатков соседей(t)

Утечки здесь нет: остатки берутся только по видимым наблюдениям чужих полигонов,
а протокол валидации маскирует скрытые строки во всех полигонах сразу.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.dataset import PolygonView
from src.ml.registry import BaseMethod, register
from src.core.restore import predict_at, restore_on_grid

FALLBACK_NDVI = 0.31
# Меньше этого числа соседей на дату — поправка не применяется: среднее по одному
# полю само по себе шум, вычитать его вреднее, чем не вычитать ничего.
MIN_SIBLINGS = 3


class _SiblingBase(BaseMethod):
    """Общая механика: сглаживание, остатки, поправка по соседям.

    beta      — доля переносимого остатка соседей
    lam       — сила сглаживания базового ряда
    robust    — брать медиану остатков вместо среднего
    normalize — приводить остатки соседей к масштабу целевого поля
    """

    def __init__(self, beta: float = 0.5, lam: float = 1000.0,
                 robust: bool = True, normalize: bool = False):
        self.beta = beta
        self.lam = lam
        self.robust = robust
        self.normalize = normalize

    @staticmethod
    def _smooth(view: PolygonView, lam: float):
        """Сглаженный ряд полигона на сплошной сетке дней."""
        if len(view.known_ord) < 4:
            return None, None
        return restore_on_grid(view.known_ord, view.known_values, lam=lam, mix=1.0)

    def _residual_table(self, views: dict[str, PolygonView]) -> tuple[pd.DataFrame, dict]:
        """Таблица остатков: строки — дни, столбцы — полигоны.

        Считается один раз на весь прогон: сглаживание 78 рядов заметно дороже
        самого предсказания, а результат от целевой точки не зависит.
        """
        columns: dict[str, pd.Series] = {}
        smoothed: dict[str, tuple] = {}
        for pid, view in views.items():
            grid, values = self._smooth(view, self.lam)
            if grid is None:
                continue
            smoothed[pid] = (grid, values)
            fitted = values[view.known_ord - grid[0]]
            columns[pid] = pd.Series(view.known_values - fitted, index=view.known_ord)
        table = pd.DataFrame(columns)
        return table, smoothed

    def predict_points(self, points: list, views: dict[str, PolygonView], context: dict) -> np.ndarray:
        table, smoothed = self._residual_table(views)
        if table.empty:
            return np.full(len(points), FALLBACK_NDVI)

        # Масштаб остатка каждого поля: поля различаются уровнем шума, и перенос
        # остатка «как есть» с шумного поля на спокойное только вредит.
        scale = table.std(axis=0)
        scale = scale.replace(0.0, np.nan)

        arr = table.to_numpy()
        cols = np.array(table.columns)
        day_index = {int(d): i for i, d in enumerate(table.index.to_numpy())}
        col_index = {c: i for i, c in enumerate(cols)}
        scale_v = scale.to_numpy()

        out = np.empty(len(points), dtype=float)
        for k, p in enumerate(points):
            view = views[p.polygon_id]
            grid, values = smoothed.get(p.polygon_id, (None, None))
            base = (predict_at(grid, values, np.array([p.ord_day]))[0]
                    if grid is not None else FALLBACK_NDVI)

            row = day_index.get(int(p.ord_day))
            correction = 0.0
            if row is not None and self.beta != 0.0:
                neighbours = arr[row].copy()
                own = col_index.get(p.polygon_id)
                if own is not None:
                    neighbours[own] = np.nan
                mask = np.isfinite(neighbours)
                if mask.sum() >= MIN_SIBLINGS:
                    vals = neighbours[mask]
                    if self.normalize:
                        # Приводим к масштабу целевого поля: сначала в единицы
                        # собственного разброса соседа, потом обратно в наши.
                        own_scale = scale_v[own] if own is not None else np.nanmedian(scale_v)
                        vals = vals / scale_v[mask] * (own_scale if np.isfinite(own_scale) else 1.0)
                    correction = float(np.median(vals) if self.robust else np.mean(vals))
            out[k] = base + self.beta * correction
        return np.clip(out, 0.0, 1.0)


@register("sib00", "Уиттекер λ = 1000, без поправки соседей", experiment="E06")
class Sibling00(_SiblingBase):
    """Контроль: та же механика при beta = 0, должна совпасть с whit1000."""

    def __init__(self):
        super().__init__(beta=0.0)


@register("sib03", "Остаток соседей, доля 0,3", experiment="E06")
class Sibling03(_SiblingBase):
    def __init__(self):
        super().__init__(beta=0.3)


@register("sib05", "Остаток соседей, доля 0,5", experiment="E06")
class Sibling05(_SiblingBase):
    def __init__(self):
        super().__init__(beta=0.5)


@register("sib07", "Остаток соседей, доля 0,7", experiment="E06")
class Sibling07(_SiblingBase):
    def __init__(self):
        super().__init__(beta=0.7)


@register("sib05n", "Остаток соседей 0,5 с приведением масштаба", experiment="E06")
class Sibling05Norm(_SiblingBase):
    def __init__(self):
        super().__init__(beta=0.5, normalize=True)


@register("sib05m", "Остаток соседей 0,5, среднее вместо медианы", experiment="E06")
class Sibling05Mean(_SiblingBase):
    def __init__(self):
        super().__init__(beta=0.5, robust=False)


@register("sib09", "Остаток соседей, доля 0,9", experiment="E06")
class Sibling09(_SiblingBase):
    def __init__(self):
        super().__init__(beta=0.9, robust=False)


@register("sib10", "Остаток соседей, доля 1,0", experiment="E06")
class Sibling10(_SiblingBase):
    def __init__(self):
        super().__init__(beta=1.0, robust=False)


@register("sib12", "Остаток соседей, доля 1,2", experiment="E06")
class Sibling12(_SiblingBase):
    def __init__(self):
        super().__init__(beta=1.2, robust=False)


@register("sib07r", "Остаток соседей 0,7, медиана", experiment="E06")
class Sibling07Robust(_SiblingBase):
    def __init__(self):
        super().__init__(beta=0.7, robust=True)


class _SiblingIterative(_SiblingBase):
    """Усиленная версия: общая суточная поправка вычитается до сглаживания.

    В простом варианте остаток соседей добавляется к уже готовому прогнозу.
    Но та же самая суточная помеха сидит и в опорных наблюдениях самого поля,
    по которым строится сглаженный ряд. Значит её надо сначала вычесть из всего
    ряда, пересгладить очищенные значения, и только потом вернуть на дату цели.

        c(t)  — общая суточная поправка, медиана остатков по всем полям
        y'(d) = y(d) − c(d)          очищенные наблюдения
        прогноз = сглаженный y'(t) + beta · c(t)

    Поправка на дату каждого поля считается без него самого (leave-one-out),
    иначе поле частично вычитало бы собственный шум и оценка была бы смещённой.
    """

    def __init__(self, beta: float = 1.0, lam: float = 1000.0, corr_power: float = 0.0,
                 clip_lo: float = -0.25, clip_hi: float = 0.25):
        super().__init__(beta=beta, lam=lam, robust=True)
        self.corr_power = corr_power
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi

    def predict_points(self, points, views, context):
        table, smoothed = self._residual_table(views)
        if table.empty:
            return np.full(len(points), FALLBACK_NDVI)

        days, corr, n_eff = _corrections_from_table(
            table, corr_power=self.corr_power, clip_lo=self.clip_lo, clip_hi=self.clip_hi)
        day_index = {int(d): i for i, d in enumerate(days)}

        # Пересглаживание очищенных рядов: та же суточная помеха сидит и в опорных
        # наблюдениях самого поля, по которым строится сглаженная кривая
        cleaned: dict[str, tuple] = {}
        for pid, view in views.items():
            c = corr.get(pid)
            if c is None or len(view.known_ord) < 4:
                continue
            idx = np.array([day_index[int(d)] for d in view.known_ord])
            y = view.known_values - c[idx]
            cleaned[pid] = restore_on_grid(view.known_ord, y, lam=self.lam, mix=1.0)

        out = np.empty(len(points), dtype=float)
        for k, p in enumerate(points):
            grid, values = cleaned.get(p.polygon_id, smoothed.get(p.polygon_id, (None, None)))
            base = (predict_at(grid, values, np.array([p.ord_day]))[0]
                    if grid is not None else FALLBACK_NDVI)
            c = corr.get(p.polygon_id)
            row = day_index.get(int(p.ord_day))
            add = float(c[row]) if (c is not None and row is not None) else 0.0
            out[k] = base + self.beta * add
        return np.clip(out, 0.0, 1.0)


@register("sibit10", "Суточная поправка до сглаживания, доля 1,0", experiment="E06")
class SiblingIter10(_SiblingIterative):
    def __init__(self):
        super().__init__(beta=1.0)


@register("sibit08", "Суточная поправка до сглаживания, доля 0,8", experiment="E06")
class SiblingIter08(_SiblingIterative):
    def __init__(self):
        super().__init__(beta=0.8)


@register("sibit10_l300", "Суточная поправка, λ = 300", experiment="E06")
class SiblingIterL300(_SiblingIterative):
    def __init__(self):
        super().__init__(beta=1.0, lam=300.0)


@register("sibit10_l3000", "Суточная поправка, λ = 3000", experiment="E06")
class SiblingIterL3000(_SiblingIterative):
    def __init__(self):
        super().__init__(beta=1.0, lam=3000.0)


# ---------------------------------------------------------------------------
# Публичный интерфейс для остальных экспериментов
# ---------------------------------------------------------------------------

def _corrections_from_table(table: pd.DataFrame, corr_power: float = 0.0,
                            clip_lo: float = -0.25, clip_hi: float = 0.25,
                            corr_floor: float = 0.1):
    """Суточная поправка для каждого поля по таблице остатков.

    corr_power — степень, в которую возводится корреляция остатков соседа с
    остатками целевого поля, чтобы получить его вес. Ноль означает равные веса.
    Замерено: степень 3 даёт −0,0035 к RMSE относительно равных весов, плато 2–4,
    и это больше, чем даёт вся очистка ряда. Смысл прямой — сосед, который
    исторически ошибается вместе с нами, несёт про нашу ошибку больше информации,
    чем поле за сто километров. Плавное взвешивание обгоняет отбор top-k соседей
    при любом k: слабые соседи не мусор, их вклад просто должен быть меньше.

    Подрезка остатков несимметрична: распределение остатков скошено влево
    (асимметрия −1,23, за пределами ±0,35 сорок шесть провалов против шести
    всплесков), потому что облачность занижает NDVI сильнее, чем что-либо его
    завышает. После взвешивания по корреляции способ подрезки почти перестаёт
    влиять, но оставлен как дешёвая страховка от грубого брака.

    Возвращает (days, corr, n_eff): дни, словарь поправок по полигонам и
    эффективное число соседей, сформировавших поправку.
    """
    days = table.index.to_numpy()
    arr = table.to_numpy()
    finite = np.isfinite(arr).astype(float)
    clipped = np.where(finite > 0, np.clip(arr, clip_lo, clip_hi), 0.0)

    n = arr.shape[1]
    if corr_power > 0:
        # Диагональ обнуляется — это и есть leave-one-out: поле не участвует
        # в собственной поправке и не вычитает свой же шум.
        C = table.corr(min_periods=30).to_numpy()
        C = np.nan_to_num(C, nan=0.0)
        # Нижний порог вместо обнуления: поле, у которого все соседи слабо
        # коррелированы, иначе осталось бы вообще без поправки, а так получает
        # обычное среднее как запасной путь. Находка E07, +0,0005 на трёх зёрнах.
        W = np.clip(C, corr_floor, None) ** corr_power
        np.fill_diagonal(W, 0.0)
    else:
        W = np.ones((n, n))
        np.fill_diagonal(W, 0.0)

    # Два матричных умножения вместо цикла по полям: дни × поля на поля × поля
    num = clipped @ W.T
    den = finite @ W.T
    cnt = finite @ (W > 0).T

    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    values = np.where(cnt >= MIN_SIBLINGS, values, 0.0)

    corr = {pid: values[:, j] for j, pid in enumerate(table.columns)}
    n_eff = {pid: cnt[:, j] for j, pid in enumerate(table.columns)}
    return days, corr, n_eff


def residual_table(views: dict[str, PolygonView], lam: float = 1000.0):
    """Остатки всех полей от их сглаженных рядов: строки — дни, столбцы — поля."""
    return _SiblingBase(lam=lam)._residual_table(views)


def daily_correction(views: dict[str, PolygonView], lam: float = 1000.0,
                     corr_power: float = 3.0):
    """Общая суточная поправка — то, что все поля ошибаются в один день вместе.

    Возвращает (days, corr), где corr[pid] — массив поправок по дням days,
    посчитанный БЕЗ вклада самого поля (leave-one-out) и со взвешиванием соседей
    по корреляции их остатков с остатками этого поля.

    Этой функцией пользуются эксперименты E02-E07, чтобы строиться поверх
    очищенного ряда, а не поверх сырого.
    """
    table, _ = residual_table(views, lam=lam)
    days, corr, _ = _corrections_from_table(table, corr_power=corr_power)
    return days, corr


def sibling_counts(views: dict[str, PolygonView], lam: float = 1000.0,
                   corr_power: float = 3.0):
    """Сколько соседей сформировало поправку на каждый день. Признак для бустинга:
    там, где соседей мало, поправке верить нельзя, и модель должна это различать."""
    table, _ = residual_table(views, lam=lam)
    days, _, n_eff = _corrections_from_table(table, corr_power=corr_power)
    return days, n_eff


def cleaned_series(views: dict[str, PolygonView], lam: float = 1000.0,
                   corr_power: float = 3.0):
    """Наблюдения полей за вычетом общей суточной помехи.

    Возвращает (clean, days, corr): clean[pid] = (дни, очищенные значения).
    Подставляй clean вместо view.known_values — и любой метод восстановления
    начинает работать с ряда, из которого убрана предсказуемая часть шума.
    Не забудь прибавить поправку обратно на дату цели.
    """
    days, corr = daily_correction(views, lam=lam, corr_power=corr_power)
    day_index = {int(d): i for i, d in enumerate(days)}
    clean = {}
    for pid, view in views.items():
        if pid not in corr or len(view.known_ord) == 0:
            continue
        idx = np.array([day_index[int(d)] for d in view.known_ord])
        clean[pid] = (view.known_ord, view.known_values - corr[pid][idx])
    return clean, days, corr


def correction_at(days: np.ndarray, corr: dict, polygon_id: str, ord_days: np.ndarray) -> np.ndarray:
    """Поправка на конкретные даты. Вне известных дней — ноль."""
    day_index = {int(d): i for i, d in enumerate(days)}
    c = corr.get(polygon_id)
    if c is None:
        return np.zeros(len(ord_days))
    return np.array([c[day_index[int(t)]] if int(t) in day_index else 0.0 for t in ord_days])


@register("sibw3", "Суточная поправка, вес соседа corr^3", experiment="E06b")
class SiblingWeighted3(_SiblingIterative):
    """Соседи взвешены по корреляции остатков. Прирост −0,0035 к равным весам."""

    def __init__(self):
        super().__init__(beta=1.0, lam=1000.0, corr_power=3.0, clip_lo=-0.15, clip_hi=0.25)


@register("sibw3_l500", "Суточная поправка, вес corr^3, λ = 500", experiment="E06b")
class SiblingWeighted3L500(_SiblingIterative):
    def __init__(self):
        super().__init__(beta=1.0, lam=500.0, corr_power=3.0, clip_lo=-0.15, clip_hi=0.25)


@register("sibw2_l500", "Суточная поправка, вес corr^2, λ = 500", experiment="E06b")
class SiblingWeighted2L500(_SiblingIterative):
    def __init__(self):
        super().__init__(beta=1.0, lam=500.0, corr_power=2.0, clip_lo=-0.15, clip_hi=0.25)


@register("sibw4_l500", "Суточная поправка, вес corr^4, λ = 500", experiment="E06b")
class SiblingWeighted4L500(_SiblingIterative):
    def __init__(self):
        super().__init__(beta=1.0, lam=500.0, corr_power=4.0, clip_lo=-0.15, clip_hi=0.25)
