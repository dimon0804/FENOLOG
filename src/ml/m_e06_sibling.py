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

    def __init__(self, beta: float = 1.0, lam: float = 1000.0, rounds: int = 1):
        super().__init__(beta=beta, lam=lam, robust=True)
        self.rounds = rounds

    def predict_points(self, points, views, context):
        table, smoothed = self._residual_table(views)
        if table.empty:
            return np.full(len(points), FALLBACK_NDVI)

        arr = table.to_numpy()
        cols = list(table.columns)
        col_index = {c: i for i, c in enumerate(cols)}
        days = table.index.to_numpy()
        day_index = {int(d): i for i, d in enumerate(days)}

        # Суточная поправка без учёта самого поля: сумма и счётчик по строке,
        # из которых вычитается вклад текущего столбца. Медиана здесь была бы
        # устойчивее, но её leave-one-out версия стоит на порядок дороже,
        # а разброс по столбцам уже подрезан отсечением выбросов ниже.
        finite = np.isfinite(arr)
        clipped = np.where(finite, np.clip(arr, -0.25, 0.25), 0.0)
        row_sum = clipped.sum(axis=1)
        row_cnt = finite.sum(axis=1)

        def correction_for(col: int) -> np.ndarray:
            """Суточная поправка для одного поля, без его собственного вклада."""
            s = row_sum - clipped[:, col]
            n = row_cnt - finite[:, col]
            out = np.where(n >= MIN_SIBLINGS, s / np.maximum(n, 1), 0.0)
            return out

        # Пересглаживание очищенных рядов
        cleaned: dict[str, tuple] = {}
        for pid, view in views.items():
            ci = col_index.get(pid)
            if ci is None or len(view.known_ord) < 4:
                continue
            corr = correction_for(ci)
            idx = np.array([day_index[int(d)] for d in view.known_ord])
            y = view.known_values - corr[idx]
            cleaned[pid] = restore_on_grid(view.known_ord, y, lam=self.lam, mix=1.0)

        out = np.empty(len(points), dtype=float)
        for k, p in enumerate(points):
            grid, values = cleaned.get(p.polygon_id, smoothed.get(p.polygon_id, (None, None)))
            base = (predict_at(grid, values, np.array([p.ord_day]))[0]
                    if grid is not None else FALLBACK_NDVI)
            ci = col_index.get(p.polygon_id)
            row = day_index.get(int(p.ord_day))
            add = 0.0
            if ci is not None and row is not None:
                n = row_cnt[row] - finite[row, ci]
                if n >= MIN_SIBLINGS:
                    add = (row_sum[row] - clipped[row, ci]) / n
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

def residual_table(views: dict[str, PolygonView], lam: float = 1000.0):
    """Остатки всех полей от их сглаженных рядов: строки — дни, столбцы — поля."""
    return _SiblingBase(lam=lam)._residual_table(views)


def daily_correction(views: dict[str, PolygonView], lam: float = 1000.0):
    """Общая суточная поправка — то, что все поля ошибаются в один день вместе.

    Возвращает (days, corr), где corr[pid] — массив поправок по дням days,
    посчитанный БЕЗ вклада самого поля pid (leave-one-out). Выбросы подрезаются
    по ±0,25: одно грубо испорченное поле не должно тянуть за собой всю группу.

    Этой функцией пользуются эксперименты E02-E05, чтобы строиться поверх
    очищенного ряда, а не поверх сырого.
    """
    table, _ = residual_table(views, lam=lam)
    days = table.index.to_numpy()
    arr = table.to_numpy()
    finite = np.isfinite(arr)
    clipped = np.where(finite, np.clip(arr, -0.25, 0.25), 0.0)
    row_sum = clipped.sum(axis=1)
    row_cnt = finite.sum(axis=1)

    corr = {}
    for j, pid in enumerate(table.columns):
        s = row_sum - clipped[:, j]
        n = row_cnt - finite[:, j]
        corr[pid] = np.where(n >= MIN_SIBLINGS, s / np.maximum(n, 1), 0.0)
    return days, corr


def cleaned_series(views: dict[str, PolygonView], lam: float = 1000.0):
    """Наблюдения полей за вычетом общей суточной помехи.

    Возвращает (clean, days, corr): clean[pid] = (дни, очищенные значения).
    Подставляй clean вместо view.known_values — и любой метод восстановления
    начинает работать с ряда, из которого убрана предсказуемая часть шума.
    Не забудь прибавить поправку обратно на дату цели.
    """
    days, corr = daily_correction(views, lam=lam)
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
