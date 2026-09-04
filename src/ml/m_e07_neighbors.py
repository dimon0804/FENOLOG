"""E07. Регрессия по соседям вместо простого среднего.

Продолжение E06. Там суточная помеха оценивается усечённым средним остатков всех
соседей: каждому полю один и тот же вес, каждому полю один и тот же коэффициент
переноса, поля с числом соседей меньше трёх остаются без поправки вовсе.

Слабое место видно из самих чисел E06: медиана парной корреляции остатков 0,246,
а корреляция со средним по погодной группе 0,521. То есть соседи заведомо
неравноценны — часть полей снята тем же пролётом и повторяет помеху почти один
в один, часть не связана с целевым полем ничем. Усреднять их с одинаковым весом
значит разбавлять сигнал шумом непохожих полей.

Что проверено (подробности и отрицательные результаты — в reports/exp_e07.md):

  веса по корреляции остатков       +0,0030 к RMSE, переносится на все зёрна
  свой коэффициент переноса на поле  0,0000, оценки всех полей лежат около 0,9
  ridge по отдельным соседям        +0,0015, устойчиво хуже корреляционных весов
  погодные группы как жёсткий фильтр -0,0087, соседей внутри группы слишком мало
  плавное сжатие по числу соседей    0,0000, порог трёх соседей и так почти не бьёт

ПРИНЯТО В РАБОТУ: nb_corr3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.restore import predict_at, restore_on_grid
from src.ml.dataset import PolygonView
from src.ml.m_e06_sibling import MIN_SIBLINGS, residual_table
from src.ml.registry import BaseMethod, register

FALLBACK_NDVI = 0.31
# Остаток больше этого по модулю — брак съёмки, а не общая помеха. Обрезка та же,
# что в E06: одно грубо испорченное поле не должно тянуть за собой всю группу.
CLIP = 0.25


class _NeighborRegression(BaseMethod):
    """Суточная поправка по соседям с настраиваемыми весами и коэффициентом переноса.

    Схема наследует E06 и меняет в ней только способ агрегировать соседей:

        R[d, q]  = остаток поля q на день d от его сглаженного ряда (λ = lam_res)
        c_p(d)   = Σ_q W[p, q] · R[d, q] / Σ_q W[p, q]      по видимым соседям q ≠ p
        y'_p(d)  = y_p(d) − β_p · c_p(d)                     очищенные наблюдения
        прогноз  = сглаженный y'_p в точке t (λ = lam_fit) + β_p · c_p(t)

    Диагональ W всегда нулевая, поэтому оценка помехи для поля никогда не содержит
    его собственного вклада (leave-one-out) — иначе поле вычитало бы часть своего
    же шума и оценка была бы смещена в его пользу.

    Утечки нет: R строится только по видимым наблюдениям, а протокол валидации
    маскирует скрытые строки одновременно во всех полигонах.
    """

    def __init__(self, weight_mode: str = "corr", power: float = 3.0, floor: float = 0.1,
                 min_periods: int = 30, alpha: float = 0.3, beta_mode: str = "one",
                 tau: float = 50.0, shrink_k: float | None = None,
                 lam_res: float = 1000.0, lam_fit: float = 1000.0):
        self.weight_mode = weight_mode
        self.power = power
        self.floor = floor
        self.min_periods = min_periods
        self.alpha = alpha
        self.beta_mode = beta_mode
        self.tau = tau
        self.shrink_k = shrink_k
        self.lam_res = lam_res
        self.lam_fit = lam_fit

    # ---------------------------------------------------------------- веса

    def _weights_corr(self, arr: np.ndarray, cols: list) -> np.ndarray:
        """Вес соседа — его корреляция остатков с целевым полем, возведённая в степень.

        Степень 3 — не подгонка ради подгонки, а способ разнести соседей по
        полезности: при линейном весе поле с корреляцией 0,2 получает половину
        веса поля с корреляцией 0,4, при кубическом — восьмую часть. Оптимум
        пологий, степени 3 и 4 дают одно и то же (см. отчёт).

        Порог floor поднимает слишком слабые и отрицательные корреляции до 0,1.
        Отрицательная корреляция почти наверняка шум оценки, а не физика, и
        переносить остаток с обратным знаком вреднее, чем не переносить. Ноль
        здесь хуже мягкого порога: если у поля все соседи оказались слабыми,
        нулевые веса оставили бы его совсем без поправки, а floor сохраняет
        обычное среднее как запасной вариант.
        """
        C = pd.DataFrame(arr, columns=cols).corr(min_periods=self.min_periods).to_numpy()
        C = np.nan_to_num(C, nan=0.0)
        W = np.clip(C, self.floor, None) ** self.power
        np.fill_diagonal(W, 0.0)
        return W

    def _weights_ridge(self, clipped: np.ndarray, finite: np.ndarray, cols: list) -> np.ndarray:
        """Ridge-регрессия остатка поля на остатки отдельных соседей.

        Соседей до 77 при паре сотен наблюдений — без регуляризации переобучение
        мгновенное. Пропуски заполнены нулями: для центрированных остатков это
        читается как «сосед не даёт сигнала», а не как «сосед показал ноль».
        Оставлено как проверенная альтернатива корреляционным весам, в отчёте
        проигрывает им на всех трёх зёрнах.
        """
        P = len(cols)
        W = np.zeros((P, P))
        X_all = clipped * finite
        for p in range(P):
            rows = finite[:, p]
            if rows.sum() < 40:
                continue
            X = np.delete(X_all[rows], p, axis=1)
            G = X.T @ X + self.alpha * np.eye(X.shape[1])
            W[p, np.arange(P) != p] = np.linalg.solve(G, X.T @ clipped[rows, p])
        return W

    # -------------------------------------------------------------- поправка

    def _correction(self, clipped: np.ndarray, finite: np.ndarray, W: np.ndarray) -> np.ndarray:
        """C[d, p] — оценка общей суточной помехи для поля p на день d.

        Всё выражается через два матричных произведения: взвешенная сумма остатков
        доступных соседей и сумма их весов. Считать это построчно в цикле по точкам
        было бы на два порядка дороже, а результат от целевой точки не зависит.
        """
        Xw = (clipped * finite) @ W.T           # взвешенная сумма остатков соседей
        Nw = finite.astype(float) @ W.T         # сумма весов доступных соседей
        n = finite.astype(float) @ (W.T > 0)    # сколько соседей вообще доступно

        if self.weight_mode == "ridge":
            # У ridge веса — уже коэффициенты регрессии, нормировать их на сумму нельзя
            C = Xw
        else:
            C = np.where(Nw > 0, Xw / np.where(Nw > 0, Nw, 1.0), 0.0)

        if self.shrink_k is None:
            # Поведение E06: меньше трёх соседей — поправки нет
            return np.where(n >= MIN_SIBLINGS, C, 0.0)
        # Плавная альтернатива: среднее по двум полям само шумное, но выбрасывать
        # его целиком расточительно, достаточно ослабить
        return C * (n / (n + float(self.shrink_k)))

    def _betas(self, C: np.ndarray, arr: np.ndarray, finite: np.ndarray) -> np.ndarray:
        """Коэффициент переноса помехи: единый или свой на каждое поле.

        Оценка по видимым дням поля — регрессия его остатка на его же суточную
        поправку без свободного члена. Sxx здесь играет роль информации Фишера по
        β, поэтому сжатие к общему значению записывается как ridge с весом tau в
        тех же единицах: у поля с короткой историей Sxx мал и оценка сама уезжает
        к общей, а не остаётся шумной.
        """
        P = arr.shape[1]
        Sxy, Sxx = np.zeros(P), np.zeros(P)
        for p in range(P):
            m = finite[:, p]
            c = C[m, p]
            Sxy[p] = float(c @ np.clip(arr[m, p], -CLIP, CLIP))
            Sxx[p] = float(c @ c)
        beta_g = Sxy.sum() / Sxx.sum() if Sxx.sum() > 0 else 1.0
        if self.beta_mode == "one":
            return np.ones(P)
        if self.beta_mode == "global":
            return np.full(P, beta_g)
        return (Sxy + self.tau * beta_g) / np.maximum(Sxx + self.tau, 1e-9)

    # -------------------------------------------------------------- прогноз

    def predict_points(self, points: list, views: dict[str, PolygonView], context: dict) -> np.ndarray:
        table, _ = residual_table(views, lam=self.lam_res)
        if table.empty:
            return np.full(len(points), FALLBACK_NDVI)

        arr = table.to_numpy()
        finite = np.isfinite(arr)
        clipped = np.where(finite, np.clip(arr, -CLIP, CLIP), 0.0)
        cols = list(table.columns)
        days = table.index.to_numpy()

        if self.weight_mode == "ridge":
            W = self._weights_ridge(clipped, finite, cols)
        elif self.weight_mode == "flat":
            W = np.ones((len(cols), len(cols))) - np.eye(len(cols))
        else:
            W = self._weights_corr(arr, cols)

        C = self._correction(clipped, finite, W)
        beta = self._betas(C, arr, finite)

        col_index = {c: i for i, c in enumerate(cols)}
        day_index = {int(d): i for i, d in enumerate(days)}

        # Пересглаживание очищенных рядов — один раз на весь прогон, а не на точку
        cleaned: dict[str, tuple] = {}
        for pid, view in views.items():
            ci = col_index.get(pid)
            if ci is None or len(view.known_ord) < 4:
                continue
            idx = np.array([day_index[int(d)] for d in view.known_ord])
            y = view.known_values - beta[ci] * C[idx, ci]
            cleaned[pid] = restore_on_grid(view.known_ord, y, lam=self.lam_fit, mix=1.0)

        out = np.empty(len(points), dtype=float)
        for k, p in enumerate(points):
            grid, values = cleaned.get(p.polygon_id, (None, None))
            base = (predict_at(grid, values, np.array([p.ord_day]))[0]
                    if grid is not None else FALLBACK_NDVI)
            ci, row = col_index.get(p.polygon_id), day_index.get(int(p.ord_day))
            add = beta[ci] * C[row, ci] if (ci is not None and row is not None) else 0.0
            out[k] = base + add
        return np.clip(out, 0.0, 1.0)


@register("nb_corr3", "Поправка соседей, веса по корреляции^3", experiment="E07",
          tags=("current",))
class NeighborCorr3(_NeighborRegression):
    """Рабочая конфигурация E07: +0,0030 к sibit10, устойчиво на трёх зёрнах."""

    def __init__(self):
        super().__init__(weight_mode="corr", power=3.0, floor=0.1)


@register("nb_corr3_l700", "Поправка соседей, корреляция^3, λ = 700", experiment="E07")
class NeighborCorr3L700(_NeighborRegression):
    """То же с чуть более слабым пересглаживанием: дно по λ пологое, 700 и 1000 равны."""

    def __init__(self):
        super().__init__(weight_mode="corr", power=3.0, floor=0.1, lam_fit=700.0)


@register("nb_corr1", "Поправка соседей, веса по корреляции^1", experiment="E07")
class NeighborCorr1(_NeighborRegression):
    """Разложение выигрыша: линейный вес даёт только треть от эффекта степени 3."""

    def __init__(self):
        super().__init__(weight_mode="corr", power=1.0, floor=0.0)


@register("nb_ridge", "Поправка соседей, ridge по отдельным полям", experiment="E07")
class NeighborRidge(_NeighborRegression):
    """Полноценная регрессия на соседей. Проигрывает корреляционным весам."""

    def __init__(self):
        super().__init__(weight_mode="ridge", alpha=0.3)


@register("nb_flat_beta", "Поправка соседей, свой коэффициент переноса на поле",
          experiment="E07")
class NeighborFlatBeta(_NeighborRegression):
    """Проверка гипотезы 1: своя β на поле при равных весах. Прироста нет."""

    def __init__(self):
        super().__init__(weight_mode="flat", beta_mode="field", tau=50.0)
