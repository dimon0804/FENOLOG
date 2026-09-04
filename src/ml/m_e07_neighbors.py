"""E07. Регрессия по соседям: что осталось после корреляционных весов.

Контекст. E06 оценивает общую суточную помеху усечённым средним остатков всех
соседей: равный вес каждому полю, единый коэффициент переноса 1,0, жёсткий порог
«меньше трёх соседей — поправки нет». Взвешивание соседей по корреляции остатков
(`max(corr, 0) ** 3`) закрыто параллельно агентом E02 и вошло в общую базу
`e02s_best`; здесь оно воспроизведено независимо и служит точкой отсчёта, а не
результатом.

Задача E07 — четыре ручки, которых корреляционные веса не касаются. Все четыре
проверены на трёх зёрнах поверх корреляционных весов и все четыре пусты:

  свой коэффициент переноса β на каждое поле   -0,0000   разброс β реален, толку нет
  плавное сжатие n/(n+k) вместо порога в 3      +0,0002   ниже порога значимости
  ridge по остаткам отдельных соседей          -0,0008   проигрывает corr^3
  погодные группы поверх корреляции            +0,0001   сигнал уже забран корреляцией

Единственная непустая находка — мягкий порог корреляции: поднимать веса ниже 0,1
до 0,1 вместо обнуления даёт +0,0005 устойчиво на трёх зёрнах. Тоже ниже порога
0,002, отдаётся владельцу общей базы как готовая правка.

ОТКЛОНЕНО: прирост в пределах шума. Подробности в reports/exp_e07.md.
Методы оставлены в реестре как воспроизводимое подтверждение замеров.
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
# Сила пересглаживания очищенного ряда. На сыром ряде оптимум 1000, поверх
# очищенного он уезжает к 500: у ряда с вычтенной общей помехой меньше шума,
# и сглаживать его надо мягче. Плато 400-800, замерено на трёх зёрнах.
LAM_FIT = 500.0


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
                 lam_res: float = 1000.0, lam_fit: float = LAM_FIT):
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
        """Вес соседа — корреляция его остатков с остатками целевого поля в степени.

        Степень разносит соседей по полезности: при линейном весе поле с
        корреляцией 0,2 получает половину веса поля с корреляцией 0,4, при
        кубическом — восьмую часть. Оптимум пологий, степени 3 и 4 равны.

        Порог floor поднимает слабые и отрицательные корреляции до 0,1, а не
        обнуляет их. Разница именно в вырожденном случае: при нулевых весах поле,
        у которого все соседи оказались слабыми, осталось бы совсем без поправки,
        а мягкий порог оставляет ему обычное среднее как запасной вариант. На трёх
        зёрнах это стоит +0,0005 против обнуления.
        """
        C = pd.DataFrame(arr, columns=cols).corr(min_periods=self.min_periods).to_numpy()
        C = np.nan_to_num(C, nan=0.0)
        W = np.clip(C, self.floor, None) ** self.power
        np.fill_diagonal(W, 0.0)
        return W

    def _weights_ridge(self, clipped: np.ndarray, finite: np.ndarray, cols: list) -> np.ndarray:
        """Ridge-регрессия остатка поля на остатки отдельных соседей.

        Мотивация была в том, что корреляционные веса игнорируют скоррелированность
        соседей между собой, а регрессия её учитывает. На практике проигрывает:
        соседей до 77 при паре сотен наблюдений на поле, и регуляризация, которой
        хватает против переобучения, вместе с ним съедает и полезный сигнал.

        Пропуски заполнены нулями: для центрированных остатков это читается как
        «сосед не даёт сигнала», а не как «сосед показал ноль».
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
        доступных соседей и сумма их весов. Построчный цикл по контрольным точкам
        стоил бы на два порядка дороже, а результат от целевой точки не зависит.
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
        # его целиком расточительно, достаточно ослабить. Прироста не даёт —
        # дней с числом соседей меньше трёх всего 8,6 %.
        return C * (n / (n + float(self.shrink_k)))

    def _betas(self, C: np.ndarray, arr: np.ndarray, finite: np.ndarray) -> np.ndarray:
        """Коэффициент переноса помехи: единый или свой на каждое поле.

        Оценка по видимым дням поля — регрессия его остатка на его же суточную
        поправку без свободного члена. Sxx здесь играет роль информации Фишера по
        β, поэтому сжатие к общему значению записывается как ridge с весом tau в
        тех же единицах: у поля с короткой историей Sxx мал и оценка сама уезжает
        к общей, а не остаётся шумной.

        Замер отрицательный, и это содержательный результат: разброс β по полям
        реален (10-90 % от 0,60 до 1,13), но МНК минимизирует ошибку на видимых
        днях, где у поля есть собственное наблюдение, а поправка нужна на скрытом
        дне, где его нет. Оптимум одной задачи не является оптимумом другой,
        поэтому подогнанное β = 0,896 работает не лучше наивной единицы.
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


# --------------------------------------------------------------------------
# Регистрация. Всё на одной λ = 500, иначе таблица сравнивает разные вещи.
# --------------------------------------------------------------------------

@register("nb_corr3", "Соседи: корреляция^3, мягкий порог 0,1", experiment="E07")
class NeighborCorr3(_NeighborRegression):
    """Опорная конфигурация E07: корреляционные веса плюс мягкий порог."""

    def __init__(self):
        super().__init__(weight_mode="corr", power=3.0, floor=0.1)


@register("nb_corr3_f0", "Соседи: корреляция^3, жёсткое обнуление", experiment="E07")
class NeighborCorr3Floor0(_NeighborRegression):
    """Вариант E02 без мягкого порога — замер того, что даёт сам порог."""

    def __init__(self):
        super().__init__(weight_mode="corr", power=3.0, floor=0.0)


@register("nb_corr3_sh", "Соседи: корреляция^3 плюс сжатие n/(n+0,5)", experiment="E07")
class NeighborCorr3Shrink(_NeighborRegression):
    """Гипотеза 2: плавное сжатие вместо жёсткого порога в три соседа."""

    def __init__(self):
        super().__init__(weight_mode="corr", power=3.0, floor=0.1, shrink_k=0.5)


@register("nb_corr3_beta", "Соседи: корреляция^3 плюс своя β на поле", experiment="E07")
class NeighborCorr3Beta(_NeighborRegression):
    """Гипотеза 1: свой коэффициент переноса на каждое поле со сжатием к общему."""

    def __init__(self):
        super().__init__(weight_mode="corr", power=3.0, floor=0.1,
                         beta_mode="field", tau=50.0)


@register("nb_ridge", "Соседи: ridge по остаткам отдельных полей", experiment="E07")
class NeighborRidge(_NeighborRegression):
    """Гипотеза 3: полноценная регрессия вместо корреляционных весов."""

    def __init__(self):
        super().__init__(weight_mode="ridge", alpha=0.3)
