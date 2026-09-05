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

import hashlib

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
    # Веса по обратному квадрату собственного шума сенсора, измеренного на парах
    # соседних дней внутри одного сенсора: s2 0.039, landsat 0.054. Отношение
    # (0.039/0.054)^2 = 0.52. Для MODIS пар с шагом в день в наборе нет вовсе,
    # поэтому его вес не измерен, а взят по здравому смыслу (250 м против 10 м).
    "noise": {"s2": 1.0, "landsat": 0.52, "modis": 0.35, "unknown": 0.52},
    "flat":  {"s2": 1.0, "landsat": 0.8, "modis": 0.7, "unknown": 0.8},
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
            sensors = _sensor_of_known(view)
            table = SENSOR_WEIGHTS[self.weighted.removesuffix("_shuffle")]
            weights = np.array([table[s] for s in sensors], dtype=float)
            if self.weighted.endswith("_shuffle"):
                # Контроль подмены: тот же набор весов, но розданный наблюдениям
                # случайно. Если перемешанные веса дают тот же выигрыш, значит
                # дело не в сенсоре, а просто в том, что средний вес меньше
                # единицы, то есть в неявно усиленном сглаживании.
                # Зерно берётся из устойчивого хеша, а не из встроенного hash():
                # для строк тот случаен в каждом процессе (PYTHONHASHSEED), и
                # перемешивание получалось разным от запуска к запуску. Метод
                # контрольный, но он входит признаком в итоговую модель, поэтому
                # из-за одной этой строки переставал воспроизводиться весь
                # submission: расхождение до 0,0022 при RMSE 0,0596.
                seed = int.from_bytes(
                    hashlib.blake2s(view.polygon_id.encode("utf-8"), digest_size=4).digest(),
                    "big",
                )
                np.random.default_rng(seed).shuffle(weights)

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
_variant("e02_wsens", "E02 веса сенсоров 1 / 0.6 / 0.3", weighted="mid")
_variant("e02_wsoft", "E02 веса сенсоров 1 / 0.8 / 0.5", weighted="soft")
_variant("e02_whard", "E02 веса сенсоров 1 / 0.4 / 0.12", weighted="hard")
_variant("e02_ws2", "E02 веса сенсоров 1 / 0.25 / 0.02", weighted="s2only")

# --- очистка и веса вместе: складываются ли выигрыши -------------------------
_variant("e02_w_med3", "E02 веса 1/0.6/0.3 + медиана-3", weighted="mid", k=3)
_variant("e02_w_soft10", "E02 веса 1/0.6/0.3 + мягкая медиана 0.10", weighted="mid", k=3, threshold=0.10)
_variant("e02_w_soft20", "E02 веса 1/0.6/0.3 + мягкая медиана 0.20", weighted="mid", k=3, threshold=0.20)
_variant("e02_w_soft10_clamp", "E02 веса + мягкая медиана 0.10 + придавливание",
         weighted="mid", k=3, threshold=0.10, clip="clamp")

# --- контроли к весам: не переоткрыли ли мы просто подбор λ ------------------
# Средний вес меньше единицы эквивалентен увеличению λ. Поэтому взвешивание
# обязано выигрывать не только у λ=1000, но и у перенастроенной λ без весов,
# и у тех же весов, розданных наблюдениям вслепую.
_variant("e02_lam1250", "E02 контроль λ=1250 без весов", lam=1250.0)
_variant("e02_lam1500", "E02 контроль λ=1500 без весов", lam=1500.0)
_variant("e02_lam2000", "E02 контроль λ=2000 без весов", lam=2000.0)
_variant("e02_wshuf", "E02 контроль: веса 1/0.8/0.5 перемешаны", weighted="soft_shuffle")
_variant("e02_wsoft_l1250", "E02 веса 1/0.8/0.5 при λ=1250", weighted="soft", lam=1250.0)
_variant("e02_wsoft_l800", "E02 веса 1/0.8/0.5 при λ=800", weighted="soft", lam=800.0)

# --- финальная настройка победившей ветки ------------------------------------
_variant("e02_wsoft_l600", "E02 веса 1/0.8/0.5 при λ=600", weighted="soft", lam=600.0)
_variant("e02_wnoise", "E02 веса по шуму сенсора 1/0.52/0.35, λ=800", weighted="noise", lam=800.0)
_variant("e02_wflat", "E02 веса 1/0.8/0.7 при λ=800", weighted="flat", lam=800.0)
_variant("e02_wsoft_clamp", "E02 веса 1/0.8/0.5, λ=800, придавливание брака",
         weighted="soft", lam=800.0, clip="clamp")
_variant("e02_wsoft_clamp_s20", "E02 веса 1/0.8/0.5, λ=800, придавливание + мягкая медиана 0.20",
         weighted="soft", lam=800.0, clip="clamp", k=3, threshold=0.20)

# Разбор победившей комбинации на составляющие и подбор порога.
_variant("e02_clamp_s20", "E02 придавливание + мягкая медиана 0.20 без весов",
         lam=800.0, clip="clamp", k=3, threshold=0.20)
_variant("e02_wsoft_clamp_s15", "E02 победитель, порог 0.15",
         weighted="soft", lam=800.0, clip="clamp", k=3, threshold=0.15)
_variant("e02_wsoft_clamp_s25", "E02 победитель, порог 0.25",
         weighted="soft", lam=800.0, clip="clamp", k=3, threshold=0.25)
_variant("e02_wsoft_clamp_s20t", "E02 победитель, порог 0.20, окно по календарю ±8 дней",
         weighted="soft", lam=800.0, clip="clamp", k=3, threshold=0.20, mode="by_time")
_variant("e02_wsoft_clamp_s20_k5", "E02 победитель, порог 0.20, окно 5 наблюдений",
         weighted="soft", lam=800.0, clip="clamp", k=5, threshold=0.20)

# Календарное окно оказалось лучше индексного — подбираем его ширину и порог.
def _cal(key, title, **kw):
    base = dict(weighted="soft", lam=800.0, clip="clamp", k=3, threshold=0.20, mode="by_time")
    base.update(kw)
    _variant(key, title, **base)


_cal("e02_cal_w4", "E02 календарное окно ±4 дня, порог 0.20", half_window=4)
_cal("e02_cal_w6", "E02 календарное окно ±6 дней, порог 0.20", half_window=6)
_cal("e02_cal_w12", "E02 календарное окно ±12 дней, порог 0.20", half_window=12)
_cal("e02_cal_w20", "E02 календарное окно ±20 дней, порог 0.20", half_window=20)
_cal("e02_cal_s15", "E02 календарное окно ±8 дней, порог 0.15", threshold=0.15)
_cal("e02_cal_s25", "E02 календарное окно ±8 дней, порог 0.25", threshold=0.25)
_cal("e02_cal_k5", "E02 календарное окно ±8 дней, 5 наблюдений, порог 0.20", k=5)
_cal("e02_cal_k5_w12", "E02 календарное окно ±12 дней, 5 наблюдений, порог 0.20", k=5, half_window=12)


# ===========================================================================
# E02b. Очистка поверх E06: ряд, из которого вычтена общая суточная помеха
# ===========================================================================
#
# E06 показал, что часть «собственного шума наблюдения» общая для полей, снятых
# одним пролётом, и предсказуема по соседям. После вычитания этой суточной
# помехи точка отсчёта сдвинулась с 0.0799 до 0.0712, и мерить очистку от старой
# базы стало нельзя: оба метода бьют по одному и тому же шуму, и часть выигрыша
# оказалась бы посчитана дважды.
#
# Здесь то же самое собирается поверх очищенного ряда. Плюс главный поворот
# темы: сама суточная поправка считается робастно. Медианный фильтр трогает
# единицы процентов наблюдений, а поправка применяется к 95 % дней — значит её
# устойчивость к выбросам стоит дороже, чем аккуратность фильтра по трём точкам.

from src.ml.m_e06_sibling import MIN_SIBLINGS, residual_table  # noqa: E402


def _sensor_weight_table(views, table) -> np.ndarray:
    """Вес каждой ячейки таблицы остатков по сенсору, давшему это наблюдение.

    Смысл в том, что соседи неравноценны: остаток поля, снятого Sentinel-2,
    несёт общую суточную помеху с собственным шумом 0.039, а снятого Landsat —
    с 0.054. При усреднении по соседям вес обратно пропорционален дисперсии
    шума, то есть тот же набор весов, что и для наблюдений собственного ряда.
    """
    day_index = {int(d): i for i, d in enumerate(table.index.to_numpy())}
    out = np.zeros((len(table.index), len(table.columns)), dtype=float)
    weights = SENSOR_WEIGHTS["soft"]
    for j, pid in enumerate(table.columns):
        view = views[pid]
        sensors = _sensor_of_known(view)
        rows = np.array([day_index[int(d)] for d in view.known_ord])
        out[rows, j] = [weights[s] for s in sensors]
    return out


def _daily_corrections(table, sens_w, *, agg="mean", clip_lo=-0.25, clip_hi=0.25,
                       corr_power=0.0, top_k=12, min_sib=MIN_SIBLINGS,
                       corr_floor=0.0) -> dict:
    """Общая суточная поправка для каждого поля, посчитанная без него самого.

    Leave-one-out обязателен: иначе поле частично вычитало бы собственный шум,
    поправка была бы смещена в его сторону, а на контрольной точке собственного
    наблюдения нет и этого смещения не будет. То есть ряд, на котором строится
    кривая, и точка, в которой она оценивается, жили бы в разных распределениях.

    agg:
      'mean'    — взвешенное среднее подрезанных остатков (E06 = flat + ±0.25);
      'median'  — медиана без подрезки, прямолинейный робастный вариант;
      'iqr'     — отбраковка по межквартильному размаху строки, затем среднее;
      'topkmed' — медиана по top_k соседям, наиболее коррелированным с целевым.

    sens_w      — матрица весов ячеек (None = все веса равны);
    corr_power  — степень, в которую возводится корреляция соседа с целевым
                  полем при взвешивании; 0 отключает взвешивание по корреляции.
    """
    arr = table.to_numpy()
    finite = np.isfinite(arr)
    cols = list(table.columns)
    n_row = finite.sum(axis=1)
    out: dict = {}

    if agg == "median":
        for j, pid in enumerate(cols):
            a = np.where(finite, arr, np.nan)
            a[:, j] = np.nan
            with np.errstate(all="ignore"):
                m = np.nanmedian(a, axis=1)
            out[pid] = np.where(n_row - finite[:, j] >= min_sib, np.nan_to_num(m), 0.0)
        return out

    if agg == "topkmed":
        C = np.nan_to_num(table.corr(min_periods=30).to_numpy())
        clipped = np.where(finite, np.clip(arr, clip_lo, clip_hi), np.nan)
        for j, pid in enumerate(cols):
            order = np.argsort(-C[j])
            order = order[order != j][:top_k]
            with np.errstate(all="ignore"):
                m = np.nanmedian(clipped[:, order], axis=1)
            out[pid] = np.where(n_row - finite[:, j] >= min_sib, np.nan_to_num(m), 0.0)
        return out

    # Дальше — взвешенное среднее. Различаются только тем, какие ячейки в него
    # попадают (маска) и с какими весами.
    if agg == "iqr":
        # Границы отбраковки считаем по всей строке. Собственное наблюдение поля
        # в них попадает, но на датах контрольных точек его нет по построению,
        # а на прочих датах вклад одного столбца из полутора десятков в квартиль
        # пренебрежимо мал.
        with np.errstate(all="ignore"):
            q1 = np.nanpercentile(np.where(finite, arr, np.nan), 25, axis=1)
            q3 = np.nanpercentile(np.where(finite, arr, np.nan), 75, axis=1)
        fence = 1.5 * (q3 - q1)
        keep = finite & (arr >= (q1 - fence)[:, None]) & (arr <= (q3 + fence)[:, None])
        vals = np.where(keep, arr, 0.0)
    else:
        keep = finite
        vals = np.where(finite, np.clip(arr, clip_lo, clip_hi), 0.0)

    w = keep.astype(float) if sens_w is None else keep.astype(float) * sens_w
    num = vals * w

    if corr_power > 0:
        # Вес соседа = его корреляция с целевым полем в заданной степени.
        # Отрицательные корреляции обнуляем: сосед, который «ошибается наоборот»,
        # физического смысла как источник общей помехи не имеет.
        C = np.nan_to_num(table.corr(min_periods=30).to_numpy())
        # Нижний порог корреляции вместо обнуления. Находка E07: при обнулении
        # поле, у которого все соседи слабо коррелированы, остаётся вообще без
        # поправки; мягкий порог оставляет ему обычное среднее как запасной путь.
        # Даёт ровно +0.0005 на всех трёх зёрнах протокола без исключений.
        W = np.clip(C, corr_floor, None) ** corr_power
        np.fill_diagonal(W, 0.0)      # своё поле в свою поправку не входит никогда
        S = num @ W
        N = w @ W
        for j, pid in enumerate(cols):
            ok = (n_row - finite[:, j] >= min_sib) & (N[:, j] > 1e-9)
            out[pid] = np.where(ok, S[:, j] / np.maximum(N[:, j], 1e-9), 0.0)
        return out

    s_all = num.sum(axis=1)
    w_all = w.sum(axis=1)
    for j, pid in enumerate(cols):
        s = s_all - num[:, j]
        wn = w_all - w[:, j]
        ok = (n_row - finite[:, j] >= min_sib) & (wn > 1e-9)
        out[pid] = np.where(ok, s / np.maximum(wn, 1e-9), 0.0)
    return out


class _E02Sibling(_CleanWhittaker):
    """Очистка E02 поверх суточной поправки E06.

    Порядок: посчитать суточную помеху по соседям → вычесть её из наблюдений →
    применить свою очистку (придавливание брака, мягкая медиана по календарному
    окну) → сгладить со взвешиванием по сенсору → вернуть поправку обратно на
    дате цели.

    Вычитать помеху надо именно до сглаживания, а не прибавлять к готовому
    прогнозу: та же помеха сидит в опорных наблюдениях, по которым строится
    кривая, и пока она там, сглаживание за ней тянется.
    """

    def __init__(self, agg="mean", clip_lo=-0.25, clip_hi=0.25, corr_power=0.0,
                 corr_floor=0.0,
                 top_k=12, sensor_neighbours=False, beta=1.0, lam_res=1000.0, **kw):
        super().__init__(**kw)
        self.agg = agg
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.corr_power = corr_power
        self.corr_floor = corr_floor
        self.top_k = top_k
        self.sensor_neighbours = sensor_neighbours
        self.beta = beta
        self.lam_res = lam_res

    def predict_points(self, points, views, context):
        table, _ = residual_table(views, lam=self.lam_res)
        if table.empty:
            return np.full(len(points), FALLBACK_NDVI)
        sens_w = _sensor_weight_table(views, table) if self.sensor_neighbours else None
        corr = _daily_corrections(table, sens_w, agg=self.agg,
                                  clip_lo=self.clip_lo, clip_hi=self.clip_hi,
                                  corr_power=self.corr_power, top_k=self.top_k,
                                  corr_floor=self.corr_floor)
        days = table.index.to_numpy()
        day_index = {int(d): i for i, d in enumerate(days)}

        out = np.empty(len(points), dtype=float)
        by_polygon: dict[str, list[int]] = {}
        for i, p in enumerate(points):
            by_polygon.setdefault(p.polygon_id, []).append(i)

        for pid, idx in by_polygon.items():
            view = views[pid]
            targets = np.array([points[i].ord_day for i in idx], dtype=np.int64)
            c = corr.get(pid)
            if c is None or len(view.known_ord) < 2:
                out[np.array(idx)] = super().predict(view, targets)
                continue

            pos = np.array([day_index[int(d)] for d in view.known_ord])
            clean_view = PolygonView(
                polygon_id=view.polygon_id,
                crop_type=view.crop_type,
                frame=view.frame,
                known_ord=view.known_ord,
                # Ряд без общей суточной помехи — на нём и работает вся очистка E02
                known_values=view.known_values - c[pos],
                known_source=view.known_source,
            )
            base = super().predict(clean_view, targets)
            add = np.array([c[day_index[int(t)]] if int(t) in day_index else 0.0
                            for t in targets])
            out[np.array(idx)] = np.clip(base + self.beta * add, 0.0, 1.0)
        return out


def _sib(key: str, title: str, **kwargs):
    """Регистрация варианта «E02 поверх E06»: одна строка на один замер."""

    @register(key, title, experiment="E02b")
    class _V(_E02Sibling):
        def __init__(self, _kw=kwargs):
            super().__init__(**_kw)

    return _V


# Очистка ряда, признанная лучшей в первой половине эксперимента.
E02_CLEAN = dict(clip="clamp", k=3, threshold=0.20, mode="by_time", half_window=8)

# --- контроль: скелет обязан воспроизвести sibit10 ---------------------------
_sib("e02s_ctl", "E02b контроль = sibit10 (без очистки, без весов)",
     clip="drop", k=0, weighted=False, lam=1000.0)

# --- разложение очистки E02 поверх E06 --------------------------------------
_sib("e02s_w", "E02b только веса сенсоров, λ=1000", weighted="soft", lam=1000.0)
_sib("e02s_clamp", "E02b только придавливание брака", clip="clamp", lam=1000.0)
_sib("e02s_med", "E02b только мягкая медиана ±8 дн, порог 0.20",
     clip="drop", k=3, threshold=0.20, mode="by_time", half_window=8, lam=1000.0)
_sib("e02s_full1000", "E02b полная очистка + веса, λ=1000", weighted="soft", lam=1000.0, **E02_CLEAN)

# --- переподбор λ на очищенном ряде -----------------------------------------
_sib("e02s_full500", "E02b полная очистка + веса, λ=500", weighted="soft", lam=500.0, **E02_CLEAN)
_sib("e02s_full800", "E02b полная очистка + веса, λ=800", weighted="soft", lam=800.0, **E02_CLEAN)
_sib("e02s_full1500", "E02b полная очистка + веса, λ=1500", weighted="soft", lam=1500.0, **E02_CLEAN)
_sib("e02s_full2500", "E02b полная очистка + веса, λ=2500", weighted="soft", lam=2500.0, **E02_CLEAN)

# --- робастность самой суточной поправки ------------------------------------
_R = dict(weighted="soft", lam=1000.0, **E02_CLEAN)
_sib("e02s_pmed", "E02b поправка = медиана остатков", agg="median", **_R)
_sib("e02s_piqr", "E02b поправка = среднее после отбраковки по IQR", agg="iqr", **_R)
_sib("e02s_ptop8", "E02b поправка = медиана 8 самых коррелированных соседей", agg="topkmed", top_k=8, **_R)
_sib("e02s_ptop20", "E02b поправка = медиана 20 самых коррелированных соседей", agg="topkmed", top_k=20, **_R)
_sib("e02s_pcorr1", "E02b поправка взвешена корреляцией, степень 1", corr_power=1.0, **_R)
_sib("e02s_pcorr2", "E02b поправка взвешена корреляцией, степень 2", corr_power=2.0, **_R)
_sib("e02s_psens", "E02b поправка взвешена сенсором соседа", sensor_neighbours=True, **_R)
_sib("e02s_psens_corr1", "E02b поправка взвешена сенсором и корреляцией",
     sensor_neighbours=True, corr_power=1.0, **_R)

# --- подрезка остатков соседей: симметричная против асимметричной ------------
_sib("e02s_clip10", "E02b подрезка остатков ±0.10", clip_lo=-0.10, clip_hi=0.10, **_R)
_sib("e02s_clip15", "E02b подрезка остатков ±0.15", clip_lo=-0.15, clip_hi=0.15, **_R)
_sib("e02s_clip50", "E02b подрезка остатков ±0.50 (почти без подрезки)",
     clip_lo=-0.50, clip_hi=0.50, **_R)
_sib("e02s_asym_dn", "E02b подрезка [-0.15, +0.25]: провалы режем жёстче",
     clip_lo=-0.15, clip_hi=0.25, **_R)
_sib("e02s_asym_up", "E02b подрезка [-0.35, +0.15]: всплески режем жёстче",
     clip_lo=-0.35, clip_hi=0.15, **_R)


# --- взвешивание соседей корреляцией оказалось главным рычагом ---------------
# Сосед, чей остаток исторически ходит вместе с остатком целевого поля, снят тем
# же пролётом и накрыт той же дымкой; сосед с нулевой корреляцией приносит в
# среднее только свой собственный шум. Степень регулирует резкость отбора:
# при степени 1 вес пропорционален корреляции, при 4 фактически остаются
# несколько ближайших по поведению полей.
_sib("e02s_pcorr3", "E02b поправка взвешена корреляцией, степень 3", corr_power=3.0, **_R)
_sib("e02s_pcorr4", "E02b поправка взвешена корреляцией, степень 4", corr_power=4.0, **_R)
_sib("e02s_pcorr6", "E02b поправка взвешена корреляцией, степень 6", corr_power=6.0, **_R)
_sib("e02s_pcorr8", "E02b поправка взвешена корреляцией, степень 8", corr_power=8.0, **_R)

_R3 = dict(weighted="soft", corr_power=3.0, **E02_CLEAN)
_sib("e02s_c3_l500", "E02b корреляция^3, λ=500", lam=500.0, **_R3)
_sib("e02s_c3_l800", "E02b корреляция^3, λ=800", lam=800.0, **_R3)
_sib("e02s_c3_l1500", "E02b корреляция^3, λ=1500", lam=1500.0, **_R3)

# --- подрезка остатков соседей поверх корреляционных весов -------------------
_sib("e02s_c3_clip10", "E02b корреляция^3, подрезка ±0.10", lam=800.0, clip_lo=-0.10, clip_hi=0.10, **_R3)
_sib("e02s_c3_clip15", "E02b корреляция^3, подрезка ±0.15", lam=800.0, clip_lo=-0.15, clip_hi=0.15, **_R3)
_sib("e02s_c3_clip50", "E02b корреляция^3, подрезка ±0.50", lam=800.0, clip_lo=-0.50, clip_hi=0.50, **_R3)
_sib("e02s_c3_asym_dn", "E02b корреляция^3, подрезка [-0.15, +0.25]",
     lam=800.0, clip_lo=-0.15, clip_hi=0.25, **_R3)
_sib("e02s_c3_asym_up", "E02b корреляция^3, подрезка [-0.35, +0.15]",
     lam=800.0, clip_lo=-0.35, clip_hi=0.15, **_R3)
_sib("e02s_c3_asym_dn10", "E02b корреляция^3, подрезка [-0.10, +0.25]",
     lam=800.0, clip_lo=-0.10, clip_hi=0.25, **_R3)
_sib("e02s_c3_iqr", "E02b корреляция^3, отбраковка по IQR", lam=800.0, agg="iqr", **_R3)
_sib("e02s_c3_sens", "E02b корреляция^3 и вес сенсора соседа",
     lam=800.0, sensor_neighbours=True, **_R3)

# --- финал: оптимум по λ на очищенном ряде сместился вниз --------------------
# Чем меньше шума осталось в опорных наблюдениях, тем меньше нужно сглаживания:
# λ = 1000 на сыром ряде, 800 при взвешивании по сенсору, 500 поверх E06.
_sib("e02s_c3_l300", "E02b корреляция^3, λ=300", lam=300.0, **_R3)
_sib("e02s_c3_l400", "E02b корреляция^3, λ=400", lam=400.0, **_R3)
_sib("e02s_c3_l650", "E02b корреляция^3, λ=650", lam=650.0, **_R3)
_sib("e02s_best", "E02b итог: корреляция^3, λ=500, подрезка [-0.15, +0.25]",
     lam=500.0, clip_lo=-0.15, clip_hi=0.25, **_R3)
_sib("e02s_best_nomed", "E02b итог без медианного фильтра", lam=500.0, clip_lo=-0.15,
     clip_hi=0.25, weighted="soft", corr_power=3.0, clip="clamp")
_sib("e02s_best_now", "E02b итог без весов сенсора", lam=500.0, clip_lo=-0.15,
     clip_hi=0.25, corr_power=3.0, **E02_CLEAN)


# Находка E07: нижний порог корреляции 0.1 вместо обнуления. Отдельный ключ, чтобы
# уже опубликованные в журнале числа e02s_best остались воспроизводимыми.
_sib("e02s_best_f10", "E02b итог с нижним порогом корреляции 0,1",
     lam=500.0, clip_lo=-0.15, clip_hi=0.25, corr_floor=0.1, **_R3)
