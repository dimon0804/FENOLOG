"""Сборка табличных признаков для надстройки-бустинга (эксперимент E05).

Идея эксперимента. Ни один метод восстановления не выигрывает везде: на разрыве
в один-два дня побеждает слабое сглаживание, на разрыве в два месяца — линейная
интерполяция и климатическая норма. Значит нужен не ещё один сглаживатель, а
арбитр: модель, которая по обстановке вокруг точки решает, кому из методов
сегодня верить. Признаки ниже описывают ровно эту обстановку.

Два блока:
  * FeatureBuilder — превращает список HoldoutPoint в матрицу признаков;
  * build_extra_points — строит дополнительный обучающий набор поверх строк
    train_dataset.csv, потому что 3000 контрольных точек для бустинга мало.

Важное ограничение протокола, которое здесь соблюдается буквально: у контрольной
точки замаскировано ВСЁ, кроме id, даты и культуры. Поэтому ни климатология, ни
погода, ни doy из самой строки не берутся никогда — только из соседних строк
того же полигона. Единственное исключение — `_doy` / `_year` / `_ord`, они
вычислены из даты, а дата видна.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml import holdout as H
from src.ml.dataset import PolygonView, build_views, mask_rows
from src.ml.holdout import HoldoutPoint

# Окна, в которых считается плотность наблюдений. Верхняя граница 60 дней взята
# по максимальной длине разрыва, которая ещё встречается массово.
COUNT_WINDOWS = (7, 15, 30, 60)
# Окна усреднения погоды. Погода в самой контрольной строке стёрта, поэтому
# усредняем по соседним датам; ±3 дня — это уже почти «погода в этот день».
WEATHER_WINDOWS = (3, 7, 15)


# --------------------------------------------------------------------------- #
# Мелкие численные помощники
# --------------------------------------------------------------------------- #

def _window_stats(sorted_ord: np.ndarray, values: np.ndarray,
                  targets: np.ndarray, half: int):
    """Количество, среднее и разброс значений в окне ±half дней вокруг каждой цели.

    Через префиксные суммы, а не циклом: точек с учётом расширенного обучающего
    набора десятки тысяч, а окон четыре штуки на каждую.
    """
    n_out = np.zeros(len(targets), dtype=float)
    mean_out = np.full(len(targets), np.nan)
    std_out = np.full(len(targets), np.nan)
    if len(sorted_ord) == 0:
        return n_out, mean_out, std_out

    lo = np.searchsorted(sorted_ord, targets - half, side="left")
    hi = np.searchsorted(sorted_ord, targets + half, side="right")
    cs = np.concatenate([[0.0], np.cumsum(values)])
    cs2 = np.concatenate([[0.0], np.cumsum(values ** 2)])
    n = (hi - lo).astype(float)
    s = cs[hi] - cs[lo]
    s2 = cs2[hi] - cs2[lo]
    ok = n > 0
    mean_out[ok] = s[ok] / n[ok]
    ok2 = n > 1
    var = s2[ok2] / n[ok2] - mean_out[ok2] ** 2
    std_out[ok2] = np.sqrt(np.clip(var, 0.0, None))
    return n, mean_out, std_out


def _doy_lookup(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Таблица «день года -> значение» по одному полигону, длиной 367.

    Собирается по всем годам полигона: у строки-цели поле стёрто, но у той же
    даты в других годах оно на месте. Дырки заполняются линейной интерполяцией
    по дню года — иначе поля, у которых есть только сезон 2025, остались бы
    вообще без климатологии.
    """
    out = np.full(367, np.nan)
    if column not in frame.columns:
        return out
    vals = frame[column].to_numpy(dtype=float)
    doy = frame["_doy"].to_numpy(dtype=np.int64)
    ok = np.isfinite(vals) & (doy >= 1) & (doy <= 366)
    if not ok.any():
        return out
    agg = pd.Series(vals[ok]).groupby(doy[ok]).mean()
    out[agg.index.to_numpy()] = agg.to_numpy()
    s = pd.Series(out[1:367])
    out[1:367] = s.interpolate(limit_direction="both").to_numpy()
    return out


class _PolygonCache:
    """Предрасчёт по одному полигону: справочники дня года и ряды погоды.

    Строится один раз на полигон и переиспользуется всеми точками этого поля —
    без этого сборка признаков по 20 тысячам точек занимала бы минуты.
    """

    __slots__ = ("clim_mean", "clim_std", "n_ref", "temp_norm", "precip_norm",
                 "w_ord", "temp", "precip", "known_ord", "known_val")

    def __init__(self, view: PolygonView):
        f = view.frame
        self.clim_mean = _doy_lookup(f, "ndvi_climatology_mean")
        self.clim_std = _doy_lookup(f, "ndvi_climatology_std")
        self.n_ref = _doy_lookup(f, "n_reference_years")
        self.temp_norm = _doy_lookup(f, "era5_temp_c")
        self.precip_norm = _doy_lookup(f, "era5_precip_mm")

        # Ряд погоды по датам, где она вообще известна. У скрытых строк погода
        # стёрта, поэтому такие даты сюда не попадают — это и есть требование
        # «брать погоду из соседних дат».
        ords = f["_ord"].to_numpy(dtype=np.int64)
        temp = f["era5_temp_c"].to_numpy(dtype=float) if "era5_temp_c" in f else np.full(len(f), np.nan)
        prec = f["era5_precip_mm"].to_numpy(dtype=float) if "era5_precip_mm" in f else np.full(len(f), np.nan)
        ok = np.isfinite(temp) | np.isfinite(prec)
        self.w_ord = ords[ok]
        self.temp = np.nan_to_num(temp[ok], nan=0.0)
        self.precip = np.nan_to_num(prec[ok], nan=0.0)

        self.known_ord = view.known_ord
        self.known_val = view.known_values


def _safe_doy(doy: np.ndarray) -> np.ndarray:
    return np.clip(doy, 1, 366).astype(np.int64)


# --------------------------------------------------------------------------- #
# Сборка матрицы признаков
# --------------------------------------------------------------------------- #

class FeatureBuilder:
    """Превращает список контрольных точек в таблицу признаков.

    crop_map фиксируется снаружи, чтобы у обучающей и контрольной части кодировка
    культуры совпадала: категориальный признак в LightGBM — это просто целое
    число, и разъехавшаяся кодировка молча испортила бы модель.
    """

    def __init__(self, views: dict[str, PolygonView], crop_map: dict[str, int] | None = None):
        self.views = views
        if crop_map is None:
            crops = sorted({v.crop_type for v in views.values() if v.crop_type})
            crop_map = {c: i for i, c in enumerate(crops)}
        self.crop_map = crop_map
        self._cache: dict[str, _PolygonCache] = {}

    def _cache_for(self, polygon_id: str) -> _PolygonCache:
        c = self._cache.get(polygon_id)
        if c is None:
            c = _PolygonCache(self.views[polygon_id])
            self._cache[polygon_id] = c
        return c

    def transform(self, points: list[HoldoutPoint], base_preds: pd.DataFrame) -> pd.DataFrame:
        """base_preds — предсказания всех методов без обучения по тем же точкам.

        Колонки берутся какие есть, а не фиксированным списком: параллельные
        эксперименты добавляют в реестр новые методы, и надстройка обязана
        усиливаться от них сама, без правок здесь.
        """
        n = len(points)
        ords = np.array([p.ord_day for p in points], dtype=np.int64)
        pids = np.array([p.polygon_id for p in points])
        left_d = np.array([p.left_dist for p in points], dtype=float)
        right_d = np.array([p.right_dist for p in points], dtype=float)

        cols: dict[str, np.ndarray] = {}

        # --- 1. Предсказания базовых методов: главный блок признаков ---------
        P = base_preds.to_numpy(dtype=float)
        for j, name in enumerate(base_preds.columns):
            cols[f"p_{name}"] = P[:, j]
        # Разброс между методами — прямая мера того, насколько ситуация спорная.
        # Там, где все методы согласны, спорить не о чем и модель просто их
        # повторит; вся польза бустинга живёт в точках с большим разбросом.
        cols["p_mean"] = np.nanmean(P, axis=1)
        cols["p_std"] = np.nanstd(P, axis=1)
        cols["p_range"] = np.nanmax(P, axis=1) - np.nanmin(P, axis=1)
        cols["p_median"] = np.nanmedian(P, axis=1)

        # --- 2. Геометрия разрыва -------------------------------------------
        span = left_d + right_d
        cols["left_dist"] = left_d
        cols["right_dist"] = right_d
        cols["gap_span"] = span
        cols["gap_min"] = np.minimum(left_d, right_d)
        cols["gap_max"] = np.maximum(left_d, right_d)
        # Асимметрия разрыва: при 1 слева и 30 справа доверять надо левому соседу,
        # а при 15/15 — сглаживанию. Одним расстоянием это не выражается.
        cols["gap_ratio"] = np.minimum(left_d, right_d) / np.maximum(np.maximum(left_d, right_d), 1.0)
        cols["log_left"] = np.log1p(np.clip(left_d, 0, None))
        cols["log_right"] = np.log1p(np.clip(right_d, 0, None))
        cols["log_span"] = np.log1p(np.clip(span, 0, None))
        cols["month"] = np.array([p.month for p in points], dtype=float)

        # --- 3. Всё, что требует ряда полигона ------------------------------
        left_val = np.full(n, np.nan)
        right_val = np.full(n, np.nan)
        clim_mean = np.full(n, np.nan)
        clim_std = np.full(n, np.nan)
        n_ref = np.full(n, np.nan)
        clim_left = np.full(n, np.nan)
        clim_right = np.full(n, np.nan)
        cnt = {w: np.zeros(n) for w in COUNT_WINDOWS}
        wmean = {w: np.full(n, np.nan) for w in COUNT_WINDOWS}
        wstd = {w: np.full(n, np.nan) for w in COUNT_WINDOWS}
        temp_w = {w: np.full(n, np.nan) for w in WEATHER_WINDOWS}
        prec_w = {w: np.full(n, np.nan) for w in WEATHER_WINDOWS}
        temp_dev = {w: np.full(n, np.nan) for w in WEATHER_WINDOWS}
        prec_dev = {w: np.full(n, np.nan) for w in WEATHER_WINDOWS}
        crop_code = np.full(n, -1.0)

        order = np.argsort(pids, kind="stable")
        for pid in pd.unique(pids[order]):
            idx = order[pids[order] == pid]
            if pid not in self.views:
                continue
            c = self._cache_for(pid)
            t = ords[idx]

            # Соседние известные значения. Тот же поиск, что и в базовых методах,
            # но здесь нужны сами значения, а не результат интерполяции.
            if len(c.known_ord):
                pos = np.searchsorted(c.known_ord, t, side="left")
                has_l = pos > 0
                has_r = pos < len(c.known_ord)
                left_val[idx[has_l]] = c.known_val[pos[has_l] - 1]
                right_val[idx[has_r]] = c.known_val[pos[has_r]]
                # Дни года соседей нужны, чтобы сравнить их с нормой ИМЕННО их дня,
                # а не дня цели: за 30 дней норма успевает заметно уехать.
                l_ord = np.full(len(idx), np.nan)
                r_ord = np.full(len(idx), np.nan)
                l_ord[has_l] = c.known_ord[pos[has_l] - 1]
                r_ord[has_r] = c.known_ord[pos[has_r]]
            else:
                l_ord = np.full(len(idx), np.nan)
                r_ord = np.full(len(idx), np.nan)

            doy = _safe_doy(np.array([pd.Timestamp.fromordinal(int(x)).timetuple().tm_yday for x in t]))
            clim_mean[idx] = c.clim_mean[doy]
            clim_std[idx] = c.clim_std[doy]
            n_ref[idx] = c.n_ref[doy]
            for arr, src, dst in ((l_ord, c.clim_mean, clim_left), (r_ord, c.clim_mean, clim_right)):
                ok = np.isfinite(arr)
                if ok.any():
                    d = _safe_doy(np.array([pd.Timestamp.fromordinal(int(x)).timetuple().tm_yday
                                            for x in arr[ok]]))
                    dst[idx[ok]] = src[d]

            for w in COUNT_WINDOWS:
                a, b, s = _window_stats(c.known_ord, c.known_val, t, w)
                cnt[w][idx], wmean[w][idx], wstd[w][idx] = a, b, s

            for w in WEATHER_WINDOWS:
                _, tm, _ = _window_stats(c.w_ord, c.temp, t, w)
                _, pm, _ = _window_stats(c.w_ord, c.precip, t, w)
                temp_w[w][idx] = tm
                prec_w[w][idx] = pm
                # Отклонение от нормы этого дня года: сама температура почти
                # дублирует день года, а вот аномалия несёт новую информацию.
                temp_dev[w][idx] = tm - c.temp_norm[doy]
                prec_dev[w][idx] = pm - c.precip_norm[doy]

            crop = self.views[pid].crop_type
            crop_code[idx] = self.crop_map.get(crop, -1) if crop else -1

        cols["left_val"] = left_val
        cols["right_val"] = right_val
        cols["neigh_diff"] = right_val - left_val
        cols["neigh_mean"] = 0.5 * (left_val + right_val)
        # Наклон ряда на разрыве: единственный признак, который отличает
        # «стабильное плато» от «фаза роста, посередине провал» без модели.
        cols["neigh_slope"] = (right_val - left_val) / np.maximum(span, 1.0)
        cols["clim_mean"] = clim_mean
        cols["clim_std"] = clim_std
        cols["n_reference_years"] = n_ref
        cols["left_anom"] = left_val - clim_left
        cols["right_anom"] = right_val - clim_right
        cols["neigh_anom"] = 0.5 * (left_val + right_val) - clim_mean
        # Отклонение соседей от нормы, приведённое к разбросу нормы: у поля с
        # маленьким clim_std то же абсолютное отклонение значит гораздо больше.
        cols["neigh_anom_z"] = cols["neigh_anom"] / np.where(clim_std > 1e-6, clim_std, np.nan)

        for w in COUNT_WINDOWS:
            cols[f"n_obs_{w}"] = cnt[w]
            cols[f"obs_mean_{w}"] = wmean[w]
            cols[f"obs_std_{w}"] = wstd[w]
        for w in WEATHER_WINDOWS:
            cols[f"temp_{w}"] = temp_w[w]
            cols[f"precip_{w}"] = prec_w[w]
            cols[f"temp_dev_{w}"] = temp_dev[w]
            cols[f"precip_dev_{w}"] = prec_dev[w]

        doy_all = np.array([pd.Timestamp.fromordinal(int(x)).timetuple().tm_yday for x in ords],
                           dtype=float)
        cols["doy"] = doy_all
        # Гармоники дня года вместо самого doy: 31 декабря и 1 января должны быть
        # рядом, а деревьям линейный doy этого не сообщает.
        cols["doy_sin"] = np.sin(2 * np.pi * doy_all / 365.0)
        cols["doy_cos"] = np.cos(2 * np.pi * doy_all / 365.0)
        cols["year"] = np.array([pd.Timestamp.fromordinal(int(x)).year for x in ords], dtype=float)
        cols["crop_type"] = crop_code

        X = pd.DataFrame(cols)
        return X


CATEGORICAL = ["crop_type"]


# --------------------------------------------------------------------------- #
# Расширение обучающей выборки за счёт train_dataset.csv
# --------------------------------------------------------------------------- #

def build_extra_points(df: pd.DataFrame, templates: pd.DataFrame,
                       n_replicates: int = 4, seed0: int = 4242,
                       hide_frac: float = 0.20, n_scored: int = 6000):
    """Строит дополнительные обучающие точки, центрируясь на строках train_dataset.

    Зачем. Оцениваемых контрольных точек всего три тысячи — для градиентного
    бустинга с полусотней признаков это мало. При этом в train_dataset 30 520
    строк с непустым primary_ndvi, и пары «полигон + дата» с тестом не
    пересекаются ни разу. Значит поверх train можно построить второй, полностью
    независимый от метрики контрольный набор той же геометрии и получить в разы
    больше обучающих примеров.

    Как. holdout.build_holdout умеет ставить центр шаблона только на строку с
    _source == "test". Файл holdout.py — общий, править его нельзя, поэтому
    источник переворачивается в копии таблицы: train становится "test" и
    наоборот. Возвращаемые точки и индексы скрытых строк от этого не зависят,
    а маскирование делается уже по исходной таблице с настоящими источниками —
    иначе сломался бы метод whit1000_notrain, который читает known_source.

    Каждая реплика — независимый розыгрыш с своим зерном на одной и той же
    таблице. Это дешевле, чем задирать hide_frac: при большой доле скрытых
    значений ряд разреживается, и обучающие примеры перестают быть похожи на
    боевые.

    Возвращает список (points, views, masked_df) по репликам.
    """
    flipped = df.copy()
    flipped["_source"] = np.where(df["_source"].to_numpy() == "train", "test", "train")

    out = []
    for r in range(n_replicates):
        pts, hidden = H.build_holdout(flipped, templates, hide_frac=hide_frac,
                                      seed=seed0 + r, n_scored=n_scored)
        if not pts:
            continue
        masked = mask_rows(df, hidden)
        out.append((pts, build_views(masked), masked))
    return out


def base_preds_for(points, views, masked_df, raw_df, columns) -> pd.DataFrame:
    """Считает предсказания базовых методов на дополнительных точках.

    Признаки надстройки — это в первую очередь предсказания остальных методов,
    поэтому дополнительный набор без них бесполезен. Импорт реестра сделан
    внутри функции: features.py не должен тянуть за собой модули методов при
    обычном использовании.
    """
    from src.ml.registry import REGISTRY

    ctx = {"df": masked_df, "raw": raw_df, "points": points}
    preds = {}
    for name in columns:
        spec = REGISTRY.get(name)
        if spec is None:
            continue
        m = spec.factory()
        if m.needs_fit:
            continue
        preds[name] = m.predict_points(points, views, ctx)
    return pd.DataFrame(preds).reindex(columns=list(columns))
