"""Сборка табличных признаков для надстройки-бустинга (эксперимент E05).

Идея эксперимента. Ни один метод восстановления не выигрывает везде: на разрыве
в один-два дня побеждает слабое сглаживание, на разрыве в два месяца — линейная
интерполяция и климатическая норма. Значит нужен не ещё один сглаживатель, а
арбитр: модель, которая по обстановке вокруг точки решает, кому из методов
сегодня верить. Признаки ниже описывают ровно эту обстановку.

Ограничение протокола соблюдается буквально: у контрольной точки замаскировано
ВСЁ, кроме id, даты и культуры. Поэтому климатология, погода и doy из самой
строки не берутся никогда. Доступны только `_doy` / `_year` / `_ord` — они
вычислены из даты, а дата видна.

Три вещи, которых нет в исходных колонках и которые приходится восстанавливать
самим (основание — разведка train, reports/eda_train.md):

  * климатическая норма. Формула организаторов восстановлена точно: среднее
    primary_ndvi того же полигона по всем годам КРОМЕ текущего в окне ±8 дней по
    дню года. Значит норму можно посчитать на любую дату, включая дату разрыва,
    а не только там, где она случайно нашлась в соседней строке;
  * погода. Полигоны сбиваются в погодные группы с общей метеосводкой: на одну
    дату на 78 полигонов приходится 9-12 уникальных температур. Поэтому точную
    погоду контрольной точки отдаёт полигон-собрат по группе на ту же дату;
  * выбросы. Собственный шум наблюдения 0.066, и 5.6 % точек — одиночные скачки.
    Если ближайший сосед сам похож на выброс, доверять ему нельзя, и модель
    должна об этом знать.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml import holdout as H
from src.ml.dataset import PolygonView, build_views, mask_rows
from src.ml.holdout import HoldoutPoint

# Окна, в которых считается плотность наблюдений
COUNT_WINDOWS = (7, 15, 30, 60)
# Окна усреднения погоды по соседним датам (закрывают те 75 % точек,
# которым собрат по погодной группе погоду не отдал)
WEATHER_WINDOWS = (3, 7, 15)
# Полуширина окна климатологии организаторов, восстановлена в разведке train
CLIM_HALF = 8


# --------------------------------------------------------------------------- #
# Численные помощники
# --------------------------------------------------------------------------- #

def _window_stats(sorted_ord: np.ndarray, values: np.ndarray,
                  targets: np.ndarray, half: int):
    """Количество, среднее и разброс значений в окне ±half дней вокруг каждой цели.

    Через префиксные суммы, а не циклом: с расширенным обучающим набором точек
    десятки тысяч, а окон семь штук на каждую.
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


def _ord_to_doy(ords: np.ndarray) -> np.ndarray:
    """День года по порядковому номеру дня. Через datetime64, а не циклом по Timestamp."""
    d = np.asarray(ords, dtype="int64")
    dt = (d - 719163).astype("datetime64[D]")           # 719163 = ordinal 1970-01-01
    year_start = dt.astype("datetime64[Y]").astype("datetime64[D]")
    return ((dt - year_start).astype(int) + 1).astype(np.int64)


def _ord_to_year(ords: np.ndarray) -> np.ndarray:
    d = np.asarray(ords, dtype="int64")
    dt = (d - 719163).astype("datetime64[D]")
    return dt.astype("datetime64[Y]").astype(int) + 1970


# --------------------------------------------------------------------------- #
# Погодные группы полигонов
# --------------------------------------------------------------------------- #

class WeatherGroups:
    """Разбиение полигонов на группы с общей метеосводкой и выдача погоды по дате.

    Зачем. У контрольной точки era5 стёрт полностью, но ERA5 — внешние данные с
    грубой сеткой, и несколько соседних полей делят одну ячейку. Если найти
    полигон-собрата с той же метеосводкой, его строка на ту же дату вернёт
    погоду контрольной точки точь-в-точь. По разведке так закрывается четверть
    контрольных точек — это заметно лучше, чем усреднение по соседним датам.

    Собрат ищется среди ДРУГИХ полигонов: собственная колонка из выдачи всегда
    исключается, иначе на локальной валидации модель получила бы погоду, которой
    у настоящей контрольной точки нет, и цифра разошлась бы с лидербордом.

    Группировка: два полигона в одной группе, если на всех общих датах их
    температуры совпадают побитово (совпадений требуется хотя бы 30 дат).
    """

    def __init__(self, df: pd.DataFrame):
        piv_t = df.pivot_table(index="_ord", columns="anon_polygon_id",
                               values="era5_temp_c", aggfunc="first")
        piv_p = df.pivot_table(index="_ord", columns="anon_polygon_id",
                               values="era5_precip_mm", aggfunc="first")
        piv_p = piv_p.reindex(index=piv_t.index, columns=piv_t.columns)

        self.ords = piv_t.index.to_numpy(dtype=np.int64)
        self.cols = list(piv_t.columns)
        T = piv_t.to_numpy(dtype=float)
        P = piv_p.to_numpy(dtype=float)

        # Жадная кластеризация по представителям: групп заведомо мало (около 37),
        # поэтому квадрат по представителям, а не по всем парам полигонов.
        labels: dict[str, int] = {}
        reps: list[int] = []
        for j in range(T.shape[1]):
            placed = False
            for gid, rj in enumerate(reps):
                both = np.isfinite(T[:, j]) & np.isfinite(T[:, rj])
                if both.sum() < 30:
                    continue
                if np.array_equal(T[both, j], T[both, rj]):
                    labels[self.cols[j]] = gid
                    placed = True
                    break
            if not placed:
                labels[self.cols[j]] = len(reps)
                reps.append(j)
        self.labels = labels
        self.n_groups = len(reps)

        # Для каждого полигона — ряд погоды его группы БЕЗ него самого.
        # Значения внутри группы идентичны, поэтому nanmean здесь просто
        # «взять любое доступное», а не сглаживание.
        self._temp: dict[str, np.ndarray] = {}
        self._prec: dict[str, np.ndarray] = {}
        by_group: dict[int, list[int]] = {}
        for j, name in enumerate(self.cols):
            by_group.setdefault(labels[name], []).append(j)
        with np.errstate(invalid="ignore"):
            for j, name in enumerate(self.cols):
                mates = [k for k in by_group[labels[name]] if k != j]
                if not mates:
                    self._temp[name] = np.full(len(self.ords), np.nan)
                    self._prec[name] = np.full(len(self.ords), np.nan)
                    continue
                self._temp[name] = np.nanmean(T[:, mates], axis=1)
                self._prec[name] = np.nanmean(P[:, mates], axis=1)

    def lookup(self, polygon_id: str, ords: np.ndarray):
        """Точная погода на указанные даты от собратьев по погодной группе."""
        t = self._temp.get(polygon_id)
        if t is None:
            return np.full(len(ords), np.nan), np.full(len(ords), np.nan)
        pos = np.searchsorted(self.ords, ords)
        pos = np.clip(pos, 0, len(self.ords) - 1)
        hit = self.ords[pos] == ords
        out_t = np.full(len(ords), np.nan)
        out_p = np.full(len(ords), np.nan)
        out_t[hit] = t[pos[hit]]
        out_p[hit] = self._prec[polygon_id][pos[hit]]
        return out_t, out_p

    def series(self, polygon_id: str):
        """Весь ряд погоды группы без самого полигона — для оконных усреднений.

        У 23 полигонов из 78 своей погоды нет ни на одной дате. Без подстановки
        от собратьев треть точек осталась бы вообще без погодных признаков.
        """
        t = self._temp.get(polygon_id)
        if t is None:
            return self.ords, np.full(len(self.ords), np.nan), np.full(len(self.ords), np.nan)
        return self.ords, t, self._prec[polygon_id]


# --------------------------------------------------------------------------- #
# Общая суточная помеха (надстройка над E06)
# --------------------------------------------------------------------------- #

class SiblingStats:
    """Суточная поправка по соседним полям, число соседей и разброс их остатков.

    E06 показал, что часть «шума наблюдения» общая для полей, снятых одним
    пролётом, и её можно вычесть. Для арбитра важна не только сама поправка, но
    и её надёжность: поправка, собранная по трём полям, и поправка по тридцати —
    разные вещи, а при числе соседей меньше трёх метод её вообще не применяет.
    Поэтому рядом с величиной поправки идут её n и разброс.

    Считается через публичную residual_table из m_e06_sibling — чужой файл не
    правится, он используется как библиотека.
    """

    MIN_SIBLINGS = 3

    def __init__(self, views: dict[str, PolygonView], lam: float = 1000.0):
        from src.ml.m_e06_sibling import residual_table

        table, _ = residual_table(views, lam=lam)
        self.days = table.index.to_numpy(dtype=np.int64)
        arr = table.to_numpy()
        finite = np.isfinite(arr)
        clipped = np.where(finite, np.clip(arr, -0.25, 0.25), 0.0)
        row_s = clipped.sum(axis=1)
        row_q = (clipped ** 2).sum(axis=1)
        row_n = finite.sum(axis=1).astype(float)

        self.corr: dict[str, np.ndarray] = {}
        self.nsib: dict[str, np.ndarray] = {}
        self.sd: dict[str, np.ndarray] = {}
        for j, pid in enumerate(table.columns):
            # leave-one-out: вклад самого поля из агрегата вычитается, иначе
            # признак подглядывал бы в собственное наблюдение цели
            s = row_s - clipped[:, j]
            q = row_q - clipped[:, j] ** 2
            n = row_n - finite[:, j]
            safe = np.maximum(n, 1)
            mean = s / safe
            self.corr[pid] = np.where(n >= self.MIN_SIBLINGS, mean, 0.0)
            self.nsib[pid] = n
            self.sd[pid] = np.sqrt(np.clip(q / safe - mean ** 2, 0.0, None))

    def at(self, polygon_id: str, ords: np.ndarray):
        """Поправка, число соседей и разброс остатков на указанные даты."""
        n = len(ords)
        c = self.corr.get(polygon_id)
        if c is None:
            return np.zeros(n), np.zeros(n), np.full(n, np.nan)
        pos = np.searchsorted(self.days, ords)
        pos = np.clip(pos, 0, len(self.days) - 1)
        hit = self.days[pos] == ords
        out_c = np.zeros(n)
        out_n = np.zeros(n)
        out_s = np.full(n, np.nan)
        out_c[hit] = c[pos[hit]]
        out_n[hit] = self.nsib[polygon_id][pos[hit]]
        out_s[hit] = self.sd[polygon_id][pos[hit]]
        return out_c, out_n, out_s


# --------------------------------------------------------------------------- #
# Предрасчёт по полигону
# --------------------------------------------------------------------------- #

class _PolygonCache:
    """Всё, что считается по полигону один раз и переиспользуется всеми его точками."""

    __slots__ = ("years", "year_pos", "P_sum", "P_cnt", "P_sq",
                 "w_ord", "temp", "precip", "temp_norm", "precip_norm",
                 "known_ord", "known_val", "out_dev", "out_span")

    def __init__(self, view: PolygonView, mate_weather=None):
        f = view.frame
        self.known_ord = view.known_ord
        self.known_val = view.known_values

        # --- сетка для климатологии: сумма/счёт/сумма квадратов по (год, doy) ---
        # Норма организаторов — leave-one-out по годам, поэтому нужен разрез
        # именно по годам: из общей суммы окна вычитается вклад текущего года.
        doy = _ord_to_doy(self.known_ord)
        yr = _ord_to_year(self.known_ord)
        self.years = np.unique(yr) if len(yr) else np.array([], dtype=np.int64)
        self.year_pos = {int(y): i for i, y in enumerate(self.years)}
        ny = max(len(self.years), 1)
        g_sum = np.zeros((ny, 368))
        g_cnt = np.zeros((ny, 368))
        g_sq = np.zeros((ny, 368))
        if len(yr):
            yi = np.array([self.year_pos[int(y)] for y in yr])
            np.add.at(g_sum, (yi, doy), self.known_val)
            np.add.at(g_cnt, (yi, doy), 1.0)
            np.add.at(g_sq, (yi, doy), self.known_val ** 2)
        self.P_sum = np.cumsum(g_sum, axis=1)
        self.P_cnt = np.cumsum(g_cnt, axis=1)
        self.P_sq = np.cumsum(g_sq, axis=1)

        # --- ряд погоды самого полигона по датам, где она видна ---
        ords = f["_ord"].to_numpy(dtype=np.int64)
        temp = f["era5_temp_c"].to_numpy(dtype=float) if "era5_temp_c" in f else np.full(len(f), np.nan)
        prec = f["era5_precip_mm"].to_numpy(dtype=float) if "era5_precip_mm" in f else np.full(len(f), np.nan)
        # Своей погоды нет вовсе у 23 полигонов из 78, поэтому там, где она
        # отсутствует, подставляем сводку собратьев по погодной группе. Для
        # скрытой строки это ровно та же ситуация, что и у настоящей контрольной
        # точки: своя погода стёрта, чужая на месте.
        if mate_weather is not None:
            m_ord, m_temp, m_prec = mate_weather
            pos = np.searchsorted(m_ord, ords)
            pos = np.clip(pos, 0, len(m_ord) - 1)
            hit = (len(m_ord) > 0) & (m_ord[pos] == ords)
            need_t = ~np.isfinite(temp) & hit
            temp = np.where(need_t, m_temp[pos], temp)
            need_p = ~np.isfinite(prec) & hit
            prec = np.where(need_p, m_prec[pos], prec)

        ok = np.isfinite(temp) | np.isfinite(prec)
        self.w_ord = ords[ok]
        self.temp = np.nan_to_num(temp[ok], nan=0.0)
        self.precip = np.nan_to_num(prec[ok], nan=0.0)

        # Норма погоды по дню года — чтобы у модели была аномалия, а не просто
        # температура (температура почти дублирует день года и сама по себе
        # ничего нового не сообщает).
        self.temp_norm = _doy_profile(_ord_to_doy(self.w_ord), self.temp)
        self.precip_norm = _doy_profile(_ord_to_doy(self.w_ord), self.precip)

        # --- насколько каждое наблюдение похоже на одиночный выброс ---
        # dev — отклонение от середины своих соседей, span — расхождение самих
        # соседей. Большое dev при малом span и есть классический выброс.
        k = len(self.known_val)
        self.out_dev = np.full(k, np.nan)
        self.out_span = np.full(k, np.nan)
        if k >= 3:
            mid = 0.5 * (self.known_val[:-2] + self.known_val[2:])
            self.out_dev[1:-1] = np.abs(self.known_val[1:-1] - mid)
            self.out_span[1:-1] = np.abs(self.known_val[:-2] - self.known_val[2:])

    def clim_at(self, doys: np.ndarray, years: np.ndarray):
        """Климатическая норма, её разброс и число опорных лет на заданные даты.

        Формула организаторов: среднее primary_ndvi полигона в окне ±8 дней по
        дню года по всем годам, кроме года самой точки. Считается по видимым
        наблюдениям — ровно то, что доступно в момент инференса.
        """
        n = len(doys)
        if len(self.years) == 0:
            nan = np.full(n, np.nan)
            return nan, nan.copy(), np.zeros(n)
        lo = np.clip(doys - CLIM_HALF, 1, 366) - 1
        hi = np.clip(doys + CLIM_HALF, 1, 366)
        tot_s = self.P_sum[:, hi].sum(axis=0) - self.P_sum[:, lo].sum(axis=0)
        tot_c = self.P_cnt[:, hi].sum(axis=0) - self.P_cnt[:, lo].sum(axis=0)
        tot_q = self.P_sq[:, hi].sum(axis=0) - self.P_sq[:, lo].sum(axis=0)

        yi = np.array([self.year_pos.get(int(y), -1) for y in years])
        has = yi >= 0
        own_s = np.zeros(n)
        own_c = np.zeros(n)
        own_q = np.zeros(n)
        if has.any():
            idx = yi[has]
            own_s[has] = self.P_sum[idx, hi[has]] - self.P_sum[idx, lo[has]]
            own_c[has] = self.P_cnt[idx, hi[has]] - self.P_cnt[idx, lo[has]]
            own_q[has] = self.P_sq[idx, hi[has]] - self.P_sq[idx, lo[has]]

        s, c, q = tot_s - own_s, tot_c - own_c, tot_q - own_q
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        var = np.where(c > 1, (q - c * mean ** 2) / np.maximum(c - 1, 1), np.nan)
        return mean, np.sqrt(np.clip(var, 0.0, None)), c


def _doy_profile(doys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Средний профиль величины по дню года, длиной 367, с заполнением дырок."""
    out = np.full(367, np.nan)
    if len(doys) == 0:
        return out
    agg = pd.Series(values).groupby(np.clip(doys, 1, 366)).mean()
    out[agg.index.to_numpy()] = agg.to_numpy()
    out[1:367] = pd.Series(out[1:367]).interpolate(limit_direction="both").to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Сборка матрицы признаков
# --------------------------------------------------------------------------- #

class FeatureBuilder:
    """Превращает список контрольных точек в таблицу признаков.

    crop_map и погодные группы задаются снаружи и переиспользуются между
    основным и расширенным наборами: категориальный признак в LightGBM — просто
    целое число, и разъехавшаяся кодировка молча испортила бы модель.
    """

    def __init__(self, views: dict[str, PolygonView],
                 weather: WeatherGroups | None = None,
                 crop_map: dict[str, int] | None = None,
                 siblings: "SiblingStats | None" = None):
        self.views = views
        self.weather = weather
        # Суточная поправка E06 считается по тем же views, что и всё остальное:
        # у расширенного набора маскировка своя, значит и поправка обязана быть
        # своей, иначе признак разъедется между обучением и предсказанием.
        self.siblings = siblings if siblings is not None else SiblingStats(views)
        if crop_map is None:
            crops = sorted({v.crop_type for v in views.values() if v.crop_type})
            crop_map = {c: i for i, c in enumerate(crops)}
        self.crop_map = crop_map
        self._cache: dict[str, _PolygonCache] = {}

    def _cache_for(self, polygon_id: str) -> _PolygonCache:
        c = self._cache.get(polygon_id)
        if c is None:
            mate = self.weather.series(polygon_id) if self.weather is not None else None
            c = _PolygonCache(self.views[polygon_id], mate_weather=mate)
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
        # Там, где все методы согласны, спорить не о чем; вся польза надстройки
        # живёт в точках с большим разбросом.
        with np.errstate(invalid="ignore"):
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
        # Асимметрия: при 1 слева и 30 справа верить надо левому соседу, а при
        # 15/15 — сглаживанию. Одним расстоянием это не выражается.
        cols["gap_ratio"] = np.minimum(left_d, right_d) / np.maximum(np.maximum(left_d, right_d), 1.0)
        cols["log_left"] = np.log1p(np.clip(left_d, 0, None))
        cols["log_right"] = np.log1p(np.clip(right_d, 0, None))
        cols["log_span"] = np.log1p(np.clip(span, 0, None))
        cols["month"] = np.array([p.month for p in points], dtype=float)

        # --- 3. Всё, что требует ряда полигона ------------------------------
        blank = lambda: np.full(n, np.nan)  # noqa: E731
        left_val, right_val = blank(), blank()
        left2_val, right2_val = blank(), blank()
        clim_m, clim_s, clim_n = blank(), blank(), np.zeros(n)
        clim_left, clim_right = blank(), blank()
        left_out_dev, right_out_dev = blank(), blank()
        left_out_span, right_out_span = blank(), blank()
        wx_temp, wx_prec = blank(), blank()
        sib_c, sib_n, sib_s = np.zeros(n), np.zeros(n), blank()
        sib_cl, sib_cr = np.zeros(n), np.zeros(n)
        sib_nl, sib_nr = np.zeros(n), np.zeros(n)
        cnt = {w: np.zeros(n) for w in COUNT_WINDOWS}
        wmean = {w: blank() for w in COUNT_WINDOWS}
        wstd = {w: blank() for w in COUNT_WINDOWS}
        temp_w = {w: blank() for w in WEATHER_WINDOWS}
        prec_w = {w: blank() for w in WEATHER_WINDOWS}
        temp_dev = {w: blank() for w in WEATHER_WINDOWS}
        prec_dev = {w: blank() for w in WEATHER_WINDOWS}
        crop_code = np.full(n, -1.0)

        doy_all = _ord_to_doy(ords)
        year_all = _ord_to_year(ords)

        order = np.argsort(pids, kind="stable")
        sorted_pids = pids[order]
        bounds = np.flatnonzero(np.r_[True, sorted_pids[1:] != sorted_pids[:-1], True])
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            idx = order[b0:b1]
            pid = sorted_pids[b0]
            if pid not in self.views:
                continue
            c = self._cache_for(pid)
            t = ords[idx]
            doy = doy_all[idx]

            # --- соседние известные значения и их «выбросность» ---
            l_ord = np.full(len(idx), np.nan)
            r_ord = np.full(len(idx), np.nan)
            if len(c.known_ord):
                pos = np.searchsorted(c.known_ord, t, side="left")
                has_l = pos > 0
                has_r = pos < len(c.known_ord)
                li = pos[has_l] - 1
                ri = pos[has_r]
                left_val[idx[has_l]] = c.known_val[li]
                right_val[idx[has_r]] = c.known_val[ri]
                l_ord[has_l] = c.known_ord[li]
                r_ord[has_r] = c.known_ord[ri]
                left_out_dev[idx[has_l]] = c.out_dev[li]
                left_out_span[idx[has_l]] = c.out_span[li]
                right_out_dev[idx[has_r]] = c.out_dev[ri]
                right_out_span[idx[has_r]] = c.out_span[ri]
                # Вторые соседи: дают наклон ряда по каждую сторону разрыва
                # отдельно, а не только хорду между ближайшими.
                l2 = li - 1
                ok2 = l2 >= 0
                left2_val[idx[has_l][ok2]] = c.known_val[l2[ok2]]
                r2 = ri + 1
                ok3 = r2 < len(c.known_val)
                right2_val[idx[has_r][ok3]] = c.known_val[r2[ok3]]

            # --- климатическая норма, пересчитанная по формуле организаторов ---
            m, s, cn = c.clim_at(doy, year_all[idx])
            clim_m[idx], clim_s[idx], clim_n[idx] = m, s, cn
            # Норму соседей берём на ИХ день года: за месяц норма успевает уехать
            for arr, dst in ((l_ord, clim_left), (r_ord, clim_right)):
                ok = np.isfinite(arr)
                if ok.any():
                    o = arr[ok].astype(np.int64)
                    mm, _, _ = c.clim_at(_ord_to_doy(o), _ord_to_year(o))
                    dst[idx[ok]] = mm

            for w in COUNT_WINDOWS:
                a, bmean, sstd = _window_stats(c.known_ord, c.known_val, t, w)
                cnt[w][idx], wmean[w][idx], wstd[w][idx] = a, bmean, sstd

            for w in WEATHER_WINDOWS:
                _, tm, _ = _window_stats(c.w_ord, c.temp, t, w)
                _, pm, _ = _window_stats(c.w_ord, c.precip, t, w)
                temp_w[w][idx] = tm
                prec_w[w][idx] = pm
                temp_dev[w][idx] = tm - c.temp_norm[doy]
                prec_dev[w][idx] = pm - c.precip_norm[doy]

            if self.weather is not None:
                et, ep = self.weather.lookup(str(pid), t)
                wx_temp[idx], wx_prec[idx] = et, ep

            # --- суточная поправка соседних полей на дату цели и на даты соседей ---
            cc, nn, ss = self.siblings.at(str(pid), t)
            sib_c[idx], sib_n[idx], sib_s[idx] = cc, nn, ss
            for arr, dst_c, dst_n in ((l_ord, sib_cl, sib_nl), (r_ord, sib_cr, sib_nr)):
                ok = np.isfinite(arr)
                if ok.any():
                    c2, n2, _ = self.siblings.at(str(pid), arr[ok].astype(np.int64))
                    dst_c[idx[ok]] = c2
                    dst_n[idx[ok]] = n2

            crop = self.views[pid].crop_type
            crop_code[idx] = self.crop_map.get(crop, -1) if crop else -1

        cols["left_val"] = left_val
        cols["right_val"] = right_val
        cols["neigh_diff"] = right_val - left_val
        cols["neigh_mean"] = 0.5 * (left_val + right_val)
        # Наклон ряда на разрыве отличает стабильное плато от фазы роста
        cols["neigh_slope"] = (right_val - left_val) / np.maximum(span, 1.0)
        cols["left2_val"] = left2_val
        cols["right2_val"] = right2_val
        cols["left_trend"] = left_val - left2_val
        cols["right_trend"] = right2_val - right_val
        # Кривизна: если ряд слева растёт, а справа падает, цель лежит выше хорды
        cols["curvature"] = (right_val - left_val) - ((left_val - left2_val) + (right2_val - right_val))
        cols["left_out_dev"] = left_out_dev
        cols["right_out_dev"] = right_out_dev
        cols["left_out_span"] = left_out_span
        cols["right_out_span"] = right_out_span
        cols["max_out_dev"] = np.fmax(left_out_dev, right_out_dev)

        cols["clim_mean"] = clim_m
        cols["clim_std"] = clim_s
        cols["clim_n"] = clim_n
        cols["left_anom"] = left_val - clim_left
        cols["right_anom"] = right_val - clim_right
        neigh_anom = 0.5 * ((left_val - clim_left) + (right_val - clim_right))
        cols["neigh_anom"] = neigh_anom
        safe_s = np.where(clim_s > 1e-6, clim_s, np.nan)
        # Аномалия, приведённая к разбросу нормы: то же абсолютное отклонение у
        # поля с маленьким clim_std значит гораздо больше. Это же ndvi_zscore.
        cols["neigh_anom_z"] = neigh_anom / safe_s
        # Климатологический якорь: норма дня цели плюс средняя аномалия соседей.
        # Готовое предсказание «в стиле E04», на длинных разрывах оно сильнее
        # любой интерполяции, и деревьям проще взять его целиком, чем собирать
        # из трёх признаков.
        cols["clim_anchor"] = clim_m + neigh_anom
        cols["clim_anchor_gap"] = (clim_m + neigh_anom) - cols["p_mean"]
        # Наклон самой нормы на разрыве: показывает, куда обязана идти кривая
        cols["clim_slope"] = (clim_right - clim_left) / np.maximum(span, 1.0)

        for w in COUNT_WINDOWS:
            cols[f"n_obs_{w}"] = cnt[w]
            cols[f"obs_mean_{w}"] = wmean[w]
            cols[f"obs_std_{w}"] = wstd[w]
        for w in WEATHER_WINDOWS:
            cols[f"temp_{w}"] = temp_w[w]
            cols[f"precip_{w}"] = prec_w[w]
            cols[f"temp_dev_{w}"] = temp_dev[w]
            cols[f"precip_dev_{w}"] = prec_dev[w]
        cols["wx_temp"] = wx_temp
        cols["wx_precip"] = wx_prec
        cols["wx_has"] = np.isfinite(wx_temp).astype(float)

        cols["sib_corr"] = sib_c
        cols["sib_n"] = sib_n
        cols["sib_std"] = sib_s
        # Ниже трёх соседей поправка не применяется вовсе — это принципиально
        # другой режим работы методов E06, и модель должна его различать.
        cols["sib_applied"] = (sib_n >= SiblingStats.MIN_SIBLINGS).astype(float)
        cols["sib_corr_left"] = sib_cl
        cols["sib_corr_right"] = sib_cr
        cols["sib_n_left"] = sib_nl
        cols["sib_n_right"] = sib_nr
        # Соседи слева и справа сняты в разные пролёты. Если их общие помехи
        # разного знака, хорда между ними перекошена — вот величина перекоса.
        cols["sib_corr_diff"] = sib_cr - sib_cl
        cols["sib_corr_gap"] = sib_c - 0.5 * (sib_cl + sib_cr)

        cols["doy"] = doy_all.astype(float)
        # Гармоники дня года вместо голого doy: 31 декабря и 1 января должны быть
        # рядом, а деревьям линейный doy этого не сообщает.
        cols["doy_sin"] = np.sin(2 * np.pi * doy_all / 365.0)
        cols["doy_cos"] = np.cos(2 * np.pi * doy_all / 365.0)
        cols["year"] = year_all.astype(float)
        cols["crop_type"] = crop_code

        return pd.DataFrame(cols)


CATEGORICAL = ["crop_type"]


# --------------------------------------------------------------------------- #
# Расширение обучающей выборки за счёт train_dataset.csv
# --------------------------------------------------------------------------- #

def build_extra_points(df: pd.DataFrame, templates: pd.DataFrame,
                       n_replicates: int = 4, seed0: int = 4242,
                       hide_frac: float = 0.20, n_scored: int = 6000):
    """Строит дополнительные обучающие точки, центрируясь на строках train_dataset.

    Зачем. Оцениваемых контрольных точек всего три тысячи — для бустинга с
    полусотней признаков мало. При этом в train_dataset 30 520 строк с непустым
    primary_ndvi против 17 641 в тесте, и пары «полигон + дата» с тестом не
    пересекаются ни разу. Значит поверх train можно построить второй, полностью
    независимый от метрики контрольный набор той же геометрии.

    Как. holdout.build_holdout ставит центр шаблона только на строку с
    _source == "test". Файл общий, править его нельзя, поэтому источник
    переворачивается в копии таблицы: train становится "test" и наоборот.
    Возвращаемые точки и индексы скрытых строк от этого не зависят, а
    маскирование делается уже по исходной таблице с настоящими источниками —
    иначе сломался бы метод, который читает known_source.

    Каждая реплика — независимый розыгрыш со своим зерном на одной и той же
    таблице. Это честнее, чем задирать hide_frac: при большой доле скрытых
    значений ряд разреживается, и обучающие примеры перестают быть похожи на
    боевые.

    Возвращает список троек (points, views, masked_df) по репликам.
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
    поэтому дополнительный набор без них бесполезен. Импорт реестра внутри
    функции: features.py не должен тянуть за собой модули методов при обычном
    использовании.
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
