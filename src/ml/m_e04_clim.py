"""E04. Климатологический якорь: восстанавливаем отклонение от нормы, а не сам NDVI.

Гипотеза. На длинном разрыве любая интерполяция тянет прямую через сезонный
подъём или спад: за 189 дней культура успевает пройти весь цикл, и прямая между
октябрём и апрелем не имеет с реальностью ничего общего. Если вместо самого NDVI
интерполировать отклонение от климатической нормы, форма сезонной кривой придёт
из нормы, а на интерполяцию останется медленно меняющийся остаток — аномалия
конкретного сезона.

Схема одна и та же во всех вариантах:
    r(t) = y(t) − norm(t)        по видимым наблюдениям
    r̂    = сглаживание Уиттекера остатка на посуточной сетке
    ŷ(t) = r̂(t) + norm(t)        в точке цели

Варианты отличаются только тем, откуда берётся norm:
    own    — собственная история полигона, пересчитанная формулой организаторов
             (окно ±8 дней по дню года, leave-one-out по годам)
    file   — готовая колонка ndvi_climatology_mean, снятая с видимых строк
    crop   — норма по типу культуры (src/core/crop_climatology.py)
    hybrid — своя норма где есть, норма культуры где своей нет

Про утечку. Норма строится строго по видимым наблюдениям: у скрытых строк
primary_ndvi уже стёрт протоколом, а колонка ndvi_climatology_mean у них
замаскирована, поэтому в норму не попадает ни одно спрятанное значение.
Заимствование чужих полигонов (норма культуры) идёт по тем же замаскированным
данным, так что и там подглядеть нечего.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.crop_climatology import CropClimatology, DOY_WINDOW
from src.core.restore import clip_outliers, predict_at, restore_on_grid, whittaker_smooth
from src.ml.dataset import PolygonView
from src.ml.m_e06_sibling import cleaned_series, correction_at
from src.ml.registry import BaseMethod, register

FALLBACK_NDVI = 0.31
# Минимум наблюдений в окне ±8 дней после исключения текущего года, иначе норма не определена
MIN_OBS_IN_WINDOW = 3
# Минимум сезонов, при котором leave-one-out вообще имеет смысл
MIN_YEARS_FOR_OWN = 2

# Кэши на время одного прогона валидации. Норма полигона считается один раз и
# переиспользуется всеми вариантами метода: без этого 366 окон × 78 полигонов
# пересчитывались бы по десять раз и прогон вырос бы в минуты.
_OWN_NORM_CACHE: dict[tuple, tuple] = {}
_FILE_NORM_CACHE: dict[tuple, np.ndarray] = {}
_CROP_CLIM_CACHE: dict[int, CropClimatology] = {}
_CLEAN_CACHE: dict[tuple, tuple] = {}


def _cleaned(views: dict, lam: float):
    """Очищенные от общей суточной помехи ряды (E06), посчитанные один раз на прогон."""
    key = (id(views), lam)
    cached = _CLEAN_CACHE.get(key)
    if cached is None:
        cached = cleaned_series(views, lam=lam)
        _CLEAN_CACHE[key] = cached
    return cached


# --------------------------------------------------------------------- нормы

def _own_loo_norm(view: PolygonView, window: int = DOY_WINDOW):
    """Норма полигона по его же истории: leave-one-out по годам, окно ±window дней.

    Возвращает (таблица year→строка, индекс годов, матрица 366×n_years).
    Формула ровно та, которой организаторы считали ndvi_climatology_mean
    (восстановлена по обучающему набору с MAE 0,0016), поэтому пересчитанная
    норма совпадает с эталонной там, где эталон виден, и продолжает её туда,
    где он замаскирован — то есть в сами контрольные точки.

    Исключение текущего года принципиально: иначе аномальный сезон входит в свою
    же норму, остаток занижается по модулю и якорь частично сам себя съедает.
    """
    key = (id(view.frame), view.polygon_id, window)
    cached = _OWN_NORM_CACHE.get(key)
    if cached is not None:
        return cached

    f = view.frame
    seen = f["primary_ndvi"].notna().to_numpy()
    doy = f["_doy"].to_numpy(dtype=np.int64)[seen]
    year = f["_year"].to_numpy(dtype=np.int64)[seen]
    val = clip_outliers(f["primary_ndvi"].to_numpy(dtype=float)[seen])
    ok = np.isfinite(val)
    doy, year, val = doy[ok], year[ok], val[ok]

    years = np.unique(year)
    if len(years) < MIN_YEARS_FOR_OWN or len(val) == 0:
        result = (np.full((366, max(len(years), 1)), np.nan), years)
        _OWN_NORM_CACHE[key] = result
        return result

    yidx = np.searchsorted(years, year)
    n_y = len(years)
    norm = np.full((366, n_y), np.nan)

    # Считаем суммы и счётчики в окне по дню года сразу по всем годам, а затем
    # вычитаем вклад текущего года — это и есть leave-one-out за один проход.
    for i in range(366):
        d = i + 1
        diff = np.abs(doy - d)
        diff = np.minimum(diff, 366 - diff)
        m = diff <= window
        if not m.any():
            continue
        sums = np.bincount(yidx[m], weights=val[m], minlength=n_y)
        cnts = np.bincount(yidx[m], minlength=n_y).astype(float)
        tot_s, tot_c = sums.sum(), cnts.sum()
        rest_s, rest_c = tot_s - sums, tot_c - cnts
        good = rest_c >= MIN_OBS_IN_WINDOW
        norm[i, good] = rest_s[good] / rest_c[good]

    result = (norm, years)
    _OWN_NORM_CACHE[key] = result
    return result


def _own_norm_at(view: PolygonView, doy: np.ndarray, year: np.ndarray) -> np.ndarray:
    """Значение собственной нормы на парах (день года, год)."""
    norm, years = _own_loo_norm(view)
    if norm.size == 0 or not np.isfinite(norm).any():
        return np.full(len(doy), np.nan)
    out = np.full(len(doy), np.nan)
    pos = np.searchsorted(years, year)
    inside = (pos < len(years)) & (years[np.clip(pos, 0, len(years) - 1)] == year)
    if inside.any():
        out[inside] = norm[np.clip(doy[inside], 1, 366) - 1, pos[inside]]
    # Год вне истории полигона (в наборе не встречается, но метод обязан быть
    # определён): берём среднее по всем годам — leave-one-out вырождается в норму.
    if (~inside).any():
        flat = np.nanmean(norm, axis=1)
        out[~inside] = flat[np.clip(doy[~inside], 1, 366) - 1]
    return out


def _file_norm_curve(view: PolygonView) -> np.ndarray:
    """Норма из готовой колонки ndvi_climatology_mean, свёрнутая по дню года.

    У контрольных точек колонка замаскирована, поэтому норму на дату цели взять
    напрямую нельзя — она собирается с видимых строк того же дня года (других
    лет). Побочный эффект: leave-one-out размывается, норма получается общей на
    все годы. Именно этот путь и сравнивается с пересчётом по формуле.
    """
    key = (id(view.frame), view.polygon_id)
    cached = _FILE_NORM_CACHE.get(key)
    if cached is not None:
        return cached

    f = view.frame
    col = "ndvi_climatology_mean"
    curve = np.full(366, np.nan)
    if col in f.columns and f[col].notna().any():
        sub = f.loc[f[col].notna(), ["_doy", col]]
        agg = sub.groupby("_doy")[col].mean()
        idx = np.clip(agg.index.to_numpy(dtype=int), 1, 366) - 1
        curve[idx] = agg.to_numpy(dtype=float)
        # Дни года без видимой строки закрываем интерполяцией по сезону:
        # дыры в норме превратились бы в дыры в предсказании.
        known = np.isfinite(curve)
        if known.sum() >= 2:
            grid = np.arange(366)
            curve = np.interp(grid, grid[known], curve[known])
            edge = (grid < grid[known].min()) | (grid > grid[known].max())
            curve[edge] = np.nan
    _FILE_NORM_CACHE[key] = curve
    return curve


def _crop_climatology(context: dict) -> CropClimatology:
    """Норма по культуре, построенная по замаскированной таблице прогона."""
    df = context.get("df")
    key = id(df)
    cached = _CROP_CLIM_CACHE.get(key)
    if cached is None:
        cached = CropClimatology().fit(df)
        _CROP_CLIM_CACHE[key] = cached
    return cached


# ---------------------------------------------------------- восстановление

def _restore_residual(ords: np.ndarray, res: np.ndarray, lam: float, mix: float, targets: np.ndarray):
    """Сглаживание остатка на посуточной сетке и снятие значений в точках цели.

    Отдельная реализация вместо core.restore.restore_on_grid нужна из-за одной
    детали: restore_on_grid перед сглаживанием выбрасывает всё за пределами
    [−0.2, 1.0] как брак съёмки. Для остатка это неверно — он живёт вокруг нуля,
    и отрицательные значения у него совершенно законны.
    """
    grid = np.arange(min(ords.min(), targets.min()), max(ords.max(), targets.max()) + 1, dtype=np.int64)
    values = np.full(grid.shape, np.nan)
    values[ords - grid[0]] = res
    known = ~np.isnan(values)
    if known.sum() < 4:
        out = np.interp(targets, ords, res)
        return out
    linear = np.interp(grid, grid[known], values[known])
    smooth = whittaker_smooth(np.nan_to_num(values), known.astype(float), lam=lam)
    restored = mix * smooth + (1.0 - mix) * linear
    return restored[np.clip(targets - grid[0], 0, len(grid) - 1)]


def _plain_whittaker(ords: np.ndarray, vals: np.ndarray, targets: np.ndarray,
                     lam: float, mix: float) -> np.ndarray:
    """Запасной путь без якоря — ровно тот же код, что у whit1000 и sibit10.

    Здесь важно не «примерно то же сглаживание», а буквально то же: там, где
    якорь выключен порогом или у полигона нет нормы, вариант обязан совпадать
    с базовым методом до последнего знака. Иначе разница в общей таблице будет
    смесью эффекта якоря и случайного расхождения двух реализаций.
    """
    if len(ords) < 2:
        return np.full(len(targets), FALLBACK_NDVI)
    grid, restored = restore_on_grid(ords, vals, lam=lam, mix=mix)
    return predict_at(grid, restored, targets)


def _gap_span(view: PolygonView, targets: np.ndarray) -> np.ndarray:
    """Длина разрыва вокруг каждой цели: расстояние до соседа слева плюс справа.

    Односторонний случай удваивается — та же условность, что в протоколе
    валидации, чтобы порог включения якоря читался в тех же единицах, что и
    отчётный разрез по длине разрыва.
    """
    ko = view.known_ord
    if len(ko) == 0:
        return np.full(len(targets), 10_000)
    pos = np.searchsorted(ko, targets, side="left")
    left = np.where(pos > 0, targets - ko[np.clip(pos - 1, 0, len(ko) - 1)], -1)
    right = np.where(pos < len(ko), ko[np.clip(pos, 0, len(ko) - 1)] - targets, -1)
    return np.where(left < 0, right * 2, np.where(right < 0, left * 2, left + right))


# ---------------------------------------------------------------- каркас якоря

class _ClimAnchor(BaseMethod):
    """Общий каркас якоря. Наследники задают только источник нормы и пороги.

    source    — own | file | crop | hybrid | crop_nohist
    lam, mix  — сглаживание ОСТАТКА. Оптимум у него свой: остаток — медленная
                аномалия сезона поверх шума наблюдения, его надо сглаживать
                сильнее, чем сам NDVI, у которого есть настоящая быстрая динамика
    base_lam,
    base_mix  — сглаживание запасного пути. По умолчанию λ=1000 — текущий лучший
                метод, чтобы выключенный якорь не приносил своей собственной потери
    gap_min   — минимальная длина разрыва, начиная с которой якорь включается;
                на коротком разрыве соседи и так рядом, а ошибка нормы добавляется
                к ошибке остатка, поэтому порог — отдельная настройка, а не догма
    shrink    — если задано, остаток на длинном разрыве стягивается к нулю с
                характерным масштабом shrink дней: чем дальше сосед, тем меньше
                доверия аномалии сезона и тем ближе ответ к чистой норме
    weight    — доля якоря в ответе на тех точках, где он включён. Единица —
                чистый якорь, ноль — база. Оптимум лежит около половины: якорь
                несёт сигнал, которого в базе нет, но собственная ошибка нормы
                у него больше, и брать его целиком — переплата
    sibling   — строиться поверх E06: сначала снять с ряда общую суточную помеху
                (её видно по соседним полям), и только потом вычитать норму.
                Порядок важен: якорь работает с аномалией сезона, а суточная
                помеха к сезону отношения не имеет и в остатке только мешает
    """

    def __init__(self, source: str, lam: float = 1000.0, mix: float = 1.0,
                 gap_min: int = 0, shrink: float | None = None,
                 base_lam: float = 1000.0, base_mix: float = 1.0,
                 weight: float = 1.0, sibling: bool = False):
        self.source = source
        self.lam = lam
        self.mix = mix
        self.base_lam = base_lam
        self.base_mix = base_mix
        self.gap_min = gap_min
        self.shrink = shrink
        self.weight = weight
        self.sibling = sibling
        self._crop: CropClimatology | None = None
        self._clean: dict = {}
        self._days = None
        self._corr: dict = {}

    # Норма культуры и суточная поправка нужны на весь набор сразу, поэтому
    # строятся здесь, а не внутри predict, куда контекст не доходит.
    def predict_points(self, points, views, context):
        if self.source in ("crop", "hybrid", "crop_nohist"):
            self._crop = _crop_climatology(context)
        if self.sibling:
            self._clean, self._days, self._corr = _cleaned(views, self.base_lam)
        return super().predict_points(points, views, context)

    def _norm_for(self, view: PolygonView, doy: np.ndarray, year: np.ndarray) -> np.ndarray:
        """Норма выбранного источника на парах (день года, год)."""
        if self.source == "own":
            return _own_norm_at(view, doy, year)
        if self.source == "file":
            return _file_norm_curve(view)[np.clip(doy, 1, 366) - 1]
        if self.source == "crop":
            return self._crop.norm(view.crop_type, doy)[0] if self._crop else np.full(len(doy), np.nan)
        if self.source == "crop_nohist":
            # Норма культуры только там, где своей истории нет вовсе
            _, years = _own_loo_norm(view)
            if len(years) >= MIN_YEARS_FOR_OWN:
                return np.full(len(doy), np.nan)
            return self._crop.norm(view.crop_type, doy)[0] if self._crop else np.full(len(doy), np.nan)
        if self.source == "hybrid":
            own = _own_norm_at(view, doy, year)
            miss = ~np.isfinite(own)
            if miss.any() and self._crop is not None:
                own[miss] = self._crop.norm(view.crop_type, doy[miss])[0]
            return own
        raise ValueError(f"неизвестный источник нормы: {self.source!r}")

    def _doy_year(self, view: PolygonView, ords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """День года и год для произвольных дат ряда, снятые с самой таблицы.

        Колонки _doy/_year посчитаны из даты до маскирования, поэтому они видны
        и у контрольных строк — это календарь, а не признак наблюдения.
        """
        f = view.frame
        lut = pd.Series(
            np.arange(len(f)), index=f["_ord"].to_numpy(dtype=np.int64)
        )
        pos = lut.reindex(ords).to_numpy()
        doy = np.empty(len(ords), dtype=np.int64)
        year = np.empty(len(ords), dtype=np.int64)
        found = np.isfinite(pos)
        p = pos[found].astype(int)
        doy[found] = f["_doy"].to_numpy()[p]
        year[found] = f["_year"].to_numpy()[p]
        if (~found).any():
            # Даты вне таблицы полигона в наборе не встречаются, но метод должен
            # оставаться определённым: считаем календарь напрямую.
            ts = pd.to_datetime([pd.Timestamp.fromordinal(int(o)) for o in ords[~found]])
            doy[~found] = ts.dayofyear
            year[~found] = ts.year
        return doy, year

    def predict(self, view: PolygonView, target_ords: np.ndarray) -> np.ndarray:
        targets = np.asarray(target_ords, dtype=np.int64)

        # Ряд, с которого начинаем: сырой либо очищенный от суточной помехи (E06).
        # Поправку, снятую с ряда, надо вернуть на дате цели — иначе прогноз
        # окажется в шкале «идеальной съёмки», а сравнивают его с реальной.
        src_ord, src_val = self._clean.get(view.polygon_id, (view.known_ord, view.known_values))
        corr_t = (correction_at(self._days, self._corr, view.polygon_id, targets)
                  if self.sibling else np.zeros(len(targets)))

        fallback = np.clip(
            _plain_whittaker(src_ord, src_val, targets, self.base_lam, self.base_mix) + corr_t,
            0.0, 1.0,
        )

        y = clip_outliers(src_val)
        ok = np.isfinite(y)
        if ok.sum() < 4:
            return fallback
        ords, vals = src_ord[ok], y[ok]

        k_doy, k_year = self._doy_year(view, ords)
        t_doy, t_year = self._doy_year(view, targets)
        k_norm = self._norm_for(view, k_doy, k_year)
        t_norm = self._norm_for(view, t_doy, t_year)

        # Наблюдения без нормы из остатка выбывают: подставлять им ноль значило бы
        # объявить их «ровно в норме» и притянуть к себе всё сглаживание.
        good = np.isfinite(k_norm)
        if good.sum() < 4 or not np.isfinite(t_norm).any():
            return fallback

        res = vals[good] - k_norm[good]
        r_hat = _restore_residual(ords[good], res, self.lam, self.mix, targets)

        if self.shrink:
            # Чем длиннее разрыв, тем меньше веса аномалии сезона и тем больше — норме
            span = _gap_span(view, targets).astype(float)
            r_hat = r_hat * np.exp(-span / float(self.shrink))

        anchored = np.clip(r_hat + t_norm + corr_t, 0.0, 1.0)
        # Выпуклая смесь базы и якоря: оба уже лежат в [0, 1], значит и смесь тоже
        mixed = (1.0 - self.weight) * fallback + self.weight * anchored

        use = np.isfinite(t_norm)
        if self.gap_min > 0:
            use &= _gap_span(view, targets) >= self.gap_min
        return np.where(use, mixed, fallback)


# --------------------------------------------------------------------- методы
#
# Регистрируются только те варианты, которые что-то показывают в отчёте.
# Все они строятся поверх E06 (sibling=True): мерить якорь от старой базы
# whit1000 значит считать один и тот же выигрыш дважды.


@register("e04_own_naive", "Якорь: своя норма, наивное сглаживание остатка λ=1000",
          experiment="E04")
class AnchorOwnNaive(_ClimAnchor):
    """Прямая реализация гипотезы: вычесть норму, восстановить остаток тем же λ.

    Тот самый вариант, который «должен был» работать. Оставлен в реестре
    намеренно: он показывает, что при λ, подобранном для сырого NDVI, якорь
    не нейтрален, а вреден — остаток требует своего, гораздо более сильного
    сглаживания.
    """

    def __init__(self):
        super().__init__(source="own", lam=1000.0, sibling=True)


@register("e04_own_smooth", "Якорь: своя норма, сглаживание остатка λ=100000", experiment="E04")
class AnchorOwnSmooth(_ClimAnchor):
    """Остаток — медленная аномалия сезона, ему нужно λ на два порядка больше."""

    def __init__(self):
        super().__init__(source="own", lam=100000.0, sibling=True)


@register("e04_own_g45", "Якорь: своя норма, только разрывы от 45 дней", experiment="E04")
class AnchorOwnGap45(_ClimAnchor):
    """Порог включения. На коротком разрыве соседи и так рядом, а ошибка нормы
    добавляется к ошибке остатка — там якорь может только навредить."""

    def __init__(self):
        super().__init__(source="own", lam=100000.0, gap_min=45, sibling=True)


@register("e04_file_g45", "Якорь: норма из колонки, только разрывы от 45 дней", experiment="E04")
class AnchorFileGap45(_ClimAnchor):
    """Второй путь к норме: готовая ndvi_climatology_mean, снятая с видимых строк."""

    def __init__(self):
        super().__init__(source="file", lam=100000.0, gap_min=45, sibling=True)


@register("e04_crop_g45", "Якорь: норма культуры, только разрывы от 45 дней", experiment="E04")
class AnchorCropGap45(_ClimAnchor):
    def __init__(self):
        super().__init__(source="crop", lam=100000.0, gap_min=45, sibling=True)


@register("e04_hybrid_g45", "Якорь: своя норма где есть, культурная где нет, от 45 дней",
          experiment="E04")
class AnchorHybridGap45(_ClimAnchor):
    def __init__(self):
        super().__init__(source="hybrid", lam=30000.0, gap_min=45, sibling=True)


@register("e04_crop_nohist", "Якорь: норма культуры только у полей без своей истории",
          experiment="E04")
class AnchorCropNoHistory(_ClimAnchor):
    """Ровно тот случай, ради которого норма по культуре и делалась: 20 полей
    из 78 не имеют собственной истории даже с учётом обучающего набора."""

    def __init__(self):
        super().__init__(source="crop_nohist", lam=100000.0, gap_min=45, sibling=True)


@register("e04_own_raw", "Якорь: своя норма поверх сырого ряда, без поправки E06",
          experiment="E04")
class AnchorOwnRaw(_ClimAnchor):
    """Замер якоря от старой базы whit1000 — для сопоставления с первой версией
    эксперимента, когда суточной поправки E06 ещё не существовало."""

    def __init__(self):
        super().__init__(source="own", lam=100000.0, gap_min=45, sibling=False)


@register("e04_own_g45_w50", "Якорь: своя норма, от 45 дней, доля 0,5", experiment="E04")
class AnchorOwnGap45Half(_ClimAnchor):
    def __init__(self):
        super().__init__(source="own", lam=100000.0, gap_min=45, weight=0.5, sibling=True)


@register("e04_anchor", "Якорь: норма из колонки, от 45 дней, доля 0,5", experiment="E04",
          tags=("accepted",))
class AnchorAccepted(_ClimAnchor):
    """Принятая конфигурация E04.

    Полная замена базы якорем не воспроизводится от зерна к зерну: якорь несёт
    свой сигнал, но и свою ошибку нормы, и на некоторых наборах вторая
    перевешивает первую. Половинная доля забирает первое и вдвое ослабляет
    второе — выигрыш на длинных разрывах держится на всех трёх зёрнах протокола.
    """

    def __init__(self):
        super().__init__(source="file", lam=100000.0, gap_min=45, weight=0.5, sibling=True)
