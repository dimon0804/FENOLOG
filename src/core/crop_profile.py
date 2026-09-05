"""Фенологический портрет поля и определение культуры по кривой NDVI.

Зачем это нужно. Культура задаёт всё: когда поле зеленеет, когда убирается,
какой уровень индекса для него нормальный и что считать провалом. До сих пор
сервис знал культуру только если её вписал пользователь или она нашлась в тегах
OSM, а это редкость. Поле без культуры получает усреднённую норму, характерное
окно уборки «вообще по региону» и продуктивность, которую не с чем сравнить.

Здесь культура определяется по самой кривой. Это не догадка: разные культуры
по-разному зеленеют и по-разному сходят, и различие измеримо (журнал экспериментов, E19).

Что важно понимать про точность. В этом наборе разница между культурами выражена
слабее, чем в учебнике: пик у всех приходится на конец мая, и различаются кривые
в основном уровнем и скоростью схода после пика. Поэтому:
    крупное деление (озимые против яровых и пастбищ) определяется надёжно;
    конкретная культура внутри группы — заметно хуже;
    подсолнечник по кривой не отделяется от озимой пшеницы вовсе.
Все три утверждения измерены leave-one-polygon-out, числа в журнале экспериментов, E19.
Отсюда правило модуля: он всегда возвращает уверенность и никогда не выдаёт
догадку за факт. Ниже порога уверенности честный ответ — «не определена».

## Почему здесь два разных механизма, а не один

Это выглядит избыточным, пока не посмотреть на числа. Задач на самом деле две, и
лучший инструмент у них разный — измерено, а не предположено (E19):

| Задача | Сравнение с эталоном | Градиентный бустинг |
| --- | --- | --- |
| Назвать культуру, точность по полям | 0,500 | **0,744** |
| Выбрать норму, RMSE против кривой поля | **0,0699** | 0,0741 |

Разгадка простая. Сравнение с эталоном выбирает культуру, чья средняя кривая
ближе всего к кривой поля, — а это ровно то, что нужно от нормы, и потому в
задаче нормы оно даже обходит точно известную культуру (0,0699 против 0,0704:
поле пшеницы с низким уровнем лучше описывается нормой яровых, чем нормой своей
собственной культуры). Но название культуры из этого не следует: похожесть
кривых и совпадение агрономических названий — разные вещи.

Поэтому `CropClassifier` (бустинг) отвечает на вопрос «что посеяно», а
`CropDetector` (сравнение с эталоном) — на вопрос «какой нормой мерить».
Бустинг необязателен: без обученного файла модуль работает на одном сравнении с
эталоном, честно понижая уверенность в названии.
"""
from __future__ import annotations

from datetime import date

import numpy as np

# Сетка сезона: с 1 апреля по 1 ноября с шагом 5 дней. Шаг выбран по плотности
# съёмки — снимок раз в 3-5 дней у Sentinel-2, чаще сетку дробить нечем.
SEASON_GRID = np.arange(91, 306, 5)
# Полуширина окна усреднения вокруг узла сетки
CURVE_WINDOW_DAYS = 10
# Максимальная доля пустых узлов, при которой кривая ещё считается пригодной
MAX_EMPTY_SHARE = 0.25
# Сезон обязан быть покрыт с обеих сторон: без весны не видно, зимовало ли поле,
# без осени не видно, когда оно сошло. Без этого культуру определять нечестно.
REQUIRE_BEFORE_DOY = 140
REQUIRE_AFTER_DOY = 250
# Минимум наблюдений в сезоне
MIN_SEASON_OBS = 10

# Порог уверенности, ниже которого культура считается неопределённой.
# Выбран по кривой «покрытие против точности» (журнал экспериментов, E19).
CONFIDENCE_MIN = 0.60

# Перевод сырой уверенности бустинга в честную. Модель обучена с выравниванием
# весов классов и не калибрована: на кросс-валидации она заявляет 0,95 там, где
# оказывается права в 84 случаях из ста, а на поле из обучающего набора выдаёт
# и вовсе 0,98. Показывать такое число пользователю — врать ему.
#
# Пары (сырая уверенность, измеренная доля верных ответов при этом пороге)
# взяты из прогона `python -m src.ml.crop_id`, таблица в журнале экспериментов, E19.
# Ниже 0,4 модель почти не опускается, поэтому нижняя ступень — общая точность.
RELIABILITY = ((0.80, 0.84), (0.70, 0.80), (0.60, 0.77), (0.50, 0.76), (0.00, 0.74))


def calibrate(raw: float) -> float:
    """Честная уверенность вместо сырой: доля верных ответов при таком пороге."""
    for threshold, measured in RELIABILITY:
        if raw >= threshold:
            return measured
    return RELIABILITY[-1][1]


# Крупные фенологические группы. Внутри группы кривые похожи, между группами
# различие устойчивое, поэтому спор с пользователем ведётся только на этом
# уровне: сказать «вы указали пшеницу, а это подсолнечник» данные не позволяют,
# а «вы указали озимую пшеницу, а поле весной голое» — позволяют.
CROP_GROUP = {
    "озимая пшеница": "озимые",
    "подсолнечник": "озимые",
    "зерновые": "яровые и пастбища",
    "пастбища/зерновые": "яровые и пастбища",
}
GROUP_TITLE = {
    "озимые": "озимые и другие культуры с ранним стартом",
    "яровые и пастбища": "яровые зерновые и пастбища",
}


def season_curve(dates, values, year: int | None = None) -> np.ndarray | None:
    """Кривая одного сезона на регулярной сетке или None, если сезон негоден.

    Усреднение в окне ±10 дней, а не интерполяция по ближайшим точкам: NDVI
    зашумлён на 0,07, и одиночное наблюдение в узле сетки описывает не поле, а
    погоду в день съёмки. Пустые узлы внутри покрытого диапазона достраиваются
    линейно — их мало по построению, иначе сезон отбраковывается целиком.
    """
    doy: list[float] = []
    val: list[float] = []
    for d, v in zip(dates, values):
        if v is None or not np.isfinite(v):
            continue
        dd = d if isinstance(d, date) else date.fromisoformat(str(d)[:10])
        if year is not None and dd.year != year:
            continue
        doy.append(dd.timetuple().tm_yday)
        val.append(float(v))
    if len(doy) < MIN_SEASON_OBS:
        return None
    doy_a = np.asarray(doy, dtype=float)
    # Брак за границами физического диапазона придавливается, а не выбрасывается:
    # выброс -0,3 это всё-таки «очень мало зелени», а не отсутствие наблюдения.
    val_a = np.clip(np.asarray(val, dtype=float), -0.2, 1.0)
    if doy_a.min() > REQUIRE_BEFORE_DOY or doy_a.max() < REQUIRE_AFTER_DOY:
        return None

    curve = np.full(SEASON_GRID.size, np.nan)
    for i, node in enumerate(SEASON_GRID):
        m = np.abs(doy_a - node) <= CURVE_WINDOW_DAYS
        if m.any():
            curve[i] = val_a[m].mean()
    empty = ~np.isfinite(curve)
    if empty.mean() > MAX_EMPTY_SHARE:
        return None
    if empty.any():
        idx = np.arange(curve.size)
        curve[empty] = np.interp(idx[empty], idx[~empty], curve[~empty])
    return curve


def _seg(curve: np.ndarray, lo: int, hi: int) -> float:
    """Средний уровень кривой на отрезке дней года."""
    m = (SEASON_GRID >= lo) & (SEASON_GRID <= hi)
    return float(curve[m].mean()) if m.any() else float("nan")


def phenology(curve: np.ndarray) -> dict:
    """Фенологический портрет сезона: когда поле зазеленело, вышло в пик и сошло.

    Отдаётся наружу целиком, а не только внутрь классификатора: даты выхода в пик
    и схода — это то, что агроном читает без перевода, и то, что интерфейс может
    показать рядом с графиком.
    """
    i_peak = int(np.argmax(curve))
    peak, peak_doy = float(curve[i_peak]), float(SEASON_GRID[i_peak])
    after = curve[i_peak:]
    i_min = int(np.argmin(after))
    trough, trough_doy = float(after[i_min]), float(SEASON_GRID[i_peak:][i_min])
    spring = _seg(curve, 91, 110)
    tail = curve[SEASON_GRID >= 240]

    # Сход посева: день, когда кривая опустилась ниже середины между пиком и
    # последующим минимумом. Для убираемых культур это уборка, для пастбищ —
    # летнее выгорание, поэтому название нейтральное, а не «дата уборки».
    half = (peak + trough) / 2.0
    below = after <= half
    decline_doy = (
        float(SEASON_GRID[i_peak:][int(np.argmax(below))]) if below.any()
        else float(SEASON_GRID[-1])
    )

    return {
        "apr": spring, "may": _seg(curve, 121, 150), "jun": _seg(curve, 151, 180),
        "jul": _seg(curve, 181, 210), "aug": _seg(curve, 211, 240),
        "sep": _seg(curve, 241, 270), "oct": _seg(curve, 271, 300),
        "peak": peak, "peak_doy": peak_doy,
        "amplitude": peak - trough, "trough": trough, "trough_doy": trough_doy,
        "integral": float(curve.mean()),
        "decline_doy": decline_doy,
        "drop_rate": (peak - trough) / max(trough_doy - peak_doy, 1.0),
        "rise_rate": (peak - spring) / max(peak_doy - 100.0, 1.0),
        "days_above_05": float((curve > 0.5).sum()) * 5.0,
        "days_above_06": float((curve > 0.6).sum()) * 5.0,
        "aug_ratio": _seg(curve, 211, 240) / max(peak, 1e-6),
        "oct_ratio": _seg(curve, 271, 300) / max(peak, 1e-6),
        "spring_ratio": spring / max(peak, 1e-6),
        "regrowth": float(tail.max() - tail.min()) if tail.size else 0.0,
    }


FEATURES = tuple(phenology(np.linspace(0.2, 0.7, SEASON_GRID.size)).keys())


class CropDetector:
    """Определение культуры сравнением сезонной кривой с эталонами культур.

    Главная работа этого класса — не назвать культуру, а выбрать, какой нормой
    мерить поле (см. таблицу в шапке модуля). Название он тоже возвращает, но
    как запасной вариант: если обученного классификатора рядом нет, лучше
    осторожная догадка с честной уверенностью, чем молчание.

    Эталон культуры — её средняя кривая по дню года, та же самая, что уже
    используется как запасная климатическая норма (models/crop_climatology.json).
    Отдельного файла модели у детектора нет по замыслу: его пришлось бы держать
    в согласии с нормой, а считается сходство двумя строками numpy.

    Сходство считается по двум независимым осям:
        форма  — корреляция кривых после вычитания собственного уровня;
        уровень — расхождение средних значений за сезон.
    Обе нужны. По одной только форме подсолнечник неотличим от пшеницы, по
    одному только уровню засушливый год пшеницы попадает в яровые.
    """

    # Вес уровня против формы в общем сходстве. Подобран по leave-one-polygon-out.
    LEVEL_WEIGHT = 0.5
    # Масштаб перевода расхождения уровня в сходство: 0,10 NDVI — это разница
    # между средним полем пшеницы и средним полем яровых.
    LEVEL_SCALE = 0.10
    # Резкость перевода сходства в вероятность. Сходства лежат близко друг к
    # другу (0,70 против 0,75), и без обострения ответ выглядел бы неуверенным
    # всегда. Значение подобрано по совпадению заявленной уверенности с
    # фактической долей верных ответов.
    SHARPNESS = 12.0

    def __init__(self, prototypes: dict[str, np.ndarray] | None = None):
        self._proto: dict[str, np.ndarray] = dict(prototypes or {})

    # ---------------------------------------------------------------- обучение

    @classmethod
    def from_climatology(cls, clim) -> "CropDetector":
        """Собирает эталоны из готовой нормы по культурам (CropClimatology)."""
        proto = {}
        for crop in clim.crops:
            mean, _ = clim.norm(crop, SEASON_GRID)
            mean = np.asarray(mean, dtype=float)
            if np.isfinite(mean).mean() > 0.9:
                proto[crop] = mean
        return cls(proto)

    @classmethod
    def fit(cls, df) -> "CropDetector":
        """Строит эталоны прямо из таблицы наблюдений.

        Нужен там, где нормы по культурам ещё нет: в валидации, где своё поле
        обязано быть исключено из эталонов, и при первом запуске без models/.
        """
        from src.core.crop_climatology import CropClimatology

        return cls.from_climatology(CropClimatology().fit(df))

    @property
    def crops(self) -> list[str]:
        return sorted(self._proto)

    # ------------------------------------------------------------- применение

    def scores(self, curve: np.ndarray) -> dict[str, float]:
        """Сходство кривой с каждой культурой, от 0 до 1."""
        out: dict[str, float] = {}
        c = np.asarray(curve, dtype=float)
        for crop, proto in self._proto.items():
            m = np.isfinite(proto) & np.isfinite(c)
            if int(m.sum()) < 10:
                continue
            a, b = c[m], proto[m]
            a0, b0 = a - a.mean(), b - b.mean()
            denom = float(np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum()))
            shape = float((a0 * b0).sum() / denom) if denom > 0 else 0.0
            level = float(np.exp(-abs(a.mean() - b.mean()) / self.LEVEL_SCALE))
            # Форма приводится из [-1; 1] в [0; 1]: отрицательная корреляция и
            # нулевая одинаково означают «не похоже», различать их незачем.
            out[crop] = (1 - self.LEVEL_WEIGHT) * max(shape, 0.0) + self.LEVEL_WEIGHT * level
        return out

    def _wrap(self, prob: dict[str, float], **extra) -> dict:
        """Общий вид ответа: культура, уверенность, группа, полный расклад."""
        top = max(prob, key=prob.get)
        conf = float(prob[top])
        by_group: dict[str, float] = {}
        for crop, p in prob.items():
            g = CROP_GROUP.get(crop, crop)
            by_group[g] = by_group.get(g, 0.0) + float(p)
        group = max(by_group, key=by_group.get)
        out = {
            # None означает «уверенности не хватает». Это штатный ответ, а не
            # ошибка: лучше промолчать, чем назвать культуру наугад.
            "crop": top if conf >= CONFIDENCE_MIN else None,
            "best_guess": top,
            "confidence": round(conf, 3),
            "group": group,
            "group_confidence": round(float(by_group[group]), 3),
            "scores": {c: round(float(p), 3) for c, p in sorted(prob.items())},
        }
        out.update(extra)
        return out

    def predict(self, curve: np.ndarray) -> dict:
        """Культура по кривой одного сезона."""
        sc = self.scores(curve)
        if not sc:
            return {"crop": None, "best_guess": None, "confidence": 0.0,
                    "group": None, "group_confidence": 0.0, "scores": {}}
        crops = list(sc)
        vals = np.array([sc[c] for c in crops], dtype=float)
        # Мягкий максимум вместо доли от суммы: сходства лежат близко друг к
        # другу, и без обострения любая культура получала бы четверть.
        w = np.exp(self.SHARPNESS * (vals - vals.max()))
        prob = w / w.sum()
        return self._wrap({c: float(p) for c, p in zip(crops, prob)})

    def predict_series(self, dates, values) -> dict:
        """Культура по всему ряду поля: каждый сезон голосует отдельно.

        Сезоны усредняются вероятностями, а не голосами большинства: сезон с
        плохим покрытием даёт размазанное распределение и сам по себе весит
        меньше, отдельной поправки на качество сезона не требуется.

        Важная оговорка, которую видно только на длинном ряде: культура на поле
        меняется по севообороту, а ответ здесь один на весь ряд. Поэтому рядом
        с ответом отдаётся `agreement` — доля сезонов, согласных с итогом.
        Низкое согласие при высокой уверенности как раз и означает севооборот,
        и интерфейсу есть о чём сказать вслух.
        """
        dates = list(dates)
        values = list(values)
        years = sorted({
            (d if isinstance(d, date) else date.fromisoformat(str(d)[:10])).year
            for d in dates
        })
        acc: dict[str, list[float]] = {}
        per_season: dict[int, str] = {}
        for y in years:
            curve = season_curve(dates, values, year=y)
            if curve is None:
                continue
            p = self.predict(curve)
            if not p["scores"]:
                continue
            per_season[y] = p["best_guess"]
            for crop, v in p["scores"].items():
                acc.setdefault(crop, []).append(v)
        if not per_season:
            return {"crop": None, "best_guess": None, "confidence": 0.0,
                    "group": None, "group_confidence": 0.0, "scores": {}, "seasons": 0,
                    "reason": "нет ни одного сезона с полным покрытием апреля-октября"}

        mean_prob = {c: float(np.mean(v)) for c, v in acc.items()}
        total = sum(mean_prob.values()) or 1.0
        mean_prob = {c: v / total for c, v in mean_prob.items()}
        top = max(mean_prob, key=mean_prob.get)
        agreement = sum(1 for v in per_season.values() if v == top) / len(per_season)
        return self._wrap(
            mean_prob,
            seasons=len(per_season),
            years=sorted(per_season),
            by_season={str(y): c for y, c in sorted(per_season.items())},
            agreement=round(agreement, 3),
        )


class CropClassifier:
    """Название культуры по фенологическим признакам сезона: градиентный бустинг.

    Обучается отдельной командой (`python -m src.ml.crop_id --train`) и живёт в
    models/crop_classifier.pkl. Отсутствие файла — штатная ситуация: тогда
    название берётся у CropDetector, и это честно отражается в уверенности.

    Признаки те же, что отдаёт phenology(): уровни по месяцам, дата и высота
    пика, скорость схода, длина периода выше 0,5 и так далее. Ни одного признака
    вида «сколько было снимков» — плотность съёмки описывает не культуру, а
    район и год, и модель на ней училась угадывать поле, а не растение.
    """

    def __init__(self, booster=None, classes: list[str] | None = None,
                 features: tuple[str, ...] = FEATURES):
        self._booster = booster
        self._classes = list(classes or [])
        self._features = tuple(features)

    @property
    def ready(self) -> bool:
        return self._booster is not None and bool(self._classes)

    @classmethod
    def load(cls, path) -> "CropClassifier":
        """Читает обученную модель. Ошибка чтения — не повод ронять анализ."""
        import pickle
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return cls()
        try:
            with p.open("rb") as f:
                blob = pickle.load(f)
            return cls(blob["booster"], blob["classes"], tuple(blob.get("features", FEATURES)))
        except Exception:  # noqa: BLE001 — запасной путь не имеет права падать
            return cls()

    def predict_curve(self, curve: np.ndarray) -> dict[str, float]:
        """Вероятности культур по одной сезонной кривой."""
        if not self.ready:
            return {}
        row = phenology(curve)
        x = np.array([[row[f] for f in self._features]], dtype=float)
        try:
            prob = np.asarray(self._booster.predict(x), dtype=float).ravel()
        except Exception:  # noqa: BLE001
            return {}
        if prob.size != len(self._classes):
            return {}
        return {c: float(p) for c, p in zip(self._classes, prob)}

    def predict_series(self, dates, values) -> dict[str, float]:
        """Вероятности культур по всему ряду: сезоны усредняются.

        Усреднение вероятностей, а не голосование: сезон с плохим покрытием даёт
        размазанное распределение и весит меньше сам по себе.
        """
        dates, values = list(dates), list(values)
        years = sorted({
            (d if isinstance(d, date) else date.fromisoformat(str(d)[:10])).year
            for d in dates
        })
        acc: dict[str, list[float]] = {}
        for y in years:
            curve = season_curve(dates, values, year=y)
            if curve is None:
                continue
            for crop, p in self.predict_curve(curve).items():
                acc.setdefault(crop, []).append(p)
        if not acc:
            return {}
        mean = {c: float(np.mean(v)) for c, v in acc.items()}
        total = sum(mean.values()) or 1.0
        return {c: v / total for c, v in mean.items()}


# Загруженные один раз модели. Ядро вызывают на каждом поле, а чтение файлов
# стоит дороже самого определения культуры.
_DETECTOR: CropDetector | None = None
_CLASSIFIER: CropClassifier | None = None
_LOAD_TRIED = False


def _models() -> tuple[CropDetector | None, CropClassifier]:
    """Ленивая загрузка эталонов и классификатора из models/, одна попытка."""
    global _DETECTOR, _CLASSIFIER, _LOAD_TRIED
    if _LOAD_TRIED:
        return _DETECTOR, _CLASSIFIER or CropClassifier()
    _LOAD_TRIED = True
    from pathlib import Path

    models = Path(__file__).resolve().parents[2] / "models"
    try:
        from src.core.crop_climatology import CropClimatology

        clim_path = models / "crop_climatology.json"
        if clim_path.exists():
            _DETECTOR = CropDetector.from_climatology(CropClimatology.load(clim_path))
    except Exception:  # noqa: BLE001
        _DETECTOR = None
    _CLASSIFIER = CropClassifier.load(models / "crop_classifier.pkl")
    return _DETECTOR, _CLASSIFIER


def set_models(detector: CropDetector | None = None,
               classifier: CropClassifier | None = None) -> None:
    """Подменяет модели снаружи — для тестов и для валидации без утечки."""
    global _DETECTOR, _CLASSIFIER, _LOAD_TRIED
    _LOAD_TRIED = True
    if detector is not None:
        _DETECTOR = detector
    if classifier is not None:
        _CLASSIFIER = classifier


def identify_crop(dates, values, declared: str | None = None) -> dict:
    """Что растёт на поле: единственная функция, нужная остальному ядру.

    declared — культура, названная пользователем или пришедшая из тегов OSM.
    Она НЕ отключает определение: расхождение между заявленным и увиденным —
    это самостоятельный результат. Севооборот никто не обязан вносить в сервис,
    и поле, записанное как озимая пшеница, три года спустя вполне может быть
    занято подсолнечником. Сказать об этом вслух полезнее, чем промолчать.

    Возвращает словарь:
        crop        — итоговая культура для показа (заявленная, если она есть)
        source      — откуда она: "user" | "detected" | "unknown"
        detected    — что увидено в данных (может быть None при малой уверенности)
        confidence  — уверенность в увиденном
        group       — крупная фенологическая группа и уверенность в ней
        norm_crop   — чьей нормой мерить поле (может отличаться от crop, см. E19)
        conflict    — расхождение заявленного с увиденным или None
        note        — готовая фраза для интерфейса, по-русски
    """
    detector, classifier = _models()
    out: dict = {
        "crop": declared, "source": "user" if declared else "unknown",
        "detected": None, "confidence": 0.0, "group": None, "group_confidence": 0.0,
        "norm_crop": declared, "conflict": None, "note": "", "seasons": 0,
    }
    if detector is None or not detector.crops:
        out["note"] = "эталонов культур нет, определение по кривой недоступно"
        return out

    by_curve = detector.predict_series(dates, values)
    out["seasons"] = by_curve.get("seasons", 0)
    if not by_curve.get("scores"):
        out["note"] = by_curve.get("reason", "сезон покрыт слишком редко, культура не определяется")
        return out

    # Норму выбирает сравнение с эталоном — в этой задаче оно измеримо лучше.
    out["norm_crop"] = by_curve["best_guess"]
    out["group"] = by_curve["group"]
    out["group_confidence"] = by_curve["group_confidence"]
    out["agreement"] = by_curve.get("agreement")
    out["by_season"] = by_curve.get("by_season")
    out["scores_by_curve"] = by_curve["scores"]

    # Название культуры называет бустинг, если он обучен.
    named = classifier.predict_series(dates, values)
    if named:
        top = max(named, key=named.get)
        raw = float(named[top])
        out["detected"] = top
        # Наружу уходит калиброванное число, сырое остаётся рядом для отладки.
        out["confidence"] = calibrate(raw)
        out["confidence_raw"] = round(raw, 3)
        out["scores"] = {c: round(v, 3) for c, v in sorted(named.items())}
        out["named_by"] = "бустинг"
    else:
        out["detected"] = by_curve["crop"]
        out["confidence"] = by_curve["confidence"]
        out["scores"] = by_curve["scores"]
        out["named_by"] = "сравнение с эталоном"

    # Отказ от ответа. У бустинга он редкий: на кросс-валидации 99 % полей
    # получают сырую уверенность выше 0,4, а ниже неё распределение размазано
    # по четырём классам и называть что-либо бессмысленно. У сравнения с
    # эталоном порог выше, потому что оно и называет хуже.
    if out["named_by"] == "бустинг":
        if out.get("confidence_raw", 0.0) < 0.40:
            out["detected"], out["confidence"] = None, 0.0
    elif out["detected"] and out["confidence"] < CONFIDENCE_MIN:
        out["detected"] = None

    if declared is None and out["detected"]:
        out["crop"] = out["detected"]
        out["source"] = "detected"
        out["note"] = (
            "культура не была указана; по форме сезонной кривой это похоже на "
            f"«{out['detected']}» (уверенность {out['confidence']:.2f})"
        )
    elif declared is None:
        out["note"] = (
            "культура не указана и по кривой не определяется уверенно; "
            f"поле похоже на группу «{GROUP_TITLE.get(out['group'], out['group'])}»"
        )
    else:
        # Спор с пользователем ведётся только на уровне группы: различить
        # пшеницу и подсолнечник данные не позволяют, а «озимые против яровых»
        # определяется с точностью 0,85 и спорить на этом уровне честно.
        declared_group = CROP_GROUP.get(str(declared).strip().lower())
        if (declared_group and out["group"] and declared_group != out["group"]
                and out["group_confidence"] >= 0.75):
            out["conflict"] = {
                "declared": declared, "declared_group": declared_group,
                "observed_group": out["group"],
                "confidence": out["group_confidence"],
            }
            out["note"] = (
                f"заявлена культура «{declared}» (группа «{declared_group}»), "
                f"но кривая поля ведёт себя как «{GROUP_TITLE.get(out['group'], out['group'])}» "
                f"(уверенность {out['group_confidence']:.2f}). "
                "Возможные причины: сменился севооборот, перепутано поле или "
                "контур захватывает соседний участок."
            )
        else:
            out["note"] = f"культура указана пользователем: «{declared}»"
    return out
