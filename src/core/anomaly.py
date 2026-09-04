"""Поиск периодов угнетения растительности и версия причины.

Пороги z-оценки взяты из постановки задачи:
    z >= -1          штатное развитие
    -2 <= z < -1     угнетение биомассы
    z < -2           критическая аномалия

Пороги проверены по разметке организаторов на 30 520 размеченных строках
обучающего набора: класс `status` восстанавливается из `ndvi_zscore` этими
правилами с точностью 1,000000, ошибок ноль. Менять их не нужно и нельзя —
это язык, на котором говорит заказчик.

Вся ценность модуля — не в пороге, а в том, что идёт после него: одна точка,
вернувшаяся выше порога, не должна разрывать эпизод угнетения на два события,
а одиночный шумовой выброс не должен порождать событие вовсе. Параметры склейки
и минимальной длительности подобраны перебором по согласию с разметкой,
таблица перебора — в reports/anomaly_calibration.md.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.contracts import (
    CAUSE_ABRUPT,
    CAUSE_DROUGHT,
    CAUSE_EXCESS_WATER,
    CAUSE_HEAT,
    CAUSE_UNKNOWN,
    SEVERITY_CRITICAL,
    SEVERITY_NORMAL,
    SEVERITY_SUPPRESSION,
    AnomalyPeriod,
    WeatherPoint,
)

# --- Версии причины сверх четырёх базовых -----------------------------------
# Коды живут здесь, а не в contracts.py: контракт заморожен, а список версий
# ещё будет расти. Интерфейс различает их по строке и раскрашивает карточки.
CAUSE_HARVEST = "harvest"          # уборка или скашивание — падение «в срок»
CAUSE_COLD = "cold"                # затяжной холод, возвратное похолодание
CAUSE_NON_WEATHER = "non_weather"  # погода в норме, причина агрономическая

# --- Параметры сборки периодов ----------------------------------------------
# Минимальная длительность периода. Ряд сглажен Уиттекером (lam=100), поэтому
# один шумовой снимок размазывается примерно на +-3 дня: всё, что короче пяти
# дней, неотличимо от одиночного выброса. Подтверждено перебором: при 5 днях
# F1 по классу «Угнетение биомассы» 0.773 против 0.766 при 7 днях и 0.791
# без фильтра вовсе — но без фильтра лента событий распухает на треть за счёт
# однодневных всплесков, которые постановка прямо запрещает считать аномалией.
MIN_DURATION_DAYS = 5
# Промежуток, через который два отрезка склеиваются в один период.
# Декада — минимальный срок, за который угнетённый посев способен реально
# восстановиться; более короткий возврат выше порога — это шум наблюдения,
# а не выздоровление. Перебор даёт колено кривой ровно здесь: 0 -> 10 дней
# поднимает F1 с 0.710 до 0.773, дальше прирост в третьем знаке, зато растёт
# слипание разных эпизодов в один.
MERGE_GAP_DAYS = 10
# Период должен опираться хотя бы на одно реальное наблюдение. Период целиком
# из интерполяции не подтверждён ничем; фильтр поднимает долю найденных периодов,
# попадающих в разметку, с 0.93 до 0.99, почти не трогая полноту.
MIN_OBSERVATIONS = 1
# Насколько далеко границе периода позволено уходить от крайнего снимка.
# Обрезка «ровно по наблюдениям» (запас 0) выглядит строго, но теряет четверть
# периодов: укоротившись, они не проходят фильтр длительности, F1 по угнетению
# падает с 0.773 до 0.754. Запас 5 дней даёт ровно те же числа, что и полное
# отсутствие обрезки, но не даёт событию начаться в межсезонье.
OBS_TRIM_PAD_DAYS = 5

# --- Параметры версий причины -----------------------------------------------
# Окно накопления осадков для проверки версии засухи
PRECIP_WINDOW_DAYS = 30
# Доля от нормы осадков, ниже которой говорим о дефиците влаги
DROUGHT_RATIO = 0.5
# Превышение нормы осадков, выше которого говорим о переувлажнении
EXCESS_WATER_RATIO = 2.0
# Отклонение средней температуры, при котором говорим о температурном стрессе
HEAT_ANOMALY_C = 2.5
# Отклонение вниз, при котором говорим о затяжном холоде
COLD_ANOMALY_C = -2.5
# Абсолютный порог суточной средней температуры, ниже которого рост
# зерновых практически останавливается (биологический ноль около +5 гр.)
COLD_ABSOLUTE_C = 8.0
# Холод считаем причиной только в фазе роста: после налива зерна похолодание
# уже не угнетает посев, а ускоряет естественное отмирание.
COLD_PHASE_DOY_MAX = 180
# Нижняя граница фазы активного роста. До начала марта озимые на юге России
# находятся в зимнем покое: точка роста не работает, биомасса не набирается,
# и похолодание в это время развитию не вредит. Без этой границы сервис писал
# «похолодание пришлось на фазу активного роста (день года 24)» — про январь,
# когда растение спит. Агроном такому объяснению верить перестанет.
COLD_PHASE_DOY_MIN = 60
# Падение z-оценки за короткий срок, которое считаем резким событием
ABRUPT_DROP_Z = 2.0
ABRUPT_WINDOW_DAYS = 10
# Уборка: после резкого падения ряд выходит на низкое плато, а не отскакивает
HARVEST_MIN_PLATEAU_DAYS = 14
# Коридор «погода совсем обычная»: в нём версия «причина не погодная» уверенная.
# За его пределами, но внутри порогов засухи/жары/холода/переувлажнения, версия
# та же, но уверенность ниже — погода была на грани, хотя ни один порог не взят.
NON_WEATHER_PRECIP_LO, NON_WEATHER_PRECIP_HI = 0.7, 1.6
NON_WEATHER_TEMP_C = 1.5

# Характерное окно уборки по дню года, восстановлено по обучающему набору:
# для каждого полигон-сезона взят день максимального падения восстановленного
# NDVI за декаду, затем квартили по культуре (расширены на 10 дней в обе стороны).
# Смысл окна: падение «в срок» — это уборка, точно такое же падение на месяц
# раньше — повреждение посева. Различить их можно только календарём.
HARVEST_WINDOW_BY_CROP = {
    "озимая пшеница": (159, 234),
    "зерновые": (145, 187),
    "пастбища/зерновые": (129, 179),
    "подсолнечник": (162, 218),
}
# Культура неизвестна — берём объединение окон, версия становится слабее
HARVEST_WINDOW_DEFAULT = (145, 234)


def classify(z: float) -> str:
    """Класс состояния по z-оценке."""
    if not np.isfinite(z):
        return SEVERITY_NORMAL
    if z < -2.0:
        return SEVERITY_CRITICAL
    if z < -1.0:
        return SEVERITY_SUPPRESSION
    return SEVERITY_NORMAL


def find_periods(
    dates: list[date],
    z: np.ndarray,
    min_duration: int = MIN_DURATION_DAYS,
    merge_gap: int = MERGE_GAP_DAYS,
    observed: np.ndarray | None = None,
    min_observations: int = MIN_OBSERVATIONS,
    trim_pad: int = OBS_TRIM_PAD_DAYS,
) -> list[tuple[int, int, str]]:
    """Находит устойчивые отрезки отрицательного отклонения.

    Три шага, и порядок между ними принципиален:
      1. отмечаем всё, что ушло ниже -1;
      2. СНАЧАЛА склеиваем соседние отрезки через короткий промежуток;
      3. и только ПОТОМ отбрасываем короткие.
    Если поменять шаги местами, длинный эпизод, разорванный шумом на три куска
    по четыре дня, будет выброшен целиком — именно это и происходило раньше:
    детектор находил лишь 37 % эталонных эпизодов вместо 61 %.

    Возвращает список (индекс начала, индекс конца включительно, класс серьёзности).
    Класс отрезка определяется по самой глубокой точке внутри него.
    """
    z = np.asarray(z, dtype=float)
    below = np.isfinite(z) & (z < -1.0)
    ordinals = np.array([d.toordinal() for d in dates])

    spans: list[list[int]] = []
    start: int | None = None
    for i, flag in enumerate(below):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append([start, i - 1])
            start = None
    if start is not None:
        spans.append([start, len(below) - 1])
    if not spans:
        return []

    # Шаг 2: склейка коротких разрывов
    merged: list[list[int]] = [spans[0]]
    for a, b in spans[1:]:
        gap_days = int(ordinals[a] - ordinals[merged[-1][1]]) - 1
        if gap_days <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    out = []
    for a, b in merged:
        if observed is not None:
            # Шаг 3: период должен опираться на реальные снимки, а его границы —
            # не убегать далеко в интерполяцию. Сетка сплошная и покрывает
            # межсезонье, поэтому без обрезки событие легко начинается 25 марта,
            # когда первый снимок сезона сделан 1 апреля. Оставляем запас
            # в trim_pad дней: отклонение почти наверняка началось до снимка,
            # который его зафиксировал, и обрезать «в ноль» значит систематически
            # занижать длительность.
            hit = np.flatnonzero(observed[a : b + 1])
            if len(hit) < min_observations:
                continue
            first, last = a + int(hit[0]), a + int(hit[-1])
            while a < first and int(ordinals[first] - ordinals[a]) > trim_pad:
                a += 1
            while b > last and int(ordinals[b] - ordinals[last]) > trim_pad:
                b -= 1
        duration = int(ordinals[b] - ordinals[a]) + 1
        if duration < min_duration:
            continue
        out.append((a, b, classify(float(np.nanmin(z[a : b + 1])))))
    return out


def _weather_frame(weather: list[WeatherPoint]) -> pd.DataFrame:
    """Приводит погодные точки к таблице, отсортированной по дате."""
    if not weather:
        return pd.DataFrame(columns=["date", "temp_c", "precip_mm"])
    df = pd.DataFrame(
        [{"date": w.date, "temp_c": w.temp_c, "precip_mm": w.precip_mm} for w in weather]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _harvest_window(crop_type: str | None) -> tuple[int, int]:
    """Характерное окно уборки по дню года для культуры."""
    if crop_type is None:
        return HARVEST_WINDOW_DEFAULT
    return HARVEST_WINDOW_BY_CROP.get(crop_type.strip().lower(), HARVEST_WINDOW_DEFAULT)


def attribute_cause(
    start: date,
    end: date,
    z_series: np.ndarray,
    dates: list[date],
    weather: list[WeatherPoint],
    crop_type: str | None = None,
) -> tuple[str, float, dict, str]:
    """Определяет наиболее вероятную причину периода.

    Версии проверяются по убыванию убедительности свидетельства, а не по частоте:
    резкое падение видно прямо в ряде и не требует погоды, дефицит осадков
    измеряется числом, а «причина не погодная» — самая слабая из содержательных
    версий и стоит последней, но она честнее «неизвестно»: она означает, что
    погоду мы проверили и она ни при чём.

    Возвращает (причина, уверенность от 0 до 1, свидетельства, фраза для интерфейса).
    """
    evidence: dict = {}
    duration = (end - start).days + 1

    # --- Версии, читаемые прямо из ряда: резкое падение и уборка ---
    idx = [i for i, d in enumerate(dates) if start <= d <= end]
    drop = np.nan
    if idx:
        head = idx[0]
        back = max(0, head - ABRUPT_WINDOW_DAYS)
        before = np.nanmax(z_series[back : head + 1]) if head > back else np.nan
        after = np.nanmin(z_series[head : min(head + 3, len(z_series))])
        drop = float(before - after) if np.isfinite(before) and np.isfinite(after) else np.nan
        if np.isfinite(drop):
            evidence["z_drop_10d"] = round(drop, 2)

    if np.isfinite(drop) and drop >= ABRUPT_DROP_Z:
        lo, hi = _harvest_window(crop_type)
        doy = start.timetuple().tm_yday
        evidence["start_doy"] = doy
        evidence["harvest_window"] = [lo, hi]
        in_window = lo <= doy <= hi
        # Уборка отличается от повреждения двумя вещами: она случается в срок
        # и после неё индекс не отскакивает, а стоит на низком плато до конца
        # сезона. Повреждение вне срока — тревога, уборка в срок — не тревога,
        # и путать их дорого: ложные уборочные события обесценивают ленту.
        if in_window and duration >= HARVEST_MIN_PLATEAU_DAYS:
            conf = 0.7 if crop_type else 0.5
            crop_note = " для культуры «{}»".format(crop_type) if crop_type else ""
            return (
                CAUSE_HARVEST,
                conf,
                evidence,
                "Индекс обвалился на {:.1f} стандартных отклонения и после этого {} дней "
                "держится на низком уровне. День года {} попадает в характерное окно "
                "уборки{} ({}-{}). Скорее всего, это уборка или скашивание, "
                "а не повреждение посева.".format(
                    drop, duration, doy, crop_note, lo, hi
                ),
            )
        return (
            CAUSE_ABRUPT,
            0.6,
            evidence,
            "Резкое падение индекса на {:.1f} стандартных отклонения за декаду. "
            "Похоже на одномоментное событие: уборку, потраву или механическое "
            "повреждение посева.".format(drop),
        )

    wf = _weather_frame(weather)
    if wf.empty:
        return CAUSE_UNKNOWN, 0.0, evidence, "Погодные данные недоступны, причина не определена."

    period_mask = (wf["date"] >= pd.Timestamp(start)) & (wf["date"] <= pd.Timestamp(end))
    window_start = pd.Timestamp(end) - pd.Timedelta(days=PRECIP_WINDOW_DAYS)
    window_mask = (wf["date"] >= window_start) & (wf["date"] <= pd.Timestamp(end))

    # Норма считается по тому же окну дня года в остальные годы истории
    doy_lo, doy_hi = window_start.dayofyear, pd.Timestamp(end).dayofyear
    hist = wf[(wf["date"].dt.dayofyear >= doy_lo) & (wf["date"].dt.dayofyear <= doy_hi)]
    hist = hist[hist["date"].dt.year != pd.Timestamp(end).year]

    precip_actual = float(wf.loc[window_mask, "precip_mm"].sum()) if window_mask.any() else np.nan
    precip_norm = (
        float(hist.groupby(hist["date"].dt.year)["precip_mm"].sum().mean())
        if not hist.empty
        else np.nan
    )
    temp_actual = float(wf.loc[period_mask, "temp_c"].mean()) if period_mask.any() else np.nan
    # Температурная норма считается по дням года САМОГО ПЕРИОДА, а не по
    # последнему 30-дневному окну. Иначе стодневный период с июня по октябрь
    # сравнивается с октябрьской нормой и получает аномалию +11 градусов
    # на ровном месте — ошибка, которая ловится только на длинных периодах.
    p_lo, p_hi = start.timetuple().tm_yday, end.timetuple().tm_yday
    doy_col = wf["date"].dt.dayofyear
    in_phase = (doy_col >= p_lo) & (doy_col <= p_hi) if p_lo <= p_hi else (
        (doy_col >= p_lo) | (doy_col <= p_hi)
    )
    hist_t = wf[in_phase & (wf["date"].dt.year != pd.Timestamp(end).year)]
    temp_norm = float(hist_t["temp_c"].mean()) if not hist_t.empty else np.nan
    temp_min = float(wf.loc[period_mask, "temp_c"].min()) if period_mask.any() else np.nan

    if np.isfinite(precip_actual):
        evidence["precip_30d_mm"] = round(precip_actual, 1)
    if np.isfinite(precip_norm):
        evidence["precip_30d_norm_mm"] = round(precip_norm, 1)
    if np.isfinite(temp_actual):
        evidence["temp_mean_c"] = round(temp_actual, 1)
    if np.isfinite(temp_norm):
        evidence["temp_norm_c"] = round(temp_norm, 1)
    if np.isfinite(temp_min):
        evidence["temp_min_c"] = round(temp_min, 1)

    ratio = precip_actual / precip_norm if np.isfinite(precip_norm) and precip_norm > 0 else np.nan
    temp_anom = (
        temp_actual - temp_norm if np.isfinite(temp_actual) and np.isfinite(temp_norm) else np.nan
    )
    if np.isfinite(ratio):
        evidence["precip_ratio"] = round(float(ratio), 2)
    if np.isfinite(temp_anom):
        evidence["temp_anomaly_c"] = round(float(temp_anom), 1)

    # --- Версия: дефицит влаги ---
    if np.isfinite(ratio) and ratio < DROUGHT_RATIO:
        hot = np.isfinite(temp_anom) and temp_anom > HEAT_ANOMALY_C
        tail = " При этом температура выше нормы на {:.1f} градуса.".format(temp_anom) if hot else ""
        return (
            CAUSE_DROUGHT,
            0.85 if hot else 0.7,
            evidence,
            "Осадков за 30 дней выпало {:.0f} мм при норме {:.0f} мм, это {:.0f} процентов "
            "от нормы. Наиболее вероятен дефицит влаги.{}".format(
                precip_actual, precip_norm, ratio * 100, tail
            ),
        )

    # --- Версия: температурный стресс при достаточном увлажнении ---
    if np.isfinite(temp_anom) and temp_anom > HEAT_ANOMALY_C:
        return (
            CAUSE_HEAT,
            0.6,
            evidence,
            "Средняя температура за период выше нормы на {:.1f} градуса при достаточном "
            "увлажнении. Вероятен температурный стресс.".format(temp_anom),
        )

    # --- Версия: затяжной холод в фазе роста ---
    # Работает только до дня года 180: позже похолодание застаёт посев уже
    # созревающим и угнетением не является. Требуется либо заметный минус
    # к норме, либо абсолютно холодный период — рост зерновых практически
    # останавливается ниже +8 градусов среднесуточных.
    start_doy = start.timetuple().tm_yday
    cold_rel = np.isfinite(temp_anom) and temp_anom <= COLD_ANOMALY_C
    if cold_rel and start_doy > COLD_PHASE_DOY_MAX:
        # Тот же холод, но в конце сезона: посев уже созрел, и похолодание не
        # угнетает его, а ускоряет естественное отмирание. Формулировка обязана
        # это различать, иначе сервис поднимает тревогу на нормальном явлении.
        return (
            CAUSE_COLD,
            0.35,
            evidence,
            "Температура за период на {:.1f} градуса ниже нормы, но это конец сезона "
            "(день года {}). Для созревающего посева холод означает ускоренное "
            "естественное отмирание, а не угнетение. Отдельного вмешательства "
            "обычно не требует.".format(abs(temp_anom), start_doy),
        )
    if cold_rel and start_doy < COLD_PHASE_DOY_MIN and np.isfinite(temp_actual):
        # Зимний покой: холод есть, угнетения нет. Отдельная формулировка, потому
        # что молчать тоже неправильно — пользователь видит просадку и ждёт
        # объяснения, почему она не повод для тревоги.
        return (
            CAUSE_COLD,
            0.3,
            evidence,
            "Средняя температура за период {:.1f} градуса, день года {} — это зимний "
            "покой. Озимые в это время не растут, низкий индекс отражает состояние "
            "почвы и остатков, а не угнетение посева. Тревоги не требует.".format(
                temp_actual, start_doy
            ),
        )
    if COLD_PHASE_DOY_MIN <= start_doy <= COLD_PHASE_DOY_MAX and np.isfinite(temp_actual):
        cold_abs = temp_actual <= COLD_ABSOLUTE_C
        if cold_rel or cold_abs:
            if cold_rel and cold_abs:
                conf, why = 0.65, (
                    "Средняя температура за период всего {:.1f} градуса, это на {:.1f} "
                    "ниже нормы".format(temp_actual, abs(temp_anom))
                )
            elif cold_rel:
                conf, why = 0.55, (
                    "Средняя температура за период на {:.1f} градуса ниже нормы".format(
                        abs(temp_anom)
                    )
                )
            else:
                conf, why = 0.4, (
                    "Средняя температура за период всего {:.1f} градуса".format(temp_actual)
                )
            return (
                CAUSE_COLD,
                conf,
                evidence,
                "{}. Похолодание пришлось на фазу активного роста (день года {}): "
                "набор биомассы приостановился. Такое угнетение обычно обратимо, "
                "но сдвигает развитие на одну-две недели.".format(why, start_doy),
            )

    # --- Версия: переувлажнение ---
    if np.isfinite(ratio) and ratio > EXCESS_WATER_RATIO:
        return (
            CAUSE_EXCESS_WATER,
            0.55,
            evidence,
            "Осадков выпало в {:.1f} раза больше нормы. Возможно переувлажнение "
            "и подтопление посевов.".format(ratio),
        )

    # --- Версия: причина не погодная ---
    # Отличается от «неизвестно» тем, что погода не просто не объяснила
    # отклонение, а измеримо в норме по обоим показателям. Это полезный вывод:
    # он снимает с погоды подозрение и отправляет агронома искать на поле.
    # До этой строки не дошла ни одна погодная версия. Если оба показателя
    # при этом измерены, значит погода проверена и оправдана — это утверждение,
    # а не отсутствие ответа.
    if np.isfinite(ratio) and np.isfinite(temp_anom):
        tight = (
            NON_WEATHER_PRECIP_LO <= ratio <= NON_WEATHER_PRECIP_HI
            and abs(temp_anom) <= NON_WEATHER_TEMP_C
        )
        return (
            CAUSE_NON_WEATHER,
            0.5 if tight else 0.35,
            evidence,
            "Осадков {:.0f} мм при норме {:.0f} мм ({:.0f} процентов), температура "
            "отличается от нормы на {:.1f} градуса — ни засухи, ни жары, ни холода, "
            "ни переувлажнения за эти {} дней не было. Погода отклонение не объясняет: "
            "проверяйте агротехнику, состояние семян, болезни и вредителей.".format(
                precip_actual, precip_norm, ratio * 100, temp_anom, duration
            ),
        )

    return (
        CAUSE_UNKNOWN,
        0.2,
        evidence,
        "Отклонение от нормы устойчивое, но погодных данных на этот период нет, "
        "поэтому версию причины проверить не на чем. Нужен осмотр поля.",
    )


def build_periods(
    dates: list[date],
    z: np.ndarray,
    weather: list[WeatherPoint],
    crop_type: str | None = None,
    norm_is_crop: bool = False,
    observed: np.ndarray | None = None,
) -> list[AnomalyPeriod]:
    """Полный проход: находит периоды и объясняет каждый.

    `norm_is_crop` означает, что норма взята не по истории самого поля, а средняя
    по типу культуры. Такая норма грубее (RMSE между двумя полями одной культуры
    0.108 при собственном шуме наблюдения 0.066), поэтому уверенность в версии
    понижается, а в тексте появляется прямая оговорка. Честность оценки здесь
    важнее её красоты: пользователь должен понимать, на что он смотрит.
    """
    out: list[AnomalyPeriod] = []
    for a, b, severity in find_periods(dates, z, observed=observed):
        seg = z[a : b + 1]
        cause, conf, evidence, text = attribute_cause(
            dates[a], dates[b], z, dates, weather, crop_type
        )
        if norm_is_crop:
            conf = round(conf * 0.7, 2)
            evidence["norm_source"] = "crop"
            text += (
                " Внимание: у этого поля нет собственной истории наблюдений, норма "
                "взята средняя по культуре. Оценка ориентировочная, глубина отклонения "
                "может быть завышена или занижена."
            )
        out.append(
            AnomalyPeriod(
                start=dates[a],
                end=dates[b],
                severity=severity,
                duration_days=(dates[b] - dates[a]).days + 1,
                min_zscore=round(float(np.nanmin(seg)), 2),
                mean_zscore=round(float(np.nanmean(seg)), 2),
                cause=cause,
                cause_confidence=conf,
                evidence=evidence,
                explanation=text,
            )
        )
    return out
