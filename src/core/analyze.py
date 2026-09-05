"""Точка входа доменного ядра: превращает собранные данные в готовый анализ.

Это единственная функция, которую вызывает API. Слой провайдеров ничего не знает
про то, как считается норма и как ищутся аномалии, а ядро ничего не знает про то,
откуда взялись наблюдения.
"""
from __future__ import annotations

from pathlib import Path

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.contracts import (
    CAUSE_ABRUPT,
    CAUSE_DROUGHT,
    CAUSE_EXCESS_WATER,
    CAUSE_HEAT,
    AnalysisResult,
    SeriesInput,
    SeriesPoint,
)
from src.core.anomaly import (
    CAUSE_COLD,
    CAUSE_HARVEST,
    CAUSE_NON_WEATHER,
    build_periods,
)
from src.core.climatology import (
    MIN_STD,
    fit_climatology_loo,
    has_enough_history,
    lookup_norm,
    zscore,
)
from src.core.restore import restore_on_grid

# Шаг выдачи итогового ряда наружу: посуточно график слишком тяжёлый для фронтенда
OUTPUT_STEP_DAYS = 1
# Потолок модуля z-оценки, см. пояснение в месте применения
Z_LIMIT = 8.0
# Вегетационный сезон (месяцы включительно): вне его события не ищутся
SEASON_MONTHS = (4, 10)

# Норма по типу культуры — запасной путь для полей без собственной истории.
# Модуль делается параллельно, поэтому импорт мягкий: пока файла нет, ядро
# работает как раньше (без z-оценки на таких полях), а когда он появится —
# заводится само, без правок здесь.
try:  # pragma: no cover - зависит от наличия соседнего модуля
    from src.core.crop_climatology import CropClimatology  # type: ignore
except ImportError:  # pragma: no cover
    CropClimatology = None  # type: ignore

# Журнал агротехнических работ. Заказчик кейса просил поле, куда пользователь
# вписывает, что и когда вносил на поле. Для ядра это второй, независимый от
# погоды источник объяснений: гербицид за неделю до просадки объясняет её лучше
# любой метеосводки, а уборка внутри периода вообще снимает тревогу.
try:  # pragma: no cover
    from src.core.agrolog import explain_with_agro  # type: ignore
except ImportError:  # pragma: no cover
    explain_with_agro = None  # type: ignore

# Определение культуры по форме сезонной кривой. Раньше культура была известна
# только со слов пользователя, а не сказать её мог кто угодно: из тегов OSM она
# приходит у считанных полей. Без культуры поле получает усреднённую норму,
# «общерегиональное» окно уборки и продуктивность, которую не с чем сравнить.
# Модуль закрывает эту дыру и заодно ловит расхождение: поле, записанное как
# озимая пшеница, три года спустя вполне может быть занято другим — севооборот
# в сервис никто не вносит.
try:  # pragma: no cover
    from src.core.crop_profile import identify_crop  # type: ignore
except ImportError:  # pragma: no cover
    identify_crop = None  # type: ignore

# Сравнение с соседними полями. Отвечает на вопрос, который агроном задаёт
# первым: «это у меня одного или у всех?». Сравнивать при этом можно только с
# полями своей культуры, поэтому модуль опирается на определение культуры.
try:  # pragma: no cover
    from src.core.peers import compare_with_peers  # type: ignore
except ImportError:  # pragma: no cover
    compare_with_peers = None  # type: ignore

# Прогноз развития поля вперёд. Заказчик кейса сформулировал прямо: «прогнозы
# должна давать программа». Модуль опирается на климатическую норму и переносит
# текущее отклонение с затуханием; полезный горизонт около 30 дней, дальше
# прогноз сходится к норме — это измерено ретроспективно, а не предположено.
try:  # pragma: no cover
    from src.core.forecast import forecast_season  # type: ignore
except ImportError:  # pragma: no cover
    forecast_season = None  # type: ignore

# Оценка поля как объекта риска — для страховой и банка, кредитующего хозяйство.
# Заказчик кейса назвал их покупателем прямо: «создаём источник независимых
# данных и оценки». Балл собирается из устойчивости год к году, стрессовой
# нагрузки, продуктивности и тренда; веса расставлены по прогнозной проверке,
# а не на глаз.
try:  # pragma: no cover
    from src.core.scoring import field_score  # type: ignore
except ImportError:  # pragma: no cover
    field_score = None  # type: ignore

# Готовая норма по культуре, если её кто-то загрузил и положил сюда.
# Слой API вызывает set_crop_climatology() один раз при старте.
_CROP_CLIM = None
_AUTOLOAD_TRIED = False


def set_crop_climatology(model) -> None:
    """Подключает готовую норму по типу культуры (обучается вне ядра).

    Ядро не знает, откуда взялась модель: из файла, из обучающего набора или из
    заглушки в тесте. Ему достаточно методов has() и norm() из контракта
    CropClimatology.
    """
    global _CROP_CLIM
    _CROP_CLIM = model


def _autoload_crop_climatology():
    """Одна попытка подобрать норму по культуре из models/, дальше не пробуем.

    Отрицательный результат кэшируется: если файла нет, незачем стучаться в
    файловую систему на каждом полигоне.
    """
    global _CROP_CLIM, _AUTOLOAD_TRIED
    if _AUTOLOAD_TRIED or CropClimatology is None:
        return _CROP_CLIM
    _AUTOLOAD_TRIED = True
    path = Path(__file__).resolve().parents[2] / "models" / "crop_climatology.json"
    if not path.exists():
        return None
    try:
        _CROP_CLIM = CropClimatology.load(path)
    except Exception:
        # Запасной путь не имеет права ронять основной сценарий анализа
        _CROP_CLIM = None
    return _CROP_CLIM


def _crop_norm(crop_type: str | None, doys: np.ndarray):
    """Пробует получить норму по культуре. Возвращает None, если её нет.

    Три причины отказа, все штатные: модуль не приземлился, модель не загружена,
    у культуры нет нормы. Во всех случаях ядро продолжает работать без z-оценки.
    """
    model = _CROP_CLIM
    if model is None:
        # Ленивая загрузка из models/. Без неё ядро молчит на 59 полигонах из 78
        # только потому, что вызывающая сторона забыла вызвать set_crop_climatology.
        # Явная установка модели по-прежнему имеет приоритет: она выполняется
        # раньше и сюда мы уже не попадём.
        model = _autoload_crop_climatology()
    if model is None:
        return None
    try:
        if not model.has(crop_type):
            return None
        mean, std = model.norm(crop_type, doys)
    except Exception:
        # Запасной путь не имеет права ронять основной сценарий анализа
        return None
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if mean.size != doys.size or not np.isfinite(mean).any():
        return None
    return mean, std



def _identify(df: pd.DataFrame, declared: str | None) -> dict:
    """Определяет культуру по ряду наблюдений. Ошибка здесь не роняет анализ.

    Отдельная функция, а не пара строк внутри analyze(), по одной причине: у
    определения культуры три штатных исхода отказа (модуля нет, эталонов нет,
    сезон покрыт слишком редко), и все три должны заканчиваться одинаково —
    работаем с тем, что сказал пользователь, и честно пишем об этом в meta.
    """
    fallback = {
        "crop": declared, "source": "user" if declared else "unknown",
        "detected": None, "confidence": 0.0, "norm_crop": declared,
        "conflict": None, "note": "определение культуры недоступно",
    }
    if identify_crop is None:
        return fallback
    try:
        return identify_crop(df["date"].dt.date.tolist(), df["ndvi"].tolist(), declared)
    except Exception:  # noqa: BLE001 — определение культуры вспомогательное
        return fallback


def _norm_note(clim_kind: str, crop_type: str | None, norm_crop: str | None) -> str:
    """Объясняет словами, откуда взялась норма. Нужна ровно в одном случае.

    Случай такой: пользователь назвал культуру, а норма подобрана по другой.
    Без объяснения это выглядит ошибкой сервиса, а на деле это измеренный выбор
    (E19): норма — это кластер похожих кривых, и полю точнее подходит ближайший
    кластер, а не тот, что подписан его агрономическим названием. Поле озимой
    пшеницы с низким уровнем описывается нормой подсолнечника лучше, чем нормой
    своей культуры, и z-оценка от этого становится честнее, а не хуже.
    """
    if clim_kind == "polygon":
        return "норма построена по собственной истории поля"
    if clim_kind != "crop":
        return "нормы нет: не хватает истории наблюдений"
    if norm_crop and crop_type and norm_crop != crop_type:
        return (
            f"у поля нет своей истории, поэтому норма взята по культуре. Эталон "
            f"подобран по форме кривой — ближе всего подошёл «{norm_crop}», хотя "
            f"заявлена культура «{crop_type}». Это не утверждение о том, что "
            f"растёт на поле: так точнее описывается его уровень и календарь."
        )
    return (
        f"у поля нет своей истории, норма взята средняя по культуре "
        f"«{norm_crop or crop_type}» — оценка ориентировочная"
    )


def _apply_agro_journal(anomalies, agro_events, crop_type) -> None:
    """Дополняет найденные периоды сведениями из журнала работ.

    Меняет три вещи и только их: дописывает объяснение, поправляет уверенность и
    складывает свидетельства. Границы периода и класс серьёзности journal не
    трогает — они получены из данных, а журнал вводит человек, и ошибка в нём не
    должна переписывать измерение.

    Отдельный случай — подсказка `agro_suggest_cause`. Модуль журнала не меняет
    версию причины сам, но если в период попала запись об уборке, он говорит об
    этом прямо, и здесь версия переписывается: плановая работа объясняет просадку
    лучше, чем «резкое одномоментное событие».
    """
    if not agro_events or explain_with_agro is None:
        return
    for a in anomalies:
        try:
            extra, delta, evidence = explain_with_agro(
                a.cause, a.cause_confidence, a.start, a.end, agro_events, crop_type
            )
        except Exception:  # noqa: BLE001 — журнал не имеет права ронять анализ
            continue
        if not extra and not delta:
            continue
        # Журнал может не просто дополнить объяснение, а переписать саму версию:
        # уборка внутри периода объясняет просадку лучше, чем «резкое событие».
        # В этом случае поправку надо брать для НОВОЙ версии, а не для старой —
        # иначе получается бессмыслица: журнал подтвердил уборку, а уверенность
        # в ней упала, потому что применилась поправка «минус к погодной версии».
        suggested = (evidence or {}).get("agro_suggest_cause")
        if suggested and suggested != a.cause:
            try:
                extra2, delta2, evidence2 = explain_with_agro(
                    suggested, a.cause_confidence, a.start, a.end, agro_events, crop_type
                )
                if extra2:
                    extra, delta, evidence = extra2, delta2, (evidence2 or evidence)
            except Exception:  # noqa: BLE001
                pass
            a.cause = suggested

        if extra:
            a.explanation = (a.explanation.rstrip() + " " + extra).strip()
        a.cause_confidence = float(min(0.95, max(0.05, a.cause_confidence + delta)))
        if evidence:
            a.evidence.update(evidence)


def _compare_peers(df, crop_type, anomalies, peers):
    """Сравнение с соседними полями. Отсутствие соседей — штатный исход."""
    if not peers or compare_with_peers is None:
        return None
    try:
        return compare_with_peers(
            df["date"].dt.date.tolist(), df["ndvi"].tolist(), peers,
            crop_type=crop_type, anomalies=anomalies,
        )
    except Exception:  # noqa: BLE001 — сравнение вспомогательное
        return None


def _apply_peer_scope(anomalies, peer_report: dict) -> None:
    """Дописывает к периодам, районное это явление или только на этом поле.

    Уверенность в версии причины при этом двигается, и в обе стороны. Логика
    простая и проверяемая: если соседи той же культуры просели вместе с полем,
    погодная версия становится вероятнее — погода на район ложится одинаково.
    Если соседи стоят целые, погодная версия слабеет, а «причина не погодная»
    наоборот усиливается: под одним и тем же дождём одно поле просело, а пять
    соседних нет.
    """
    weather_causes = {CAUSE_DROUGHT, CAUSE_HEAT, CAUSE_EXCESS_WATER, CAUSE_COLD}
    # Версии, для которых вывод «районное или локальное» бессмыслен. Уборка
    # локальна по своей природе: каждое хозяйство убирает в свой день, и
    # советовать после неё «проверьте вредителей» — значит пугать на ровном
    # месте. Факт про соседей при этом остаётся: он говорит о ходе уборки
    # в районе, и это полезно само по себе.
    scope_irrelevant = {CAUSE_HARVEST, CAUSE_ABRUPT}
    by_start = {p["start"]: p for p in peer_report.get("periods", [])}
    for a in anomalies:
        info = by_start.get(str(a.start))
        if not info:
            continue
        district = info["scope"] == "район"
        tail = info["fact"] if a.cause in scope_irrelevant else f"{info['fact']} {info['verdict']}"
        a.explanation = (a.explanation.rstrip() + " " + tail).strip()
        a.evidence["peers_checked"] = info["peers_checked"]
        a.evidence["peers_depressed"] = info["peers_depressed"]
        a.evidence["scope"] = info["scope"]
        if a.cause in weather_causes:
            delta = 0.10 if district else -0.15
        elif a.cause == CAUSE_NON_WEATHER:
            delta = -0.10 if district else 0.10
        else:
            delta = 0.0
        if delta:
            a.cause_confidence = float(min(0.95, max(0.05, a.cause_confidence + delta)))


def analyze(inp: SeriesInput, output_step: int = OUTPUT_STEP_DAYS,
            agro_events: list | None = None,
            forecast_days: int = 30,
            peers: list | None = None) -> AnalysisResult:
    """Собирает ряд, восстанавливает пропуски, считает норму и находит аномалии.

    agro_events — журнал работ по полю (список AgroEvent из src/core/agrolog.py).
    Необязательный: контракт SeriesInput заморожен, поэтому журнал приходит
    отдельным аргументом, и старый вызов analyze(inp) работает как раньше.

    forecast_days — горизонт прогноза. По умолчанию 30, а не 60: ретроспектива
    показала, что дальше тридцати дней прогноз сходится к климатической норме
    и перестаёт нести собственное знание о поле.
    """
    obs = [o for o in inp.observations if o.ndvi is not None and np.isfinite(o.ndvi)]
    if not obs:
        return AnalysisResult(
            polygon_id=inp.polygon_id,
            meta={"n_obs": 0, "error": "нет ни одного пригодного наблюдения"},
        )

    df = pd.DataFrame(
        [{"date": pd.Timestamp(o.date), "ndvi": float(o.ndvi), "source": o.source} for o in obs]
    ).sort_values("date")
    # Если на одну дату пришло несколько сенсоров, берём медиану
    df = df.groupby("date", as_index=False).agg(ndvi=("ndvi", "median"), source=("source", "first"))

    # Культура определяется до всего остального: от неё зависит и норма, и окно
    # уборки, и эталон продуктивности. Заявленная пользователем культура имеет
    # приоритет в названии, но не отменяет проверку — расхождение с тем, что
    # видно в данных, это самостоятельный и полезный результат.
    crop_info = _identify(df, inp.crop_type)
    crop_type = crop_info.get("crop") or inp.crop_type
    # Норму выбирает не название, а сходство кривых: измерено, что так ближе к
    # настоящей кривой поля, чем даже по точно известной культуре (E19).
    norm_crop = crop_info.get("norm_crop") or crop_type

    t_days = df["date"].map(pd.Timestamp.toordinal).to_numpy()
    # Сглаживание здесь намеренно мягче, чем в задаче восстановления пропусков,
    # и это измеренный выбор, а не умолчание.
    #
    # Для метрики восстановления оптимум lam=1000 без примеси линейной: RMSE
    # 0,0794 против 0,0834 у смеси lam=100 / 50 на 50. Но у детекции аномалий
    # другая цель и другой эталон. Организаторы считают z-оценку по СЫРЫМ
    # значениям, а не по сглаженной кривой: чем сильнее сглаживание, тем дальше
    # ряд от сырых данных и тем хуже совпадение с их разметкой. Замерено на
    # 6282 размеченных точках: согласие 0,9417 при мягком сглаживании против
    # 0,8868 при lam=1000. Пять с половиной процентных пунктов согласия дороже
    # гладкости кривой, поэтому здесь остаётся мягкий вариант.
    #
    # Лучшая конфигурация проекта (0,0596) здесь неприменима принципиально:
    # она опирается на суточную поправку по соседним полям, а сервис разбирает
    # одно поле за раз, и соседей у него нет.
    grid, restored = restore_on_grid(t_days, df["ndvi"].to_numpy())
    grid_dates = [date.fromordinal(int(d)) for d in grid]

    # Норма считается по САМИМ НАБЛЮДЕНИЯМ, а не по восстановленному ряду.
    # Восстановленный ряд плотнее в тех местах, где снимков было больше, и тянет
    # норму на себя; кроме того, зимние месяцы в нём — чистая экстраполяция.
    # Организаторы считают норму по наблюдениям, и повторение их формулы даёт
    # согласие z-оценок, а значит и согласие классов с их разметкой.
    enough = has_enough_history(df["date"])
    clim = fit_climatology_loo(df["date"], df["ndvi"]) if enough else None

    doys = np.array([d.timetuple().tm_yday for d in grid_dates])
    years = np.array([d.year for d in grid_dates])
    clim_kind = "polygon"
    if clim is not None and len(clim):
        clim_mean, clim_std = lookup_norm(clim, years, doys)
        clim_years = int(np.nanmax(clim["n_years"].to_numpy()))
    else:
        # Своей истории нет — пробуем норму по типу культуры. Она грубее
        # (медиана корреляции кривых внутри культуры 0.95 против 0.86 между
        # культурами, но RMSE между двумя полями пшеницы 0.108), поэтому
        # помечается отдельно и понижает уверенность в версии причины.
        crop = _crop_norm(norm_crop, doys)
        if crop is None and norm_crop != crop_type:
            # Подобранного эталона может не оказаться в норме (культура редкая
            # или норма собрана на другом наборе). Тогда честный запасной шаг —
            # культура, которую назвал пользователь, а не отказ от нормы вовсе.
            norm_crop = crop_type
            crop = _crop_norm(norm_crop, doys)
        if crop is not None:
            clim_mean, clim_std = crop
            clim_std = np.where(np.isfinite(clim_std) & (clim_std > MIN_STD), clim_std, MIN_STD)
            clim_kind = "crop"
            clim_years = 0
        else:
            clim_mean = np.full(len(grid), np.nan)
            clim_std = np.full(len(grid), np.nan)
            clim_kind = "none"
            clim_years = 0

    z = (
        zscore(restored, clim_mean, clim_std)
        if clim_kind != "none"
        else np.full(len(grid), np.nan)
    )
    # Потолок на модуль z-оценки. Отклонение больше восьми сигм на природных
    # данных означает не рекордную аномалию, а негодную норму: слишком короткая
    # история, смена культуры или дыра в наблюдениях. Ограничение не прячет
    # событие — оно остаётся критическим, — но убирает из интерфейса дикие
    # формулировки вида «падение на 22 стандартных отклонения».
    z = np.clip(z, -Z_LIMIT, Z_LIMIT)

    # Периоды угнетения ищутся только в вегетационный сезон. Вне его низкий
    # индекс — это снег, голая почва и растительные остатки, а не состояние
    # посева: спутниковый слой маскирует снег, поэтому зимние значения ряда
    # почти целиком экстраполяция. Организаторы по той же причине ограничили
    # свой набор апрелем-октябрём. Без этого фильтра сервис объявлял
    # критическую аномалию в январе — на поле, где просто лежал снег.
    #
    # Сам ряд и z-оценка за зиму остаются в ответе: график должен быть
    # непрерывным, скрывается только поиск событий.
    season = np.array([SEASON_MONTHS[0] <= d.month <= SEASON_MONTHS[1] for d in grid_dates])
    z_for_search = np.where(season, z, np.nan)

    observed_map = dict(zip(df["date"].dt.date, df["ndvi"]))
    source_map = dict(zip(df["date"].dt.date, df["source"]))

    series: list[SeriesPoint] = []
    for i in range(0, len(grid_dates), output_step):
        d = grid_dates[i]
        has_obs = d in observed_map
        series.append(
            SeriesPoint(
                date=d,
                observed=float(observed_map[d]) if has_obs else None,
                restored=round(float(restored[i]), 4),
                climatology_mean=None if np.isnan(clim_mean[i]) else round(float(clim_mean[i]), 4),
                climatology_std=None if np.isnan(clim_std[i]) else round(float(clim_std[i]), 4),
                zscore=None if np.isnan(z[i]) else round(float(z[i]), 2),
                is_restored=not has_obs,
                source=source_map.get(d),
            )
        )

    # Флаг «на этот день есть реальный снимок» нужен детектору: период,
    # собранный целиком из интерполяции, не подтверждён ни одним наблюдением.
    observed_flags = np.array([d in observed_map for d in grid_dates])

    anomalies = build_periods(
        grid_dates,
        z_for_search,
        inp.weather,
        crop_type=crop_type,
        norm_is_crop=(clim_kind == "crop"),
        observed=observed_flags,
    )

    _apply_agro_journal(anomalies, agro_events, crop_type)

    # Сравнение с соседями идёт после поиска периодов: по каждому найденному
    # периоду проверяется, просели ли в те же дни соседние поля своей культуры.
    peer_report = _compare_peers(df, crop_type, anomalies, peers)
    if peer_report:
        _apply_peer_scope(anomalies, peer_report)

    # Прогноз кладём в meta, а не в отдельное поле результата: контракт
    # AnalysisResult заморожен, а meta для того и существует. Слой API отдаёт
    # его как есть, интерфейс рисует пунктиром продолжение кривой.
    forecast = None
    if forecast_season is not None and clim_kind != "none":
        try:
            forecast = forecast_season(
                grid_dates, restored, clim_mean, clim_std,
                crop_type=crop_type, horizon_days=forecast_days,
                n_reference_years=clim_years or None,
                clim_source=clim_kind,
                last_observation=max(observed_map) if observed_map else None,
            )
        except Exception:  # noqa: BLE001 — прогноз не имеет права ронять анализ
            forecast = None

    score = None
    if field_score is not None and clim_kind != "none":
        try:
            score = field_score(
                grid_dates, restored, clim_mean, clim_std, z, anomalies,
                crop_type=crop_type, climatology_source=clim_kind,
                observed=observed_flags,
            )
        except Exception:  # noqa: BLE001 — оценка не имеет права ронять анализ
            score = None

    return AnalysisResult(
        polygon_id=inp.polygon_id,
        series=series,
        anomalies=anomalies,
        meta={
            "n_obs": len(df),
            "sources": sorted(set(df["source"].dropna())),
            "first_date": str(grid_dates[0]),
            "last_date": str(grid_dates[-1]),
            "climatology_years": clim_years,
            "has_climatology": clim_kind != "none",
            # Откуда взялась норма: "polygon" — собственная история поля,
            # "crop" — средняя по типу культуры (грубее, честно помечаем),
            # "none" — нормы нет, аномалии не ищутся.
            "climatology_source": clim_kind,
            # Культура, с которой работал анализ. Может отличаться от того, что
            # прислал вызывающий: если он не прислал ничего, она определена по
            # кривой. Откуда она взялась, говорит crop_source.
            "crop_type": crop_type,
            "crop_declared": inp.crop_type,
            "crop_source": crop_info.get("source"),
            # Полный разбор определения культуры: что увидено, с какой
            # уверенностью, чьей нормой меряем и спорит ли увиденное с
            # заявленным. Интерфейсу нужна и фраза note, и поле conflict.
            "crop_detection": crop_info,
            "climatology_crop": norm_crop if clim_kind == "crop" else None,
            "climatology_note": _norm_note(clim_kind, crop_type, norm_crop),
            # Сравнение с соседними полями: место поля в округе и разбор каждого
            # периода на «районное явление» против «только на этом поле». None,
            # если соседей не подали или их оказалось слишком мало.
            "peers": peer_report,
            # Прогноз развития поля вперёд от последней даты ряда. None, если
            # нормы нет или модуль прогноза недоступен — интерфейс тогда просто
            # не рисует продолжение кривой.
            "forecast": forecast,
            # Оценка поля как объекта риска: балл 0..100, буква, разбор по
            # сезонам и честные оговорки в flags. None, если нормы нет.
            "score": score,
        },
    )
