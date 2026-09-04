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

from src.contracts import AnalysisResult, SeriesInput, SeriesPoint
from src.core.anomaly import build_periods
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

# Прогноз развития поля вперёд. Заказчик кейса сформулировал прямо: «прогнозы
# должна давать программа». Модуль опирается на климатическую норму и переносит
# текущее отклонение с затуханием; полезный горизонт около 30 дней, дальше
# прогноз сходится к норме — это измерено ретроспективно, а не предположено.
try:  # pragma: no cover
    from src.core.forecast import forecast_season  # type: ignore
except ImportError:  # pragma: no cover
    forecast_season = None  # type: ignore

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


def analyze(inp: SeriesInput, output_step: int = OUTPUT_STEP_DAYS,
            agro_events: list | None = None,
            forecast_days: int = 30) -> AnalysisResult:
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

    t_days = df["date"].map(pd.Timestamp.toordinal).to_numpy()
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
        crop = _crop_norm(inp.crop_type, doys)
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
        crop_type=inp.crop_type,
        norm_is_crop=(clim_kind == "crop"),
        observed=observed_flags,
    )

    _apply_agro_journal(anomalies, agro_events, inp.crop_type)

    # Прогноз кладём в meta, а не в отдельное поле результата: контракт
    # AnalysisResult заморожен, а meta для того и существует. Слой API отдаёт
    # его как есть, интерфейс рисует пунктиром продолжение кривой.
    forecast = None
    if forecast_season is not None and clim_kind != "none":
        try:
            forecast = forecast_season(
                grid_dates, restored, clim_mean, clim_std,
                crop_type=inp.crop_type, horizon_days=forecast_days,
                n_reference_years=clim_years or None,
                clim_source=clim_kind,
                last_observation=max(observed_map) if observed_map else None,
            )
        except Exception:  # noqa: BLE001 — прогноз не имеет права ронять анализ
            forecast = None

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
            "crop_type": inp.crop_type,
            # Прогноз развития поля вперёд от последней даты ряда. None, если
            # нормы нет или модуль прогноза недоступен — интерфейс тогда просто
            # не рисует продолжение кривой.
            "forecast": forecast,
        },
    )
