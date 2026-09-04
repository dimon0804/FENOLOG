"""Поиск периодов угнетения растительности и версия причины.

Пороги z-оценки взяты из постановки задачи:
    z >= -1          штатное развитие
    -2 <= z < -1     угнетение биомассы
    z < -2           критическая аномалия

Одиночная выпавшая точка аномалией не считается: при шуме наблюдения около 0.07
это дало бы поток ложных тревог. Период засчитывается только при устойчивости.
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

# Минимальная длительность периода, при которой он считается аномалией
MIN_DURATION_DAYS = 7
# Окно накопления осадков для проверки версии засухи
PRECIP_WINDOW_DAYS = 30
# Доля от нормы осадков, ниже которой говорим о дефиците влаги
DROUGHT_RATIO = 0.5
# Превышение нормы осадков, выше которого говорим о переувлажнении
EXCESS_WATER_RATIO = 2.0
# Отклонение средней температуры, при котором говорим о температурном стрессе
HEAT_ANOMALY_C = 2.5
# Падение z-оценки за короткий срок, которое считаем резким событием
ABRUPT_DROP_Z = 2.0
ABRUPT_WINDOW_DAYS = 10


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
) -> list[tuple[int, int, str]]:
    """Находит непрерывные отрезки отрицательного отклонения.

    Возвращает список (индекс начала, индекс конца включительно, класс серьёзности).
    Класс отрезка определяется по самой глубокой точке внутри него.
    """
    z = np.asarray(z, dtype=float)
    below = np.isfinite(z) & (z < -1.0)

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(below):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(below) - 1))

    out = []
    for a, b in spans:
        duration = (dates[b] - dates[a]).days + 1
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


def attribute_cause(
    start: date,
    end: date,
    z_series: np.ndarray,
    dates: list[date],
    weather: list[WeatherPoint],
) -> tuple[str, float, dict, str]:
    """Определяет наиболее вероятную причину периода.

    Правила проверяются по убыванию убедительности. Возвращает
    (причина, уверенность от 0 до 1, свидетельства, готовая фраза для интерфейса).
    """
    evidence: dict = {}

    # Версия первая: резкое одномоментное падение — уборка, потрава, повреждение
    idx = [i for i, d in enumerate(dates) if start <= d <= end]
    if idx:
        head = idx[0]
        back = max(0, head - ABRUPT_WINDOW_DAYS)
        before = np.nanmax(z_series[back : head + 1]) if head > back else np.nan
        after = np.nanmin(z_series[head : min(head + 3, len(z_series))])
        drop = float(before - after) if np.isfinite(before) and np.isfinite(after) else np.nan
        if np.isfinite(drop):
            evidence["z_drop_10d"] = round(drop, 2)
        if np.isfinite(drop) and drop >= ABRUPT_DROP_Z:
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
    temp_norm = float(hist["temp_c"].mean()) if not hist.empty else np.nan

    if np.isfinite(precip_actual):
        evidence["precip_30d_mm"] = round(precip_actual, 1)
    if np.isfinite(precip_norm):
        evidence["precip_30d_norm_mm"] = round(precip_norm, 1)
    if np.isfinite(temp_actual):
        evidence["temp_mean_c"] = round(temp_actual, 1)
    if np.isfinite(temp_norm):
        evidence["temp_norm_c"] = round(temp_norm, 1)

    ratio = precip_actual / precip_norm if np.isfinite(precip_norm) and precip_norm > 0 else np.nan
    temp_anom = (
        temp_actual - temp_norm if np.isfinite(temp_actual) and np.isfinite(temp_norm) else np.nan
    )
    if np.isfinite(ratio):
        evidence["precip_ratio"] = round(float(ratio), 2)
    if np.isfinite(temp_anom):
        evidence["temp_anomaly_c"] = round(float(temp_anom), 1)

    # Версия вторая: дефицит влаги
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

    # Версия третья: температурный стресс при достаточном увлажнении
    if np.isfinite(temp_anom) and temp_anom > HEAT_ANOMALY_C:
        return (
            CAUSE_HEAT,
            0.6,
            evidence,
            "Средняя температура за период выше нормы на {:.1f} градуса при достаточном "
            "увлажнении. Вероятен температурный стресс.".format(temp_anom),
        )

    # Версия четвёртая: переувлажнение
    if np.isfinite(ratio) and ratio > EXCESS_WATER_RATIO:
        return (
            CAUSE_EXCESS_WATER,
            0.55,
            evidence,
            "Осадков выпало в {:.1f} раза больше нормы. Возможно переувлажнение "
            "и подтопление посевов.".format(ratio),
        )

    return (
        CAUSE_UNKNOWN,
        0.2,
        evidence,
        "Отклонение от нормы устойчивое, но погодных причин не найдено. Нужен осмотр поля: "
        "возможны болезни, вредители или нарушения агротехники.",
    )


def build_periods(
    dates: list[date],
    z: np.ndarray,
    weather: list[WeatherPoint],
) -> list[AnomalyPeriod]:
    """Полный проход: находит периоды и объясняет каждый."""
    out: list[AnomalyPeriod] = []
    for a, b, severity in find_periods(dates, z):
        seg = z[a : b + 1]
        cause, conf, evidence, text = attribute_cause(dates[a], dates[b], z, dates, weather)
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
