"""Контракт между слоем данных и доменным ядром.

Владелец файла: Дмитрий. После первого коммита структуры НЕ МЕНЯЮТСЯ —
на них уже опирается слой провайдеров и API. Любое изменение согласуется в чате.

Слой провайдеров (зона Никиты) собирает SeriesInput.
Доменное ядро (зона Дмитрия) превращает его в AnalysisResult.
API вызывает ровно одну функцию — analyze(). Больше ничего из ядра наружу не торчит.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# Классы состояния растительности по z-оценке (пороги из постановки задачи)
SEVERITY_NORMAL = "normal"        # z >= -1   — штатное развитие
SEVERITY_SUPPRESSION = "suppression"  # -2 <= z < -1 — угнетение биомассы
SEVERITY_CRITICAL = "critical"    # z < -2    — критическая аномалия

# Версии причины аномалии
CAUSE_DROUGHT = "drought"          # дефицит осадков
CAUSE_HEAT = "heat"                # температурный стресс
CAUSE_EXCESS_WATER = "excess_water"  # переувлажнение
CAUSE_ABRUPT = "abrupt"            # резкое одномоментное событие (уборка, повреждение)
CAUSE_UNKNOWN = "unknown"


@dataclass
class Observation:
    """Одно спутниковое наблюдение по полигону, уже агрегированное по контуру."""
    date: date
    ndvi: float | None
    evi: float | None = None
    ndwi: float | None = None
    source: str = "unknown"  # "s2" | "landsat" | "modis"


@dataclass
class WeatherPoint:
    """Суточная метеосводка по центроиду полигона."""
    date: date
    temp_c: float | None = None
    precip_mm: float | None = None


@dataclass
class SeriesInput:
    """Что слой провайдеров отдаёт доменному ядру."""
    polygon_id: str
    geometry: dict | None = None          # GeoJSON Polygon
    observations: list[Observation] = field(default_factory=list)
    weather: list[WeatherPoint] = field(default_factory=list)
    crop_type: str | None = None


@dataclass
class SeriesPoint:
    """Точка итогового ряда. Именно это рисует фронтенд."""
    date: date
    observed: float | None        # исходное наблюдение, если оно было
    restored: float               # восстановленное значение
    climatology_mean: float | None
    climatology_std: float | None
    zscore: float | None
    is_restored: bool             # True, если наблюдения на эту дату не было
    source: str | None            # сенсор-источник наблюдения


@dataclass
class AnomalyPeriod:
    """Найденный период нетипичного поведения растительности."""
    start: date
    end: date
    severity: str
    duration_days: int
    min_zscore: float
    mean_zscore: float
    cause: str = CAUSE_UNKNOWN
    cause_confidence: float = 0.0
    evidence: dict = field(default_factory=dict)
    explanation: str = ""         # готовая фраза для интерфейса, по-русски


@dataclass
class AnalysisResult:
    """Единственный ответ ядра наружу."""
    polygon_id: str
    series: list[SeriesPoint] = field(default_factory=list)
    anomalies: list[AnomalyPeriod] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
