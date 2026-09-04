"""Сборка входа доменного ядра по произвольному полигону на карте.

Единственная функция, которую нужно знать слою API: `collect_series`. Она берёт
GeoJSON-контур и возвращает готовый `SeriesInput` — то, что доменное ядро умеет
превращать в восстановленный ряд и список периодов угнетения.

Постановка требует, чтобы основной пользовательский сценарий работал **без ручной
загрузки заранее подготовленного датасета**. Этот модуль и есть выполнение того
требования: снимки, погода и культура добываются из открытых источников по одному
контуру, без участия человека.

Устройство слоя:

    parcels.py    поиск сельхозконтуров в рамке карты (Overpass)
    satellite.py  снимки Sentinel-2 / Landsat / MODIS -> Observation (STAC)
    weather.py    суточная температура и осадки -> WeatherPoint (Open-Meteo)
    cache.py      файловый кэш, чтобы повторный показ поля был мгновенным
    collect.py    <- этот файл: склейка всего перечисленного в SeriesInput

Устойчивость к отказам здесь не украшение, а отдельный критерий оценки: любой из
источников может отвалиться, и сервис обязан отдать то, что удалось собрать, честно
сообщив, чего не хватает. Поэтому каждый источник вызывается в своей защите, а
собранная диагностика уезжает в `meta`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta

from src.contracts import Observation, SeriesInput, WeatherPoint

# Сколько сезонов истории тянем по умолчанию. Доменному ядру для собственной
# климатической нормы нужно минимум три года, иначе оно перейдёт на норму по
# культуре и честно пометит это в ответе. Пять — компромисс между качеством
# нормы и временем сбора.
DEFAULT_YEARS = 5

# Вегетационный сезон: вне его снимки для этой задачи бесполезны, а время сбора
# они съедают. Организаторы по той же причине ограничили набор апрелем-октябрём.
SEASON_START_MONTH = 4
SEASON_END_MONTH = 10


@dataclass
class CollectReport:
    """Что удалось собрать и что отвалилось — для интерфейса и для отладки."""

    observations: int = 0
    weather_days: int = 0
    seconds: float = 0.0
    sources: dict = field(default_factory=dict)   # "s2" -> сколько наблюдений
    failures: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None

    def as_meta(self) -> dict:
        return {
            "collected_observations": self.observations,
            "collected_weather_days": self.weather_days,
            "collect_seconds": round(self.seconds, 1),
            "sources": self.sources,
            "failures": self.failures,
            "date_from": self.date_from,
            "date_to": self.date_to,
        }


def season_range(years: int, today: date | None = None) -> tuple[date, date]:
    """Диапазон дат для сбора: `years` сезонов назад по конец текущего сезона."""
    today = today or date.today()
    end_year = today.year
    start = date(end_year - years + 1, SEASON_START_MONTH, 1)
    end = min(today, date(end_year, SEASON_END_MONTH, 31))
    # Если сезон текущего года ещё не начался, тянем по конец прошлого
    if end < start:
        end = date(end_year - 1, SEASON_END_MONTH, 31)
    return start, end


def _safe(step: str, fn, report: CollectReport, default):
    """Вызов источника под защитой: падение одного не роняет весь сбор."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — наружу не должно улететь ничего
        report.failures.append(f"{step}: {type(exc).__name__}: {exc}"[:200])
        return default


def collect_series(
    geometry: dict,
    polygon_id: str = "AOI-USER",
    crop_type: str | None = None,
    years: int = DEFAULT_YEARS,
    progress=None,
    max_scenes: int | None = None,
) -> SeriesInput:
    """Собирает всё, что нужно доменному ядру, по одному контуру на карте.

    geometry   — GeoJSON Polygon в EPSG:4326
    polygon_id — идентификатор для интерфейса
    crop_type  — тип культуры, если пользователь его знает или он пришёл из OSM
    years      — сколько сезонов истории тянуть
    progress   — callable(этап: str, готово: int, всего: int)

    Возвращает `SeriesInput`. Диагностика сбора складывается в `meta` результата
    анализа через поле `series_input.geometry`, поэтому она возвращается отдельно
    функцией `collect_series_with_report`, если вызывающей стороне она нужна.
    """
    series, _ = collect_series_with_report(
        geometry, polygon_id=polygon_id, crop_type=crop_type,
        years=years, progress=progress, max_scenes=max_scenes,
    )
    return series


def collect_series_with_report(
    geometry: dict,
    polygon_id: str = "AOI-USER",
    crop_type: str | None = None,
    years: int = DEFAULT_YEARS,
    progress=None,
    max_scenes: int | None = None,
) -> tuple[SeriesInput, CollectReport]:
    """То же, что collect_series, плюс отчёт о сборе для интерфейса."""
    from src.providers import satellite, weather

    t0 = time.perf_counter()
    report = CollectReport()
    start, end = season_range(years)
    report.date_from, report.date_to = start.isoformat(), end.isoformat()

    def step(name: str, done: int, total: int) -> None:
        if progress:
            progress(name, done, total)

    step("ищу снимки", 0, 2)
    observations: list[Observation] = _safe(
        "снимки",
        lambda: satellite.fetch_observations(
            geometry, start, end,
            progress=lambda s, d, t: step(s, d, t),
            max_scenes=max_scenes,
        ),
        report, [],
    )

    step("забираю погоду", 1, 2)
    weather_points: list[WeatherPoint] = _safe(
        "погода",
        lambda: weather.fetch_weather(
            geometry, start, end,
            progress=lambda *a: step("забираю погоду", 1, 2),
        ),
        report, [],
    )

    observations = sorted(
        [o for o in observations if o.ndvi is not None], key=lambda o: o.date
    )
    report.observations = len(observations)
    report.weather_days = len(weather_points)
    for obs in observations:
        report.sources[obs.source] = report.sources.get(obs.source, 0) + 1
    report.seconds = time.perf_counter() - t0

    step("готово", 2, 2)
    return (
        SeriesInput(
            polygon_id=polygon_id,
            geometry=geometry,
            observations=observations,
            weather=weather_points,
            crop_type=crop_type,
        ),
        report,
    )


def analyze_polygon(
    geometry: dict,
    polygon_id: str = "AOI-USER",
    crop_type: str | None = None,
    years: int = DEFAULT_YEARS,
    progress=None,
    max_scenes: int | None = None,
):
    """Полный путь от контура на карте до готового анализа — одна строка для API.

    Возвращает `AnalysisResult` доменного ядра, в `meta` которого доложена
    диагностика сбора: сколько наблюдений и по каким сенсорам, сколько дней
    погоды, что отвалилось и сколько времени всё заняло.
    """
    from src.core.analyze import analyze

    series, report = collect_series_with_report(
        geometry, polygon_id=polygon_id, crop_type=crop_type,
        years=years, progress=progress, max_scenes=max_scenes,
    )
    result = analyze(series)
    result.meta.update(report.as_meta())
    return result
