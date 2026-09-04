"""Протокол источников данных.

Заготовка, дальше владелец файла — Никита. Смысл слоя: доменное ядро не знает,
откуда взялись наблюдения, а провайдеры не знают, что с ними будут делать.
Любой источник можно подменить, не трогая ядро.

Требование рубрики: сервис должен работать при недоступности части источников.
Поэтому каждый провайдер обязан отвечать на is_available() и не падать, а возвращать
пустой список, если данных нет.
"""
from __future__ import annotations

from datetime import date
from typing import Protocol

from src.contracts import Observation, WeatherPoint


class SatelliteProvider(Protocol):
    """Источник спутниковых наблюдений по контуру."""

    name: str

    def is_available(self) -> bool:
        """Быстрая проверка живости источника, без тяжёлых запросов."""
        ...

    def fetch(self, geometry: dict, start: date, end: date) -> list[Observation]:
        """Наблюдения по полигону за период.

        geometry — GeoJSON Polygon в WGS84.
        Внутри обязательно: маскирование облаков и агрегация по контуру (медиана
        по пикселям внутри полигона). Наружу отдаются уже готовые индексы.
        """
        ...


class WeatherProvider(Protocol):
    """Источник метеоданных по центроиду полигона."""

    name: str

    def is_available(self) -> bool: ...

    def fetch(self, geometry: dict, start: date, end: date) -> list[WeatherPoint]: ...


class FieldProvider(Protocol):
    """Источник готовых сельскохозяйственных контуров."""

    name: str

    def is_available(self) -> bool: ...

    def discover(self, bbox: tuple[float, float, float, float], limit: int = 200) -> list[dict]:
        """Контуры полей в прямоугольнике карты.

        bbox — (min_lon, min_lat, max_lon, max_lat).
        Возвращает список GeoJSON Feature с полями id и geometry.
        """
        ...
