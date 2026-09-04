"""Тела запросов.

Только вход. Ответы наружу уходят обычными словарями: они собираются из
датаклассов ядра, и дублировать замороженный контракт ещё раз в pydantic
означало бы держать две правды об одной структуре.

Смысл проверок здесь — отсечь заведомо негодный запрос до того, как он займёт
поток фонового сбора на несколько минут.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.api import config


class AnalyzeRequest(BaseModel):
    """Запуск анализа по произвольному контуру, нарисованному на карте."""

    # Геометрия принимается как есть и разбирается в geometry.py: с карты она
    # приходит то Polygon, то Feature, то FeatureCollection.
    geometry: dict
    crop_type: str | None = None
    years: int = Field(default=config.DEFAULT_YEARS, ge=1, le=config.MAX_YEARS)
    # Потолок числа сцен: пользователь может попросить полную историю, заплатив
    # временем. None — взять значение по умолчанию из настроек.
    max_scenes: int | None = Field(default=None, ge=1, le=5000)
    # Имя нужно, только если контур сразу сохраняется в список участков.
    name: str | None = None
    save: bool = False


class PolygonCreate(BaseModel):
    """Сохранение контура в список участков."""

    geometry: dict
    name: str | None = None
    crop_type: str | None = None
    # Откуда контур: нарисован пользователем или выбран из найденных в OSM.
    source: str = "drawn"
    external_id: str | None = None


class PolygonPatch(BaseModel):
    """Переименование участка и уточнение культуры."""

    name: str | None = None
    crop_type: str | None = None


class PolygonAnalyzeRequest(BaseModel):
    """Повторный запуск анализа по уже сохранённому участку."""

    years: int = Field(default=config.DEFAULT_YEARS, ge=1, le=config.MAX_YEARS)
    max_scenes: int | None = Field(default=None, ge=1, le=5000)
    # Культура, если пользователь уточнил её только сейчас. None — взять из
    # сохранённой записи.
    crop_type: str | None = None


class DiscoverRequest(BaseModel):
    """Поиск сельхозконтуров. Рамка либо задана явно, либо берётся у геометрии.

    Форма с телом запроса нужна для случая, когда регион выбран поиском и его
    граница приходит полигоном: рамка из неё считается на сервере.
    """

    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    geometry: dict | None = None
    limit: int = Field(default=50, ge=1, le=300)
