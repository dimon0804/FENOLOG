"""Разбор и проверка геометрии, пришедшей с карты.

Фронтенд рисует полигоны разными инструментами, и наружу они выходят по-разному:
голым GeoJSON-объектом, Feature с полем properties, иногда FeatureCollection из
одного элемента. Ядру и слою сбора нужен один вид — Polygon или MultiPolygon,
поэтому вход приводится к нему в одном месте, а не в каждом обработчике.

Здесь же проверки, которые должны сработать до запуска фоновой задачи: пустая
или вывернутая геометрия обязана дать понятный отказ сразу, а не через две
минуты сбора и падение где-то в rasterio.
"""
from __future__ import annotations

import math

# Полигон размером с область собирать бессмысленно: снимков сотни, медиана по
# такому контуру не описывает ни одно поле, а ждать пользователь будет минутами.
# Предел выбран с запасом — крупные хозяйства укладываются свободно.
MAX_AREA_HA = 200_000.0

# Совсем мелкий контур меньше пикселя Sentinel-2 (10 м) не даст ни одного
# валидного значения — лучше сказать об этом сразу.
MIN_AREA_HA = 0.05


class GeometryError(ValueError):
    """Геометрия непригодна для анализа. Текст уходит пользователю как есть."""


def normalize_geometry(raw: object) -> dict:
    """Любой разумный GeoJSON с карты -> Polygon или MultiPolygon.

    Бросает GeometryError с русским текстом: сообщение показывается в интерфейсе.
    """
    if not isinstance(raw, dict):
        raise GeometryError("Геометрия должна быть объектом GeoJSON")

    kind = raw.get("type")

    if kind == "FeatureCollection":
        features = raw.get("features") or []
        if not features:
            raise GeometryError("В FeatureCollection нет ни одного контура")
        if len(features) > 1:
            raise GeometryError("Ожидается один контур, а не набор")
        return normalize_geometry(features[0])

    if kind == "Feature":
        return normalize_geometry(raw.get("geometry"))

    if kind not in ("Polygon", "MultiPolygon"):
        raise GeometryError(
            f"Поддерживаются только Polygon и MultiPolygon, получено: {kind!r}"
        )

    coordinates = raw.get("coordinates")
    if not coordinates:
        raise GeometryError("У контура нет координат")

    rings = coordinates if kind == "Polygon" else [r for poly in coordinates for r in poly]
    for ring in rings:
        if not isinstance(ring, (list, tuple)) or len(ring) < 4:
            raise GeometryError("Контур должен состоять минимум из трёх точек")
        for point in ring:
            _check_point(point)

    return {"type": kind, "coordinates": coordinates}


def _check_point(point: object) -> None:
    """Точка внутри допустимых широт и долгот.

    Перепутанные местами широта и долгота — самая частая ошибка при ручной
    подготовке GeoJSON, и она даёт не отказ, а пустой ряд где-то в Атлантике.
    Ловим её здесь: широта больше 90 невозможна физически.
    """
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise GeometryError("Точка контура должна быть парой [долгота, широта]")
    lon, lat = float(point[0]), float(point[1])
    if not (-180.0 <= lon <= 180.0):
        raise GeometryError(f"Долгота вне допустимого диапазона: {lon}")
    if not (-90.0 <= lat <= 90.0):
        raise GeometryError(
            f"Широта вне допустимого диапазона: {lat}. "
            "Вероятно, координаты переставлены местами — в GeoJSON порядок [долгота, широта]"
        )


def iter_points(geometry: dict):
    """Все точки контура подряд, независимо от вложенности колец."""
    coordinates = geometry["coordinates"]
    rings = coordinates if geometry["type"] == "Polygon" else [
        r for poly in coordinates for r in poly
    ]
    for ring in rings:
        for point in ring:
            yield float(point[0]), float(point[1])


def bbox_of(geometry: dict) -> tuple[float, float, float, float]:
    """Охватывающая рамка (запад, юг, восток, север)."""
    lons, lats = zip(*iter_points(geometry))
    return min(lons), min(lats), max(lons), max(lats)


def centroid_of(geometry: dict) -> tuple[float, float]:
    """Центр охватывающей рамки. Для площади погоды по центроиду этого хватает."""
    west, south, east, north = bbox_of(geometry)
    return (west + east) / 2.0, (south + north) / 2.0


def area_ha(geometry: dict) -> float:
    """Площадь контура в гектарах.

    Считаем по формуле шнурков в градусах и переводим в метры через поправку на
    широту: cos(широта) для долготы, постоянные 111 320 м на градус для широты.
    Точного перепроецирования здесь не нужно — площадь показывается пользователю
    справочно, а ошибка такого приближения на размере поля меньше процента.
    """
    coordinates = geometry["coordinates"]
    polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates

    total = 0.0
    for polygon in polygons:
        # Первое кольцо — внешнее, остальные дырки: вычитаем их.
        for index, ring in enumerate(polygon):
            signed = _ring_area_m2(ring)
            total += signed if index == 0 else -signed
    return abs(total) / 10_000.0


def _ring_area_m2(ring) -> float:
    points = [(float(p[0]), float(p[1])) for p in ring]
    mean_lat = sum(y for _, y in points) / len(points)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(mean_lat))

    # Переводим градусы в метры до подсчёта площади, а не после: множители по
    # осям разные, и вынести их за скобки не получилось бы.
    metric = [(x * m_per_deg_lon, y * m_per_deg_lat) for x, y in points]

    area = 0.0
    for (x1, y1), (x2, y2) in zip(metric, metric[1:] + metric[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _ha(value: float) -> str:
    """Гектары с пробелом между разрядами — так число читается с одного взгляда."""
    return f"{value:,.0f}".replace(",", " ")


def validate_for_analysis(geometry: dict) -> dict:
    """Полная проверка контура перед постановкой задачи на анализ."""
    geometry = normalize_geometry(geometry)
    area = area_ha(geometry)
    if area > MAX_AREA_HA:
        raise GeometryError(
            f"Контур слишком велик: {_ha(area)} га при пределе {_ha(MAX_AREA_HA)} га. "
            "Выделите отдельное поле, а не весь район"
        )
    if area < MIN_AREA_HA:
        raise GeometryError(
            "Контур меньше одного пикселя снимка — по нему не получится посчитать индекс"
        )
    return geometry


def parse_bbox(raw: str) -> tuple[float, float, float, float]:
    """Рамка карты из строки запроса «запад,юг,восток,север»."""
    parts = [p.strip() for p in (raw or "").split(",")]
    if len(parts) != 4:
        raise GeometryError("bbox задаётся четырьмя числами: запад,юг,восток,север")
    try:
        west, south, east, north = (float(p) for p in parts)
    except ValueError:
        raise GeometryError("bbox должен состоять из чисел") from None
    if west >= east or south >= north:
        raise GeometryError("Углы bbox перепутаны: нужно запад,юг,восток,север")
    _check_point((west, south))
    _check_point((east, north))
    return west, south, east, north
