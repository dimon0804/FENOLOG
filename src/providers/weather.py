"""Суточная погода по центроиду полигона — Open-Meteo Archive (реанализ ERA5).

Почему именно этот источник. Доменному ядру нужна не текущая погода, а длинная
однородная история: версия причины аномалии выбирается сравнением погоды за
период отклонения с нормой по дню года, а норма считается по прошлым годам того
же поля. Прогнозные API дают несколько дней и не годятся. Open-Meteo Archive
раздаёт реанализ ERA5 с 1940 года, без ключа и без лимита на регион — это ровно
тот же продукт, из которого собраны колонки era5_temp_c / era5_precip_mm в
выданном наборе, то есть живые данные и офлайн-набор физически согласованы.

Ключевые решения:

* Берётся центроид контура, а не среднее по пикселям. Шаг сетки ERA5-Land около
  9 км, типовое поле — сотни метров: внутри одной ячейки полигон помещается
  целиком, и усреднение по контуру дало бы то же число при N запросах вместо
  одного.
* Пропуск в ответе (null) превращается в None в WeatherPoint, а не в исключение
  и не в ноль. Ноль здесь опасен: для осадков это осмысленное значение «сухо»,
  и подмена пропуска нулём напрямую исказила бы версию «дефицит влаги».
* Дни, которых API не вернул совсем, не выдумываются: ряд отдаётся как есть,
  ядро само работает по датам.
"""
from __future__ import annotations

from datetime import date, timedelta

import requests

from src.contracts import WeatherPoint
from src.providers.cache import cached

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Реанализ отстаёт от реального времени: последние несколько суток ещё не
# посчитаны. Запрос за этот хвост возвращает ошибку целиком, а не частичный ряд,
# поэтому конец диапазона подрезаем сами — лучше отдать на пять дней меньше,
# чем не отдать ничего.
ARCHIVE_LAG_DAYS = 5

# Порог, после которого диапазон режется на куски.
#
# Значение выставлено по замеру, а не наугад, и первая версия была неправильной.
# Изначально стояло 6 лет из осторожности «длинный запрос не пройдёт». Замер с
# этой машины: один запрос на 16 лет — 0,9 с, на 26 лет — 0,8 с, на 46 лет —
# 1,0 с (16 802 суток). То есть время ответа от длины диапазона практически не
# зависит: платим за соединение, а не за объём. Зато нарезка на 16 кусков по
# годам стоила 13,3 с вместо 0,9 — в четырнадцать раз дороже на ровном месте.
# Поэтому режем только совсем экстремальные диапазоны, а нарезка по годам живёт
# ниже как запасной путь: если единый запрос всё же не прошёл, годы добираются
# по одному и сбой стоит одного года, а не всей истории.
CHUNK_THRESHOLD_DAYS = 366 * 50
REQUEST_TIMEOUT = 60


def _centroid(geometry: dict) -> tuple[float, float]:
    """Центроид GeoJSON-геометрии -> (lat, lon).

    shapely считает центроид в градусах как в плоскости. Для полигона размером с
    поле искажение проекции — доли метра, метрическое перепроецирование ради
    точки запроса к сетке в 9 км было бы бессмысленной работой.
    """
    from shapely.geometry import shape

    geom = shape(geometry)
    if not geom.is_valid:
        geom = geom.buffer(0)  # самопересечения в нарисованных пользователем контурах
    point = geom.centroid
    if point.is_empty:
        raise ValueError("пустая геометрия: центроид не определён")
    return float(point.y), float(point.x)


def _split_by_year(start: date, end: date) -> list[tuple[date, date]]:
    """Разрезать диапазон по календарным годам, сохранив исходные края."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        year_end = date(cursor.year, 12, 31)
        chunk_end = min(year_end, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _request_range(lat: float, lon: float, start: date, end: date) -> dict:
    """Один запрос к архиву. Возвращает блок daily или пустой словарь при отказе."""
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "UTC",  # даты снимков тоже в UTC — иначе ряды разъедутся на сутки
    }
    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        # Сбой куска не должен ронять весь ряд: остальные годы приедут, а по
        # выпавшему просто не будет точек. Требование рубрики — сервис живёт
        # при частичной недоступности источников.
        return {}
    return payload.get("daily") or {}


def _to_points(daily: dict) -> list[WeatherPoint]:
    """Разложить параллельные массивы Open-Meteo в список WeatherPoint."""
    times = daily.get("time") or []
    temps = daily.get("temperature_2m_mean") or []
    precs = daily.get("precipitation_sum") or []

    points: list[WeatherPoint] = []
    for i, day in enumerate(times):
        try:
            parsed = date.fromisoformat(day)
        except (TypeError, ValueError):
            continue
        temp = temps[i] if i < len(temps) else None
        prec = precs[i] if i < len(precs) else None
        points.append(
            WeatherPoint(
                date=parsed,
                temp_c=float(temp) if temp is not None else None,
                precip_mm=float(prec) if prec is not None else None,
            )
        )
    return points


@cached("weather_openmeteo", ttl_days=30)
def fetch_weather(geometry: dict, start: date, end: date, progress=None) -> list[WeatherPoint]:
    """Суточная погода по центроиду полигона: температура и осадки.

    geometry — GeoJSON Polygon в WGS84. Диапазон запрашивается целиком: чем
    длиннее история, тем устойчивее климатическая норма по дню года, на которую
    опирается объяснение причины.

    progress — необязательный колбэк (доля 0..1, текст), чтобы веб-интерфейс
    показывал ход загрузки многолетнего ряда, а не молчал полминуты.
    """

    def report(done: float, message: str) -> None:
        if progress is not None:
            try:
                progress(done, message)
            except Exception:
                pass  # сломанный индикатор прогресса не повод терять данные

    if start > end:
        start, end = end, start

    # Подрезаем хвост, которого в реанализе ещё нет.
    limit = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    if end > limit:
        end = limit
    if start > end:
        report(1.0, "запрошенный период целиком в будущем для реанализа")
        return []

    try:
        lat, lon = _centroid(geometry)
    except Exception:
        report(1.0, "не удалось определить центроид полигона")
        return []

    span_days = (end - start).days + 1
    chunks = [(start, end)] if span_days <= CHUNK_THRESHOLD_DAYS else _split_by_year(start, end)

    points: list[WeatherPoint] = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        report(i / len(chunks), f"погода {chunk_start.year}")
        points.extend(_to_points(_request_range(lat, lon, chunk_start, chunk_end)))

    # Если единственный большой запрос не прошёл — вторая попытка кусками.
    # Часто длинный диапазон отваливается по таймауту, а по годам проходит.
    if not points and len(chunks) == 1 and span_days > 366:
        for i, (chunk_start, chunk_end) in enumerate(_split_by_year(start, end)):
            report(i / max(1, end.year - start.year + 1), f"повтор: погода {chunk_start.year}")
            points.extend(_to_points(_request_range(lat, lon, chunk_start, chunk_end)))

    # Куски склеены встык, но при повторе диапазоны могут пересечься — страхуемся
    # от дублей по дате и заодно гарантируем ядру строгий календарный порядок.
    unique: dict[date, WeatherPoint] = {}
    for point in points:
        unique[point.date] = point
    result = [unique[day] for day in sorted(unique)]

    report(1.0, f"погода: {len(result)} суток")
    return result


def is_available() -> bool:
    """Быстрая проверка живости источника — один короткий запрос без разбора данных."""
    try:
        probe = date.today() - timedelta(days=30)
        response = requests.get(
            ARCHIVE_URL,
            params={
                "latitude": 47.2,
                "longitude": 39.7,
                "start_date": probe.isoformat(),
                "end_date": probe.isoformat(),
                "daily": "temperature_2m_mean",
                "timezone": "UTC",
            },
            timeout=15,  # 10 с иногда не хватало на холодное TLS-соединение
        )
        return response.ok
    except Exception:
        return False
