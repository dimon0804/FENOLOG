"""Спутниковые наблюдения NDVI по произвольному контуру — живой сбор из открытых архивов.

Зачем этот файл. Постановка требует, чтобы сервис работал по любому полю на карте,
а не по заранее выгруженному датасету. Значит нужно уметь за приемлемое время
превратить GeoJSON-полигон и диапазон дат в ряд наблюдений — ровно тот контракт,
который дальше ест доменное ядро (`src.contracts.Observation`).

Как устроено, коротко:

    STAC-поиск сцен  ->  чтение ТОЛЬКО окна под полигоном  ->  маска облаков
    ->  маска контура  ->  медиана по пикселям  ->  склейка сенсоров по дате

Ключевые решения и их причины расписаны у соответствующих функций. Три из них
определяют, будет ли результат вообще осмысленным:

1. Окно вместо сцены. Сцена Sentinel-2 — это 10980x10980 пикселей на канал, около
   200 МБ. Поле 500x500 м — это 50x50 пикселей. Читаем через HTTP range-запросы
   ровно нужный прямоугольник COG: секунда вместо минуты, и так по каждой из сотни
   сцен сезона.
2. Маска облаков. Без неё ряд NDVI превращается в шум: облако даёт NDVI около нуля,
   тень — заниженное значение, и «периоды угнетения» будут находиться на погоде,
   а не на растительности.
3. Масштабирование. Sentinel-2 L2A с baseline 04.00 (январь 2022) хранит отражение
   со сдвигом -1000. Если его не вычесть, NDVI поедет тем сильнее, чем темнее
   поверхность, — и сезонная кривая сломается на стыке 2021/2022 годов.

Устойчивость. Ни одна ошибка внутри не выходит наружу: недоступная коллекция,
битый ассет, таймаут, просроченная подпись — всё логируется и пропускается.
Функция возвращает то, что удалось собрать; пустой список — допустимый ответ.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from src.contracts import Observation

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Настройки источника
# --------------------------------------------------------------------------------------

# Planetary Computer: единственный из трёх каталогов, где Sentinel-2, Landsat и MODIS
# лежат рядом, отдаются как COG и не требуют регистрации. Copernicus Data Space и
# AWS Element84 держим как запасные (см. STAC_FALLBACKS) — у них другие имена ассетов,
# поэтому все обращения к ассетам идут через словари синонимов ниже.
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
STAC_FALLBACKS = ("https://earth-search.aws.element84.com/v1",)

COLLECTION_S2 = "sentinel-2-l2a"
COLLECTION_LANDSAT = "landsat-c2-l2"
COLLECTION_MODIS = "modis-13Q1-061"

# Приоритет сенсоров при склейке одной даты. Ровно так собран `primary_ndvi`
# в наборе организаторов (проверено: при наличии s2_ndvi он совпадает точно),
# и доменное ядро взвешивает наблюдения по этой же шкале 1 / 0,8 / 0,5.
SOURCE_PRIORITY = {"s2": 0, "landsat": 1, "modis": 2}

# Порог годности сцены: доля незамаскированных пикселей внутри контура.
# Меньше — значит поле закрыто облаком почти целиком, и медиана по остатку
# считается по краю облака, то есть по мусору. Такую сцену честнее выбросить.
MIN_VALID_FRACTION = 0.20

# Сцены, где облачность выше этого порога по всему снимку, не открываем вообще:
# шанс, что именно наш контур чист, мизерный, а открытие стоит времени.
# Порог намеренно щадящий — настоящая фильтрация всё равно попиксельная.
MAX_SCENE_CLOUD_COVER = 95.0

# 6-8 потоков: узкое место — сетевая задержка range-запросов, а не CPU и не канал.
# Больше потоков упираются в троттлинг хранилища и начинают давать таймауты.
DOWNLOAD_WORKERS = 7

# Общий бюджет времени на скачивание. Веб-запрос не может висеть вечно:
# что успели — то и отдаём, это лучше, чем ошибка в интерфейсе.
TOTAL_TIMEOUT_S = 900.0

# Верхний предел на размер читаемого окна. Полигон может оказаться не полем,
# а целым районом; тогда читаем прореженно, а не валимся по памяти и времени.
# 400 000 — это 632x632 пикселя на контур: для медианы по площади с запасом,
# а поле обычного размера (50x50 пикселей) сюда даже близко не подходит.
MAX_WINDOW_PIXELS = 400_000

# Настройки GDAL для чтения COG по сети. Без GDAL_DISABLE_READDIR_ON_OPEN каждый
# open() дополнительно листает «каталог» контейнера — это десятки лишних запросов.
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.jp2,.vrt",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_CONNECTTIMEOUT": "20",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "33554432",
}

# Синонимы ассетов: Planetary Computer называет каналы Sentinel-2 как B04/B08,
# Element84 — red/nir. Перебираем варианты, чтобы код пережил смену каталога.
S2_ASSETS = {
    "red": ("red", "B04"),
    "nir": ("nir", "B08"),
    "blue": ("blue", "B02"),
    "green": ("green", "B03"),
    "scl": ("SCL", "scl"),
}
LANDSAT_ASSETS = {
    "red": ("red",),
    "nir": ("nir08", "nir"),
    "blue": ("blue",),
    "green": ("green",),
    "qa": ("qa_pixel",),
}
MODIS_ASSETS = {
    "ndvi": ("250m_16_days_NDVI",),
    "evi": ("250m_16_days_EVI",),
    "qa": ("250m_16_days_pixel_reliability",),
}

# Классы Scene Classification Layer у Sentinel-2 L2A, которые нельзя пускать в медиану.
# 0 — нет данных, 1 — насыщение/дефект, 3 — тень облака, 8/9 — облако средней и высокой
# вероятности, 10 — перистые, 11 — снег. Снег добавлен к обязательному списку намеренно:
# NDVI по снегу — это не измерение растительности, а ноль, и в ряду он выглядит как
# внезапный обвал биомассы, то есть провоцирует ложную «критическую аномалию».
SCL_BAD = (0, 1, 3, 8, 9, 10, 11)

# Биты qa_pixel у Landsat Collection 2 Level-2:
# 0 — заполнитель за краем снимка, 1 — расширенное облако, 3 — облако, 4 — тень облака,
# 2 — перистые, 5 — снег/лёд.
# Биты 0, 2 и 5 в минимальном списке не значатся, но без них ряд ломается измеримо:
# на контрольном поле сцена LC08 от 25.10.2025 давала NDVI -0,023 при соседних значениях
# 0,25-0,29 — её пиксели помечены cirrus + snow и не помечены cloud. Бит 0 отсекает нули
# за краем снимка, которые иначе утаскивают медиану вниз.
LANDSAT_QA_BITS = (0, 1, 2, 3, 4, 5)

# Пригодность пикселя MOD13Q1: 0 — хорошие данные, 1 — пригодные с оговоркой,
# 2 — снег/лёд, 3 — облако, -1 — заполнитель. Берём только 0 и 1.
MODIS_RELIABILITY_OK = (0, 1)

ProgressFn = Callable[[str, int, int], None]


# --------------------------------------------------------------------------------------
# Публичный интерфейс
# --------------------------------------------------------------------------------------

def fetch_observations(
    geometry: dict,
    start: date,
    end: date,
    progress: ProgressFn | None = None,
    max_scenes: int | None = None,
) -> list[Observation]:
    """Наблюдения NDVI по контуру, отсортированные по дате.

    geometry  — GeoJSON Polygon/MultiPolygon (или Feature) в EPSG:4326.
    progress  — callable(этап, готово, всего); вызывается по мере скачивания.
    max_scenes— ограничение числа сцен для быстрой демонстрации; сцены при этом
                прореживаются равномерно по времени, чтобы сезонная кривая
                осталась узнаваемой, а не обрезалась по началу периода.

    Никогда не бросает исключений: при полном отказе источников возвращает [].
    """
    t0 = time.perf_counter()
    try:
        shape = _to_shapely(geometry)
    except Exception as exc:  # некорректная геометрия — не повод падать наружу
        log.error("satellite: не разобрал геометрию: %s", exc)
        return []

    if shape is None or shape.is_empty:
        log.error("satellite: пустая геометрия")
        return []

    bbox = tuple(shape.bounds)
    _emit(progress, "ищу сцены", 0, 1)

    items = _search_all(bbox, start, end)
    if not items:
        log.warning("satellite: сцен не найдено за %s..%s", start, end)
        _emit(progress, "скачиваю сцены", 0, 0)
        return []

    items = _thin_scenes(items, max_scenes)
    total = len(items)
    log.info("satellite: к обработке %d сцен", total)

    done = 0
    lock = threading.Lock()
    collected: list[tuple[Observation, int]] = []
    _emit(progress, "скачиваю сцены", 0, total)

    # Параллелим по сценам: каждая сцена — независимая пачка range-запросов,
    # ошибка одной не должна касаться остальных.
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(_process_item, it, shape): it for it in items}
        try:
            for fut in as_completed(futures, timeout=TOTAL_TIMEOUT_S):
                item = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:  # страховка: worker и так ловит всё сам
                    log.warning("satellite: сцена %s упала: %s", _safe_id(item), exc)
                    result = None
                if result is not None:
                    collected.append(result)
                with lock:
                    done += 1
                    _emit(progress, "скачиваю сцены", done, total)
        except TimeoutError:
            log.warning(
                "satellite: исчерпан бюджет %.0f с, обработано %d из %d сцен",
                TOTAL_TIMEOUT_S, done, total,
            )
            for fut in futures:
                fut.cancel()

    # 16-дневный композит MODIS может начинаться до `start` и заканчиваться после `end`:
    # STAC отдаёт его по пересечению интервалов. Обрезаем ряд запрошенным периодом,
    # иначе ядро получит наблюдения за пределами окна анализа.
    observations = [o for o in _merge_by_date(collected) if start <= o.date <= end]
    log.info(
        "satellite: %d наблюдений из %d сцен за %.1f с",
        len(observations), total, time.perf_counter() - t0,
    )
    return observations


class PlanetaryComputerSatelliteProvider:
    """Обёртка под протокол `src.providers.base.SatelliteProvider`.

    Нужна API-слою: он работает со списком провайдеров единообразно и должен уметь
    спросить живость источника до того, как повесит пользователя на минуту ожидания.
    """

    name = "planetary-computer"

    def __init__(self, max_scenes: int | None = None) -> None:
        self.max_scenes = max_scenes

    def is_available(self) -> bool:
        """Лёгкая проверка: только корневой документ STAC, без поиска сцен."""
        try:
            import requests

            r = requests.get(STAC_URL, timeout=8)
            return r.status_code == 200
        except Exception as exc:
            log.warning("satellite: каталог недоступен: %s", exc)
            return False

    def fetch(self, geometry: dict, start: date, end: date) -> list[Observation]:
        return fetch_observations(geometry, start, end, max_scenes=self.max_scenes)


# --------------------------------------------------------------------------------------
# Геометрия
# --------------------------------------------------------------------------------------

def _to_shapely(geometry: dict):
    """GeoJSON -> shapely. Терпим Feature и FeatureCollection: фронтенд рисует по-разному."""
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    if geometry is None:
        return None
    gtype = geometry.get("type")
    if gtype == "Feature":
        return _to_shapely(geometry["geometry"])
    if gtype == "FeatureCollection":
        parts = [_to_shapely(f) for f in geometry.get("features", [])]
        parts = [p for p in parts if p is not None and not p.is_empty]
        return unary_union(parts) if parts else None
    return shapely_shape(geometry)


def _project_shape(shape, dst_crs):
    """Полигон из WGS84 в проекцию сцены (UTM у Sentinel-2/Landsat, синусоидальная у MODIS).

    Без этого шага окно чтения посчиталось бы по градусам в метровой сетке и уехало
    бы за пределы снимка. Трансформер кэшируется pyproj-ом, накладных расходов нет.
    """
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform

    dst = CRS.from_user_input(dst_crs)
    if dst.to_epsg() == 4326:
        return shape
    tr = Transformer.from_crs(CRS.from_epsg(4326), dst, always_xy=True)
    return shapely_transform(lambda x, y, z=None: tr.transform(x, y), shape)


# --------------------------------------------------------------------------------------
# Поиск сцен
# --------------------------------------------------------------------------------------

def _open_catalog():
    """Открываем STAC-каталог, при отказе основного пробуем запасные."""
    from pystac_client import Client

    last: Exception | None = None
    for url in (STAC_URL, *STAC_FALLBACKS):
        try:
            return Client.open(url), url
        except Exception as exc:
            last = exc
            log.warning("satellite: каталог %s не открылся: %s", url, exc)
    raise RuntimeError(f"ни один STAC-каталог не доступен: {last}")


def _search_all(bbox: Sequence[float], start: date, end: date) -> list[Any]:
    """Сцены всех трёх коллекций за период.

    Коллекции опрашиваются независимо и каждая в своём try: MODIS может быть
    недоступен, а Sentinel-2 при этом работать — сервис обязан пережить такое.
    """
    try:
        catalog, url = _open_catalog()
    except Exception as exc:
        log.error("satellite: поиск невозможен: %s", exc)
        return []

    period = f"{start.isoformat()}/{end.isoformat()}"
    found: list[Any] = []
    for collection in (COLLECTION_S2, COLLECTION_LANDSAT, COLLECTION_MODIS):
        try:
            search = catalog.search(
                collections=[collection],
                bbox=list(bbox),
                datetime=period,
            )
            items = list(search.items())
        except Exception as exc:
            log.warning("satellite: коллекция %s недоступна (%s): %s", collection, url, exc)
            continue

        kept = [it for it in items if _scene_cloud_ok(it)]
        log.info("satellite: %s — найдено %d, взято %d", collection, len(items), len(kept))
        found.extend(kept)
    return found


def _scene_cloud_ok(item: Any) -> bool:
    """Отсев сцен, закрытых облаком почти целиком (см. MAX_SCENE_CLOUD_COVER)."""
    cc = item.properties.get("eo:cloud_cover")
    if cc is None:
        return True
    try:
        return float(cc) <= MAX_SCENE_CLOUD_COVER
    except (TypeError, ValueError):
        return True


def _scene_rank(item: Any) -> tuple[int, float]:
    """Чем сцена лучше для демонстрации: сначала сенсор, потом облачность."""
    collection = getattr(item, "collection_id", "") or ""
    if collection.startswith("sentinel-2"):
        source = "s2"
    elif collection.startswith("landsat"):
        source = "landsat"
    else:
        source = "modis"
    cloud = item.properties.get("eo:cloud_cover")
    try:
        cloud = float(cloud)
    except (TypeError, ValueError):
        cloud = 100.0
    return SOURCE_PRIORITY.get(source, 9), cloud


def _thin_scenes(items: list[Any], max_scenes: int | None) -> list[Any]:
    """Прореживание для быстрой демонстрации.

    Наивное «первые N сцен» показало бы апрель-май и никакой сезонности, а «каждая
    N-я» с равным шансом вытянула бы сцену под сплошным облаком, и половина точек
    демо отвалилась бы на маске. Поэтому режем период на max_scenes равных корзин и
    в каждой берём лучшую сцену: сначала по приоритету сенсора, потом по облачности.
    Так на выходе и равномерная по времени сетка, и максимум пригодных наблюдений.
    """
    items = sorted(items, key=lambda it: (_item_date(it) or date.min))
    if max_scenes is None or max_scenes <= 0 or len(items) <= max_scenes:
        return items

    first = _item_date(items[0]) or date.min
    last = _item_date(items[-1]) or date.max
    span = max(1, (last - first).days)
    buckets: dict[int, Any] = {}
    for item in items:
        d = _item_date(item) or first
        idx = min(max_scenes - 1, (d - first).days * max_scenes // (span + 1))
        best = buckets.get(idx)
        if best is None or _scene_rank(item) < _scene_rank(best):
            buckets[idx] = item
    return [buckets[i] for i in sorted(buckets)]


def _item_date(item: Any) -> date | None:
    """Дата сцены. У MODIS `datetime` пустой — это композит за 16 дней.

    Для композита берём дату начала окна: именно так подписан сам гранул
    (MOD13Q1.A2025177 — 177-й день года) и так же подписаны колонки modis_* в
    наборе организаторов. Смещать на середину окна нельзя — разъедется склейка.
    """
    try:
        if item.datetime is not None:
            return item.datetime.date()
        raw = item.properties.get("start_datetime") or item.properties.get("end_datetime")
        if raw:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
    except Exception:
        pass
    return None


def _safe_id(item: Any) -> str:
    try:
        return str(item.id)
    except Exception:
        return "<item>"


# --------------------------------------------------------------------------------------
# Чтение окна и маски
# --------------------------------------------------------------------------------------

def _asset_href(item: Any, names: Iterable[str]) -> str | None:
    """Ссылка на ассет по первому подошедшему синониму имени, уже подписанная.

    Подписываем непосредственно перед чтением, а не при поиске: подпись Planetary
    Computer живёт около часа, а сбор длинного ряда может идти дольше.
    """
    for name in names:
        asset = item.assets.get(name)
        if asset is None:
            continue
        href = asset.href
        try:
            import planetary_computer as pc

            return pc.sign(href)
        except Exception:
            # Не PC-ссылка (Element84 отдаёт открытые URL) или подпись не удалась —
            # пробуем как есть, хуже уже не будет.
            return href
    return None


def _asset_scale_offset(item: Any, names: Iterable[str]) -> tuple[float | None, float | None]:
    """scale/offset из raster:bands, если каталог их объявил (Landsat, MODIS).

    Предпочитаем объявленные значения зашитым константам: это единственный способ
    пережить смену коллекции без правки кода.
    """
    for name in names:
        asset = item.assets.get(name)
        if asset is None:
            continue
        bands = (asset.extra_fields or {}).get("raster:bands") or []
        if bands and isinstance(bands[0], dict):
            return bands[0].get("scale"), bands[0].get("offset")
    return None, None


def _reference_grid(href: str, shape_wgs84):
    """Опорная сетка чтения: целочисленное окно самого детального канала под контуром.

    Все остальные каналы сцены приводятся к этой же сетке, иначе NDVI пришлось бы
    считать по массивам разного размера (у Sentinel-2 каналы 10 м, а SCL — 20 м).
    Берём запас в один пиксель по периметру: край полигона почти никогда не совпадает
    с границей пикселя, и без запаса теряется крайний ряд.
    """
    import rasterio
    from rasterio import windows
    from rasterio.windows import Window

    with rasterio.open(href) as ds:
        # Изредка попадается ассет без привязки (битый или подменённый превью-файл).
        # Считать по нему окно бессмысленно: без геотрансформации полигон ляжет
        # в левый верхний угол и медиана будет посчитана по чужой территории.
        if ds.crs is None or ds.transform.is_identity:
            log.debug("satellite: ассет без геопривязки, сцена пропущена: %s", href[:120])
            return None
        shape_native = _project_shape(shape_wgs84, ds.crs)
        minx, miny, maxx, maxy = shape_native.bounds
        win = windows.from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        win = Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2)
        try:
            win = win.intersection(Window(0, 0, ds.width, ds.height))
        except Exception:
            return None  # контур вне снимка (сцена задевает bbox краем)
        if win.width < 1 or win.height < 1:
            return None

        transform = windows.transform(win, ds.transform)
        height, width = int(win.height), int(win.width)

        # Полигон размером с район: читаем прореженно, чтобы не съесть память.
        # На точность медианы это не влияет — только на её пространственное разрешение.
        if height * width > MAX_WINDOW_PIXELS:
            factor = int(math.ceil(math.sqrt(height * width / MAX_WINDOW_PIXELS)))
            height = max(1, height // factor)
            width = max(1, width // factor)
            transform = transform * transform.scale(
                int(win.width) / width, int(win.height) / height
            )
            log.info("satellite: окно прорежено в %d раз(а)", factor)

        return {
            "crs": ds.crs,
            "transform": transform,
            "shape": (height, width),
            "shape_native": shape_native,
        }


def _read_to_grid(href: str, grid: dict, resampling_name: str) -> tuple[np.ndarray, Any]:
    """Читает окно ассета и приводит его к опорной сетке.

    Читаем родное целочисленное окно, покрывающее опорное, и только потом
    пересэмплируем: так не возникает полупиксельного сдвига между каналами
    разного разрешения, из-за которого NDVI на краю поля уезжает.
    Отражение — билинейно, маски качества — только ближайшим соседом:
    интерполировать номера классов бессмысленно.
    """
    import rasterio
    from rasterio import windows
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds
    from rasterio.warp import reproject
    from rasterio.windows import Window

    height, width = grid["shape"]
    bounds = array_bounds(height, width, grid["transform"])
    resampling = getattr(Resampling, resampling_name)

    with rasterio.open(href) as ds:
        win = windows.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3],
                                  transform=ds.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        win = Window(win.col_off - 1, win.row_off - 1, win.width + 2, win.height + 2)
        win = win.intersection(Window(0, 0, ds.width, ds.height))
        src = ds.read(1, window=win)
        src_transform = windows.transform(win, ds.transform)
        nodata = ds.nodata
        src_crs = ds.crs

    if src.shape == (height, width) and src_transform.almost_equals(grid["transform"]):
        return src, nodata

    dst = np.zeros((height, width), dtype=src.dtype)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=grid["transform"],
        dst_crs=grid["crs"],
        resampling=resampling,
        src_nodata=nodata,
        dst_nodata=nodata,
    )
    return dst, nodata


def _polygon_mask(grid: dict) -> np.ndarray:
    """Булев массив «пиксель внутри контура».

    Три уровня деградации, потому что полигоны бывают меньше пикселя:
    строгий охват -> all_touched (любое касание) -> всё окно целиком.
    Последнее корректно потому, что окно и так минимальное покрытие контура:
    для MODIS с пикселем 250 м поле 500x500 м — это буквально 2x2 пикселя,
    и требовать полного попадания центра пикселя внутрь контура нельзя.
    """
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    height, width = grid["shape"]
    geom = mapping(grid["shape_native"])
    try:
        inside = geometry_mask([geom], out_shape=(height, width),
                               transform=grid["transform"], invert=True)
        if inside.any():
            return inside
        inside = geometry_mask([geom], out_shape=(height, width),
                               transform=grid["transform"], invert=True,
                               all_touched=True)
        if inside.any():
            return inside
    except Exception as exc:
        log.debug("satellite: маска контура не построилась: %s", exc)
    return np.ones((height, width), dtype=bool)


# --------------------------------------------------------------------------------------
# Обработка одной сцены
# --------------------------------------------------------------------------------------

def _process_item(item: Any, shape_wgs84) -> tuple[Observation, int] | None:
    """Сцена -> одно наблюдение (или None).

    Возвращает пару (наблюдение, число валидных пикселей): счётчик нужен при склейке,
    когда на одну дату пришли две сцены одного сенсора (поле на стыке MGRS-тайлов).
    Весь метод — один большой try: сцена не должна ронять сбор.
    """
    import rasterio

    collection = getattr(item, "collection_id", "") or ""
    obs_date = _item_date(item)
    if obs_date is None:
        return None

    try:
        with rasterio.Env(**GDAL_ENV):
            if collection.startswith("sentinel-2"):
                return _process_s2(item, shape_wgs84, obs_date)
            if collection.startswith("landsat"):
                return _process_landsat(item, shape_wgs84, obs_date)
            if "13Q1" in collection or collection.startswith("modis"):
                return _process_modis(item, shape_wgs84, obs_date)
    except Exception as exc:
        log.warning("satellite: сцена %s пропущена: %s: %s",
                    _safe_id(item), type(exc).__name__, exc)
        return None

    log.debug("satellite: неизвестная коллекция %s", collection)
    return None


def _s2_offset(item: Any) -> float:
    """Аддитивный сдвиг отражения Sentinel-2 L2A (BOA_ADD_OFFSET).

    С baseline 04.00 (продукция с 25.01.2022) ESA хранит отражение со сдвигом -1000,
    чтобы уместить отрицательные значения после атмосферной коррекции в беззнаковый
    тип. Каталог этот сдвиг отдельным полем не отдаёт, поэтому выводим его из номера
    baseline. Цена ошибки высока: без вычета NDVI смещается нелинейно и тем сильнее,
    чем темнее поверхность, — весенние значения занижаются заметнее летних.
    """
    raw = item.properties.get("s2:processing_baseline")
    try:
        return -1000.0 if float(raw) >= 4.0 else 0.0
    except (TypeError, ValueError):
        # Baseline не указан: ориентируемся по дате перехода ESA.
        d = _item_date(item)
        return -1000.0 if d is not None and d >= date(2022, 1, 25) else 0.0


def _process_s2(item: Any, shape_wgs84, obs_date: date):
    red_href = _asset_href(item, S2_ASSETS["red"])
    scl_href = _asset_href(item, S2_ASSETS["scl"])
    nir_href = _asset_href(item, S2_ASSETS["nir"])
    if not (red_href and nir_href):
        return None

    grid = _reference_grid(red_href, shape_wgs84)
    if grid is None:
        return None
    inside = _polygon_mask(grid)
    n_inside = int(inside.sum())
    if n_inside == 0:
        return None

    # SCL читаем вторым — до тяжёлых каналов: если поле под облаком, дальше не идём
    # и экономим три range-запроса на сцену. На сотне сцен это минуты.
    clear = np.ones(grid["shape"], dtype=bool)
    if scl_href:
        scl, _ = _read_to_grid(scl_href, grid, "nearest")
        clear = ~np.isin(scl, SCL_BAD)
    else:
        log.debug("satellite: у %s нет SCL, маска облаков пропущена", _safe_id(item))

    valid = inside & clear
    if valid.sum() < MIN_VALID_FRACTION * n_inside:
        return None

    offset = _s2_offset(item)
    red = _scale_s2(_read_to_grid(red_href, grid, "bilinear"), offset)
    nir = _scale_s2(_read_to_grid(nir_href, grid, "bilinear"), offset)

    # Отражение 0 в исходных данных = «нет данных»; после сдвига это -0.1,
    # поэтому проверяем сырое значение через NaN, проставленный в _scale_s2.
    valid &= np.isfinite(red) & np.isfinite(nir)
    if valid.sum() < MIN_VALID_FRACTION * n_inside:
        return None

    ndvi = _median_index(_ndvi(nir, red), valid)
    if ndvi is None:
        return None

    # EVI и NDWI — «сколько получится»: контракт разрешает None, а лишний отказ
    # из-за отсутствующего синего канала стоил бы всей сцены.
    evi = ndwi = None
    blue_href = _asset_href(item, S2_ASSETS["blue"])
    green_href = _asset_href(item, S2_ASSETS["green"])
    try:
        if blue_href:
            blue = _scale_s2(_read_to_grid(blue_href, grid, "bilinear"), offset)
            evi = _median_index(_evi(nir, red, blue), valid & np.isfinite(blue))
        if green_href:
            green = _scale_s2(_read_to_grid(green_href, grid, "bilinear"), offset)
            ndwi = _median_index(_ndwi(green, nir), valid & np.isfinite(green))
    except Exception as exc:
        log.debug("satellite: %s — доп. индексы не посчитаны: %s", _safe_id(item), exc)

    return Observation(date=obs_date, ndvi=ndvi, evi=evi, ndwi=ndwi, source="s2"), int(valid.sum())


def _scale_s2(read_result: tuple[np.ndarray, Any], offset: float) -> np.ndarray:
    """Целые числа Sentinel-2 -> физическое отражение 0..1. Ноль = нет данных -> NaN."""
    arr, _nodata = read_result
    out = arr.astype(np.float32)
    out[arr == 0] = np.nan
    return (out + offset) / 10000.0


def _process_landsat(item: Any, shape_wgs84, obs_date: date):
    red_href = _asset_href(item, LANDSAT_ASSETS["red"])
    nir_href = _asset_href(item, LANDSAT_ASSETS["nir"])
    qa_href = _asset_href(item, LANDSAT_ASSETS["qa"])
    if not (red_href and nir_href):
        return None

    grid = _reference_grid(red_href, shape_wgs84)
    if grid is None:
        return None
    inside = _polygon_mask(grid)
    n_inside = int(inside.sum())
    if n_inside == 0:
        return None

    clear = np.ones(grid["shape"], dtype=bool)
    if qa_href:
        qa, _ = _read_to_grid(qa_href, grid, "nearest")
        qa = qa.astype(np.uint16)
        bad = np.zeros(qa.shape, dtype=bool)
        for bit in LANDSAT_QA_BITS:
            bad |= (qa & (1 << bit)) != 0
        clear = ~bad
    valid = inside & clear
    if valid.sum() < MIN_VALID_FRACTION * n_inside:
        return None

    scale, offset = _asset_scale_offset(item, LANDSAT_ASSETS["red"])
    scale = float(scale) if scale else 0.0000275
    offset = float(offset) if offset is not None else -0.2

    red = _scale_landsat(_read_to_grid(red_href, grid, "bilinear"), scale, offset)
    nir = _scale_landsat(_read_to_grid(nir_href, grid, "bilinear"), scale, offset)
    valid &= np.isfinite(red) & np.isfinite(nir)
    if valid.sum() < MIN_VALID_FRACTION * n_inside:
        return None

    ndvi = _median_index(_ndvi(nir, red), valid)
    if ndvi is None:
        return None

    evi = ndwi = None
    try:
        blue_href = _asset_href(item, LANDSAT_ASSETS["blue"])
        green_href = _asset_href(item, LANDSAT_ASSETS["green"])
        if blue_href:
            blue = _scale_landsat(_read_to_grid(blue_href, grid, "bilinear"), scale, offset)
            evi = _median_index(_evi(nir, red, blue), valid & np.isfinite(blue))
        if green_href:
            green = _scale_landsat(_read_to_grid(green_href, grid, "bilinear"), scale, offset)
            ndwi = _median_index(_ndwi(green, nir), valid & np.isfinite(green))
    except Exception as exc:
        log.debug("satellite: %s — доп. индексы не посчитаны: %s", _safe_id(item), exc)

    return (Observation(date=obs_date, ndvi=ndvi, evi=evi, ndwi=ndwi, source="landsat"),
            int(valid.sum()))


def _scale_landsat(read_result: tuple[np.ndarray, Any], scale: float, offset: float) -> np.ndarray:
    """Landsat C2 L2: значение * 0.0000275 - 0.2. Ноль — служебный заполнитель."""
    arr, nodata = read_result
    out = arr.astype(np.float32)
    fill = 0 if nodata is None else nodata
    out[arr == fill] = np.nan
    return out * scale + offset


def _process_modis(item: Any, shape_wgs84, obs_date: date):
    """MOD13Q1: NDVI и EVI уже посчитаны производителем, каналов отражения нет.

    Поэтому NDWI здесь принципиально недоступен — оставляем None, это законный
    случай по контракту. Пикселей внутри поля мало (250 м), поэтому медиана
    считается по 2-4 значениям; ядро учитывает это весом сенсора 0,5.
    """
    ndvi_href = _asset_href(item, MODIS_ASSETS["ndvi"])
    if not ndvi_href:
        return None

    grid = _reference_grid(ndvi_href, shape_wgs84)
    if grid is None:
        return None
    inside = _polygon_mask(grid)
    n_inside = int(inside.sum())
    if n_inside == 0:
        return None

    valid = inside.copy()
    qa_href = _asset_href(item, MODIS_ASSETS["qa"])
    if qa_href:
        rel, _ = _read_to_grid(qa_href, grid, "nearest")
        valid &= np.isin(rel.astype(np.int16), MODIS_RELIABILITY_OK)
    if valid.sum() < MIN_VALID_FRACTION * n_inside:
        return None

    scale, _ = _asset_scale_offset(item, MODIS_ASSETS["ndvi"])
    scale = float(scale) if scale else 0.0001

    raw, nodata = _read_to_grid(ndvi_href, grid, "nearest")
    arr = raw.astype(np.float32)
    fill = -3000 if nodata is None else nodata
    arr[raw == fill] = np.nan
    ndvi_grid = arr * scale
    valid &= np.isfinite(ndvi_grid)
    if valid.sum() < MIN_VALID_FRACTION * n_inside:
        return None
    ndvi = _median_index(ndvi_grid, valid)
    if ndvi is None:
        return None

    evi = None
    try:
        evi_href = _asset_href(item, MODIS_ASSETS["evi"])
        if evi_href:
            raw_evi, evi_nodata = _read_to_grid(evi_href, grid, "nearest")
            a = raw_evi.astype(np.float32)
            a[raw_evi == (-3000 if evi_nodata is None else evi_nodata)] = np.nan
            evi = _median_index(a * scale, valid & np.isfinite(a))
    except Exception as exc:
        log.debug("satellite: %s — EVI не посчитан: %s", _safe_id(item), exc)

    return (Observation(date=obs_date, ndvi=ndvi, evi=evi, ndwi=None, source="modis"),
            int(valid.sum()))


# --------------------------------------------------------------------------------------
# Индексы и агрегация
# --------------------------------------------------------------------------------------

def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Деление, где нулевой знаменатель даёт NaN, а не предупреждение и inf."""
    out = np.full(num.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-6)
    out[ok] = num[ok] / den[ok]
    return out


def _ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return _safe_div(nir - red, nir + red)


def _evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """EVI = 2.5*(NIR-RED)/(NIR + 6*RED - 7.5*BLUE + 1).

    Знаменатель может пройти через ноль на аномальных пикселях (дымка, вода),
    и тогда EVI улетает на порядки. Подрезаем результат физическим диапазоном —
    выбрасывать пиксель целиком было бы хуже: NDVI по нему обычно корректен.
    """
    evi = 2.5 * _safe_div(nir - red, nir + 6.0 * red - 7.5 * blue + 1.0)
    return np.clip(evi, -1.0, 1.5)


def _ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return _safe_div(green - nir, green + nir)


def _median_index(values: np.ndarray, valid: np.ndarray) -> float | None:
    """Медиана индекса по пикселям внутри контура.

    Медиана, а не среднее: одно недомаскированное облако или пиксель дороги
    сдвигает среднее заметно, медиану — почти нет. Плюс отсекаем нефизичные
    значения: за пределами [-1, 1] у нормированной разности лежит только брак.
    """
    sel = values[valid & np.isfinite(values)]
    sel = sel[(sel >= -1.0) & (sel <= 1.0)]
    if sel.size == 0:
        return None
    return float(np.median(sel))


def _merge_by_date(collected: list[tuple[Observation, int]]) -> list[Observation]:
    """Склейка наблюдений одной даты по приоритету сенсоров.

    Приоритет Sentinel-2 -> Landsat -> MODIS: так собран `primary_ndvi` у
    организаторов, и на это же рассчитано доменное ядро. Внутри одного сенсора
    (поле на стыке двух тайлов или двух витков) побеждает сцена, у которой внутри
    контура осталось больше валидных пикселей — она надёжнее.
    """
    best: dict[date, tuple[Observation, int]] = {}
    for obs, n_valid in collected:
        if obs.ndvi is None:
            continue
        current = best.get(obs.date)
        if current is None:
            best[obs.date] = (obs, n_valid)
            continue
        rank_new = (SOURCE_PRIORITY.get(obs.source, 9), -n_valid)
        rank_old = (SOURCE_PRIORITY.get(current[0].source, 9), -current[1])
        if rank_new < rank_old:
            best[obs.date] = (obs, n_valid)
    return [best[d][0] for d in sorted(best)]


def _emit(progress: ProgressFn | None, stage: str, done: int, total: int) -> None:
    """Прогресс наружу. Ошибка в чужом коллбэке не должна ронять сбор данных."""
    if progress is None:
        return
    try:
        progress(stage, done, total)
    except Exception as exc:
        log.debug("satellite: коллбэк прогресса упал: %s", exc)


# --------------------------------------------------------------------------------------
# Самопроверка: реальное поле под Ростовом-на-Дону
# --------------------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def square(lon: float, lat: float, side_m: float = 500.0) -> dict:
        """Квадрат заданной стороны в метрах вокруг точки, в градусах WGS84."""
        dlat = side_m / 2 / 111320.0
        dlon = side_m / 2 / (111320.0 * math.cos(math.radians(lat)))
        return {"type": "Polygon", "coordinates": [[
            [lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
            [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat],
        ]]}

    def run(label: str, poly: dict) -> None:
        t0 = time.perf_counter()
        obs = fetch_observations(
            poly, date(2025, 4, 1), date(2025, 10, 31),
            progress=lambda stage, done, total: print(f"\r{stage}: {done}/{total}   ", end=""),
        )
        elapsed = time.perf_counter() - t0
        print()
        counts: dict[str, int] = {}
        for o in obs:
            counts[o.source] = counts.get(o.source, 0) + 1
        print(f"[{label}] наблюдений: {len(obs)}, источники: {counts}, время: {elapsed:.1f} с")
        if not obs:
            return
        vals = [o.ndvi for o in obs if o.ndvi is not None]
        print(f"    NDVI: min={min(vals):+.3f} max={max(vals):+.3f} медиана={np.median(vals):+.3f}")
        by_month: dict[int, list[float]] = {}
        for o in obs:
            by_month.setdefault(o.date.month, []).append(o.ndvi)
        print("    медиана NDVI по месяцам: " + "  ".join(
            f"{m:02d}:{np.median(v):+.2f}" for m, v in sorted(by_month.items())))
        print("     дата        NDVI   источник")
        rows = obs[:10] + ([None] + obs[-10:] if len(obs) > 20 else [])
        for o in rows:
            print("     ..." if o is None else f"     {o.date}  {o.ndvi:+.3f}  {o.source}")
        print()

    # Точка из постановки. Она попадает в застройку Ростова-на-Дону (центр города —
    # 47,236 с.ш. / 39,702 в.д.), поэтому сезонной кривой там нет и быть не должно:
    # это полезная проверка на то, что модуль не «дорисовывает» вегетацию.
    run("город, 39.72/47.22 из постановки", square(39.72, 47.22))

    # Настоящее поле южнее города — на нём и проверяется физичность ряда.
    run("поле, 40.10/46.95", square(40.10, 46.95))
