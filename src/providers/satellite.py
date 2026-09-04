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
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
# --------------------------------------------------------------------------------------
# Реестр источников
#
# Заказчик кейса просил Европейское космическое агентство (Copernicus Data Space) и
# Google Earth Engine. GEE отпадает по инфраструктуре: он требует собственного проекта
# в Google Cloud и аутентификации под аккаунтом владельца, ключ в репозиторий не
# положишь. Остальные три каталога проверены руками, и результат проверки записан в
# поле `note` каждого источника — это не предположения, а наблюдённые коды ответов.
# --------------------------------------------------------------------------------------

# Ключи сенсоров, которыми оперирует весь модуль.
SENSOR_S2 = "s2"
SENSOR_LANDSAT = "landsat"
SENSOR_MODIS = "modis"

# Приоритет сенсоров при склейке одной даты. Ровно так собран `primary_ndvi`
# в наборе организаторов (проверено: при наличии s2_ndvi он совпадает точно),
# и доменное ядро взвешивает наблюдения по этой же шкале 1 / 0,8 / 0,5.
SOURCE_PRIORITY = {SENSOR_S2: 0, SENSOR_LANDSAT: 1, SENSOR_MODIS: 2}

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

# Имена ассетов у трёх каталогов разные, и это единственное, что реально мешает
# переключаться между ними. Planetary Computer зовёт каналы B04 / SCL, Element84 —
# red / scl, Copernicus — B04_10m / SCL_20m. Держим все синонимы одним списком и
# берём первый подошедший: тогда один и тот же код читает любой каталог.
S2_ASSETS = {
    "red": ("red", "B04", "B04_10m"),
    "nir": ("nir", "B08", "B08_10m"),
    "blue": ("blue", "B02", "B02_10m"),
    "green": ("green", "B03", "B03_10m"),
    "scl": ("SCL", "scl", "SCL_20m"),
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

# Переключатель источника: auto | pc | cdse | element84.
ENV_SOURCE = "FENOLOG_SATELLITE_SOURCE"

# Учётные данные Copernicus. Каталог открыт, пиксели — нет (см. заметку у _CDSE),
# поэтому источник включается только при одном из двух наборов переменных.
ENV_CDSE_S3_KEY = "CDSE_S3_ACCESS_KEY"
ENV_CDSE_S3_SECRET = "CDSE_S3_SECRET_KEY"
ENV_CDSE_USER = "CDSE_USERNAME"
ENV_CDSE_PASSWORD = "CDSE_PASSWORD"

CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
                  "/protocol/openid-connect/token")
CDSE_S3_ENDPOINT = "eodata.dataspace.copernicus.eu"


@dataclass
class _Source:
    """Описание одного STAC-каталога: где искать, как называются ассеты, чем платить.

    Всё, чем каталоги отличаются друг от друга, собрано здесь. Остальной код
    источник не различает и работает только через это описание — именно поэтому
    добавление четвёртого каталога не потребует правок в чтении и маскировании.
    """
    key: str
    title: str
    stac_url: str
    collections: dict          # сенсор -> id коллекции в этом каталоге
    note: str                  # результат ручной проверки доступа, для отчёта и логов
    needs_key: bool = False    # пиксели закрыты авторизацией
    prefer_alternate_https: bool = False  # ассет отдаёт s3://, ссылка лежит в alternate

    def sensor_of(self, item: Any) -> str | None:
        """Обратное отображение: id коллекции сцены -> ключ сенсора."""
        cid = getattr(item, "collection_id", "") or ""
        for sensor, name in self.collections.items():
            if cid == name:
                return sensor
        # Запасной разбор по имени: у каталогов встречаются варианты вроде
        # "sentinel-2-c1-l2a", которые в явной таблице не перечислить.
        low = cid.lower()
        if "sentinel-2" in low:
            return SENSOR_S2
        if "landsat" in low:
            return SENSOR_LANDSAT
        if "13q1" in low or "modis" in low:
            return SENSOR_MODIS
        return None


# Planetary Computer — основной. Три сенсора, всё в COG, подпись бесплатная и без
# регистрации. За всё время проверок ни разу не отказал.
_PC = _Source(
    key="pc",
    title="Microsoft Planetary Computer",
    stac_url="https://planetarycomputer.microsoft.com/api/stac/v1",
    collections={SENSOR_S2: "sentinel-2-l2a",
                 SENSOR_LANDSAT: "landsat-c2-l2",
                 SENSOR_MODIS: "modis-13Q1-061"},
    note="открыт полностью: и поиск, и пиксели, без регистрации, все три сенсора",
)

# Copernicus Data Space Ecosystem — портал ЕКА, тот самый источник из просьбы заказчика.
# Каталог открыт (поиск отвечает за 1,3 с), но КАЖДЫЙ ассет с пикселями закрыт:
#   - основной href это s3://eodata/..., "auth:refs": ["s3"] — нужен ключ объектного
#     хранилища CDSE, GDAL без него отвечает InvalidCredentials;
#   - alternate.https на zipper.dataspace.copernicus.eu помечен "auth:refs": ["oidc"] и
#     без токена отдаёт HTTP 401 {"code":"DAT-ZIP-604","message":"Token not found"};
#   - OData-ссылка Products(...)/$value — тот же 401.
# Плюс данные там лежат в SAFE/JPEG2000, а не в COG: даже с ключом чтение окна будет
# заметно дороже, чем у двух других каталогов.
# Поэтому источник включается только при заданных учётных данных, а без них честно
# сообщает «требует ключа» и уступает очередь следующему каталогу.
# Landsat здесь только Level-1 (сырые DN без атмосферной коррекции) — другой уровень
# обработки, к Level-2 несопоставим, поэтому не подключён.
_CDSE = _Source(
    key="cdse",
    title="Copernicus Data Space Ecosystem (ЕКА)",
    stac_url="https://catalogue.dataspace.copernicus.eu/stac",
    collections={SENSOR_S2: "sentinel-2-l2a"},
    note=("каталог открыт, пиксели требуют ключа: s3 -> InvalidCredentials, "
          "https -> 401 DAT-ZIP-604 'Token not found'"),
    needs_key=True,
    prefer_alternate_https=True,
)

# AWS Element84 — открытая витрина тех же продуктов ЕКА, перепакованных в COG.
# Sentinel-2 читается без ключей (проверено), а Landsat лежит в requester-pays бакете
# usgs-landsat и без ключей AWS не открывается, поэтому не подключён.
_ELEMENT84 = _Source(
    key="element84",
    title="AWS Element84 Earth Search",
    stac_url="https://earth-search.aws.element84.com/v1",
    collections={SENSOR_S2: "sentinel-2-l2a"},
    note="Sentinel-2 открыт без ключей; Landsat в requester-pays бакете, не подключён",
)

SOURCES = {src.key: src for src in (_PC, _CDSE, _ELEMENT84)}

# Порядок автоматического перебора: сначала полный каталог, затем ЕКА (если дали ключ),
# затем открытая витрина тех же данных.
SOURCE_ORDER = ("pc", "cdse", "element84")

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
    source: str | None = None,
) -> list[Observation]:
    """Наблюдения NDVI по контуру, отсортированные по дате.

    geometry  — GeoJSON Polygon/MultiPolygon (или Feature) в EPSG:4326.
    progress  — callable(этап, готово, всего); вызывается по мере скачивания.
    max_scenes— ограничение числа сцен для быстрой демонстрации; сцены при этом
                прореживаются равномерно по времени, чтобы сезонная кривая
                осталась узнаваемой, а не обрезалась по началу периода.
    source    — принудительный выбор каталога: "pc" | "cdse" | "element84" | "auto".
                По умолчанию берётся переменная окружения FENOLOG_SATELLITE_SOURCE,
                а если и её нет — "auto": каталоги перебираются по SOURCE_ORDER,
                пока какой-нибудь не отдаст наблюдения.

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

    chain = _resolve_sources(source)
    if not chain:
        log.error("satellite: не осталось ни одного пригодного источника")
        return []

    # Перебор источников до первого, который реально отдал наблюдения. Пустой ответ
    # приравнивается к отказу: каталог может быть жив, но не покрывать эту территорию
    # (у Element84, например, нет Landsat), и тогда честнее пойти дальше по цепочке,
    # чем возвращать пользователю пустой график при живом запасном источнике.
    for attempt, src in enumerate(chain, 1):
        ok, reason = _source_ready(src)
        if not ok:
            log.warning("satellite: источник %s пропущен — %s", src.title, reason)
            continue
        log.info("satellite: источник %s (%d из %d)", src.title, attempt, len(chain))
        observations = _collect_from_source(shape, start, end, progress, max_scenes, src)
        if observations:
            log.info(
                "satellite: %d наблюдений из %s за %.1f с",
                len(observations), src.title, time.perf_counter() - t0,
            )
            return observations
        log.warning("satellite: источник %s не дал наблюдений, перехожу к следующему",
                    src.title)

    log.warning("satellite: ни один источник не дал наблюдений за %s..%s", start, end)
    return []


def _collect_from_source(
    shape,
    start: date,
    end: date,
    progress: ProgressFn | None,
    max_scenes: int | None,
    src: _Source,
) -> list[Observation]:
    """Полный цикл сбора по одному каталогу. Наружу исключений не выпускает."""
    _emit(progress, "ищу сцены", 0, 1)
    try:
        items = _search_all(tuple(shape.bounds), start, end, src)
    except Exception as exc:
        log.warning("satellite: поиск в %s не удался: %s", src.title, exc)
        return []
    if not items:
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
        futures = {pool.submit(_process_item, it, shape, src): it for it in items}
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
    return [o for o in _merge_by_date(collected) if start <= o.date <= end]


# --------------------------------------------------------------------------------------
# Выбор источника и учётные данные
# --------------------------------------------------------------------------------------

def _resolve_sources(source: str | None) -> list[_Source]:
    """Во что превращается запрошенный источник: список каталогов в порядке перебора.

    Явно названный каталог — это именно он и только он: если пользователь выбрал ЕКА,
    подсовывать ему молча другие данные нельзя, он их выбирал осознанно. Молчаливая
    подмена источника — это как раз то, из-за чего потом невозможно объяснить, откуда
    в отчёте взялись числа. Автоматический режим, наоборот, идёт по цепочке.
    """
    requested = (source or os.environ.get(ENV_SOURCE) or "auto").strip().lower()
    if requested in ("auto", "", "any"):
        return [SOURCES[k] for k in SOURCE_ORDER if k in SOURCES]
    if requested in SOURCES:
        return [SOURCES[requested]]
    log.error("satellite: неизвестный источник %r, известны: %s; беру auto",
              requested, ", ".join(SOURCES))
    return [SOURCES[k] for k in SOURCE_ORDER if k in SOURCES]


def _cdse_credentials() -> tuple[str, dict]:
    """Какими учётными данными Copernicus мы располагаем.

    Возвращает ("s3" | "oidc" | "", подробности). Порядок предпочтения — S3: это
    прямой доступ к объектному хранилищу без промежуточного сервиса выдачи файлов,
    он и быстрее, и не упирается в срок жизни токена.
    """
    key = os.environ.get(ENV_CDSE_S3_KEY)
    secret = os.environ.get(ENV_CDSE_S3_SECRET)
    if key and secret:
        return "s3", {"key": key, "secret": secret}
    user = os.environ.get(ENV_CDSE_USER)
    password = os.environ.get(ENV_CDSE_PASSWORD)
    if user and password:
        return "oidc", {"user": user, "password": password}
    return "", {}


# Токен Keycloak живёт около 10 минут, а сбор сезона занимает десятки секунд и идёт
# в несколько потоков. Поэтому держим один токен на процесс под замком и обновляем
# заранее, за минуту до формального истечения.
_cdse_token_lock = threading.Lock()
_cdse_token_value: str | None = None
_cdse_token_expires: float = 0.0


def _cdse_token() -> str | None:
    """Bearer-токен Copernicus по логину и паролю. None, если не выдали."""
    global _cdse_token_value, _cdse_token_expires
    mode, creds = _cdse_credentials()
    if mode != "oidc":
        return None
    with _cdse_token_lock:
        if _cdse_token_value and time.time() < _cdse_token_expires:
            return _cdse_token_value
        try:
            import requests

            r = requests.post(
                CDSE_TOKEN_URL,
                data={
                    "grant_type": "password",
                    "username": creds["user"],
                    "password": creds["password"],
                    "client_id": "cdse-public",
                },
                timeout=20,
            )
            if r.status_code != 200:
                log.error("satellite: Copernicus не выдал токен, HTTP %s: %s",
                          r.status_code, r.text[:200])
                return None
            payload = r.json()
            _cdse_token_value = payload["access_token"]
            _cdse_token_expires = time.time() + float(payload.get("expires_in", 600)) - 60
            log.info("satellite: получен токен Copernicus")
            return _cdse_token_value
        except Exception as exc:
            log.error("satellite: не смог получить токен Copernicus: %s", exc)
            return None


def _source_ready(src: _Source) -> tuple[bool, str]:
    """Можно ли вообще пытаться читать пиксели из этого каталога.

    Проверка дешёвая и делается ДО поиска: бессмысленно тратить секунды на STAC-запрос,
    если файлы всё равно не откроются. Сообщение возвращается человеческим текстом —
    оно уходит в лог и дальше в интерфейс, вместо молчаливого пустого графика.
    """
    if not src.needs_key:
        return True, ""
    if src.key == "cdse":
        mode, _ = _cdse_credentials()
        if mode == "s3":
            return True, ""
        if mode == "oidc":
            return (True, "") if _cdse_token() else (
                False, f"{src.title}: логин есть, но токен не выдан")
        return False, (
            f"{src.title}: источник требует ключа. Каталог открыт, но пиксели закрыты "
            f"({src.note}). Зарегистрируйтесь на dataspace.copernicus.eu и задайте "
            f"{ENV_CDSE_S3_KEY}/{ENV_CDSE_S3_SECRET} (ключи S3) либо "
            f"{ENV_CDSE_USER}/{ENV_CDSE_PASSWORD} (учётная запись)")
    return True, ""


def _source_gdal_env(src: _Source) -> dict:
    """Настройки GDAL под конкретный каталог: общие плюс то, чем платим за доступ."""
    env = dict(GDAL_ENV)
    if src.key != "cdse":
        return env
    mode, creds = _cdse_credentials()
    if mode == "s3":
        # Хранилище CDSE — S3-совместимое, но не Amazon: нужен свой адрес, выключенный
        # virtual-hosting (бакет в пути, а не в имени хоста) и фиктивный регион.
        env.update({
            "AWS_ACCESS_KEY_ID": creds["key"],
            "AWS_SECRET_ACCESS_KEY": creds["secret"],
            "AWS_S3_ENDPOINT": CDSE_S3_ENDPOINT,
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_HTTPS": "YES",
            "AWS_REGION": "default",
            "AWS_NO_SIGN_REQUEST": "NO",
        })
    elif mode == "oidc":
        token = _cdse_token()
        if token:
            env["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {token}"
    return env


class StacSatelliteProvider:
    """Обёртка под протокол `src.providers.base.SatelliteProvider`.

    Нужна API-слою: он работает со списком провайдеров единообразно и должен уметь
    спросить живость источника до того, как повесит пользователя на минуту ожидания.
    Один класс на все каталоги — различия целиком описаны объектом _Source.
    """

    def __init__(self, source: str = "auto", max_scenes: int | None = None) -> None:
        self.source = source
        self.max_scenes = max_scenes
        src = SOURCES.get(source)
        self.name = src.key if src else "stac-auto"
        self.title = src.title if src else "автовыбор источника"

    def is_available(self) -> bool:
        """Лёгкая проверка: корневой документ STAC плюс наличие ключей, без поиска сцен."""
        for src in _resolve_sources(self.source):
            ok, reason = _source_ready(src)
            if not ok:
                log.info("satellite: %s недоступен — %s", src.title, reason)
                continue
            try:
                import requests

                if requests.get(src.stac_url, timeout=8).status_code == 200:
                    return True
            except Exception as exc:
                log.warning("satellite: каталог %s недоступен: %s", src.title, exc)
        return False

    def fetch(self, geometry: dict, start: date, end: date) -> list[Observation]:
        return fetch_observations(geometry, start, end,
                                  max_scenes=self.max_scenes, source=self.source)


class PlanetaryComputerSatelliteProvider(StacSatelliteProvider):
    """Основной источник: три сенсора, без регистрации."""

    def __init__(self, max_scenes: int | None = None) -> None:
        super().__init__("pc", max_scenes)


class CopernicusSatelliteProvider(StacSatelliteProvider):
    """Источник ЕКА. Работает только при заданных учётных данных Copernicus.

    Без ключа `is_available()` вернёт False с внятной причиной в логе — это осознанно
    лучше, чем притвориться живым и отдать пустой ряд.
    """

    def __init__(self, max_scenes: int | None = None) -> None:
        super().__init__("cdse", max_scenes)


class Element84SatelliteProvider(StacSatelliteProvider):
    """Открытая витрина продуктов ЕКА на AWS. Только Sentinel-2, зато без ключей."""

    def __init__(self, max_scenes: int | None = None) -> None:
        super().__init__("element84", max_scenes)


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

def _open_catalog(src: _Source):
    """Открываем STAC-каталог конкретного источника."""
    from pystac_client import Client

    return Client.open(src.stac_url)


def _search_all(bbox: Sequence[float], start: date, end: date, src: _Source) -> list[Any]:
    """Сцены всех коллекций источника за период.

    Коллекции опрашиваются независимо и каждая в своём try: MODIS может быть
    недоступен, а Sentinel-2 при этом работать — сервис обязан пережить такое.
    Отдельно ловим 429: Copernicus прикрыт WAF и на пачку быстрых запросов отвечает
    «Rate limit exceeded», из-за чего часть коллекций молча выпала бы из выдачи.
    """
    try:
        catalog = _open_catalog(src)
    except Exception as exc:
        log.error("satellite: каталог %s не открылся: %s", src.title, exc)
        return []

    period = f"{start.isoformat()}/{end.isoformat()}"
    found: list[Any] = []
    for sensor, collection in src.collections.items():
        try:
            search = catalog.search(
                collections=[collection],
                bbox=list(bbox),
                datetime=period,
            )
            items = list(search.items())
        except Exception as exc:
            detail = str(exc)
            if "429" in detail or "Rate limit" in detail:
                log.warning("satellite: %s ограничил частоту запросов на коллекции %s",
                            src.title, collection)
            else:
                log.warning("satellite: коллекция %s недоступна в %s: %s",
                            collection, src.title, detail[:200])
            continue

        kept = [it for it in items if _scene_cloud_ok(it)]
        log.info("satellite: %s/%s — найдено %d, взято %d",
                 src.key, collection, len(items), len(kept))
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

def _asset_href(item: Any, names: Iterable[str], src: _Source) -> str | None:
    """Ссылка на ассет по первому подошедшему синониму имени, готовая к чтению.

    Три каталога отдают ссылки тремя разными способами:
      - Planetary Computer — https, но требует подписи; подписываем непосредственно
        перед чтением, а не при поиске, потому что подпись живёт около часа, а сбор
        длинного ряда может идти дольше;
      - Element84 — открытый https, ничего делать не нужно;
      - Copernicus — основной href это s3://, а пригодная для GDAL https-ссылка
        спрятана в extra_fields["alternate"]["https"]; при доступе по ключам S3,
        наоборот, нужен именно s3://.
    """
    for name in names:
        asset = item.assets.get(name)
        if asset is None:
            continue
        href = asset.href

        if src.prefer_alternate_https:
            mode, _ = _cdse_credentials()
            alternate = (asset.extra_fields or {}).get("alternate") or {}
            https_href = (alternate.get("https") or {}).get("href")
            # С ключами S3 читаем напрямую из хранилища, с токеном — по https.
            if mode == "s3" and href.startswith("s3://"):
                return href
            if https_href:
                return https_href
            return href

        if src.key == "pc":
            try:
                import planetary_computer as pc

                return pc.sign(href)
            except Exception as exc:
                # Подпись не удалась — пробуем как есть, хуже уже не будет.
                log.debug("satellite: подпись не удалась (%s), читаю без неё", exc)
                return href
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

def _process_item(item: Any, shape_wgs84, src: _Source) -> tuple[Observation, int] | None:
    """Сцена -> одно наблюдение (или None).

    Возвращает пару (наблюдение, число валидных пикселей): счётчик нужен при склейке,
    когда на одну дату пришли две сцены одного сенсора (поле на стыке MGRS-тайлов).
    Весь метод — один большой try: сцена не должна ронять сбор.
    """
    import rasterio

    sensor = src.sensor_of(item)
    obs_date = _item_date(item)
    if obs_date is None or sensor is None:
        log.debug("satellite: сцена %s без даты или без сенсора", _safe_id(item))
        return None

    try:
        with rasterio.Env(**_source_gdal_env(src)):
            if sensor == SENSOR_S2:
                return _process_s2(item, shape_wgs84, obs_date, src)
            if sensor == SENSOR_LANDSAT:
                return _process_landsat(item, shape_wgs84, obs_date, src)
            if sensor == SENSOR_MODIS:
                return _process_modis(item, shape_wgs84, obs_date, src)
    except Exception as exc:
        detail = str(exc)
        # Отказ по правам разбираем отдельно: это не «битая сцена», а незакрытый
        # доступ, и пользователю надо сказать именно это.
        if any(m in detail for m in ("401", "403", "InvalidCredentials",
                                     "Token not found", "AccessDenied")):
            log.warning("satellite: %s не отдал пиксели без ключа (%s): %s",
                        src.title, _safe_id(item), detail[:160])
        else:
            log.warning("satellite: сцена %s пропущена: %s: %s",
                        _safe_id(item), type(exc).__name__, detail[:160])
        return None

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


def _process_s2(item: Any, shape_wgs84, obs_date: date, src: _Source):
    red_href = _asset_href(item, S2_ASSETS["red"], src)
    scl_href = _asset_href(item, S2_ASSETS["scl"], src)
    nir_href = _asset_href(item, S2_ASSETS["nir"], src)
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
    blue_href = _asset_href(item, S2_ASSETS["blue"], src)
    green_href = _asset_href(item, S2_ASSETS["green"], src)
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


def _process_landsat(item: Any, shape_wgs84, obs_date: date, src: _Source):
    red_href = _asset_href(item, LANDSAT_ASSETS["red"], src)
    nir_href = _asset_href(item, LANDSAT_ASSETS["nir"], src)
    qa_href = _asset_href(item, LANDSAT_ASSETS["qa"], src)
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
        blue_href = _asset_href(item, LANDSAT_ASSETS["blue"], src)
        green_href = _asset_href(item, LANDSAT_ASSETS["green"], src)
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


def _process_modis(item: Any, shape_wgs84, obs_date: date, src: _Source):
    """MOD13Q1: NDVI и EVI уже посчитаны производителем, каналов отражения нет.

    Поэтому NDWI здесь принципиально недоступен — оставляем None, это законный
    случай по контракту. Пикселей внутри поля мало (250 м), поэтому медиана
    считается по 2-4 значениям; ядро учитывает это весом сенсора 0,5.
    """
    ndvi_href = _asset_href(item, MODIS_ASSETS["ndvi"], src)
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
    qa_href = _asset_href(item, MODIS_ASSETS["qa"], src)
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
        evi_href = _asset_href(item, MODIS_ASSETS["evi"], src)
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
