"""Маска пашни ESA WorldCereal — второй, независимый от OpenStreetMap источник.

Зачем он в проекте. Постановка кейса называет ESA WorldCereal рядом с
OpenStreetMap, а OSM — источник рукотворный: поле там есть ровно тогда, когда
его кто-то обвёл. WorldCereal рукотворным не является: это глобальная
классификация пашни по Sentinel-1/2 за 2021 год с шагом 10 метров. Два источника
такой разной природы ошибаются по-разному, и именно поэтому их согласие — довод,
а не удвоенное утверждение.

Чем он НЕ является — это главное, что нужно понимать про этот файл.
WorldCereal — растр, а не векторный слой границ. В нём нет объекта «поле №17»:
есть пиксель 10x10 м со значением 100 (временные культуры) или 0 (всё
остальное). Вторым источником *контуров* он поэтому быть не может: чтобы достать
из растра границу конкретного участка, нужна векторизация маски и разделение
смежных полей, которые в маске слиты в одно пятно, — отдельная задача с
сегментацией, и решать её ради подсказок на карте бессмысленно. Так что здесь
WorldCereal делает то, на что он годится по своей природе: отвечает на вопрос
«какая доля присланного контура размечена как пашня».

Как берутся данные, и почему именно так.

    Terrascope STAC  ->  найти AEZ-снимок над контуром
    публичный TiTiler  ->  зональная статистика по контуру

Прямое чтение GeoTIFF, как в satellite.py, здесь невозможно, и это проверено:
ассеты лежат на services.terrascope.be, который на HEAD и на range-GET отвечает
401 без токена Terrascope. Регистрация нам запрещена требованием
воспроизводимости, поэтому файл ходит не за пикселями, а за уже посчитанной
статистикой — на titiler.terrascope.be, который стоит перед тем же хранилищем
и открыт без ключа (проверено: 200 на /point, /info и /statistics).

Побочная выгода такого пути важнее, чем кажется: считает сервер. Рамка 7x7 км —
847 тысяч пикселей — обрабатывается за 2,1 с, и ни rasterio, ни GDAL, ни лишние
зависимости для этого не нужны, только requests.

Измерено на живых запросах (доля пикселей пашни в квадрате 600 м):

    поле под Ростовом 39,31 / 47,30      0,969
    поле в Айове -93,5 / 42,0            0,893
    Цимлянское водохранилище             0,000
    Брянский лес, Бузулукский бор        0,000
    центр Ростова / центр Москвы    0,000 / 0,008
    Сахара, тайга Коми                   0,000

Разделение полное, и оно закрывает ровно ту дыру, которая названа вслух в
проверке по NDVI: мелкий пруд с заросшим берегом по спутниковой подписи похож на
поле, а в маске пашни он ноль.

Ограничения, из-за которых низкая доля здесь никогда не приговор:

* Продукт называется temporary crops — однолетние культуры. Сады, виноградники,
  многолетние травы и пастбища в него не входят намеренно, а у нас они считаются
  сельхозугодьями (см. FARM_LANDUSE в parcels.py). Ноль на винограднике — не
  ошибка маски, а её определение.
* Год один — 2021. Поле, распаханное позже или заброшенное раньше, маска
  покажет по состоянию на 2021-й.
* Покрытие не абсолютно: за пределами агрозон (Антарктида, Гренландия — оба
  проверены) поиск возвращает ноль снимков. Это ответ «не знаю», а не «не пашня».
"""
from __future__ import annotations

import os

import requests

from src.providers.cache import cached

# Каталог Terrascope: единственный из проверенных, где продукты WorldCereal
# описаны как STAC-коллекции. В Microsoft Planetary Computer, откуда мы берём
# снимки, WorldCereal отсутствует — в его 136 коллекциях есть esa-worldcover и
# esa-cci-lc, но не WorldCereal (проверено запросом /collections).
STAC_SEARCH_URL = "https://stac.terrascope.be/search"

# Публичный TiTiler перед тем же хранилищем. Он и делает всю работу с растром.
TITILER_URL = "https://titiler.terrascope.be"

# Из шести продуктов WorldCereal берём базовый — маску временных культур.
# Остальные (wintercereals, springcereals, maize, irrigation) проверены на том же
# ростовском поле и дают 0,05-0,08: они выпускались не для всех агрозон и сезонов,
# и трактовать их низкие значения как «не озимая пшеница» нельзя. Продукт
# activecropland отпал по другой причине: его значения выходят за шкалу 0/100,
# то есть кодировка у него не бинарная, и доля по среднему для него не считается.
COLLECTION = "esa-worldcereal-temporarycrops-10m-2021-v1"
ASSET = "classification"

# Значение пикселя «пашня» в этом продукте. Проверено гистограммой: в маске
# ровно два класса, 0 и 100, поэтому среднее по контуру, делённое на 100, и есть
# искомая доля площади.
CROPLAND_VALUE = 100.0

HTTP_HEADERS = {"User-Agent": "fenolog/1.0 (vegetation monitoring service)"}

# Прокси только для этого источника, и только если он задан переменной
# окружения FENOLOG_CROPLAND_PROXY. Причина: Terrascope отвечает не из всех
# сетей — с рабочей машины он открыт, а с сервера соединение не устанавливается
# вовсе. Остальные источники (снимки, погода, контуры) ходят напрямую: гнать их
# через один узкий канал значит замедлить весь сбор ради одной проверки.
#
# Учётные данные держим в переменной окружения, а не в коде: репозиторий
# публичный, и вписанный в него пароль от прокси — это выданный всем пароль.
_PROXY = os.environ.get("FENOLOG_CROPLAND_PROXY", "").strip()
PROXIES = {"http": _PROXY, "https": _PROXY} if _PROXY else None

STAC_TIMEOUT = 25          # поиск лёгкий: замерено 0,5-1,0 с
STATS_TIMEOUT = 90         # статистика тяжелее, но и она укладывалась в 2,1 с
MAX_ITEMS = 3              # на стыке агрозон снимки перекрываются, больше трёх не бывает

# Одна повторная попытка на снимок. Не перестраховка: на прогоне по
# виноградникам Тамани запрос сорвался ровно один раз, а повтор тут же вернул
# нормальный ответ. Без повтора единственный случайный обрыв означал бы «маска
# недоступна» там, где она доступна. Двух попыток достаточно — если сервис лежит,
# третья только задержит ответ.
STATS_ATTEMPTS = 2

# Ниже этой доли валидных пикселей ответу не верим: значит, контур почти целиком
# вышел за край снимка агрозоны, и доля посчитана по случайному его уголку.
MIN_VALID_PERCENT = 50.0


def _ring(geometry: dict) -> list[list[float]]:
    """Внешнее кольцо GeoJSON-геометрии. Принимаем и Feature, и голую геометрию."""
    geom = geometry.get("geometry", geometry)
    coords = geom.get("coordinates") or []
    if geom.get("type") == "MultiPolygon":
        # Контракт проекта — Polygon, но чужой GeoJSON приходит любым: берём
        # первый кусок, этого хватает для поиска нужного снимка агрозоны.
        coords = coords[0] if coords else []
    return list(coords[0]) if coords else []


def _bbox(geometry: dict) -> tuple[float, float, float, float] | None:
    ring = _ring(geometry)
    if len(ring) < 3:
        return None
    lons = [float(p[0]) for p in ring]
    lats = [float(p[1]) for p in ring]
    return min(lons), min(lats), max(lons), max(lats)


def _as_feature(geometry: dict) -> dict:
    """Обернуть геометрию в Feature: TiTiler принимает статистику только так."""
    geom = geometry.get("geometry", geometry)
    return {"type": "Feature", "properties": {}, "geometry": geom}


def _find_items(bbox: tuple[float, float, float, float]) -> list[str]:
    """Идентификаторы снимков WorldCereal, накрывающих рамку. [] — не нашлось."""
    body = {"collections": [COLLECTION], "bbox": list(bbox), "limit": MAX_ITEMS}
    response = requests.post(STAC_SEARCH_URL, json=body,
                             headers=HTTP_HEADERS, timeout=STAC_TIMEOUT,
                             proxies=PROXIES)
    if response.status_code != 200:
        return []
    features = response.json().get("features") or []
    return [f["id"] for f in features if f.get("id")]


def _item_fraction(item_id: str, feature: dict) -> tuple[float, int, float] | None:
    """Доля пашни, число пикселей и процент валидных для одного снимка.

    Считает сервер: мы отправляем контур, получаем среднее по пикселям внутри
    него. Так как класс пашни закодирован числом 100, а фон нулём, среднее,
    делённое на 100, — это в точности доля площади под маской.
    """
    url = (f"{TITILER_URL}/collections/{COLLECTION}/items/{item_id}"
           f"/statistics?assets={ASSET}")
    response = requests.post(url, json=feature, headers=HTTP_HEADERS,
                             timeout=STATS_TIMEOUT, proxies=PROXIES)
    if response.status_code != 200:
        return None
    stats = (response.json().get("properties") or {}).get("statistics") or {}
    band = stats.get("b1") or (next(iter(stats.values())) if stats else None)
    if not band or band.get("mean") is None:
        return None
    valid_percent = float(band.get("valid_percent", 100.0))
    fraction = float(band["mean"]) / CROPLAND_VALUE
    # Среднее приходит с плавающей точкой и на краю снимка может чуть выйти за
    # единицу из-за ресемплинга — подрезаем, наружу уходит честная доля.
    fraction = min(max(fraction, 0.0), 1.0)
    return fraction, int(band.get("valid_pixels") or 0), valid_percent


@cached("cropland_worldcereal", ttl_days=180)
def cropland_fraction(geometry: dict) -> dict | None:
    """Доля площади контура, размеченная как пашня в маске ESA WorldCereal 2021.

    Возвращает:
      {"covered": True,  "fraction": 0.0..1.0, "pixels": int,
       "valid_percent": float, "items": [id, ...], "source": "esa-worldcereal",
       "product": ..., "year": 2021}
      {"covered": False, "fraction": None, ...}  — точка вне покрытия продукта
      None                                        — источник не ответил

    Три исхода вместо двух здесь принципиальны: «маска говорит ноль» и «маски
    для этого места нет» — разные утверждения, и склеивать их в один False
    значило бы объявлять непокрытые районы непашней.

    TTL кэша 180 дней против обычных 30: продукт выпущен один раз за 2021 год и
    меняться уже не будет, ходить за ним в сеть повторно незачем.

    Наружу не выпускается ни одно исключение: недоступный каталог, таймаут,
    неожиданный формат ответа — всё это означает «подтвердить нечем», а не
    аварию сервиса.
    """
    bbox = _bbox(geometry)
    if bbox is None:
        return None

    try:
        items = _find_items(bbox)
    except Exception:  # noqa: BLE001
        return None

    if not items:
        return {"covered": False, "fraction": None, "pixels": 0, "items": [],
                "source": "esa-worldcereal", "product": COLLECTION, "year": 2021}

    feature = _as_feature(geometry)
    best: tuple[float, int, float] | None = None
    used: list[str] = []
    for item_id in items:
        measured = None
        for _ in range(STATS_ATTEMPTS):
            try:
                measured = _item_fraction(item_id, feature)
            except Exception:  # noqa: BLE001
                measured = None
            if measured is not None:
                break
        if measured is None or measured[2] < MIN_VALID_PERCENT:
            continue
        used.append(item_id)
        # Из перекрывающихся снимков берём наибольшую долю, а не среднюю.
        # Причина в устройстве продукта: он нарезан по агроэкологическим зонам, и
        # на стыке зон один и тот же участок попадает в два снимка, у каждого из
        # которых своя модель и свой сезон. Пиксель, признанный пашней хотя бы
        # одной из моделей, — пашня; усреднение же занижало бы долю ровно на
        # границах зон, то есть создавало бы ложные сомнения там, где их нет.
        if best is None or measured[0] > best[0]:
            best = measured

    if best is None:
        return None

    fraction, pixels, valid_percent = best
    return {
        "covered": True,
        "fraction": round(fraction, 3),
        "pixels": pixels,
        "valid_percent": round(valid_percent, 1),
        "items": used,
        "source": "esa-worldcereal",
        "product": COLLECTION,
        "year": 2021,
    }


def is_available() -> bool:
    """Живость источника: проверяем всю цепочку, а не только каталог.

    Одного ответа STAC мало: каталог и хранилище пикселей — разные сервисы, и
    рабочий каталог при закрытом TiTiler дал бы ложное «источник доступен».
    Поэтому пробуем и найти снимок, и прочитать по нему одну точку. Рамка взята
    крошечная над заведомо покрытым районом, весь зонд стоит около секунды.
    """
    try:
        items = _find_items((39.30, 47.29, 39.32, 47.31))
        if not items:
            return False
        url = (f"{TITILER_URL}/collections/{COLLECTION}/items/{items[0]}"
               f"/point/39.31,47.30?assets={ASSET}")
        response = requests.get(url, headers=HTTP_HEADERS, timeout=STAC_TIMEOUT,
                                proxies=PROXIES)
        return response.status_code == 200
    except Exception:  # noqa: BLE001
        return False
