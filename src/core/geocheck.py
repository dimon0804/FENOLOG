"""Проверка координат поля: действительно ли там поле и то ли это поле.

Зачем. Сервис принимает контур от пользователя — нарисованный мышью на карте,
вставленный из GeoJSON, выгруженный из чужой системы. И дальше он считает по
нему всё: норму, аномалии, оценку риска. Если контур попал в лесополосу, в
пруд, на соседний участок или просто сдвинут на двести метров, сервис всё равно
выдаст красивый график и уверенную аномалию — про лесополосу. Пользователю при
этом ничего не покажется странным: цифры-то есть.

Поэтому координаты нужно проверять, а не принимать на веру. Проверка здесь идёт
по четырём независимым каналам, и это принципиально: совпадение независимых
источников — довод, а один источник — просто утверждение.

    1. Сама геометрия. Замкнут ли контур, не вывернут ли он, правдоподобна ли
       площадь, не перепутаны ли местами широта и долгота. Сети не требует.
    2. Спутник. Что показывает NDVI внутри контура за несколько сезонов. У
       пашни размах по сезону большой (медиана 0,56 на выданном наборе), у леса
       и застройки маленький, у воды индекс отрицательный. Пороги измерены, а не
       назначены: числа и методика в журнале экспериментов, E21.
    3. Карта. Есть ли по этим координатам зарегистрированный сельхозконтур в
       открытых данных и насколько он совпадает с присланным. Это ответ на
       вопрос «нашлось ли поле независимо от того, что нам прислали».
    4. Маска пашни ESA WorldCereal. Какая доля контура размечена как пашня в
       глобальной классификации по Sentinel-1/2 за 2021 год. Канал добавлен
       именно четвёртым, потому что он независим от трёх остальных сразу: он не
       смотрит на NDVI этого сезона (в отличие от канала 2) и не зависит от того,
       обвёл ли кто-то это поле руками (в отличие от канала 3). Его вклад
       предметный: он закрывает названную вслух дыру канала 2 — мелкий пруд по
       спутниковой подписи неотличим от поля, а в маске пашни он ноль.

Отдельно проверяется смещение: контур сравнивается со своими же копиями,
сдвинутыми на пару сотен метров. Если сдвинутая копия ведёт себя как поле
заметно убедительнее исходной, контур скорее всего съехал.

Модуль ничего не качает сам. Источники приходят вызовами, которые передаёт слой
провайдеров: доменное ядро не должно знать ни про STAC, ни про Overpass.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np

# --- Пороги геометрии ------------------------------------------------------
# Минимальная площадь: меньше 0,5 га — это огород, а не поле, и один пиксель
# Sentinel-2 (10 м) покрывает уже сотую его часть, то есть усреднять нечего.
MIN_AREA_HA = 0.5
# Верхняя граница: поля больше 5000 га в России единичны, а вот случайно
# обведённый мышью район — обычное дело.
MAX_AREA_HA = 5000.0
# Предельная вытянутость: 30 к 1 — это уже не поле, а лесополоса или дорога.
MAX_ASPECT = 30.0

# --- Пороги спутниковой подписи --------------------------------------------
# Все три числа измерены, а не назначены: 78 полей выданного набора против
# размеченных в OSM леса, застройки и воды рядом с тестовым полем. Методика и
# полная таблица в журнале экспериментов, E21, здесь итог.
#
# Сезонный размах NDVI. У пашни медиана 0,62, пятый процентиль 0,18: поле
# обязано за сезон зазеленеть и сойти.
CROP_RANGE_MIN = 0.19
# Пик. Ниже 0,35 не поднимается только то, на чём ничего не растёт.
CROP_PEAK_MIN = 0.35
# Минимум за сезон — главный разделитель, и это было неочевидно. Размах у леса
# (0,27-0,39) попадает в нижний хвост пашни, а вот минимум нет: у пашни медиана
# 0,156 и 95-й процентиль 0,315, потому что между уборкой и всходами поле
# голое, а лес и застройка не опускаются ниже 0,36 никогда. Порог 0,35 выбран
# по замеру: пашня узнаётся в 91 % случаев при нуле ложных срабатываний на лесу
# и застройке. Порог 0,30 даёт 86 % узнавания, порог 0,40 — те же 91 %, но
# начинает принимать лес за поле.
CROP_TROUGH_MAX = 0.35
# Вода. Открытая вода даёт устойчиво отрицательный NDVI: на Цимлянском
# водохранилище медиана -0,27 при пике 0,32. Порог по медиане, а не по пику:
# отдельные летние снимки цветущей воды дотягивают до 0,3.
WATER_MEDIAN_MAX = 0.0

# --- Пороги совпадения с картой --------------------------------------------
# Доля пересечения, при которой контур считается подтверждённым по карте.
MAP_MATCH_MIN = 0.50
# Сдвиги для проверки на смещение, в метрах
SHIFT_METERS = 200.0

# --- Пороги маски пашни ESA WorldCereal -------------------------------------
# Доля площади контура под маской, начиная с которой участок считается пашней
# подтверждённым. Порог не назначен, а взят с запасом от измеренных значений:
# на настоящих полях маска даёт 0,89-0,97, на воде, лесе и застройке 0,000-0,008
# (замеры в шапке src/providers/cropland.py). Разрыв между этими группами такой,
# что любой порог от 0,2 до 0,8 разделил бы их одинаково; 0,50 выбран потому, что
# он же означает «пашни в контуре больше половины» — величина, которую можно
# объяснить пользователю словами.
CROPLAND_CONFIRM_MIN = 0.50
# Доля, ниже которой маска говорит «пашни здесь нет». Не ноль, потому что край
# контура почти всегда прихватывает дорогу или лесополосу.
CROPLAND_DENY_MAX = 0.10


# ---------------------------------------------------------------- геометрия

def _ring(geometry: dict) -> list[tuple[float, float]]:
    """Внешнее кольцо GeoJSON-полигона. Пустой список, если разобрать нечего."""
    if not isinstance(geometry, dict):
        return []
    geom = geometry.get("geometry", geometry)
    coords = geom.get("coordinates") or []
    if geom.get("type") == "MultiPolygon":
        coords = coords[0] if coords else []
    ring = coords[0] if coords else []
    return [(float(x), float(y)) for x, y in ring if isinstance(x, (int, float))]


def _utm_epsg(lon: float, lat: float) -> int:
    """Код UTM-зоны для точки: в ней площади и расстояния считаются в метрах."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _metric_ring(ring: list[tuple[float, float]]) -> np.ndarray:
    """Кольцо в метрах — локальная равнопромежуточная проекция вокруг центра.

    Полноценный pyproj здесь не нужен: контур поля — это километр-другой, на
    таком масштабе поправка на кривизну Земли меньше точности самих координат.
    Зато модуль остаётся без обязательной зависимости и работает всегда.
    """
    arr = np.asarray(ring, dtype=float)
    lat0 = float(arr[:, 1].mean())
    k = math.cos(math.radians(lat0))
    x = (arr[:, 0] - arr[0, 0]) * 111_320.0 * k
    y = (arr[:, 1] - arr[0, 1]) * 110_540.0
    return np.column_stack([x, y])


def _area_ha(ring: list[tuple[float, float]]) -> float:
    """Площадь контура в гектарах по формуле шнурков в метрической проекции."""
    if len(ring) < 4:
        return 0.0
    m = _metric_ring(ring)
    x, y = m[:, 0], m[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return area / 10_000.0


def _segments_cross(p1, p2, p3, p4) -> bool:
    """Пересекаются ли два отрезка (нужно для поиска самопересечений)."""
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def check_geometry(geometry: dict) -> dict:
    """Проверка контура без обращения к внешним источникам.

    Возвращает словарь с площадью, списком проблем и признаком пригодности.
    Проблемы разделены на «нельзя работать» (ok=False) и предупреждения:
    вытянутый контур странен, но считать по нему можно, а вывернутая наизнанку
    широта делает дальнейший разбор бессмысленным.
    """
    out: dict = {"ok": False, "area_ha": 0.0, "problems": [], "warnings": [],
                 "centroid": None, "points": 0}
    ring = _ring(geometry)
    out["points"] = len(ring)
    if len(ring) < 4:
        out["problems"].append("контур состоит меньше чем из трёх точек")
        return out

    lons = np.array([p[0] for p in ring], dtype=float)
    lats = np.array([p[1] for p in ring], dtype=float)
    if not (np.all(np.abs(lats) <= 90) and np.all(np.abs(lons) <= 180)):
        # Самая частая ошибка при вставке чужого GeoJSON: координаты записаны
        # как (широта, долгота), а GeoJSON требует (долгота, широта). Обычно
        # это видно сразу — «широта» вылезает за 90 градусов.
        if np.all(np.abs(lons) <= 90) and np.all(np.abs(lats) <= 180):
            out["problems"].append(
                "широта и долгота, похоже, переставлены местами: в GeoJSON "
                "порядок (долгота, широта)"
            )
        else:
            out["problems"].append("координаты выходят за допустимые пределы")
        return out

    if abs(ring[0][0] - ring[-1][0]) > 1e-9 or abs(ring[0][1] - ring[-1][1]) > 1e-9:
        out["warnings"].append("контур не замкнут, замыкаем сами")
        ring = ring + [ring[0]]

    out["centroid"] = (round(float(lons.mean()), 6), round(float(lats.mean()), 6))
    area = _area_ha(ring)
    out["area_ha"] = round(area, 2)
    if area < MIN_AREA_HA:
        out["problems"].append(
            f"площадь {area:.2f} га — меньше {MIN_AREA_HA} га; на таком участке "
            f"усреднять по спутнику нечего, один пиксель уже сравним с полем"
        )
    elif area > MAX_AREA_HA:
        out["problems"].append(
            f"площадь {area:.0f} га — больше {MAX_AREA_HA:.0f} га; похоже, "
            f"обведено не поле, а район"
        )

    m = _metric_ring(ring)
    width = float(m[:, 0].max() - m[:, 0].min())
    height = float(m[:, 1].max() - m[:, 1].min())
    short, long_ = sorted((max(width, 1.0), max(height, 1.0)))
    if long_ / short > MAX_ASPECT:
        out["warnings"].append(
            f"контур вытянут как {long_ / short:.0f} к 1 — так выглядят "
            f"лесополосы и дороги, а не поля"
        )

    n = len(ring) - 1
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(ring[i], ring[i + 1], ring[j], ring[j + 1]):
                out["warnings"].append("контур самопересекается, площадь может быть неверной")
                break
        else:
            continue
        break

    out["ok"] = not out["problems"]
    return out


# ------------------------------------------------------- спутниковая подпись

def cover_signature(dates, values) -> dict | None:
    """Сезонная подпись участка: размах, пик, минимум, средний уровень.

    Считается по всем сезонам сразу, а не по одному: единственный сезон может
    оказаться паром, и поле будет объявлено пустырём. Берётся лучший сезон —
    участок считается пашней, если он вёл себя как пашня хотя бы раз.
    """
    from src.core.crop_profile import season_curve

    dates, values = list(dates), list(values)
    if not dates:
        return None
    # Медиана по ВСЕМ наблюдениям, а не по сезонной кривой: она нужна для
    # распознавания воды, а вода не имеет сезона и в кривую может не попасть.
    all_vals = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    median = float(np.median(all_vals)) if all_vals.size else float("nan")

    best = None
    for year in sorted({d.year for d in dates}):
        curve = season_curve(dates, values, year=year)
        if curve is None:
            continue
        sig = {
            "year": year,
            "range": float(curve.max() - curve.min()),
            "peak": float(curve.max()),
            "trough": float(curve.min()),
            "mean": float(curve.mean()),
            "median_all": round(median, 3),
        }
        if best is None or sig["range"] > best["range"]:
            best = sig
    if best is None and all_vals.size >= 5:
        # Полной сезонной кривой нет, но наблюдения есть. Для воды этого уже
        # достаточно: она распознаётся по медиане, а сезона у неё нет.
        best = {"year": None, "range": float("nan"), "peak": float(all_vals.max()),
                "trough": float(all_vals.min()), "mean": float(all_vals.mean()),
                "median_all": round(median, 3)}
    return best


def classify_cover(sig: dict | None) -> tuple[str, str]:
    """Что это за участок по спутниковой подписи: (класс, объяснение).

    Классы: «пашня», «многолетняя растительность», «вода», «без растительности»,
    «не определено». Последний — честный ответ, когда снимков не хватило.
    """
    if not sig:
        return "не определено", (
            "снимков не хватило: за проверяемые сезоны не собралось ни одной "
            "полной кривой апрель-октябрь"
        )
    if sig.get("median_all", 1.0) < WATER_MEDIAN_MAX:
        return "вода", (
            f"медианный индекс за все снимки {sig['median_all']:.2f} — "
            f"отрицательный индекс даёт только водная поверхность"
        )
    if not np.isfinite(sig["range"]):
        return "не определено", (
            "снимки есть, но ни одного сезона с покрытием апрель-октябрь: "
            "по обрывку ряда судить о том, что это за участок, нельзя"
        )
    if (sig["range"] >= CROP_RANGE_MIN and sig["peak"] >= CROP_PEAK_MIN
            and sig["trough"] < CROP_TROUGH_MAX):
        return "пашня", (
            f"за сезон {sig['year']} индекс вырос до {sig['peak']:.2f} и сошёл до "
            f"{sig['trough']:.2f}, размах {sig['range']:.2f} — так выглядит поле "
            f"с посевом и уборкой"
        )
    if sig["peak"] < CROP_PEAK_MIN:
        return "без растительности", (
            f"индекс не поднимается выше {sig['peak']:.2f} — застройка, дорога, "
            f"голый грунт или открытая вода"
        )
    if sig["trough"] >= CROP_TROUGH_MAX:
        return "многолетняя растительность", (
            f"индекс за сезон меняется на {sig['range']:.2f}, но даже в минимуме не "
            f"опускается ниже {sig['trough']:.2f}. Поле после уборки становится "
            f"голым, а этот участок зелен круглый сезон — лес, лесополоса, сад, "
            f"залежь или газон"
        )
    return "не определено", (
        f"подпись участка не похожа ни на поле, ни на лес: размах {sig['range']:.2f}, "
        f"пик {sig['peak']:.2f}, минимум {sig['trough']:.2f}. Так выглядят мелкие "
        f"объекты, где в пиксель попадает и участок, и его окружение"
    )


# ------------------------------------------------------------ сдвиг контура

def _swap_lonlat(geometry: dict) -> dict | None:
    """Контур с переставленными местами широтой и долготой.

    None, если после перестановки координаты выходят за допустимые пределы —
    тогда проверять нечего, ошибка была бы видна и без спутника.
    """
    ring = _ring(geometry)
    if not ring:
        return None
    if any(abs(x) > 90 for x, _ in ring):
        return None
    return {"type": "Polygon", "coordinates": [[[y, x] for x, y in ring]]}


def _shift(geometry: dict, dx_m: float, dy_m: float) -> dict:
    """Копия контура, сдвинутая на заданное число метров."""
    ring = _ring(geometry)
    lat0 = float(np.mean([p[1] for p in ring]))
    dlon = dx_m / (111_320.0 * max(math.cos(math.radians(lat0)), 1e-6))
    dlat = dy_m / 110_540.0
    return {"type": "Polygon", "coordinates": [[[x + dlon, y + dlat] for x, y in ring]]}


# ------------------------------------------------------------ совпадение карт

def _ring_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Грубая оценка совпадения двух контуров: доля общей площади рамок.

    Именно рамок, а не самих полигонов. Полноценное пересечение требовало бы
    shapely, а задача здесь — не измерить площадь перекрытия, а ответить «это
    примерно тот же участок или совсем другой». Для ответа на такой вопрос
    точности рамок достаточно, а зависимость модуль не тянет.
    """
    if len(a) < 3 or len(b) < 3:
        return 0.0
    ax = [p[0] for p in a]; ay = [p[1] for p in a]
    bx = [p[0] for p in b]; by = [p[1] for p in b]
    ix = max(0.0, min(max(ax), max(bx)) - max(min(ax), min(bx)))
    iy = max(0.0, min(max(ay), max(by)) - max(min(ay), min(by)))
    inter = ix * iy
    area_a = (max(ax) - min(ax)) * (max(ay) - min(ay))
    area_b = (max(bx) - min(bx)) * (max(by) - min(by))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def verify_polygon(
    geometry: dict,
    fetch_observations=None,
    find_parcels=None,
    cropland_mask=None,
    years: int = 3,
    check_shift: bool = False,
    today: date | None = None,
) -> dict:
    """Полная проверка координат поля по всем доступным каналам.

    fetch_observations — callable(geometry, start, end) -> list[Observation].
    find_parcels       — callable(bbox) -> list[{"geometry": ...}], где bbox
                         это (запад, юг, восток, север) в градусах.
    cropland_mask      — callable(geometry) -> {"covered": bool,
                         "fraction": float | None, ...} | None. Доля контура под
                         маской пашни; None означает «источник не ответил».
    check_shift        — проверять ли смещение контура. Стоит четырёх лишних
                         загрузок снимков, поэтому по умолчанию выключено.

    Любой источник может отсутствовать: проверка тогда проходит по остальным и
    честно пишет, каких каналов не было. Отсутствие сети не должно означать
    «контур не подтверждён» — это означает «подтвердить было нечем».
    """
    today = today or date.today()
    out: dict = {
        "geometry": check_geometry(geometry),
        "satellite": None, "map": None, "cropland": None, "shift": None,
        "confirmed_by": [], "verdict": "", "problems": [], "warnings": [],
    }
    out["problems"].extend(out["geometry"]["problems"])
    out["warnings"].extend(out["geometry"]["warnings"])
    if not out["geometry"]["ok"]:
        out["verdict"] = (
            "контур не проходит проверку геометрии, дальше проверять нечего: "
            + "; ".join(out["geometry"]["problems"])
        )
        return out

    start = date(today.year - years + 1, 4, 1)
    end = min(today, date(today.year, 10, 31))

    # --- Канал 1: спутник --------------------------------------------------
    own_obs = []
    if fetch_observations is not None:
        try:
            own_obs = [o for o in fetch_observations(geometry, start, end)
                       if o.ndvi is not None]
        except Exception as exc:  # noqa: BLE001
            out["warnings"].append(f"снимки недоступны: {type(exc).__name__}")
        sig = cover_signature([o.date for o in own_obs], [o.ndvi for o in own_obs])
        kind, why = classify_cover(sig)
        out["satellite"] = {"class": kind, "reason": why, "signature": sig,
                            "observations": len(own_obs)}
        if kind == "пашня":
            out["confirmed_by"].append("спутник")
        elif kind != "не определено":
            out["problems"].append(f"по снимкам это не пашня, а {kind}: {why}")

        # Перестановка широты и долготы — самая частая и самая незаметная
        # ошибка во входных данных: GeoJSON требует (долгота, широта), а почти
        # все остальные форматы пишут наоборот. Поймать её проверкой диапазона
        # можно только когда «широта» вылезла за 90; в средних широтах России
        # оба числа остаются допустимыми, и контур молча уезжает за тысячу
        # километров. Зато его можно найти: если по исходным координатам поля
        # нет, а по переставленным есть — ошибка названа точно.
        if kind != "пашня":
            swapped = _swap_lonlat(geometry)
            if swapped is not None:
                try:
                    sw_obs = [o for o in fetch_observations(swapped, start, end)
                              if o.ndvi is not None]
                except Exception:  # noqa: BLE001
                    sw_obs = []
                sw_kind, sw_why = classify_cover(
                    cover_signature([o.date for o in sw_obs], [o.ndvi for o in sw_obs])
                )
                out["satellite"]["swapped_class"] = sw_kind
                if sw_kind == "пашня":
                    out["problems"].append(
                        "похоже, широта и долгота переставлены местами: по "
                        "присланным координатам поля нет, а по переставленным "
                        f"есть ({sw_why})"
                    )

    # --- Канал 2: карта ----------------------------------------------------
    if find_parcels is not None:
        ring = _ring(geometry)
        lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
        pad = 0.02  # примерно два километра
        bbox = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
        try:
            parcels = find_parcels(bbox) or []
        except Exception as exc:  # noqa: BLE001
            parcels = []
            out["warnings"].append(f"карта недоступна: {type(exc).__name__}")
        best, best_share = None, 0.0
        for p in parcels:
            share = _ring_overlap(ring, _ring(p.get("geometry", p)))
            if share > best_share:
                best, best_share = p, share
        out["map"] = {
            "parcels_nearby": len(parcels),
            "best_overlap": round(best_share, 2),
            "matched": bool(best_share >= MAP_MATCH_MIN),
            "crop_hint": (best or {}).get("crop_hint"),
            "name": (best or {}).get("name"),
        }
        # Три разных исхода, и путать их нельзя. Полное отсутствие пересечения
        # почти всегда означает, что поле просто не размечено в OSM: покрытие
        # там разреженное, и на юге России размечена меньшая часть угодий. А вот
        # частичное пересечение — совсем другой разговор: контур есть, он рядом,
        # но границы не сходятся, и это уже похоже на смещение или на соседний
        # участок. Одинаковое предупреждение на оба случая приучило бы
        # пользователя не читать предупреждения вовсе.
        if best_share >= MAP_MATCH_MIN:
            out["confirmed_by"].append("карта")
        elif best_share > 0.05:
            out["warnings"].append(
                f"рядом размечен сельхозконтур, но с присланным он совпадает лишь "
                f"на {best_share:.0%} — проверьте, не смещён ли контур и то ли "
                f"поле обведено"
            )
        else:
            out["warnings"].append(
                f"в открытых данных этот участок как сельхозугодье не размечен "
                f"(рядом найдено контуров: {len(parcels)}). Это не ошибка: в OSM "
                f"размечена меньшая часть полей, поэтому подтверждения по карте "
                f"может не быть и у совершенно настоящего поля"
            )

    # --- Канал 3: маска пашни ESA WorldCereal ------------------------------
    if cropland_mask is not None:
        mask = None
        try:
            mask = cropland_mask(geometry)
        except Exception as exc:  # noqa: BLE001
            out["warnings"].append(f"маска пашни недоступна: {type(exc).__name__}")
        if mask is None:
            out["cropland"] = {"covered": False, "fraction": None,
                               "status": "источник не ответил"}
        elif not mask.get("covered"):
            # Вне агрозон продукта. Это «нечем подтвердить», и говорить об этом
            # надо прямо: молчание здесь читалось бы как «маска не возражает».
            out["cropland"] = {"covered": False, "fraction": None,
                               "status": "участок вне покрытия ESA WorldCereal"}
        else:
            share = float(mask.get("fraction") or 0.0)
            out["cropland"] = {
                "covered": True,
                "fraction": round(share, 3),
                "pixels": mask.get("pixels"),
                "year": mask.get("year"),
                "status": "",
            }
            sat_class = (out["satellite"] or {}).get("class")
            if share >= CROPLAND_CONFIRM_MIN:
                out["confirmed_by"].append("маска пашни")
                out["cropland"]["status"] = (
                    f"{share:.0%} площади контура размечено как пашня")
            elif share <= CROPLAND_DENY_MAX:
                out["cropland"]["status"] = (
                    f"пашней размечено лишь {share:.0%} площади контура")
                # Низкая доля сама по себе НЕ приговор, и это не осторожность, а
                # определение продукта: WorldCereal размечает однолетние
                # культуры, а сады, виноградники и пастбища в маску не входят —
                # у нас же они полноправные угодья (FARM_LANDUSE в parcels.py).
                # Поэтому вето маска получает только вместе со вторым несогласным
                # каналом: когда спутник тоже не смог назвать участок пашней,
                # два независимых источника говорят одно и то же, и это уже
                # довод, а не особенность одного из них.
                if sat_class in (None, "не определено"):
                    out["problems"].append(
                        f"в маске пашни ESA WorldCereal этот участок пашней не "
                        f"размечен ({share:.0%} площади), а спутниковая подпись "
                        f"ничего не подтвердила — проверьте координаты"
                    )
                elif sat_class == "пашня":
                    out["warnings"].append(
                        f"спутник видит пашню, но в маске ESA WorldCereal за 2021 год "
                        f"этот участок размечен пашней лишь на {share:.0%}. Так "
                        f"выглядят сады, виноградники и многолетние травы — их эта "
                        f"маска не включает — а также поля, распаханные после 2021 года"
                    )
                else:
                    out["warnings"].append(
                        f"маска ESA WorldCereal согласна со снимками: пашни в "
                        f"контуре {share:.0%}"
                    )
            else:
                # Промежуточная доля — почти всегда контур захватил лишнее:
                # половину лесополосы, дорогу, край соседнего луга.
                out["cropland"]["status"] = (
                    f"пашней размечено {share:.0%} площади контура")
                out["warnings"].append(
                    f"по маске ESA WorldCereal пашня занимает {share:.0%} контура — "
                    f"похоже, в контур попало лишнее (лесополоса, дорога, край "
                    f"соседнего участка)"
                )

    # --- Канал 4: смещение -------------------------------------------------
    if check_shift and fetch_observations is not None and own_obs:
        own_sig = cover_signature([o.date for o in own_obs], [o.ndvi for o in own_obs])
        own_range = (own_sig or {}).get("range", 0.0)
        best_dir, best_range = None, own_range
        for dx, dy, name in ((SHIFT_METERS, 0, "восток"), (-SHIFT_METERS, 0, "запад"),
                             (0, SHIFT_METERS, "север"), (0, -SHIFT_METERS, "юг")):
            try:
                obs = [o for o in fetch_observations(_shift(geometry, dx, dy), start, end)
                       if o.ndvi is not None]
            except Exception:  # noqa: BLE001
                continue
            sig = cover_signature([o.date for o in obs], [o.ndvi for o in obs])
            if sig and sig["range"] > best_range:
                best_dir, best_range = name, sig["range"]
        out["shift"] = {
            "own_range": round(own_range, 3),
            "best_direction": best_dir,
            "best_range": round(best_range, 3),
            # Порог в 25 % не случаен: собственный шум сезонного размаха между
            # соседними участками одного поля около 10 %, и меньшая разница
            # означала бы, что мы ловим шум, а не смещение.
            "suspected": bool(best_dir and best_range > own_range * 1.25),
        }
        if out["shift"]["suspected"]:
            out["warnings"].append(
                f"участок в {SHIFT_METERS:.0f} м к {best_dir}у ведёт себя как поле "
                f"убедительнее присланного контура (размах {best_range:.2f} против "
                f"{own_range:.2f}) — контур мог съехать"
            )

    # --- Приговор ----------------------------------------------------------
    if out["problems"]:
        out["verdict"] = "координаты вызывают сомнения: " + "; ".join(out["problems"])
    elif len(out["confirmed_by"]) >= 2:
        # Источников теперь может быть три, поэтому перечисление собирается
        # по-русски: «спутник, карта и маска пашни», а не «а и б и в».
        names = out["confirmed_by"]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " и " + names[-1]
        count = {2: "двумя", 3: "тремя"}.get(len(names), str(len(names)))
        out["verdict"] = (
            f"координаты подтверждены независимо {count} источниками "
            f"({listed}): по этим координатам действительно "
            f"обрабатываемое поле площадью {out['geometry']['area_ha']:.1f} га"
        )
    elif out["confirmed_by"]:
        out["verdict"] = (
            f"координаты подтверждены одним источником ({out['confirmed_by'][0]}): "
            f"поле площадью {out['geometry']['area_ha']:.1f} га. Второго "
            f"подтверждения получить не удалось"
        )
    else:
        out["verdict"] = (
            "подтвердить координаты не удалось ни по снимкам, ни по карте. "
            "Геометрия при этом в порядке, поэтому разбор возможен, но "
            "относиться к нему стоит осторожно"
        )
    return out
