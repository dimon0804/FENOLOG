"""Суточная поправка по соседним полям для сервиса, разбирающего одно поле.

Зачем этот модуль существует.

Лучший приём проекта — снятие общей суточной помехи по соседним полям. Дымка,
тонкое облако, угол Солнца и поправки атмосферной коррекции сдвигают индекс
сразу у всей группы полей, снятых одним пролётом спутника, и эту составляющую
можно измерить и вычесть. На пакетном разборе, где все поля видны сразу, приём
даёт основной прирост: ошибка падает с 0,0794 до 0,0694.

Но сервис разбирает одно поле за раз, и соседей у него нет. Это ограничение
честно стояло в отчёте.

Здесь оно снимается. Поиск сельхозконтуров у нас уже работает — тот же, что
показывает пользователю поля на карте. Значит по выбранному полю можно найти
его соседей, собрать их ряды и посчитать ту же самую поправку. Соседи качаются
параллельно и кэшируются, поэтому первый разбор дороже одиночного примерно
вдвое по времени, а повторный — бесплатен.

Приём остаётся физически обоснованным: это не подгонка, а снятие измеренной
общей составляющей, того же рода, что атмосферная коррекция, которую поставщик
данных применяет ещё до нас.
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import date

import numpy as np

from src.contracts import Observation

log = logging.getLogger(__name__)

# Сколько соседей брать. Больше восьми не нужно: поправка это среднее, и её
# точность растёт как корень из числа полей — девятый сосед добавляет три
# процента точности за двенадцать процентов времени.
NEIGHBOUR_COUNT = 6

# Радиус поиска. Общая помеха — это атмосфера и геометрия съёмки, они меняются
# на масштабе десятков километров. Пятнадцать километров — компромисс между
# «та же самая атмосфера» и «в радиусе вообще есть поля».
NEIGHBOUR_RADIUS_KM = 15.0

# Минимум соседей, при котором поправка вообще применяется. По одному-двум полям
# среднее само по себе шум, и вычитать его вреднее, чем не вычитать ничего.
MIN_NEIGHBOURS = 3

# Потолок времени на сбор соседей. Если источники тормозят, лучше отдать разбор
# без поправки, чем заставить пользователя ждать вдвое дольше.
TIME_BUDGET_S = 150.0

# Сколько соседей разрешено качать из сети за один разбор. Остальные берутся
# только из кэша. Смысл: первый разбор в новом районе не должен стоить семь
# минут, а после разбора района или повторного захода все соседи уже лежат
# рядом и поправка достаётся бесплатно.
MAX_FRESH = 3

# Сила сглаживания, по которой считается остаток. Та же, что в пакетном режиме.
SMOOTH_LAM = 1000.0

# Подрезка остатка: одно грубо испорченное поле не должно тянуть за собой всю
# группу. Границы несимметричны — распределение остатков скошено влево, потому
# что облачность занижает индекс сильнее, чем что-либо его завышает.
CLIP_LO, CLIP_HI = -0.15, 0.25


def _bbox_around(geometry: dict, radius_km: float) -> tuple[float, float, float, float]:
    """Рамка вокруг контура, расширенная на радиус поиска."""
    from src.providers.satellite import _to_shapely

    shape = _to_shapely(geometry)
    west, south, east, north = shape.bounds
    lat = (south + north) / 2.0
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 0.1))
    return (west - dlon, south - dlat, east + dlon, north + dlat)


def _residuals(obs: list[Observation]) -> dict[date, float]:
    """Остаток каждого наблюдения от собственной сглаженной кривой поля.

    Именно остаток, а не само значение: уровень индекса у полей разный, а вот
    промах относительно своей кривой — величина сравнимая между полями.
    """
    from src.core.restore import restore_on_grid

    pts = [(o.date, o.ndvi) for o in obs if o.ndvi is not None and np.isfinite(o.ndvi)]
    if len(pts) < 8:
        return {}
    pts.sort()
    days = np.array([d.toordinal() for d, _ in pts], dtype=np.int64)
    vals = np.array([v for _, v in pts], dtype=float)
    try:
        grid, smooth = restore_on_grid(days, vals, lam=SMOOTH_LAM, mix=1.0)
    except Exception:  # noqa: BLE001 — соседа можно потерять, разбор нельзя
        return {}
    fitted = smooth[days - grid[0]]
    return {d: float(v - f) for (d, _), v, f in zip(pts, vals, fitted)}


def daily_correction(
    geometry: dict,
    start: date,
    end: date,
    progress=None,
    count: int = NEIGHBOUR_COUNT,
    radius_km: float = NEIGHBOUR_RADIUS_KM,
) -> tuple[dict[date, float], dict]:
    """Общая суточная помеха по соседним полям.

    Возвращает (поправка по датам, диагностика). Пустая поправка — штатный
    исход: соседей не нашлось, источники не ответили, времени не хватило.
    Разбор в этом случае просто идёт без неё.
    """
    t0 = time.perf_counter()
    info = {"requested": count, "used": 0, "applied": False, "reason": None}

    from src.providers.parcels import find_parcels
    from src.providers.satellite import _to_shapely, fetch_observations

    try:
        target = _to_shapely(geometry)
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"геометрия не разобрана: {type(exc).__name__}"
        return {}, info

    try:
        parcels = find_parcels(_bbox_around(geometry, radius_km), limit=count * 3)
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"поиск соседей не удался: {type(exc).__name__}"
        return {}, info

    # Отбрасываем сам целевой контур и всё, что с ним заметно перекрывается:
    # поправка должна приходить от других полей, иначе поле вычтет свой же шум.
    neighbours = []
    for p in parcels:
        try:
            shape = _to_shapely(p["geometry"])
            if shape.intersection(target).area > 0.2 * min(shape.area, target.area):
                continue
            neighbours.append(p)
        except Exception:  # noqa: BLE001
            continue
        if len(neighbours) >= count:
            break

    if len(neighbours) < MIN_NEIGHBOURS:
        info["reason"] = f"соседей найдено {len(neighbours)}, нужно {MIN_NEIGHBOURS}"
        return {}, info

    # Соседи делятся на две очереди. Те, что уже лежат в кэше, достаются
    # мгновенно и берутся все. Свежих качаем не больше MAX_FRESH: иначе первый
    # разбор в новом районе превращается в многоминутное ожидание.
    from src.providers.satellite import is_cached

    warm = [n for n in neighbours if is_cached(n["geometry"], start, end)]
    cold = [n for n in neighbours if n not in warm][:MAX_FRESH]
    queue = warm + cold
    info["cached"] = len(warm)
    info["fetched"] = len(cold)

    if progress:
        try:
            progress("собираю соседние поля", 0, len(queue))
        except Exception:  # noqa: BLE001
            pass

    tables: list[dict[date, float]] = []
    done = 0
    pool = ThreadPoolExecutor(max_workers=min(6, max(len(queue), 1)))
    try:
        futures = {pool.submit(fetch_observations, n["geometry"], start, end): n for n in queue}
        # Бюджет времени задаётся самому ожиданию, а не проверяется между
        # завершениями. Разница принципиальна: проверка в теле цикла срабатывает
        # только когда очередной сосед досчитался, и один медленный источник
        # держит разбор сколько угодно долго, ни разу не дав циклу провернуться.
        # Замерено на холодном районе: сбор шёл тринадцать минут при бюджете
        # в две с половиной.
        try:
            left = max(TIME_BUDGET_S - (time.perf_counter() - t0), 1.0)
            for fut in as_completed(futures, timeout=left):
                try:
                    res = _residuals(fut.result())
                except Exception:  # noqa: BLE001
                    res = {}
                if res:
                    tables.append(res)
                done += 1
                if progress:
                    try:
                        progress("собираю соседние поля", done, len(queue))
                    except Exception:  # noqa: BLE001
                        pass
                if time.perf_counter() - t0 > TIME_BUDGET_S:
                    info["reason"] = "бюджет времени исчерпан, поправка по собранным"
                    break
        except FuturesTimeout:
            # Штатный исход, а не сбой: уходим с теми соседями, что успели.
            info["reason"] = "бюджет времени исчерпан, поправка по собранным"
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if len(tables) < MIN_NEIGHBOURS:
        info["reason"] = info["reason"] or f"рядов собрано {len(tables)}, нужно {MIN_NEIGHBOURS}"
        return {}, info

    # Усечённое среднее остатков по каждой дате. Медиана была бы устойчивее, но
    # на восьми полях она грубее среднего, а подрезка снимает ровно ту проблему,
    # ради которой медиану обычно и берут.
    by_day: dict[date, list[float]] = {}
    for t in tables:
        for d, v in t.items():
            by_day.setdefault(d, []).append(min(max(v, CLIP_LO), CLIP_HI))

    correction = {
        d: float(np.mean(vs)) for d, vs in by_day.items() if len(vs) >= MIN_NEIGHBOURS
    }
    if not correction:
        info["reason"] = "нет дат, где сошлось хотя бы три соседа"
        return {}, info

    vals = np.array(list(correction.values()), dtype=float)
    info.update({
        "used": len(tables),
        "applied": True,
        "days": len(correction),
        "std": round(float(np.std(vals)), 4),
        "seconds": round(time.perf_counter() - t0, 1),
    })
    log.info("siblings: поправка по %d полям, %d дат, std %.4f",
             len(tables), len(correction), float(np.std(vals)))
    return correction, info


def apply_correction(obs: list[Observation], correction: dict[date, float]) -> list[Observation]:
    """Вычитает общую суточную помеху из наблюдений поля.

    Значения остаются наблюдениями: снимается измеренная общая составляющая,
    того же рода, что атмосферная коррекция поставщика данных. Даты, для которых
    поправки нет, остаются нетронутыми.
    """
    if not correction:
        return obs
    out = []
    for o in obs:
        c = correction.get(o.date)
        if c is None or o.ndvi is None:
            out.append(o)
            continue
        out.append(Observation(
            date=o.date,
            ndvi=float(min(max(o.ndvi - c, 0.0), 1.0)),
            evi=o.evi, ndwi=o.ndwi, source=o.source,
        ))
    return out
