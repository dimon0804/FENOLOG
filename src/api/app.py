"""Приложение FastAPI: маршруты сервиса и отдача интерфейса.

Один процесс на всё: HTTP, фоновые задачи и статика собранного фронтенда. Так
`docker compose up` поднимает сервис одной командой и на одном порту, без
отдельного веб-сервера и без настройки прокси — а это прямой критерий
воспроизводимости.

Обработчики, ходящие в сеть, объявлены обычным `def`, а не `async def`. Это не
небрежность: FastAPI уводит синхронные обработчики в пул потоков, и запрос к
Overpass на десятки секунд не блокирует цикл событий. `async def` с блокирующим
requests внутри остановил бы весь сервис на время запроса.

Правило по ошибкам: наружу не должно уходить ни одной пятисотой из-за внешнего
источника. Отказ источника — штатная ситуация, о ней сообщается словами.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api import config, geocoding
from src.api.geometry import GeometryError, bbox_of, normalize_geometry, parse_bbox, validate_for_analysis
from src.api.health import providers_health
from src.api.schemas import (
    AnalyzeRequest,
    DiscoverRequest,
    PolygonAnalyzeRequest,
    PolygonCreate,
    PolygonPatch,
)
from src.api.storage import store
from src.api.tasks import drop_result, load_result, load_summary, manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fenolog.api")

VERSION = "1.0"

app = FastAPI(
    title="Фенолог — мониторинг вегетационной динамики",
    version=VERSION,
    description=(
        "Сервис собирает спутниковые и метеоданные по произвольному контуру, "
        "строит ряд вегетационных индексов, восстанавливает пропуски и находит "
        "негативные аномальные периоды с интерпретацией причины."
    ),
)

# В разработке фронтенд поднимается своим сервером на другом порту, в проде он
# отдаётся отсюда же. Разрешаем всё: аутентификации в сервисе нет, защищать
# нечего, а несовпадение портов на демонстрации стоило бы половины времени.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(GeometryError)
def _geometry_error(_request, exc: GeometryError) -> JSONResponse:
    """Негодная геометрия — ошибка пользователя, а не сбой сервиса."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.on_event("startup")
def _on_startup() -> None:
    """Что важно увидеть в логе при старте.

    Отсутствие climatology по культурам — самая коварная из возможных проблем:
    сервис поднимется и будет работать, но на полях без собственной истории (это
    три четверти полигонов) покажет пустой список периодов. Выглядит как баг в
    детекции, а на деле — незабранный в образ файл.
    """
    clim = config.MODELS_DIR / "crop_climatology.json"
    if clim.exists():
        log.info("норма по культурам на месте: %s", clim)
    else:
        log.warning(
            "НЕТ %s — на полях без собственной истории периоды искаться не будут",
            clim,
        )
    log.info("интерфейс: %s", config.WEB_DIST if config.WEB_DIST.exists() else "не собран, доступен только API")


# --------------------------------------------------------------------------------------
# Живость и состояние источников
# --------------------------------------------------------------------------------------

@app.get("/health", tags=["служебное"])
def health() -> dict:
    """Живость самого сервиса. Отвечает мгновенно и никуда не ходит —
    именно это нужно docker healthcheck."""
    return {"status": "ok", "service": "fenolog", "version": VERSION}


@app.get("/api/providers/health", tags=["служебное"])
def api_providers_health(force: bool = Query(False, description="проверить заново, минуя кэш")) -> dict:
    """Состояние внешних источников и следствия их отказа."""
    return providers_health(force=force)


@app.get("/api/summary", tags=["служебное"])
def summary() -> dict:
    """Сводка по всем сохранённым участкам — то, что показывает экран «Обзор».

    Считается по выжимкам разборов, а не по самим разборам: полный ряд за пять
    сезонов весит сотни килобайт на поле, и собирать их все ради трёх чисел на
    дашборде значило бы гонять мегабайты на каждое открытие экрана.
    """
    polygons = store.list_all()
    fields, total, critical, suppression = [], 0, 0, 0
    last_analyzed = None

    for polygon in polygons:
        digest = load_summary(polygon["id"])
        if digest:
            total += digest["anomalies"]
            critical += digest["critical"]
            suppression += digest["suppression"]
        stamp = polygon.get("last_analyzed_at")
        if stamp and (last_analyzed is None or stamp > last_analyzed):
            last_analyzed = stamp
        fields.append({
            "id": polygon["id"],
            "name": polygon["name"],
            "area_ha": polygon["area_ha"],
            "crop_type": polygon.get("crop_type"),
            "source": polygon.get("source"),
            "center": polygon.get("center"),
            "last_analyzed_at": stamp,
            "summary": digest,
        })

    # Худшие поля наверх: агроному нужно сразу видеть, где хуже всего, а не
    # листать список в порядке добавления.
    fields.sort(key=lambda f: (f["summary"] or {}).get("worst_zscore") or 0.0)

    analyzed = sum(1 for f in fields if f["summary"])
    return {
        "polygons": len(polygons),
        "analyzed": analyzed,
        "pending": len(polygons) - analyzed,
        "anomalies": {"total": total, "critical": critical, "suppression": suppression},
        "last_analyzed_at": last_analyzed,
        "total_area_ha": round(sum(p["area_ha"] for p in polygons), 1),
        "fields": fields,
    }


# --------------------------------------------------------------------------------------
# Территория: поиск региона и готовые контуры
# --------------------------------------------------------------------------------------

@app.get("/api/regions/search", tags=["территория"])
def regions_search(
    q: str = Query(..., min_length=2, description="название региона, района или города"),
    limit: int = Query(5, ge=1, le=20),
) -> dict:
    """Название -> центр, рамка и, если есть, граница региона."""
    places = geocoding.search_region(q, limit=limit)
    return {
        "query": q,
        "places": places,
        # Пустая выдача бывает двух видов, и интерфейс обязан их различать:
        # «ничего не нашлось» и «геокодер недоступен». Второе — не повод
        # переспрашивать пользователя другими словами.
        "geocoder_available": bool(places) or geocoding.is_available(),
    }


@app.get("/api/polygons/discover", tags=["территория"])
def polygons_discover(
    bbox: str = Query(..., description="рамка карты: запад,юг,восток,север"),
    limit: int = Query(50, ge=1, le=300),
) -> dict:
    """Сельхозконтуры из OpenStreetMap в текущей рамке карты."""
    return _discover(parse_bbox(bbox), limit)


@app.post("/api/regions/parcels", tags=["территория"])
def regions_parcels(request: DiscoverRequest) -> dict:
    """То же, но рамку можно не считать самому: достаточно прислать геометрию.

    Нужно после поиска региона — граница приходит полигоном, и считать по нему
    рамку на фронтенде значило бы дублировать логику.
    """
    if request.bbox:
        west, south, east, north = request.bbox
        box = parse_bbox(f"{west},{south},{east},{north}")
    elif request.geometry:
        box = bbox_of(normalize_geometry(request.geometry))
    else:
        raise HTTPException(status_code=400, detail="Нужен либо bbox, либо geometry")
    return _discover(box, request.limit)


def _discover(box: tuple[float, float, float, float], limit: int) -> dict:
    from src.providers import parcels

    started = time.perf_counter()
    try:
        found = parcels.find_parcels(box, limit=limit)
    except Exception as exc:  # noqa: BLE001 — Overpass не имеет права ронять сервис
        log.warning("поиск контуров не удался: %s", exc)
        found = []
    seconds = round(time.perf_counter() - started, 1)

    # Пустая выдача Overpass — два разных события, которые выглядят одинаково:
    # «в этой рамке полей не размечено» и «источник не ответил». Разница
    # принципиальная: в первом случае пользователю надо двигать карту, во втором
    # ждать бесполезно и нужно рисовать контур руками. Разделяем их отдельной
    # лёгкой проверкой живости — она делается только на пустом ответе и потому
    # ничего не стоит в обычном сценарии.
    source_alive = True
    if not found:
        source_alive = parcels.is_available()

    if found:
        note = None
    elif source_alive:
        note = ("В этой рамке OpenStreetMap не знает сельхозконтуров. "
                "Попробуйте соседний участок карты или нарисуйте полигон сами.")
    else:
        note = ("Источник контуров сейчас недоступен. "
                "Полигон можно нарисовать вручную — дальше путь анализа тот же.")

    return {
        "bbox": list(box),
        "parcels": found,
        "count": len(found),
        "source_available": source_alive,
        "seconds": seconds,
        "note": note,
    }


# --------------------------------------------------------------------------------------
# Анализ
# --------------------------------------------------------------------------------------

@app.post("/api/analyze", status_code=202, tags=["анализ"])
def analyze_geometry(request: AnalyzeRequest) -> dict:
    """Запустить анализ произвольного контура. Отдаёт идентификатор задачи.

    Сохранять контур при этом не обязательно: постановка требует, чтобы
    нарисованный полигон анализировался сразу, до всякого сохранения.
    """
    geometry = validate_for_analysis(request.geometry)

    polygon = None
    if request.save:
        polygon = store.create(
            geometry,
            name=request.name,
            crop_type=request.crop_type,
            source=request.source,
            external_id=request.external_id,
        )

    task = manager.submit(
        geometry,
        polygon_id=polygon["id"] if polygon else None,
        polygon_name=polygon["name"] if polygon else request.name,
        crop_type=request.crop_type,
        years=request.years,
        max_scenes=request.max_scenes,
    )
    return {"task": task.public(), "polygon": polygon}


@app.post("/api/polygons/{polygon_id}/analyze", status_code=202, tags=["анализ"])
def analyze_polygon(polygon_id: str, request: PolygonAnalyzeRequest | None = None) -> dict:
    """Пересчитать сохранённый участок."""
    polygon = store.get(polygon_id)
    if polygon is None:
        raise HTTPException(status_code=404, detail="Участок не найден")

    request = request or PolygonAnalyzeRequest()
    crop_type = request.crop_type if request.crop_type is not None else polygon.get("crop_type")

    task = manager.submit(
        polygon["geometry"],
        polygon_id=polygon["id"],
        polygon_name=polygon["name"],
        crop_type=crop_type,
        years=request.years,
        max_scenes=request.max_scenes,
    )
    return {"task": task.public(), "polygon": polygon}


@app.get("/api/tasks/{task_id}", tags=["анализ"])
def task_status(task_id: str) -> dict:
    """Прогресс задачи: этап словами, проценты и список отвалившихся источников."""
    task = manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена или уже забыта")
    return task.public()


@app.get("/api/tasks/{task_id}/result", tags=["анализ"])
def task_result(task_id: str) -> dict:
    """Готовый AnalysisResult: ряд, периоды и служебное для интерфейса."""
    task = manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Задача не найдена или уже забыта")
    if task.status == "failed":
        raise HTTPException(status_code=409, detail=task.error or "Анализ не удался")
    if task.status != "done":
        # 409, а не 404: задача существует, просто ещё считается. Интерфейс по
        # этому коду понимает, что надо продолжать опрашивать прогресс.
        raise HTTPException(status_code=409, detail=f"Задача ещё выполняется: {task.stage}")
    return task.public(with_result=True)


# --------------------------------------------------------------------------------------
# Сохранённые участки
# --------------------------------------------------------------------------------------

@app.get("/api/polygons", tags=["участки"])
def polygons_list() -> dict:
    items = store.list_all()
    return {"polygons": items, "count": len(items)}


@app.post("/api/polygons", status_code=201, tags=["участки"])
def polygons_create(request: PolygonCreate) -> dict:
    return store.create(
        request.geometry,
        name=request.name,
        crop_type=request.crop_type,
        source=request.source,
        external_id=request.external_id,
    )


@app.get("/api/polygons/{polygon_id}", tags=["участки"])
def polygons_get(polygon_id: str) -> dict:
    polygon = store.get(polygon_id)
    if polygon is None:
        raise HTTPException(status_code=404, detail="Участок не найден")
    return polygon


@app.patch("/api/polygons/{polygon_id}", tags=["участки"])
def polygons_patch(polygon_id: str, request: PolygonPatch) -> dict:
    polygon = store.update(polygon_id, name=request.name, crop_type=request.crop_type)
    if polygon is None:
        raise HTTPException(status_code=404, detail="Участок не найден")
    return polygon


@app.delete("/api/polygons/{polygon_id}", tags=["участки"])
def polygons_delete(polygon_id: str) -> dict:
    if not store.delete(polygon_id):
        raise HTTPException(status_code=404, detail="Участок не найден")
    # Сохранённый результат уходит вместе с участком: иначе новый участок с тем
    # же идентификатором показал бы чужой ряд.
    drop_result(polygon_id)
    return {"deleted": polygon_id}


@app.get("/api/polygons/{polygon_id}/result", tags=["участки"])
def polygons_result(polygon_id: str) -> dict:
    """Последний посчитанный анализ участка.

    Ради этого маршрута результат и кладётся на диск: открывая сохранённое поле
    назавтра, пользователь должен увидеть прошлый разбор сразу, а не ждать нового
    сбора на минуты.
    """
    polygon = store.get(polygon_id)
    if polygon is None:
        raise HTTPException(status_code=404, detail="Участок не найден")
    result = load_result(polygon_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Участок ещё не анализировался")
    return {
        "polygon": polygon,
        "analyzed_at": polygon.get("last_analyzed_at"),
        "result": result,
    }


@app.get("/api/polygons/{polygon_id}/agro", tags=["участки"])
def polygons_agro_get(polygon_id: str) -> dict:
    """Журнал полевых работ участка."""
    polygon = store.get(polygon_id)
    if polygon is None:
        raise HTTPException(status_code=404, detail="Участок не найден")
    return {"polygon_id": polygon_id, "events": polygon.get("agro_events") or []}


@app.put("/api/polygons/{polygon_id}/agro", tags=["участки"])
def polygons_agro_put(polygon_id: str, events: list[dict]) -> dict:
    """Заменяет журнал полевых работ участка целиком.

    Каждая запись — словарь с датой и видом работ; понимаются и русские, и
    английские имена полей («дата»/«date_from», «вид»/«kind»), потому что
    агроном чаще всего вставляет свою таблицу как есть. Разбор ядра подхватит
    журнал при следующем пересчёте участка.

    Замена целиком, а не добавление по одной записи: журнал ведут таблицей, и
    выгрузить её заново проще, чем сверять, какие строки уже отправлены.
    """
    polygon = store.get(polygon_id)
    if polygon is None:
        raise HTTPException(status_code=404, detail="Участок не найден")

    # Проверяем разбором ядра: пусть ошибку в дате пользователь увидит сразу,
    # а не через пять минут в виде разбора без объяснений.
    from src.core.agrolog import load_events

    parsed = load_events(list(events or []))
    if events and not parsed:
        raise HTTPException(
            status_code=400,
            detail="Ни одна запись журнала не разобрана: нужны дата и вид работ",
        )

    store.set_agro(polygon_id, list(events or []))
    return {
        "polygon_id": polygon_id,
        "saved": len(events or []),
        "recognised": len(parsed),
        "note": "журнал учтётся при следующем пересчёте участка",
    }


@app.get("/api/polygons/{polygon_id}/report.pdf", tags=["участки"])
def polygons_report_pdf(polygon_id: str):
    """Клиентский отчёт по участку одним PDF-файлом.

    Всё, что сервис показывает на экране, написано языком метрик и рассчитано на
    подготовленного человека. Этот маршрут отдаёт тот же анализ, но для того, кто
    в теме не разбирается: фермера, агронома, оценщика банка или страховой.
    Графики и формулировки собираются в `src/reporting`.

    Сборка идёт через `build_pdf_safe`: отчёт скачивают одной кнопкой, и получить
    пятисотую ошибку вместо файла хуже, чем получить документ с честным
    сообщением о сбое.
    """
    polygon = store.get(polygon_id)
    if polygon is None:
        raise HTTPException(status_code=404, detail="Участок не найден")
    result = load_result(polygon_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Участок ещё не анализировался")

    from src.reporting.pdf import build_pdf_safe

    data = build_pdf_safe(result, polygon)
    # Имя файла уезжает в заголовок в двух видах: ASCII-запасной вариант для
    # старых клиентов и UTF-8 по RFC 5987 для нормальных браузеров, иначе
    # кириллица в названии участка превращается в мусор.
    stamp = str(polygon.get("last_analyzed_at") or "")[:10]
    ascii_name = f"fenolog-report-{polygon_id}.pdf"
    human = f"Фенолог — {polygon.get('name') or polygon_id}{(' ' + stamp) if stamp else ''}.pdf"
    disposition = (
        f'attachment; filename="{ascii_name}"; '
        f"filename*=UTF-8''{quote(human)}"
    )
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": disposition},
    )


# --------------------------------------------------------------------------------------
# Интерфейс
# --------------------------------------------------------------------------------------

# Монтируется последним: StaticFiles перехватывает все пути, и объявленный после
# него маршрут API уже не сработает.
if config.WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(config.WEB_DIST), html=True), name="web")
else:
    @app.get("/", include_in_schema=False)
    def _no_web() -> dict:
        return {
            "service": "fenolog",
            "detail": (
                "Интерфейс не собран. Соберите его командой `npm ci && npm run build` "
                "в каталоге web/ или запустите сервис через docker compose up."
            ),
            "api_docs": "/docs",
        }
