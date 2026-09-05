"""Фоновые задачи анализа с прогрессом.

Почему анализ вообще фоновый. Сбор данных по одному полю — это поход в каталог
снимков, скачивание десятков сцен и запрос архива погоды за пять сезонов, то есть
от десятков секунд до нескольких минут. Синхронный HTTP-запрос на такое время
недопустим: браузер и обратные прокси рвут соединение по таймауту, а пользователь
всё это время смотрит в пустой экран и считает сервис зависшим.

Поэтому POST только ставит задачу и сразу отдаёт её идентификатор, а интерфейс
опрашивает прогресс и показывает словами, что происходит: «ищу снимки»,
«скачано 14 из 40 сцен», «восстанавливаю ряд», «ищу аномалии».

Устройство намеренно простое: пул потоков и словарь задач в памяти. Очередь
сообщений здесь была бы отдельным сервисом в docker-compose ради одной функции.
Сбор упирается в сеть, а не в процессор, поэтому GIL работе не мешает.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone

from src.api import config
from src.api.storage import store

log = logging.getLogger("fenolog.api.tasks")

# Как этапы сбора ложатся на шкалу процентов. Названия слева приходят снизу:
# часть из collect.py, часть из satellite.py. Значения справа — границы участка
# шкалы, внутри которого этап растёт пропорционально своим «сделано из всего».
#
# Скачивание сцен занимает больше половины шкалы, потому что занимает больше
# половины времени: шкала должна показывать время, а не число шагов, иначе она
# застревает на одном месте и выглядит сломанной.
_STAGE_BANDS: dict[str, tuple[int, int]] = {
    "ищу снимки": (2, 4),
    "ищу сцены": (4, 10),
    "скачиваю сцены": (10, 62),
    "забираю погоду": (62, 68),
    # Поправка по соседним полям: ядро собирает ряды соседей тем же путём, что
    # и ряд самого поля, и на холодном районе это самая долгая часть разбора.
    # Без своей полосы шкала замирала бы здесь на одном числе дольше всего.
    "собираю соседние поля": (68, 82),
    "готово": (84, 84),
}

# Этапы после сбора — их эмитим сами, снизу о них никто не знает.
STAGE_RESTORE = "восстанавливаю ряд"
STAGE_ANOMALIES = "ищу аномалии"
STAGE_DONE = "готово"

# Куда складываем последний результат по сохранённому полю. Задачи живут в
# памяти час, а поле пользователь открывает и через день — и должен увидеть
# прошлый анализ сразу, без нового сбора.
RESULTS_DIR = config.DATA_DIR / "results"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Task:
    """Состояние одного анализа. Ровно это видит интерфейс в /api/tasks/{id}."""

    id: str
    status: str = "pending"          # pending | running | done | failed
    stage: str = "в очереди"
    percent: int = 0
    polygon_id: str | None = None
    polygon_name: str | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    # Чего не хватило: упавшие источники. Пустой список — собралось всё.
    warnings: list[str] = field(default_factory=list)
    result: dict | None = None

    def public(self, with_result: bool = False) -> dict:
        """Состояние без результата: опрос идёт раз в секунду, гонять по сети
        мегабайт ряда на каждый опрос незачем."""
        data = {k: v for k, v in asdict(self).items() if k != "result"}
        if with_result:
            data["result"] = self.result
        return data


class TaskManager:
    """Пул фоновых анализов и реестр их состояний."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=config.TASK_WORKERS, thread_name_prefix="analyze"
        )
        self._tasks: dict[str, Task] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ постановка

    def submit(
        self,
        geometry: dict,
        polygon_id: str | None = None,
        polygon_name: str | None = None,
        crop_type: str | None = None,
        years: int | None = None,
        max_scenes: int | None = None,
    ) -> Task:
        task = Task(
            id=uuid.uuid4().hex[:12],
            polygon_id=polygon_id,
            polygon_name=polygon_name,
        )
        with self._lock:
            self._forget_old()
            self._tasks[task.id] = task
        self._pool.submit(
            self._run, task, geometry, crop_type,
            years or config.DEFAULT_YEARS,
            config.MAX_SCENES if max_scenes is None else max_scenes,
        )
        return task

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _forget_old(self) -> None:
        """Убрать давно завершённые задачи, чтобы память не росла бесконечно."""
        if len(self._tasks) < 64:
            return
        limit = datetime.now(timezone.utc).timestamp() - config.TASK_TTL_SECONDS
        for task_id, task in list(self._tasks.items()):
            if task.status in ("done", "failed") and task.finished_at:
                if datetime.fromisoformat(task.finished_at).timestamp() < limit:
                    self._tasks.pop(task_id, None)

    # -------------------------------------------------------------------- работа

    def _run(self, task: Task, geometry: dict, crop_type, years: int, max_scenes) -> None:
        from src.core.analyze import analyze
        from src.providers.collect import collect_series_with_report

        task.status = "running"
        task.started_at = _now()
        task.stage = "начинаю сбор"

        try:
            series_input, report = collect_series_with_report(
                geometry,
                polygon_id=task.polygon_id or task.id,
                crop_type=crop_type,
                years=years,
                progress=lambda stage, done, total: self._progress(task, stage, done, total),
                max_scenes=max_scenes,
            )

            # Собственные этапы вокруг вызова ядра. Ради них сбор и анализ здесь
            # вызываются по отдельности, а не готовой связкой analyze_polygon():
            # восстановление ряда на длинной истории занимает заметное время, и
            # шкала, замирающая на «готово» после сбора, выглядит зависшей.
            self._set(task, STAGE_RESTORE, 88)
            result = analyze(series_input)
            self._set(task, STAGE_ANOMALIES, 95)
            result.meta.update(report.as_meta())

            task.warnings = list(report.failures)
            payload = _jsonable(result)
            # Погода едет наружу вместе с рядом. В AnalysisResult её нет — ядру
            # она нужна только чтобы назвать причину, — но панель температуры и
            # осадков под графиком обязательна: без неё фраза «дефицит осадков»
            # выглядит голословной, и проверить её пользователю нечем.
            payload["weather"] = _jsonable(series_input.weather)
            task.result = payload
            task.status = "done"
            self._set(task, STAGE_DONE, 100)

            if task.polygon_id:
                store.mark_analyzed(task.polygon_id, task.id)
                _save_result(task.polygon_id, task.result)

        except Exception as exc:  # noqa: BLE001 — задача обязана завершиться статусом
            # Наружу уходит текст, а не пятисотая: падение внешнего источника —
            # штатная ситуация, о которой пользователю надо сказать словами.
            log.exception("анализ %s не удался", task.id)
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"[:400]
            task.stage = "не удалось"
        finally:
            task.finished_at = _now()

    def _progress(self, task: Task, stage: str, done: int, total: int) -> None:
        low, high = _STAGE_BANDS.get(stage, (task.percent, task.percent))
        fraction = (done / total) if total else 0.0
        percent = int(low + (high - low) * min(max(fraction, 0.0), 1.0))

        # Человеческий текст вместо служебного имени этапа: «скачано 14 из 40 сцен»
        # понятнее, чем «скачиваю сцены 35%».
        text = stage
        if stage == "скачиваю сцены" and total:
            text = f"скачано {done} из {total} сцен"
        self._set(task, text, percent)

    @staticmethod
    def _set(task: Task, stage: str, percent: int) -> None:
        task.stage = stage
        # Шкала не должна ехать назад: этапы приходят из разных модулей, и один
        # запоздавший колбэк не имеет права отматывать прогресс к началу.
        task.percent = max(task.percent, min(percent, 100))


# --------------------------------------------------------------------------------------
# Сериализация ответа ядра
# --------------------------------------------------------------------------------------

def _jsonable(value):
    """Датаклассы ядра -> JSON-совместимые структуры.

    Изобретать формат не нужно, но три вещи asdict() не покрывает: даты, скаляры
    numpy (из pandas они лезут повсюду) и NaN. Последний особенно коварен —
    json.dumps печатает его как NaN, а JSON.parse в браузере на этом падает.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        # numpy-скаляр: .item() возвращает питоновский тип
        try:
            value = value.item()
        except Exception:
            return str(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (int, str)) or value is None:
        return value
    return str(value)


def summarize(payload: dict) -> dict:
    """Выжимка из разбора: то, что нужно сводке и списку участков.

    Полный разбор весит сотни килобайт — это ряд за пять сезонов посуточно.
    Складывать их все в одну сводку значило бы гонять по сети мегабайты ради
    трёх чисел, поэтому выжимка считается один раз при сохранении и лежит
    отдельным маленьким файлом.
    """
    anomalies = payload.get("anomalies") or []
    series = payload.get("series") or []
    observed = sum(1 for p in series if not p.get("is_restored"))
    zscores = [a.get("min_zscore") for a in anomalies if a.get("min_zscore") is not None]
    meta = payload.get("meta") or {}
    return {
        "anomalies": len(anomalies),
        "critical": sum(1 for a in anomalies if a.get("severity") == "critical"),
        "suppression": sum(1 for a in anomalies if a.get("severity") == "suppression"),
        # Худшее отклонение по полю — по нему список участков сортируется:
        # агроному нужно сразу видеть, где хуже всего.
        "worst_zscore": min(zscores) if zscores else None,
        "series_points": len(series),
        "observations": observed,
        "climatology_source": meta.get("climatology_source"),
        "date_from": meta.get("date_from"),
        "date_to": meta.get("date_to"),
        "sources": meta.get("sources") or {},
        "failures": meta.get("failures") or [],
        "last_anomaly": anomalies[0].get("end") if anomalies else None,
    }


def _save_result(polygon_id: str, payload: dict) -> None:
    """Сохранить последний результат по участку рядом с полигонами."""
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in (
            (f"{polygon_id}.json", payload),
            (f"{polygon_id}.summary.json", summarize(payload)),
        ):
            path = RESULTS_DIR / name
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
    except OSError as exc:
        # Кэш результата — удобство, а не обязательство: не сохранилось, значит
        # при следующем открытии поля будет новый сбор.
        log.warning("не удалось сохранить результат по %s: %s", polygon_id, exc)


def load_summary(polygon_id: str) -> dict | None:
    """Выжимка по участку. None — участок ещё не анализировался.

    Если выжимки нет, а полный разбор есть — считаем и дописываем. Так разборы,
    сделанные до появления выжимок, не превращаются в «поле не анализировалось»
    и пользователю не приходится пересчитывать их заново.
    """
    path = RESULTS_DIR / f"{polygon_id}.summary.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    payload = load_result(polygon_id)
    if payload is None:
        return None
    digest = summarize(payload)
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(digest, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass
    return digest


def load_result(polygon_id: str) -> dict | None:
    """Последний сохранённый результат по участку, если он есть."""
    path = RESULTS_DIR / f"{polygon_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def drop_result(polygon_id: str) -> None:
    """Удалить сохранённый результат вместе с самим участком."""
    for name in (f"{polygon_id}.json", f"{polygon_id}.summary.json"):
        try:
            (RESULTS_DIR / name).unlink(missing_ok=True)
        except OSError:
            pass


manager = TaskManager()
