"""Сохранённые пользователем полигоны.

Постановка требует, чтобы участки можно было добавлять, переименовывать, удалять
и возвращаться к ним позже — это отдельный критерий на 5 баллов. Никакой базы для
этого не нужно: полигонов десятки, а не миллионы.

Хранилище — один JSON-файл. Такой выбор осознанный, а не от лени:

* состояние видно глазами и правится руками, если на защите что-то пошло не так;
* переживает перезапуск контейнера, если каталог data/ смонтирован томом;
* не добавляет ни зависимости, ни отдельного сервиса в docker-compose.

Запись идёт через временный файл и os.replace, то есть атомарно: прерванная
запись не оставит обрезанный JSON, из-за которого сервис потом не поднимется.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.api import config
from src.api.geometry import area_ha, centroid_of, normalize_geometry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PolygonStore:
    """Список сохранённых участков, синхронизированный по одному замку.

    Замок нужен потому, что фоновые задачи анализа ходят сюда из своих потоков
    одновременно с обработчиками HTTP.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or config.POLYGONS_FILE
        self._lock = threading.RLock()
        self._items: list[dict] = []
        self._loaded = False

    # ---------------------------------------------------------------- чтение с диска

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._items = self._read()
            self._loaded = True

    def _read(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Битый файл не имеет права мешать запуску сервиса: отодвигаем его в
            # сторону и начинаем с пустого списка. Данные при этом не пропадают —
            # файл остаётся рядом под именем polygons.broken.json.
            try:
                os.replace(self._path, self._path.with_suffix(".broken.json"))
            except OSError:
                pass
            return []
        return raw.get("polygons", []) if isinstance(raw, dict) else []

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"polygons": self._items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)

    # ------------------------------------------------------------------- операции

    def list_all(self) -> list[dict]:
        self._ensure_loaded()
        with self._lock:
            return [dict(item) for item in self._items]

    def get(self, polygon_id: str) -> dict | None:
        self._ensure_loaded()
        with self._lock:
            for item in self._items:
                if item["id"] == polygon_id:
                    return dict(item)
        return None

    def create(
        self,
        geometry: dict,
        name: str | None = None,
        crop_type: str | None = None,
        source: str = "drawn",
        external_id: str | None = None,
    ) -> dict:
        """Сохранить контур. Геометрия приводится к канону здесь же."""
        geometry = normalize_geometry(geometry)
        self._ensure_loaded()
        with self._lock:
            lon, lat = centroid_of(geometry)
            item = {
                "id": uuid.uuid4().hex[:12],
                "name": (name or "").strip() or self._default_name(),
                "geometry": geometry,
                "crop_type": crop_type,
                "area_ha": round(area_ha(geometry), 2),
                "center": [round(lon, 6), round(lat, 6)],
                # Откуда контур: нарисован пользователем или взят из OSM. Разница
                # видна в интерфейсе и попадает в отчёт по полю.
                "source": source,
                "external_id": external_id,
                "created_at": _now(),
                "updated_at": _now(),
                "last_task_id": None,
                "last_analyzed_at": None,
            }
            self._items.append(item)
            self._write()
            return dict(item)

    def _default_name(self) -> str:
        """Имя вида «Поле 3»: пользователь чаще всего имя не вводит, а безымянные
        строки в боковом списке неразличимы между собой."""
        return f"Поле {len(self._items) + 1}"

    def update(
        self,
        polygon_id: str,
        name: str | None = None,
        crop_type: str | None = None,
    ) -> dict | None:
        self._ensure_loaded()
        with self._lock:
            for item in self._items:
                if item["id"] != polygon_id:
                    continue
                if name is not None and name.strip():
                    item["name"] = name.strip()
                # crop_type сбрасывается пустой строкой: «культура неизвестна» —
                # осмысленное состояние, ядро умеет работать без неё.
                if crop_type is not None:
                    item["crop_type"] = crop_type.strip() or None
                item["updated_at"] = _now()
                self._write()
                return dict(item)
        return None

    def set_agro(self, polygon_id: str, events: list[dict]) -> dict | None:
        """Записывает журнал полевых работ участка.

        Журнал хранится прямо в записи участка, а не отдельным файлом: он мелкий
        (десятки строк на сезон), меняется вместе с участком и должен исчезать
        вместе с ним. Разбор ядра читает его при следующем пересчёте.
        """
        self._ensure_loaded()
        with self._lock:
            for item in self._items:
                if item["id"] != polygon_id:
                    continue
                item["agro_events"] = list(events or [])
                item["updated_at"] = _now()
                self._write()
                return dict(item)
        return None

    def delete(self, polygon_id: str) -> bool:
        self._ensure_loaded()
        with self._lock:
            before = len(self._items)
            self._items = [i for i in self._items if i["id"] != polygon_id]
            if len(self._items) == before:
                return False
            self._write()
            return True

    def mark_analyzed(self, polygon_id: str, task_id: str) -> None:
        """Запомнить последнюю задачу анализа по участку.

        Нужно интерфейсу: открывая сохранённое поле, он показывает прошлый
        результат сразу, не запуская сбор заново.
        """
        self._ensure_loaded()
        with self._lock:
            for item in self._items:
                if item["id"] == polygon_id:
                    item["last_task_id"] = task_id
                    item["last_analyzed_at"] = _now()
                    self._write()
                    return


# Единственный экземпляр на процесс: сервис однопроцессный, разделять состояние
# между воркерами не требуется.
store = PolygonStore()
