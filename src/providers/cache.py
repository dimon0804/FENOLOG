"""Файловый кэш ответов внешних источников.

Зачем он вообще нужен. Слой провайдеров ходит в Open-Meteo, Overpass и каталоги
снимков — это секунды на запрос, а иногда десятки секунд. Демонстрация сервиса,
в которой каждый повторный клик по тому же полю снова ждёт сеть, выглядит
неработающей, хотя данные не менялись. Поэтому кэш здесь не оптимизация, а часть
поведения продукта: первый показ полигона платит за сеть, все последующие —
мгновенные.

Устройство простое намеренно. Никакой БД, никакого сервера: один файл на один
ответ в data/cache/<namespace>/<sha256>.pkl. Такой кэш переживает перезапуск
процесса и контейнера, его видно глазами, его можно снести rm -rf, и он не
добавляет ни одной зависимости.

Инвалидация — по возрасту файла (mtime) против ttl_days. Версионирования ключа
нет сознательно: если поменяется формат значения, кэш чистится целиком через
cache_clear(), это дешевле, чем тащить схему версий ради хакатона.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import shutil
import time
from datetime import date, datetime
from pathlib import Path

# Корень кэша считаем от расположения этого файла, а не от текущей директории:
# сервис запускают и как `python -m src.api`, и из docker с другим cwd, и из
# тестов — привязка к cwd давала бы каждый раз новый пустой кэш.
_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"

# Значение по умолчанию: исторические данные (архив погоды, контуры OSM) меняются
# медленно, месяц — разумный компромисс между свежестью и числом походов в сеть.
DEFAULT_TTL_DAYS = 30


def _json_default(obj: object) -> str:
    """Как сериализовать в ключ то, что json сам не умеет.

    В аргументы провайдеров регулярно приходят date/datetime (start, end периода),
    и без этого хука построение ключа падало бы на каждом вызове fetch_weather.
    Всё прочее приводим к строке: ключ не нужно уметь читать обратно, от него
    требуется только устойчивость — одинаковый вход даёт одинаковый хеш.
    """
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)  # type: ignore[return-value]
    return repr(obj)


def _fingerprint(namespace: str, key: dict) -> str:
    """Устойчивый отпечаток аргументов.

    sort_keys=True обязателен: словарь аргументов собирается из **kwargs, порядок
    ключей в нём зависит от того, как вызывающий код перечислил параметры, а нам
    нужен один и тот же хеш для одного и того же набора значений.
    """
    payload = json.dumps(
        {"ns": namespace, "key": key},
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_namespace(namespace: str) -> str:
    """Namespace попадает в имя каталога, поэтому чистим всё, кроме безопасных знаков."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in namespace)
    return cleaned or "default"


def _path_for(namespace: str, key: dict) -> Path:
    return _ROOT / _safe_namespace(namespace) / f"{_fingerprint(namespace, key)}.pkl"


def cache_get(namespace: str, key: dict) -> object | None:
    """Достать значение или None, если его нет, оно протухло или файл битый.

    TTL хранится внутри самой записи, а не задаётся при чтении: иначе один и тот же
    файл считался бы живым или мёртвым в зависимости от того, кто его спросил.
    """
    path = _path_for(namespace, key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            record = pickle.load(fh)
        ttl_days = record.get("ttl_days", DEFAULT_TTL_DAYS)
        age_days = (time.time() - record.get("saved_at", 0.0)) / 86400.0
        if ttl_days is not None and age_days > ttl_days:
            return None
        return record.get("value")
    except Exception:
        # Битый или недописанный файл (упали посреди записи, сменился формат
        # объекта) — это промах кэша, а не авария сервиса. Сносим и идём в сеть.
        try:
            path.unlink()
        except OSError:
            pass
        return None


def cache_set(namespace: str, key: dict, value: object, ttl_days: int = DEFAULT_TTL_DAYS) -> None:
    """Положить значение. Ошибки записи проглатываются: кэш не обязателен для работы.

    Пишем через временный файл и os.replace — атомарная подмена на уровне ФС.
    Без этого параллельные запросы к одному полю (а фронтенд шлёт их пачкой)
    могли бы прочитать наполовину записанный pickle.
    """
    path = _path_for(namespace, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"saved_at": time.time(), "ttl_days": ttl_days, "value": value}
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as fh:
            pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        return


def cache_clear(namespace: str | None = None) -> int:
    """Снести кэш целиком или один namespace. Возвращает число удалённых файлов."""
    target = _ROOT if namespace is None else _ROOT / _safe_namespace(namespace)
    if not target.exists():
        return 0
    removed = sum(1 for _ in target.rglob("*.pkl"))
    shutil.rmtree(target, ignore_errors=True)
    return removed


def cached(namespace: str, ttl_days: int = DEFAULT_TTL_DAYS):
    """Декоратор: кэширует результат функции по её аргументам.

    Ключ собирается из позиционных и именованных аргументов. Два исключения:

    * `progress` выкидывается из ключа — это колбэк прогресса, объект-функция,
      он меняется от вызова к вызову и не влияет на результат. Если бы он попадал
      в ключ, кэш не срабатывал бы никогда.
    * `force_refresh=True` в вызове означает «сходи в сеть заново» — нужно, когда
      пользователь жмёт «обновить». Флаг до самой функции не доходит.

    Пустой результат не кэшируется: пустой список у нас означает «источник не
    ответил» (Overpass отдал 504, погода недоступна), и запоминать эту неудачу
    на месяц было бы худшим из возможных поведений — сервис бы «залип» на сбое.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            force = bool(kwargs.pop("force_refresh", False))
            key_kwargs = {k: v for k, v in kwargs.items() if k != "progress"}
            key = {"fn": func.__qualname__, "args": list(args), "kwargs": key_kwargs}

            if not force:
                hit = cache_get(namespace, key)
                if hit is not None:
                    return hit

            result = func(*args, **kwargs)
            # Пустые списки/словари/None — признак сбоя источника, не кэшируем.
            if result is not None and not (hasattr(result, "__len__") and len(result) == 0):
                cache_set(namespace, key, result, ttl_days=ttl_days)
            return result

        wrapper.cache_namespace = namespace  # type: ignore[attr-defined]
        return wrapper

    return decorator
