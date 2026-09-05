"""Прогрев демонстрации: находит поля района, сохраняет их и разбирает.

    python -m src.cli.warmup --region rostov --fields 14
    python -m src.cli.warmup --bbox 39.6 47.1 39.9 47.3 --fields 8 --passes 1

Зачем нужно. На карте поля закрашены по состоянию: зелёные в норме, жёлтые под
наблюдением, красные требуют выезда. Но состояние берётся из результата анализа,
а пока поле не разобрано, оно серое — «нет данных». На пустом сервисе карта
выглядит серой, и главный экран не показывает того, ради чего сделан.

Скрипт закрывает это честно: он не подрисовывает цвета, а действительно
прогоняет анализ через тот же API, которым пользуется интерфейс. После прогрева
карта показывает настоящую картину района, а кэш источников оказывается тёплым.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

from src.cli.region import REGIONS

DEFAULT_API = "http://127.0.0.1:8000"


def _call(api: str, path: str, payload=None, method: str | None = None, timeout: int = 120):
    """Запрос к сервису. Ошибка одного поля не должна валить прогрев целиком."""
    url = f"{api.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _analyse_one(args, pid: str, label: str) -> tuple[bool, float, dict]:
    """Запускает разбор поля и ждёт его. Возвращает (успех, секунды, поправка)."""
    try:
        # Тела эта ручка не принимает: глубину истории задаёт сервер
        # переменной FENOLOG_YEARS, поэтому шлём пустой POST.
        task = _call(args.api, f"/api/polygons/{pid}/analyze", method="POST", timeout=60)
        tid = task.get("task", {}).get("id") or task.get("id")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label}: анализ не запущен — {type(exc).__name__}")
        return False, 0.0, {}

    t0 = time.perf_counter()
    status = "pending"
    while time.perf_counter() - t0 < args.timeout:
        try:
            st = _call(args.api, f"/api/tasks/{tid}", timeout=30)
        except Exception:  # noqa: BLE001
            time.sleep(3)
            continue
        status = st.get("status", "")
        stage = str(st.get("stage", ""))[:26]
        sys.stdout.write(f"\r  {label:<44} {stage:<26} {st.get('percent', 0):>3}%   ")
        sys.stdout.flush()
        if status in ("done", "failed"):
            break
        time.sleep(3)

    took = time.perf_counter() - t0
    siblings: dict = {}
    if status == "done":
        try:
            res = _call(args.api, f"/api/polygons/{pid}/result", timeout=30)
            siblings = (res.get("meta") or {}).get("siblings") or {}
        except Exception:  # noqa: BLE001
            pass
    return status == "done", took, siblings


def _run_pass(args, targets: list[dict], ids: dict[int, str]) -> tuple[int, int, int]:
    """Один проход по району: сохранить (только на первом) и разобрать."""
    saved = analysed = failed = 0

    for i, parcel in enumerate(targets, 1):
        name = parcel.get("name") or f"Участок {i}"
        label = f"[{i}/{len(targets)}] {name[:30]}"

        pid = ids.get(i)
        if pid is None:
            try:
                polygon = _call(args.api, "/api/polygons", {
                    "geometry": parcel["geometry"],
                    "name": name,
                    "crop_type": parcel.get("crop_hint"),
                    "source": "osm",
                    "external_id": parcel.get("id"),
                })
                pid = polygon.get("id") or polygon.get("polygon", {}).get("id")
                ids[i] = pid
                saved += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  {label}: не сохранён — {type(exc).__name__}")
                failed += 1
                continue

        ok, took, sib = _analyse_one(args, pid, label)
        if ok:
            analysed += 1
            note = ""
            if sib.get("applied"):
                note = (f"поправка по {sib.get('used')} соседям, "
                        f"{sib.get('days')} дат, размах {sib.get('std')}")
            elif sib:
                note = f"без поправки: {sib.get('reason')}"
            print(f"\r  {label:<44} готово за {took:>5.0f} с   {note}" + " " * 10)
        else:
            failed += 1
            print(f"\r  {label:<44} НЕ УДАЛОСЬ" + " " * 30)

    return saved, analysed, failed


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Прогрев демонстрации через API сервиса")
    p.add_argument("--region", choices=sorted(REGIONS), help="пресет Южного федерального округа")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("З", "Ю", "В", "С"))
    p.add_argument("--fields", type=int, default=10, help="сколько полей разобрать")
    p.add_argument("--api", default=DEFAULT_API, help="адрес сервиса")
    p.add_argument("--timeout", type=int, default=600, help="сколько ждать один анализ, секунд")
    p.add_argument("--passes", type=int, default=2, help="проходов по району, см. пояснение в коде")
    args = p.parse_args()

    if args.region:
        title, bbox = REGIONS[args.region]
    elif args.bbox:
        title, bbox = "произвольная рамка", tuple(args.bbox)
    else:
        p.error("укажите --region или --bbox")

    try:
        _call(args.api, "/health", timeout=10)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Сервис не отвечает на {args.api}: {exc}\n"
                         "Подними его: python -m uvicorn src.api.app:app --port 8000")

    print(f"Регион: {title}")
    print(f"Рамка:  {bbox}")
    print("Ищу поля…")
    found = _call(args.api, "/api/regions/parcels",
                  {"bbox": list(bbox), "limit": args.fields}, timeout=180)
    parcels = found.get("parcels", [])
    if not parcels:
        raise SystemExit("Полей не найдено. Overpass мог ограничить запросы — попробуйте позже.")
    area = sum(x.get("area_ha", 0) for x in parcels)
    print(f"Найдено: {len(parcels)}, суммарно {area:,.0f} га".replace(",", " "))
    print()

    t_all = time.perf_counter()
    targets = parcels[: args.fields]
    ids: dict[int, str] = {}

    # Почему проходов два.
    #
    # Суточная поправка снимается по соседним полям, и у неё есть бюджет времени:
    # в холодном районе шесть соседей по шесть сезонов из сети не выкачиваются за
    # отведённые две с половиной минуты, и разбор честно уходит без поправки.
    # Но соседи разбираемого поля — это и есть другие поля района. Поэтому первый
    # проход просто наполняет кэш, а на втором все соседи уже лежат рядом,
    # достаются мгновенно, и поправка применяется по-настоящему.
    for pass_no in range(1, max(args.passes, 1) + 1):
        if args.passes > 1:
            hint = "  (наполняю кэш)" if pass_no == 1 else "  (поправка по соседям)"
            print(f"--- проход {pass_no} из {args.passes}{hint}")
        saved, analysed, failed = _run_pass(args, targets, ids)
        print(f"    сохранено {saved}, разобрано {analysed}, не удалось {failed}")
        print()

    print(f"Всего {time.perf_counter() - t_all:.0f} с")

    try:
        summary = _call(args.api, "/api/summary", timeout=30)
        print()
        print("СВОДКА СЕРВИСА")
        for k, v in summary.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")
    except Exception:  # noqa: BLE001
        pass

    print()
    print("Открывай карту: район закрашен по настоящему состоянию полей.")


if __name__ == "__main__":
    main()
