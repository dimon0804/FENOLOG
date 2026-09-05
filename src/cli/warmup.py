"""Прогрев демонстрации: находит поля района, сохраняет их и разбирает.

    python -m src.cli.warmup --region rostov --fields 12
    python -m src.cli.warmup --bbox 39.6 47.1 39.9 47.3 --fields 8 --api http://127.0.0.1:8000

Зачем нужно. На карте поля закрашены по состоянию: зелёные в норме, жёлтые под
наблюдением, красные требуют выезда. Но состояние берётся из результата анализа,
а пока поле не разобрано, оно серое — «нет данных». На пустом сервисе карта
выглядит серой, и главный экран не показывает того, ради чего сделан.

Скрипт закрывает это честно: он не подрисовывает цвета, а действительно
прогоняет анализ через тот же API, которым пользуется интерфейс. После прогрева
карта показывает настоящую картину района, а кэш источников оказывается тёплым —
и суточная поправка по соседям для новых полей достаётся почти бесплатно.
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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Прогрев демонстрации через API сервиса")
    p.add_argument("--region", choices=sorted(REGIONS), help="пресет Южного федерального округа")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("З", "Ю", "В", "С"))
    p.add_argument("--fields", type=int, default=10, help="сколько полей разобрать")
    p.add_argument("--api", default=DEFAULT_API, help="адрес сервиса")
    p.add_argument("--timeout", type=int, default=600, help="сколько ждать один анализ, секунд")
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
                         f"Подними его: python -m uvicorn src.api.app:app --port 8000")

    print(f"Регион: {title}")
    print(f"Рамка:  {bbox}")
    print("Ищу поля…")
    found = _call(args.api, "/api/regions/parcels",
                  {"bbox": list(bbox), "limit": args.fields}, timeout=180)
    parcels = found.get("parcels", [])
    if not parcels:
        raise SystemExit("Полей не найдено. Overpass мог ограничить запросы — попробуйте позже.")
    print(f"Найдено: {len(parcels)}, суммарно {sum(x.get('area_ha', 0) for x in parcels):,.0f} га"
          .replace(",", " "))
    print()

    saved, analysed, failed = 0, 0, 0
    t_all = time.perf_counter()

    for i, parcel in enumerate(parcels[: args.fields], 1):
        name = parcel.get("name") or f"Участок {i}"
        try:
            polygon = _call(args.api, "/api/polygons", {
                "geometry": parcel["geometry"],
                "name": name,
                "crop_type": parcel.get("crop_hint"),
                "source": "osm",
                "external_id": parcel.get("id"),
            })
            pid = polygon.get("id") or polygon.get("polygon", {}).get("id")
            saved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{args.fields}] {name}: не сохранён — {type(exc).__name__}")
            failed += 1
            continue

        try:
            # Тела эта ручка не принимает: глубину истории задаёт сервер
            # переменной FENOLOG_YEARS, поэтому шлём пустой POST.
            task = _call(args.api, f"/api/polygons/{pid}/analyze",
                         method="POST", timeout=60)
            tid = task.get("task", {}).get("id") or task.get("id")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{args.fields}] {name}: анализ не запущен — {type(exc).__name__}")
            failed += 1
            continue

        t0 = time.perf_counter()
        status, stage = "pending", ""
        while time.perf_counter() - t0 < args.timeout:
            try:
                st = _call(args.api, f"/api/tasks/{tid}", timeout=30)
            except Exception:  # noqa: BLE001
                time.sleep(3)
                continue
            status, stage = st.get("status", ""), st.get("stage", "")
            sys.stdout.write(f"\r  [{i}/{len(parcels[:args.fields])}] {name[:28]:<28} {stage[:26]:<26} "
                             f"{st.get('percent', 0):>3}%   ")
            sys.stdout.flush()
            if status in ("done", "failed"):
                break
            time.sleep(3)

        took = time.perf_counter() - t0
        if status == "done":
            analysed += 1
            print(f"\r  [{i}/{len(parcels[:args.fields])}] {name[:28]:<28} готово за {took:>5.0f} с" + " " * 20)
        else:
            failed += 1
            print(f"\r  [{i}/{len(parcels[:args.fields])}] {name[:28]:<28} НЕ УДАЛОСЬ ({status})" + " " * 20)

    print()
    print(f"Сохранено полей: {saved}, разобрано: {analysed}, не удалось: {failed}")
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
    print("Открывай карту: карта района теперь закрашена по состоянию полей.")


if __name__ == "__main__":
    main()
