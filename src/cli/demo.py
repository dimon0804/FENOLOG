"""Сквозная демонстрация без интерфейса: от рамки на карте до разбора поля.

    python -m src.cli.demo --bbox 39.60 47.10 39.90 47.30
    python -m src.cli.demo --bbox 39.60 47.10 39.90 47.30 --analyze 1
    python -m src.cli.demo --polygon-file my_field.geojson

Показывает то же, что делает веб-сервис, только в терминале:

    1. находит сельхозконтуры в указанной рамке (OpenStreetMap);
    2. по выбранному контуру сам скачивает снимки и погоду;
    3. восстанавливает ряд, считает климатическую норму;
    4. находит периоды угнетения и объясняет их причину.

Нужна по двум причинам. Во-первых, это проверка, что автосбор действительно
работает **на произвольном регионе**, а не только на выданном наборе полигонов, —
ровно то, что оценивает критерий адаптивности. Во-вторых, это запасной путь
демонстрации: если на площадке ляжет интерфейс, сценарий можно показать отсюда.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _print_progress(stage: str, done: int, total: int) -> None:
    """Однострочный индикатор: сбор данных долгий, молчать нельзя."""
    bar = ""
    if total:
        filled = int(24 * done / max(total, 1))
        bar = "█" * filled + "·" * (24 - filled)
    sys.stdout.write(f"\r  {stage:<28} {bar} {done}/{total}   ")
    sys.stdout.flush()


def cmd_parcels(args) -> list[dict]:
    """Шаг 1: поиск сельхозконтуров в рамке карты."""
    from src.providers.parcels import find_parcels

    west, south, east, north = args.bbox
    print(f"Ищу сельхозполя в рамке {west}–{east} в.д., {south}–{north} с.ш.")
    t0 = time.perf_counter()
    parcels = find_parcels((west, south, east, north), limit=args.limit)
    print(f"\rНайдено полей: {len(parcels)} за {time.perf_counter() - t0:.1f} с" + " " * 30)

    if not parcels:
        print("Ничего не найдено. Попробуйте другую рамку или увеличьте её.")
        return []

    print()
    print(f"{'№':>3}  {'площадь, га':>12}  {'культура':<18} название")
    print("-" * 74)
    for i, p in enumerate(parcels[: args.limit], 1):
        crop = (p.get("crop_hint") or "—")[:18]
        name = (p.get("name") or "")[:28]
        print(f"{i:>3}  {p.get('area_ha', 0):>12.1f}  {crop:<18} {name}")
    return parcels


def cmd_analyze(geometry: dict, polygon_id: str, crop_type, args) -> None:
    """Шаги 2–4: сбор данных по контуру и разбор ряда доменным ядром."""
    from src.providers.collect import analyze_polygon

    print()
    print(f"Собираю данные по полю {polygon_id} за {args.years} сезона(ов).")
    print("Первый прогон долгий — скачиваются сцены. Повторный будет из кэша.")
    t0 = time.perf_counter()
    result = analyze_polygon(
        geometry, polygon_id=polygon_id, crop_type=crop_type,
        years=args.years, progress=_print_progress, max_scenes=args.max_scenes,
    )
    print(f"\rГотово за {time.perf_counter() - t0:.1f} с" + " " * 50)

    meta = result.meta
    print()
    print("СБОР ДАННЫХ")
    print(f"  период                {meta.get('date_from')} .. {meta.get('date_to')}")
    print(f"  наблюдений собрано    {meta.get('collected_observations')}")
    print(f"  по сенсорам           {meta.get('sources')}")
    print(f"  дней погоды           {meta.get('collected_weather_days')}")
    if meta.get("failures"):
        print(f"  не удалось            {'; '.join(meta['failures'])}")

    restored = sum(1 for p in result.series if p.is_restored)
    print()
    print("РЯД")
    print(f"  точек в ряду          {len(result.series)}")
    print(f"  из них восстановлено  {restored}")
    print(f"  источник нормы        {meta.get('climatology_source')}")
    if meta.get("climatology_source") == "crop":
        print("    (у поля нет собственной истории — норма средняя по культуре,")
        print("     оценка ориентировочная)")

    print()
    print(f"НАЙДЕННЫЕ ПЕРИОДЫ УГНЕТЕНИЯ: {len(result.anomalies)}")
    if not result.anomalies:
        print("  Периодов ниже нормы не найдено — поле развивалось штатно.")
    for a in sorted(result.anomalies, key=lambda x: x.min_zscore)[: args.top]:
        print()
        print(f"  {a.start} .. {a.end}   {a.duration_days} дн   "
              f"z_min {a.min_zscore:.2f}   {a.severity} / {a.cause}")
        print(f"    {a.explanation}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "polygon_id": result.polygon_id,
            "meta": meta,
            "series": [
                {"date": p.date.isoformat(), "observed": p.observed,
                 "restored": p.restored, "zscore": p.zscore,
                 "is_restored": p.is_restored}
                for p in result.series
            ],
            "anomalies": [
                {"start": a.start.isoformat(), "end": a.end.isoformat(),
                 "duration_days": a.duration_days, "severity": a.severity,
                 "cause": a.cause, "confidence": a.cause_confidence,
                 "explanation": a.explanation}
                for a in result.anomalies
            ],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print()
        print(f"Результат сохранён: {out}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        description="Сквозная демонстрация: рамка на карте -> поля -> анализ")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("ЗАПАД", "ЮГ", "ВОСТОК", "СЕВЕР"),
                   help="рамка карты в градусах, например 39.60 47.10 39.90 47.30")
    p.add_argument("--polygon-file", help="свой контур: файл с GeoJSON Polygon")
    p.add_argument("--analyze", type=int, metavar="N",
                   help="разобрать поле номер N из найденных")
    p.add_argument("--crop", help="тип культуры, если известен")
    p.add_argument("--years", type=int, default=5, help="сколько сезонов истории собирать")
    p.add_argument("--limit", type=int, default=15, help="сколько полей показать")
    p.add_argument("--max-scenes", type=int, help="ограничить число сцен (быстрая демонстрация)")
    p.add_argument("--top", type=int, default=5, help="сколько периодов показать подробно")
    p.add_argument("--output", help="сохранить результат в JSON")
    args = p.parse_args()

    if args.polygon_file:
        raw = json.loads(Path(args.polygon_file).read_text(encoding="utf-8"))
        geometry = raw.get("geometry", raw)
        cmd_analyze(geometry, Path(args.polygon_file).stem, args.crop, args)
        return

    if not args.bbox:
        p.error("укажите --bbox ЗАПАД ЮГ ВОСТОК СЕВЕР или --polygon-file")

    parcels = cmd_parcels(args)
    if args.analyze and parcels:
        if not 1 <= args.analyze <= len(parcels):
            raise SystemExit(f"поля номер {args.analyze} нет, найдено {len(parcels)}")
        chosen = parcels[args.analyze - 1]
        cmd_analyze(chosen["geometry"], chosen.get("id", "AOI-OSM"),
                    args.crop or chosen.get("crop_hint"), args)
    elif parcels:
        print()
        print("Чтобы разобрать поле, добавьте --analyze N, например --analyze 1")


if __name__ == "__main__":
    main()
