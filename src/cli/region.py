"""Пакетный разбор региона: сотни полей за один запуск.

    python -m src.cli.region --region rostov --fields 50
    python -m src.cli.region --region krasnodar --fields 300 --years 6
    python -m src.cli.region --bbox 39.6 47.1 39.9 47.3 --fields 20 --output reports/region.json

Зачем это нужно. Разбор одного поля отвечает на вопрос агронома. Но заказчик
кейса назвал покупателем страховые компании и банки, кредитующие сельское
хозяйство, — а им нужен взгляд на портфель: где в районе проблемы, сколько полей
в норме, какие требуют выезда. Этот режим и даёт такой взгляд.

Он же закрывает требование адаптивности: пресеты покрывают весь Южный федеральный
округ, и решение не привязано ни к одному заранее подготовленному набору полей.

Стоимость. Первый прогон по региону долгий: на каждое поле качаются снимки за
несколько сезонов. Дальше работает кэш, и повторный прогон по тому же региону
занимает секунды. Поэтому для демонстрации регион прогревается заранее.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

# Пресеты Южного федерального округа: рамки в градусах (запад, юг, восток, север).
# Взяты по основным земледельческим районам, а не по административным границам
# целиком — в границы попадают море, горы и города, где полей нет, и запрос к
# Overpass распухает впустую.
REGIONS: dict[str, tuple[str, tuple[float, float, float, float]]] = {
    "rostov":     ("Ростовская область, приазовская степь", (39.30, 46.80, 40.60, 47.40)),
    "krasnodar":  ("Краснодарский край, кубанская равнина",  (38.60, 45.00, 40.20, 45.90)),
    "stavropol":  ("Ставропольский край",                    (41.60, 44.90, 43.20, 45.70)),
    "volgograd":  ("Волгоградская область, юг",              (43.80, 48.30, 45.20, 49.20)),
    "astrakhan":  ("Астраханская область, дельта Волги",     (46.80, 46.00, 48.20, 46.80)),
    "adygea":     ("Адыгея",                                 (39.00, 44.60, 40.20, 45.20)),
    "kalmykia":   ("Калмыкия, восточные районы",             (44.00, 45.80, 45.40, 46.60)),
    "crimea":     ("Крым, степная часть",                    (33.60, 45.20, 35.20, 45.90)),
}

# Класс поля по итогам сезона — то, что видит человек в сводке
VERDICT_OK = "в норме"
VERDICT_WATCH = "под наблюдением"
VERDICT_PROBLEM = "требует выезда"
VERDICT_NODATA = "данных мало"


def classify(result) -> tuple[str, float]:
    """Короткий вердикт по полю и глубина худшего отклонения.

    Правило простое и объяснимое, а не подобранное: критическая аномалия в
    текущем сезоне — выезд; устойчивое угнетение — наблюдение; нет нормы —
    отдельный класс, потому что молчание из-за нехватки истории нельзя выдавать
    за «всё хорошо».
    """
    if result.meta.get("climatology_source") == "none":
        return VERDICT_NODATA, float("nan")

    # Норма может формально существовать, но не покрывать ряд: под ней меньше трёх
    # опорных лет, и z-оценка на большей части дней не определена. Отсутствие
    # аномалий в такой ситуации означает «не знаем», а не «всё хорошо», и выдавать
    # его за норму нельзя — на этой цифре страховая будет принимать решение.
    with_z = sum(1 for pt in result.series if pt.zscore is not None)
    coverage = with_z / max(len(result.series), 1)
    if coverage < 0.5:
        return VERDICT_NODATA, float("nan")

    if not result.anomalies:
        return VERDICT_OK, 0.0
    worst = min(a.min_zscore for a in result.anomalies)
    has_critical = any(a.severity == "critical" for a in result.anomalies)
    if has_critical:
        return VERDICT_PROBLEM, worst
    return VERDICT_WATCH, worst


def analyse_one(parcel: dict, years: int, max_scenes: int | None) -> dict:
    """Разбор одного поля. Ошибка одного поля не должна валить весь регион."""
    from src.providers.collect import analyze_polygon

    pid = parcel.get("id", "AOI")
    try:
        t0 = time.perf_counter()
        result = analyze_polygon(
            parcel["geometry"], polygon_id=pid,
            crop_type=parcel.get("crop_hint"), years=years,
            max_scenes=max_scenes,
        )
        verdict, worst = classify(result)
        return {
            "id": pid,
            "area_ha": round(parcel.get("area_ha", 0.0), 1),
            "crop": parcel.get("crop_hint"),
            "verdict": verdict,
            "worst_z": None if worst != worst else round(worst, 2),
            "anomalies": len(result.anomalies),
            "observations": result.meta.get("collected_observations", 0),
            "climatology": result.meta.get("climatology_source"),
            "seconds": round(time.perf_counter() - t0, 1),
            "top_cause": (max(result.anomalies, key=lambda a: -a.min_zscore).cause
                          if result.anomalies else None),
            # Оценка риска и прогноз — то, ради чего портфельный взгляд и нужен
            # банку со страховой: не «что было», а «чего ждать и насколько
            # надёжен этот объект».
            "score": (result.meta.get("score") or {}).get("score"),
            "grade": (result.meta.get("score") or {}).get("grade"),
            "risk": (result.meta.get("forecast") or {}).get("risk"),
            "forecast": (result.meta.get("forecast") or {}).get("summary"),
            "explanation": (min(result.anomalies, key=lambda a: a.min_zscore).explanation
                            if result.anomalies else ""),
            "geometry": parcel["geometry"],
        }
    except Exception as exc:  # noqa: BLE001 — одно поле не роняет регион
        return {"id": pid, "verdict": "ошибка", "error": f"{type(exc).__name__}: {exc}"[:160],
                "area_ha": round(parcel.get("area_ha", 0.0), 1), "geometry": parcel.get("geometry")}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Пакетный разбор полей региона")
    p.add_argument("--region", choices=sorted(REGIONS), help="пресет Южного федерального округа")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("З", "Ю", "В", "С"),
                   help="произвольная рамка вместо пресета")
    p.add_argument("--fields", type=int, default=25, help="сколько полей разобрать")
    p.add_argument("--years", type=int, default=4, help="сезонов истории на поле")
    p.add_argument("--workers", type=int, default=4, help="полей одновременно")
    p.add_argument("--max-scenes", type=int, help="ограничить сцены на поле (быстрая демонстрация)")
    p.add_argument("--output", help="сохранить сводку в JSON (для карты)")
    p.add_argument("--list-regions", action="store_true", help="показать доступные регионы")
    args = p.parse_args()

    if args.list_regions:
        print(f"{'ключ':<12} {'описание':<42} рамка")
        for key, (title, bbox) in REGIONS.items():
            print(f"{key:<12} {title:<42} {bbox}")
        return

    if args.region:
        title, bbox = REGIONS[args.region]
    elif args.bbox:
        title, bbox = "произвольная рамка", tuple(args.bbox)
    else:
        p.error("укажите --region или --bbox")

    from src.providers.parcels import find_parcels

    print(f"Регион: {title}")
    print(f"Рамка:  {bbox}")
    t_all = time.perf_counter()

    print("Ищу поля…")
    parcels = find_parcels(bbox, limit=args.fields)
    if not parcels:
        raise SystemExit("Полей не найдено. Overpass мог ограничить запросы — попробуйте позже "
                         "или другую рамку.")
    total_ha = sum(x.get("area_ha", 0) for x in parcels)
    print(f"Найдено полей: {len(parcels)}, суммарно {total_ha:,.0f} га".replace(",", " "))
    print()
    print(f"Разбираю {len(parcels)} полей по {args.years} сезона(ов), "
          f"{args.workers} одновременно. Первый прогон долгий, дальше из кэша.")

    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(analyse_one, x, args.years, args.max_scenes): x for x in parcels}
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            sys.stdout.write(f"\r  разобрано {done}/{len(parcels)}   ")
            sys.stdout.flush()
    print()

    order = {VERDICT_PROBLEM: 0, VERDICT_WATCH: 1, VERDICT_OK: 2, VERDICT_NODATA: 3, "ошибка": 4}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), r.get("worst_z") or 0))

    counts: dict[str, int] = {}
    ha: dict[str, float] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        ha[r["verdict"]] = ha.get(r["verdict"], 0.0) + r.get("area_ha", 0.0)

    print()
    print("=" * 78)
    print("СВОДКА ПО РЕГИОНУ")
    print("=" * 78)
    print(f"{'состояние':<20} {'полей':>7} {'доля':>7} {'площадь, га':>14}")
    print("-" * 78)
    for verdict in (VERDICT_PROBLEM, VERDICT_WATCH, VERDICT_OK, VERDICT_NODATA, "ошибка"):
        if verdict not in counts:
            continue
        share = 100 * counts[verdict] / len(rows)
        print(f"{verdict:<20} {counts[verdict]:>7} {share:>6.1f}% {ha[verdict]:>14,.0f}".replace(",", " "))
    print("-" * 78)
    print(f"{'всего':<20} {len(rows):>7} {100.0:>6.1f}% {sum(ha.values()):>14,.0f}".replace(",", " "))

    problem = [r for r in rows if r["verdict"] in (VERDICT_PROBLEM, VERDICT_WATCH)]
    if problem:
        print()
        print("ПОЛЯ, ТРЕБУЮЩИЕ ВНИМАНИЯ")
        print(f"{'поле':<26} {'га':>8} {'z':>7} {'балл':>5} {'риск':<10} причина")
        print("-" * 92)
        for r in problem[:15]:
            print(f"{r['id'][:26]:<26} {r.get('area_ha', 0):>8.1f} "
                  f"{(r.get('worst_z') if r.get('worst_z') is not None else 0):>7.2f} "
                  f"{(r.get('score') if r.get('score') is not None else 0):>5} "
                  f"{(r.get('risk') or '—'):<10} "
                  f"{r.get('top_cause') or '—'}")
        worst = problem[0]
        if worst.get("explanation"):
            print()
            print(f"Худшее поле {worst['id']}:")
            print(f"  {worst['explanation']}")

    # Портфельный срез по оценке риска: то, что банк смотрит первым
    scored = [r for r in rows if r.get("score") is not None]
    if scored:
        by_grade: dict[str, list] = {}
        for r in scored:
            by_grade.setdefault(r["grade"], []).append(r)
        print()
        print("ОЦЕНКА ПОРТФЕЛЯ")
        print(f"{'класс':<8} {'полей':>7} {'площадь, га':>14} {'средний балл':>14}")
        print("-" * 92)
        for grade in ("A", "B", "C", "D", "E"):
            g = by_grade.get(grade)
            if not g:
                continue
            avg = sum(x["score"] for x in g) / len(g)
            area = sum(x.get("area_ha", 0) for x in g)
            print(f"{grade:<8} {len(g):>7} {area:>14,.0f} {avg:>14.0f}".replace(",", " "))
        weak = [r for r in scored if r["score"] < 55]
        if weak:
            print()
            print(f"Ниже порога надёжности (балл < 55): {len(weak)} полей на "
                  f"{sum(x.get('area_ha', 0) for x in weak):,.0f} га".replace(",", " "))

    elapsed = time.perf_counter() - t_all
    obs = sum(r.get("observations", 0) for r in rows)
    print()
    print(f"Разобрано {len(rows)} полей за {elapsed:.0f} с, собрано {obs} наблюдений.")
    print(f"В среднем {elapsed / max(len(rows), 1):.1f} с на поле.")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "region": title,
            "bbox": list(bbox),
            "generated": date.today().isoformat(),
            "fields_total": len(rows),
            "area_total_ha": round(sum(ha.values()), 1),
            "counts": counts,
            "area_by_verdict": {k: round(v, 1) for k, v in ha.items()},
            "fields": rows,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Сводка сохранена: {out}")
        print("Файл содержит геометрию каждого поля — его можно отдать прямо на карту.")


if __name__ == "__main__":
    main()
