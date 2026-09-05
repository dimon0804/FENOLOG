"""Проверка координат поля: то ли это место и поле ли это вообще.

Отдельная команда, а не часть разбора, по двум причинам. Первая: проверка стоит
лишних загрузок снимков (кольцо вокруг контура, сдвинутые копии), и платить за
неё на каждом разборе незачем. Вторая: она нужна один раз — когда поле заводят.

    python -m src.cli.verify --polygon-file reports/test_field.geojson
    python -m src.cli.verify --lon 40.10 --lat 46.95 --side 600
    python -m src.cli.verify --polygon-file field.geojson --deep   # ещё и сдвиг

Что проверяется и почему именно так — в шапке src/core/geocheck.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.core.geocheck import verify_polygon


def square(lon: float, lat: float, side_m: float) -> dict:
    """Квадратный контур заданной стороны вокруг точки — для проверки по точке."""
    import math

    half = side_m / 2.0
    dlat = half / 110_540.0
    dlon = half / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
            [lon + dlon, lat + dlat], [lon - dlon, lat + dlat],
            [lon - dlon, lat - dlat],
        ]],
    }


def _providers(offline: bool):
    """Источники для проверки. В офлайне их нет, и это штатный режим.

    Доменное ядро не импортирует провайдеры само — оно получает их вызовами.
    Здесь как раз то место, где слои соединяются.
    """
    if offline:
        return None, None

    def fetch(geometry, start, end):
        from src.providers.satellite import fetch_observations

        return fetch_observations(geometry, start, end)

    def parcels(bbox):
        from src.providers.parcels import find_parcels

        return find_parcels(bbox, limit=30)

    return fetch, parcels


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Проверка координат поля")
    ap.add_argument("--polygon-file", help="GeoJSON с контуром поля")
    ap.add_argument("--lon", type=float, help="долгота точки")
    ap.add_argument("--lat", type=float, help="широта точки")
    ap.add_argument("--side", type=float, default=600.0, help="сторона квадрата, м")
    ap.add_argument("--years", type=int, default=3, help="сколько сезонов проверять")
    ap.add_argument("--deep", action="store_true",
                    help="проверить смещение контура (четыре лишние загрузки)")
    ap.add_argument("--offline", action="store_true",
                    help="только геометрия, без обращения к источникам")
    ap.add_argument("--json", action="store_true", help="печатать ответ как JSON")
    args = ap.parse_args(argv)

    if args.polygon_file:
        geometry = json.loads(Path(args.polygon_file).read_text(encoding="utf-8"))
        geometry = geometry.get("geometry", geometry)
    elif args.lon is not None and args.lat is not None:
        geometry = square(args.lon, args.lat, args.side)
    else:
        ap.error("укажите --polygon-file или пару --lon/--lat")

    fetch, parcels = _providers(args.offline)
    res = verify_polygon(geometry, fetch_observations=fetch, find_parcels=parcels,
                         years=args.years, check_shift=args.deep)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        return 0

    g = res["geometry"]
    print()
    print("ГЕОМЕТРИЯ")
    print(f"  точек в контуре       {g['points']}")
    print(f"  площадь               {g['area_ha']} га")
    print(f"  центр                 {g['centroid']}")

    if res["satellite"]:
        s = res["satellite"]
        print()
        print("СПУТНИК")
        print(f"  наблюдений            {s['observations']}")
        print(f"  что это               {s['class']}")
        print(f"  {s['reason']}")

    if res["map"]:
        m = res["map"]
        print()
        print("КАРТА")
        print(f"  контуров рядом        {m['parcels_nearby']}")
        print(f"  лучшее совпадение     {m['best_overlap']:.0%}"
              f"{'  (подтверждено)' if m['matched'] else ''}")
        if m.get("crop_hint"):
            print(f"  культура по карте     {m['crop_hint']}")

    if res["shift"]:
        sh = res["shift"]
        print()
        print("СМЕЩЕНИЕ")
        print(f"  размах у контура      {sh['own_range']}")
        if sh["best_direction"]:
            print(f"  лучше на {sh['best_range']} при сдвиге на {sh['best_direction']}")
        print(f"  подозрение на сдвиг   {'да' if sh['suspected'] else 'нет'}")

    for w in res["warnings"]:
        print(f"\n  ! {w}")
    print()
    print("ИТОГ")
    print(f"  {res['verdict']}")
    return 0 if not res["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
