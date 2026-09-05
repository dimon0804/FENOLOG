"""Детекция периодов угнетения растительности по готовому ряду наблюдений.

Вторая точка входа в решение помимо восстановления пропусков. Нужна, чтобы
детекцию аномалий можно было запустить и проверить отдельно от веб-сервиса —
это прямое требование к воспроизводимости.

    python -m src.cli.detect --polygon AOI-0001
    python -m src.cli.detect --all --output reports/anomalies.csv
    python -m src.cli.detect --polygon AOI-0043 --json

Ряд берётся из того же CSV, что и в техническом режиме восстановления. В рабочем
сервисе ровно та же функция analyze() вызывается на ряде, который слой провайдеров
собрал из спутниковых снимков и метеоданных, — код детекции при этом не меняется.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd

from src.contracts import Observation, SeriesInput, WeatherPoint
from src.core.analyze import analyze

DEFAULT_INPUT = "data/private_features.csv"


def series_from_frame(polygon_id: str, g: pd.DataFrame) -> SeriesInput:
    """Собирает вход доменного ядра из куска таблицы по одному полигону.

    Наблюдения берутся только там, где значение индекса известно: контрольные
    строки и естественные пропуски ядру подавать не нужно, оно восстановит ряд само.
    """
    g = g.sort_values("date")
    observations = [
        Observation(date=row.date.date(), ndvi=float(row.primary_ndvi))
        for row in g.itertuples()
        if pd.notna(row.primary_ndvi)
    ]
    weather = [
        WeatherPoint(
            date=row.date.date(),
            temp_c=None if pd.isna(row.era5_temp_c) else float(row.era5_temp_c),
            precip_mm=None if pd.isna(row.era5_precip_mm) else float(row.era5_precip_mm),
        )
        for row in g.itertuples()
    ]
    crop = g["crop_type"].dropna()
    return SeriesInput(
        polygon_id=polygon_id,
        observations=observations,
        weather=weather,
        crop_type=str(crop.iloc[0]) if len(crop) else None,
    )


def report_rows(polygon_id: str, result) -> list[dict]:
    """Плоская таблица найденных периодов — то, что уходит в CSV и в интерфейс."""
    return [
        {
            "anon_polygon_id": polygon_id,
            "start": a.start.isoformat(),
            "end": a.end.isoformat(),
            "duration_days": a.duration_days,
            "severity": a.severity,
            "min_zscore": round(a.min_zscore, 3),
            "mean_zscore": round(a.mean_zscore, 3),
            "cause": a.cause,
            "cause_confidence": round(a.cause_confidence, 2),
            "climatology_source": result.meta.get("climatology_source"),
            "explanation": a.explanation,
        }
        for a in result.anomalies
    ]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Поиск периодов угнетения растительности")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="CSV с рядами наблюдений")
    parser.add_argument("--polygon", help="идентификатор полигона, например AOI-0001")
    parser.add_argument("--all", action="store_true", help="прогнать все полигоны набора")
    parser.add_argument("--output", help="куда сохранить таблицу найденных периодов (CSV)")
    parser.add_argument("--json", action="store_true", help="печатать полный ответ ядра в JSON")
    args = parser.parse_args()

    if not args.polygon and not args.all:
        parser.error("укажите --polygon <id> или --all")

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    if args.polygon:
        if args.polygon not in set(df["anon_polygon_id"]):
            raise SystemExit(f"полигон {args.polygon!r} не найден в {args.input}")
        df = df[df["anon_polygon_id"] == args.polygon]

    rows: list[dict] = []
    for polygon_id, g in df.groupby("anon_polygon_id", sort=True):
        result = analyze(series_from_frame(str(polygon_id), g))

        if args.json:
            print(json.dumps(dataclasses.asdict(result), ensure_ascii=False,
                             indent=2, default=str))
            continue

        found = report_rows(str(polygon_id), result)
        rows.extend(found)
        if not args.all:
            source = result.meta.get("climatology_source")
            print(f"Полигон {polygon_id}: наблюдений {result.meta.get('n_obs')}, "
                  f"норма по источнику «{source}», найдено периодов {len(found)}")
            note = result.meta.get("climatology_note")
            if source == "crop" and note:
                print(f"  Норма: {note}")
            crop = result.meta.get("crop_detection") or {}
            if crop.get("note"):
                print(f"  Культура: {crop['note']}")
            if crop.get("conflict"):
                # Расхождение заявленного с увиденным выносится отдельной
                # строкой: в CSV оно не помещается, а для агронома это самая
                # ценная строка вывода — она означает, что данные о поле устарели.
                print("  ! Заявленная культура расходится с тем, что видно в данных.")
            print()
            for a in sorted(result.anomalies, key=lambda x: x.min_zscore):
                print(f"  {a.start} .. {a.end}  ({a.duration_days} дн)  "
                      f"z_min {a.min_zscore:.2f}  {a.severity} / {a.cause}")
                print(f"     {a.explanation}")
                print()

    if args.output and rows:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
        print(f"Записано {len(rows)} периодов по {len(set(r['anon_polygon_id'] for r in rows))} "
              f"полигонам в {out}")
    elif args.all:
        print(f"Найдено {len(rows)} периодов. Укажите --output, чтобы сохранить таблицу.")


if __name__ == "__main__":
    main()
