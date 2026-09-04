"""Технический режим: private_features.csv -> submission.csv.

Запуск:
    python -m src.cli.batch_infer --input data/private_features.csv --output submission.csv

Восстанавливаются только строки с is_synthetic_gap = True. Формат файла задан
организаторами: колонки anon_polygon_id, date, primary_ndvi_pred, разделитель запятая,
кодировка UTF-8, каждая пара «полигон + дата» встречается ровно один раз, без пропусков.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.restore import predict_at, restore_on_grid

# Значение-заглушка на случай, если у полигона вообще нет известных наблюдений
FALLBACK_NDVI = 0.31  # медиана primary_ndvi по всему выданному набору


def build_submission(df: pd.DataFrame, lam: float = 100.0, mix: float = 0.5) -> pd.DataFrame:
    """Восстанавливает контрольные точки по каждому полигону отдельно."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_ord"] = df["date"].map(pd.Timestamp.toordinal)

    parts: list[pd.DataFrame] = []
    for polygon_id, g in df.groupby("anon_polygon_id", sort=False):
        g = g.sort_values("date")
        targets = g[g["is_synthetic_gap"] == True]  # noqa: E712 — в CSV это булев столбец
        if targets.empty:
            continue

        known = g[g["primary_ndvi"].notna()]
        if len(known) < 2:
            preds = np.full(len(targets), FALLBACK_NDVI)
        else:
            grid, restored = restore_on_grid(
                known["_ord"].to_numpy(), known["primary_ndvi"].to_numpy(), lam=lam, mix=mix
            )
            preds = predict_at(grid, restored, targets["_ord"].to_numpy())

        parts.append(
            pd.DataFrame(
                {
                    "anon_polygon_id": targets["anon_polygon_id"].to_numpy(),
                    "date": targets["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                    "primary_ndvi_pred": np.asarray(preds, dtype=float),
                }
            )
        )

    if not parts:
        return pd.DataFrame(columns=["anon_polygon_id", "date", "primary_ndvi_pred"])
    return pd.concat(parts, ignore_index=True)


def validate(sub: pd.DataFrame, expected: int) -> None:
    """Проверяет файл до отправки — платформа не примёт его с ошибками формата."""
    assert list(sub.columns) == ["anon_polygon_id", "date", "primary_ndvi_pred"], "неверные колонки"
    assert len(sub) == expected, f"строк {len(sub)}, ожидалось {expected}"
    assert sub["primary_ndvi_pred"].notna().all(), "есть пропуски в предсказаниях"
    assert np.isfinite(sub["primary_ndvi_pred"]).all(), "есть бесконечные значения"
    assert not sub.duplicated(["anon_polygon_id", "date"]).any(), "есть дубликаты пары полигон+дата"


def main() -> None:
    parser = argparse.ArgumentParser(description="Восстановление контрольных точек primary_ndvi")
    parser.add_argument("--input", required=True, help="путь к private_features.csv")
    parser.add_argument("--output", default="submission.csv", help="куда положить результат")
    parser.add_argument("--lam", type=float, default=100.0, help="сила сглаживания Уиттекера")
    parser.add_argument("--mix", type=float, default=0.5, help="доля Уиттекера в смеси")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    expected = int((df["is_synthetic_gap"] == True).sum())  # noqa: E712
    sub = build_submission(df, lam=args.lam, mix=args.mix)
    validate(sub, expected)

    out = Path(args.output)
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"Записано {len(sub)} строк в {out}")
    print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
