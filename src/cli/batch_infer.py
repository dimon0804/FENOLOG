"""Технический режим: private_features.csv -> submission.csv.

Запуск:
    python -m src.cli.batch_infer --method sibw3
    python -m src.cli.batch_infer --input data/private_features.csv --output submission.csv

Восстанавливаются только строки с is_synthetic_gap = True. Формат файла задан
организаторами: колонки anon_polygon_id, date, primary_ndvi_pred, разделитель
запятая, кодировка UTF-8, каждая пара «полигон + дата» встречается ровно один раз.

Ключевое требование к этому файлу: он обязан считать контрольные точки **тем же
кодом**, что и локальная валидация. Любая своя реализация «как в валидации, только
для инференса» рано или поздно разойдётся с ней, и локальная цифра перестанет
что-либо значить. Поэтому здесь нет собственного восстановления — только сборка
представлений полигонов и вызов метода из реестра `src/ml/registry.py`.

Отличие от валидации ровно одно: там контрольные точки создаются искусственно и у
них известен ответ, здесь они уже размечены колонкой is_synthetic_gap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.dataset import build_views, load_frame, TRAIN_PATH
from src.ml.holdout import HoldoutPoint, _neighbour_distances
from src.ml.registry import REGISTRY, discover

# Значение-заглушка на случай, если у полигона вообще нет известных наблюдений
FALLBACK_NDVI = 0.31
DEFAULT_METHOD = "sibw3"


def build_targets(df: pd.DataFrame) -> list[HoldoutPoint]:
    """Контрольные точки в том же виде, в каком их видит протокол валидации.

    Расстояния до ближайших известных соседей считаются по фактическим данным:
    на них опираются методы с порогом по длине разрыва (климатологический якорь)
    и признаки бустинга.
    """
    points: list[HoldoutPoint] = []
    for polygon_id, g in df.groupby("anon_polygon_id", sort=False):
        g = g.sort_values("_ord")
        targets = g[g["is_synthetic_gap"]]
        if targets.empty:
            continue
        known_ord = g.loc[g["primary_ndvi"].notna(), "_ord"].to_numpy(dtype=np.int64)
        t_ord = targets["_ord"].to_numpy(dtype=np.int64)
        left, right = _neighbour_distances(t_ord, known_ord)
        for k in range(len(t_ord)):
            points.append(
                HoldoutPoint(
                    polygon_id=str(polygon_id),
                    ord_day=int(t_ord[k]),
                    truth=float("nan"),
                    left_dist=int(left[k]),
                    right_dist=int(right[k]),
                    month=int(targets["_month"].to_numpy()[k]),
                )
            )
    return points


def build_submission(df: pd.DataFrame, method_key: str = DEFAULT_METHOD) -> pd.DataFrame:
    """Восстанавливает все контрольные точки выбранным методом из реестра."""
    discover()
    if method_key not in REGISTRY:
        raise SystemExit(
            f"метод {method_key!r} не зарегистрирован. Доступны: {', '.join(sorted(REGISTRY))}"
        )

    views = build_views(df)
    points = build_targets(df)
    if not points:
        return pd.DataFrame(columns=["anon_polygon_id", "date", "primary_ndvi_pred"])

    method = REGISTRY[method_key].factory()
    if method.needs_fit:
        raise SystemExit(
            f"метод {method_key!r} требует обучения; обучаемые методы подаются через "
            f"свою точку входа, а не через batch_infer"
        )
    preds = method.predict_points(points, views, {"df": df})

    # Страховка: ни одного пропуска и ни одного значения вне диапазона NDVI
    preds = np.asarray(preds, dtype=float)
    preds = np.where(np.isfinite(preds), preds, FALLBACK_NDVI)
    preds = np.clip(preds, 0.0, 1.0)

    return pd.DataFrame(
        {
            "anon_polygon_id": [p.polygon_id for p in points],
            "date": [pd.Timestamp.fromordinal(p.ord_day).strftime("%Y-%m-%d") for p in points],
            "primary_ndvi_pred": preds,
        }
    )


def validate(sub: pd.DataFrame, expected: int) -> None:
    """Проверяет файл до отправки — платформа не примёт его с ошибками формата."""
    assert list(sub.columns) == ["anon_polygon_id", "date", "primary_ndvi_pred"], "неверные колонки"
    assert len(sub) == expected, f"строк {len(sub)}, ожидалось {expected}"
    assert sub["primary_ndvi_pred"].notna().all(), "есть пропуски в предсказаниях"
    assert np.isfinite(sub["primary_ndvi_pred"]).all(), "есть бесконечные значения"
    assert not sub.duplicated(["anon_polygon_id", "date"]).any(), "есть дубликаты пары полигон+дата"
    assert sub["primary_ndvi_pred"].between(0.0, 1.0).all(), "значения вне диапазона NDVI"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Восстановление контрольных точек primary_ndvi")
    parser.add_argument("--input", default="data/private_features.csv", help="путь к тестовому файлу")
    parser.add_argument("--output", default="submission.csv", help="куда положить результат")
    parser.add_argument("--method", default=DEFAULT_METHOD, help="ключ метода из реестра")
    parser.add_argument("--no-train", action="store_true",
                        help="не подмешивать наблюдения train_dataset (на метрику не влияет, "
                             "но влияет на объём истории для климатологии)")
    parser.add_argument("--list-methods", action="store_true", help="показать доступные методы")
    args = parser.parse_args()

    if args.list_methods:
        discover()
        for spec in sorted(REGISTRY.values(), key=lambda s: (s.experiment, s.name)):
            print(f"{spec.name:16} {spec.experiment:5} {spec.title}")
        return

    test = load_frame(args.input, "test")
    expected = int(test["is_synthetic_gap"].sum())

    # Наблюдения train не сокращают разрывы (замерено в E01b), но удлиняют
    # историю каждого поля, а от неё зависит и климатология, и таблица
    # корреляций между полями в суточной поправке.
    df = test
    if not args.no_train and Path(TRAIN_PATH).exists():
        train = load_frame(TRAIN_PATH, "train")
        df = pd.concat([test, train], ignore_index=True, sort=False)
        df["is_synthetic_gap"] = df["is_synthetic_gap"].fillna(False).astype(bool)
    df = df.sort_values(["anon_polygon_id", "_ord"]).reset_index(drop=True)

    sub = build_submission(df, method_key=args.method)
    validate(sub, expected)

    out = Path(args.output)
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"Метод: {args.method} — {REGISTRY[args.method].title}")
    print(f"Записано {len(sub)} строк в {out}")
    print(f"Диапазон предсказаний: {sub.primary_ndvi_pred.min():.4f} .. {sub.primary_ndvi_pred.max():.4f}")
    print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
