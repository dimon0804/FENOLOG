"""Воспроизводимый протокол локальной валидации восстановления NDVI.

    python -m src.ml.validate

Зачем он нужен. Лидерборд отдаёт одну цифру в сутки и годится только для проверки
формата файла. Все решения по методам принимаются по этой команде, и только по ней.

Как устроено.
  1. Из тестового файла снимается геометрия 3112 реальных контрольных точек:
     расстояние до ближайшего известного значения слева, справа и месяц.
  2. По этим шаблонам прячется 20 % известных значений (см. src/ml/holdout.py).
     У спрятанных строк стираются ВСЕ признаки, кроме id, даты и культуры —
     ровно так замаскированы настоящие контрольные точки.
  3. Обучаемые методы гоняются через GroupKFold по anon_polygon_id: полигон
     целиком либо в обучении, либо в контроле. Иначе бустинг подглядит соседние
     дни того же поля и локальная цифра разойдётся с лидербордом.
  4. Метрика — RMSE по спрятанным точкам и GapScore = 30 · max(0, 1 − RMSE/0.1).

Зерно 42 зашито в holdout.SEED, набор детерминирован от запуска к запуску.
Предсказания всех методов складываются в reports/validation_preds.csv — из них
собираются признаки для бустинга-надстройки и разрезы для отчёта.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.ml import holdout as H
from src.ml.dataset import build_views, load_all, mask_rows
from src.ml.registry import REGISTRY, discover

PREDS_PATH = Path("reports/validation_preds.csv")
N_FOLDS = 5


def gap_score(rmse: float) -> float:
    """Балл организаторов за восстановление: 30 баллов при RMSE 0, ноль при 0.1."""
    return round(30.0 * max(0.0, 1.0 - rmse / 0.10), 2)


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def prepare(use_train: bool, hide_frac: float, seed: int, test_path=None):
    """Готовит контрольный набор: прячет точки и стирает у них признаки.

    test_path — какой файл считать основным. По умолчанию тот, на котором
    ставился весь протокол. Возможность подменить нужна, когда организаторы
    выдают новый тест: у него своя плотность наблюдений (до 2016 года нет
    Sentinel-2 вовсе), и мерить на старом файле — значит мерить не ту задачу.
    """
    df = load_all(use_train=use_train, **({"test_path": test_path} if test_path else {}))
    templates = H.extract_templates(df)
    points, hidden_rows = H.build_holdout(df, templates, hide_frac=hide_frac, seed=seed)
    masked = mask_rows(df, hidden_rows)
    views = build_views(masked)
    return df, masked, views, points, templates, hidden_rows


def evaluate(views, points, context, only: list[str] | None = None) -> pd.DataFrame:
    """Прогоняет все зарегистрированные методы и собирает таблицу результатов."""
    truth = np.array([p.truth for p in points], dtype=float)
    groups = np.array([p.polygon_id for p in points])

    # Сначала методы без обучения: их предсказания заодно становятся признаками
    # для надстроек. Утечки тут нет — они ничего не подбирают по разметке.
    specs = [s for s in REGISTRY.values() if only is None or s.name in only]
    stateless = [s for s in specs if not s.factory().needs_fit]
    trainable = [s for s in specs if s.factory().needs_fit]

    preds: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}

    for spec in stateless:
        t0 = time.perf_counter()
        preds[spec.name] = spec.factory().predict_points(points, views, context)
        timing[spec.name] = time.perf_counter() - t0

    context = dict(context)
    context["base_preds"] = pd.DataFrame(preds)

    if trainable:
        # GroupKFold по полигонам: поле целиком либо в обучении, либо в контроле
        n_splits = min(N_FOLDS, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits)
        for spec in trainable:
            t0 = time.perf_counter()
            out = np.full(len(points), np.nan)
            for train_idx, test_idx in splitter.split(np.zeros(len(points)), groups=groups):
                method = spec.factory()
                fold_ctx = dict(context)
                fold_ctx["train_mask"] = np.isin(np.arange(len(points)), train_idx)
                method.fit(views, points, fold_ctx)
                subset = [points[i] for i in test_idx]
                out[test_idx] = method.predict_points(subset, views, fold_ctx)
            preds[spec.name] = out
            timing[spec.name] = time.perf_counter() - t0

    bins = H.gap_bin(np.array([p.left_dist for p in points]),
                     np.array([p.right_dist for p in points]))

    rows = []
    for name, p in preds.items():
        spec = REGISTRY[name]
        rows.append(
            {
                "метод": spec.title,
                "эксп.": spec.experiment,
                "RMSE": round(rmse(truth, p), 4),
                "GapScore": gap_score(rmse(truth, p)),
                "MAE": round(float(np.mean(np.abs(truth - p))), 4),
                "сек": round(timing[name], 1),
                "_name": name,
            }
        )
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    # Разрез по длине разрыва: где именно метод выигрывает, а где просто везёт
    per_bin = {}
    for name, p in preds.items():
        per_bin[REGISTRY[name].title] = {
            b: round(rmse(truth[bins == b], p[bins == b]), 4)
            for b in H.GAP_LABELS if (bins == b).sum() >= 20
        }
    breakdown = pd.DataFrame(per_bin).T

    frame = pd.DataFrame(
        {
            "anon_polygon_id": groups,
            "date": [pd.Timestamp.fromordinal(p.ord_day).strftime("%Y-%m-%d") for p in points],
            "truth": truth,
            "left_dist": [p.left_dist for p in points],
            "right_dist": [p.right_dist for p in points],
            "month": [p.month for p in points],
            "gap_bin": bins,
        }
    )
    for name, p in preds.items():
        frame[f"pred_{name}"] = p

    return table, breakdown, frame


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Локальная валидация методов восстановления")
    parser.add_argument("--no-train", action="store_true",
                        help="не подмешивать наблюдения из train_dataset.csv")
    parser.add_argument("--hide-frac", type=float, default=0.20, help="доля скрываемых значений")
    parser.add_argument("--seed", type=int, default=H.SEED, help="зерно протокола")
    parser.add_argument("--only", nargs="*", default=None, help="прогнать только эти методы")
    parser.add_argument("--no-save", action="store_true", help="не писать validation_preds.csv")
    args = parser.parse_args()

    loaded = discover()
    use_train = not args.no_train

    t0 = time.perf_counter()
    df, masked, views, points, templates, hidden_rows = prepare(use_train, args.hide_frac, args.seed)

    print("=" * 78)
    print("ПРОТОКОЛ ЛОКАЛЬНОЙ ВАЛИДАЦИИ")
    print("=" * 78)
    print(f"источник данных      : private_features.csv" + (" + train_dataset.csv" if use_train else ""))
    print(f"строк всего          : {len(df)}")
    print(f"известных значений   : {int(df['primary_ndvi'].notna().sum())}")
    print(f"скрыто строк         : {len(hidden_rows)} ({100 * len(hidden_rows) / df['primary_ndvi'].notna().sum():.1f} %)")
    print(f"оценивается точек    : {len(points)}")
    print(f"полигонов            : {len(views)}, зерно {args.seed}, GroupKFold {N_FOLDS}")
    print(f"модулей методов      : {', '.join(loaded)}")
    print()
    print("Совпадение геометрии с реальными контрольными точками:")
    print(H.describe_holdout(points, templates).to_string())
    print()

    context = {"df": masked, "raw": df, "points": points}
    table, breakdown, frame = evaluate(views, points, context, only=args.only)

    print("=" * 78)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 78)
    print(table.drop(columns=["_name"]).to_string(index=False))
    print()
    print("RMSE в разрезе длины разрыва (дней):")
    print(breakdown.to_string())
    print()

    if not args.no_save:
        PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(PREDS_PATH, index=False, encoding="utf-8")
        print(f"Предсказания сохранены: {PREDS_PATH}")
    print(f"Всего {time.perf_counter() - t0:.1f} с")


if __name__ == "__main__":
    main()
