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
from src.ml.holdout import SEED, HoldoutPoint, _neighbour_distances
from src.ml.registry import REGISTRY, discover

# Значение-заглушка на случай, если у полигона вообще нет известных наблюдений
FALLBACK_NDVI = 0.31
# Метод по умолчанию — итоговая конфигурация решения. Раньше здесь стоял
# промежуточный вариант, и запуск без флага давал файл хуже заявленного:
# способ случайно сдать не тот результат, которым нельзя разбрасываться.
DEFAULT_METHOD = "e05_lgbm"


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


def _stateless_predictions(points: list, views: dict, context: dict) -> pd.DataFrame:
    """Предсказания всех методов без обучения — они же признаки для надстроек.

    Тот же порядок и те же ключи, что в src/ml/validate.evaluate: обучаемый метод
    должен увидеть на инференсе ровно ту таблицу признаков, на которой учился.
    """
    preds: dict[str, np.ndarray] = {}
    for spec in REGISTRY.values():
        method = spec.factory()
        if method.needs_fit:
            continue
        preds[spec.name] = method.predict_points(points, views, context)
    return pd.DataFrame(preds)


def _fit_on_local_holdout(method, seed: int = SEED):
    """Обучает метод на локальном контрольном наборе целиком.

    Фолды GroupKFold нужны для честной ОЦЕНКИ: там модель не должна видеть поле,
    на котором её проверяют. Для боевой модели ограничение снимается — она обязана
    использовать все размеченные точки, какие есть. Число деревьев при этом берётся
    то, что подобрала ранняя остановка внутри самого метода.
    """
    from src.ml.validate import prepare

    df_v, masked, views_v, points_v, _templates, _hidden = prepare(
        use_train=True, hide_frac=0.20, seed=seed)
    context = {"df": masked, "raw": df_v, "points": points_v}
    context["base_preds"] = _stateless_predictions(points_v, views_v, context)
    context["train_mask"] = np.ones(len(points_v), dtype=bool)
    method.fit(views_v, points_v, context)
    return len(points_v)


def _saved_path(method):
    """Путь к сохранённой модели метода, если он такую поддерживает."""
    import sys as _sys

    module = _sys.modules.get(type(method).__module__)
    return getattr(module, "FINAL_MODEL_PATH", None)


def _load_saved(method):
    """Готовая модель метода с диска либо None.

    Механизм намеренно общий: метод объявляет у себя `from_saved()` и
    `FINAL_MODEL_PATH`, и этого достаточно. Так следующий обучаемый метод
    получит то же поведение, не трогая этот файл.
    """
    import sys as _sys

    module = _sys.modules.get(type(method).__module__)
    loader = getattr(module, "from_saved", None)
    path = getattr(module, "FINAL_MODEL_PATH", None)
    if loader is None or path is None or not Path(path).exists():
        return None
    try:
        return loader(path)
    except Exception as exc:  # noqa: BLE001
        # Битый или несовместимый файл модели не должен ронять инференс:
        # честнее переобучить и сказать об этом, чем упасть.
        print(f"Готовую модель прочитать не удалось ({type(exc).__name__}), переобучаю")
        return None


def build_submission(df: pd.DataFrame, method_key: str = DEFAULT_METHOD,
                     retrain: bool = False, save_model: bool = False) -> pd.DataFrame:
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
    context: dict = {"df": df, "points": points}

    if method.needs_fit:
        # Обучаемый метод. Если рядом лежит сохранённая модель — берём её, и
        # тогда результат воспроизводится побитово, а эксперт не ждёт двадцать
        # минут переобучения. Обучение остаётся запасным путём: нет файла или
        # запрошен --retrain.
        loaded = None if retrain else _load_saved(method)
        if loaded is not None:
            method = loaded
            print(f"Метод обучаемый: взята готовая модель {_saved_path(method)}")
        else:
            n_train = _fit_on_local_holdout(method)
            print(f"Метод обучаемый: обучен на {n_train} точках локального контроля")
            if save_model:
                import sys as _sys

                module = _sys.modules.get(type(method).__module__)
                saver = getattr(module, "save_model", None)
                if saver is None:
                    print("Метод не умеет сохранять модель — пропускаю")
                else:
                    print(f"Модель сохранена: {saver(method)}")
        context["base_preds"] = _stateless_predictions(points, views, context)
        if not hasattr(method, "predict_external"):
            raise SystemExit(
                f"метод {method_key!r} обучаемый, но не умеет считать признаки для точек "
                f"вне контрольного набора: нужен predict_external(points, views, context)"
            )
        preds = method.predict_external(points, views, context)
    else:
        preds = method.predict_points(points, views, context)

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
    parser.add_argument("--retrain", action="store_true",
                        help="переобучить модель с нуля, не беря готовую из models/")
    parser.add_argument("--save-model", action="store_true",
                        help="сохранить переобученную модель в models/ (только с --retrain)")
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

    sub = build_submission(df, method_key=args.method, retrain=args.retrain,
                           save_model=args.save_model)
    validate(sub, expected)

    out = Path(args.output)
    sub.to_csv(out, index=False, encoding="utf-8")
    print(f"Метод: {args.method} — {REGISTRY[args.method].title}")
    print(f"Записано {len(sub)} строк в {out}")
    print(f"Диапазон предсказаний: {sub.primary_ndvi_pred.min():.4f} .. {sub.primary_ndvi_pred.max():.4f}")
    print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
