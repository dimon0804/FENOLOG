"""Проверка определения культуры. Leave-one-polygon-out, эксперимент E19.

Здесь измеряется всё, что проект утверждает про распознавание культуры, и
измеряется одним прогоном, чтобы числа в отчёте нельзя было получить случайно:

    1. Точность детектора против базовой линии «всегда самая частая культура».
    2. Точность на крупных группах (озимые против яровых и пастбищ).
    3. Согласие заявленной уверенности с фактической долей верных ответов —
       без этого порог CONFIDENCE_MIN был бы подобран на глаз.
    4. Сравнение с градиентным бустингом на тех же признаках: если простое
       сравнение с эталоном проигрывает существенно, его надо менять.
    5. Главное. Выигрыш в самой задаче: насколько норма выбранной культуры
       ближе к настоящей кривой поля, чем норма «в среднем по всем культурам».
       Классификатор может быть точным и при этом бесполезным — если культуры
       различаются слабо, ошибка выбора ничего не стоит. Проверяется отдельно.

Своё поле всюду исключено: и из эталонов культур, и из обучения бустинга.
Группировка кросс-валидации по полигону, а не по строке — сезоны одного поля
похожи друг на друга, и разрезание по строкам дало бы завышенную точность.

Запуск:
    python -m src.ml.crop_id             # полный протокол
    python -m src.ml.crop_id --no-lgbm   # без сравнения с бустингом, быстрее
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.crop_climatology import (
    MIN_SEASON_DAYS,
    MIN_YEARS,
    SEASON_DOY,
    polygon_doy_curve,
)
from src.core.crop_profile import (
    CROP_GROUP,
    SEASON_GRID,
    CropDetector,
    phenology,
    season_curve,
)
from src.ml.dataset import TEST_PATH, TRAIN_PATH

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "crop_classifier.pkl"

# Порог уверенности, на которых строится таблица «покрытие против точности»
CONF_STEPS = (0.4, 0.5, 0.6, 0.7, 0.8)


def load_labelled() -> pd.DataFrame:
    """Оба файла набора одной таблицей: сезоны одного поля лежат в разных файлах.

    Разрез между train и test идёт по годам, а не по полям, поэтому для портрета
    поля их надо склеивать — иначе у 39 полигонов из истории видно только 2025 год.
    """
    cols = ["anon_polygon_id", "date", "crop_type", "primary_ndvi"]
    frames = []
    for path in (TEST_PATH, TRAIN_PATH):
        df = pd.read_csv(path, usecols=cols)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["primary_ndvi"].notna()].copy()
    df["_doy"] = df["date"].dt.dayofyear
    df["_year"] = df["date"].dt.year
    return df


def season_table(df: pd.DataFrame) -> pd.DataFrame:
    """Кривые и фенологические признаки по каждому сезону каждого поля."""
    rows = []
    for (pid, year), g in df.groupby(["anon_polygon_id", "_year"], sort=False):
        curve = season_curve(g["date"].dt.date.tolist(), g["primary_ndvi"].tolist(), year=year)
        if curve is None:
            continue
        row = {"pid": str(pid), "year": int(year), "crop": str(g["crop_type"].iloc[0]),
               "curve": curve}
        row.update(phenology(curve))
        rows.append(row)
    return pd.DataFrame(rows)


def polygon_curves(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, float, float, str]]:
    """Портрет каждого поля с достаточной историей: (среднее, разброс, уровень, размах, культура).

    Повторяет отбор из CropClimatology.fit — те же MIN_YEARS и MIN_SEASON_DAYS,
    иначе эталоны валидации отличались бы от боевых и сравнивать было бы нечего.
    """
    season = np.zeros(366, dtype=bool)
    season[SEASON_DOY[0] - 1:SEASON_DOY[1]] = True
    out = {}
    for pid, g in df.groupby("anon_polygon_id", sort=False):
        if g["_year"].nunique() < MIN_YEARS:
            continue
        mean, std = polygon_doy_curve(g["_doy"].to_numpy(int), g["primary_ndvi"].to_numpy(float))
        filled = np.isfinite(mean) & season
        if int(filled.sum()) < MIN_SEASON_DAYS:
            continue
        level = float(np.nanmean(mean[filled]))
        amp = float(np.nanstd(mean[filled]))
        out[str(pid)] = (mean, std, level, max(amp, 1e-6), str(g["crop_type"].iloc[0]))
    return out


def _crop_prototypes(curves: dict, exclude: str | None) -> dict[str, np.ndarray]:
    """Эталонные кривые культур по всем полям, кроме одного.

    Усреднение центрированное, как в боевой CropClimatology: поля одной культуры
    различаются уровнем, и сырое среднее размазывало бы форму. Пересчёт вынесен
    сюда, а не сделан вызовом CropClimatology.fit на подвыборке, ради скорости:
    портреты полей считаются один раз, а не 78 раз подряд.
    """
    by_crop: dict[str, list[str]] = {}
    for pid, (_m, _s, _l, _a, crop) in curves.items():
        if pid == exclude:
            continue
        by_crop.setdefault(crop, []).append(pid)

    proto = {}
    for crop, pids in by_crop.items():
        M = np.vstack([curves[p][0] for p in pids])
        levels = np.array([curves[p][2] for p in pids])[:, None]
        # Зимние дни года пусты у всех полей: nanmean по пустому срезу законно
        # возвращает NaN, предупреждение о нём только зашумляет вывод.
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(M - levels, axis=0) + float(levels.mean())
        proto[crop] = mean[SEASON_GRID - 1]
    return proto


def _global_prototype(curves: dict, exclude: str | None) -> np.ndarray:
    """Одна норма на все культуры — то, чем сервис пользуется, не зная культуру."""
    proto = _crop_prototypes(curves, exclude)
    return np.nanmean(np.vstack(list(proto.values())), axis=0)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(((a[m] - b[m]) ** 2).mean())) if int(m.sum()) > 20 else float("nan")


def run(use_lgbm: bool = True) -> dict:
    df = load_labelled()
    seasons = season_table(df)
    curves = polygon_curves(df)
    print(f"выборка: {len(seasons)} полигоно-лет, {seasons.pid.nunique()} полей, "
          f"{len(curves)} из них с историей от {MIN_YEARS} сезонов")

    labels = seasons.groupby("pid")["crop"].first()
    majority = labels.value_counts().idxmax()
    base_poly = float((labels == majority).mean())
    base_season = float((seasons["crop"] == majority).mean())
    print(f"базовая линия «всегда {majority}»: по сезонам {base_season:.4f}, "
          f"по полям {base_poly:.4f}")

    # --- 1. Детектор, leave-one-polygon-out -------------------------------
    per_season_pred, per_season_conf = [], []
    poly_pred, poly_conf, poly_true, poly_ids = [], [], [], []
    for pid, g in seasons.groupby("pid", sort=False):
        det = CropDetector(_crop_prototypes(curves, exclude=pid))
        if not det.crops:
            continue
        probs = []
        for curve in g["curve"]:
            p = det.predict(curve)
            per_season_pred.append((p["best_guess"], g["crop"].iloc[0]))
            per_season_conf.append(p["confidence"])
            probs.append(p["scores"])
        mean_prob = pd.DataFrame(probs).mean().to_dict()
        top = max(mean_prob, key=mean_prob.get)
        poly_pred.append(top)
        poly_conf.append(float(mean_prob[top]))
        poly_true.append(g["crop"].iloc[0])
        poly_ids.append(pid)

    poly_pred = np.array(poly_pred)
    poly_true_a = np.array(poly_true)
    poly_conf_a = np.array(poly_conf)
    acc_season = float(np.mean([p == t for p, t in per_season_pred]))
    acc_poly = float((poly_pred == poly_true_a).mean())
    grp_pred = np.array([CROP_GROUP.get(c, c) for c in poly_pred])
    grp_true = np.array([CROP_GROUP.get(c, c) for c in poly_true_a])
    acc_group = float((grp_pred == grp_true).mean())
    base_group = float(pd.Series(grp_true).value_counts(normalize=True).iloc[0])

    print(f"\nдетектор по эталонам культур:")
    print(f"  по сезонам {acc_season:.4f}")
    print(f"  по полям   {acc_poly:.4f}  (базовая {base_poly:.4f})")
    print(f"  по группам {acc_group:.4f}  (базовая {base_group:.4f})")

    print("\n  полнота по классам (поля):")
    for crop in sorted(set(poly_true_a)):
        m = poly_true_a == crop
        print(f"    {crop:<20} n={int(m.sum()):3d}  найдено {float((poly_pred[m] == crop).mean()):.3f}")

    print("\n  матрица ошибок (строки — правда):")
    cm = pd.crosstab(pd.Series(poly_true_a, name="правда"), pd.Series(poly_pred, name="детектор"))
    print("    " + cm.to_string().replace("\n", "\n    "))

    print("\n  порог уверенности: покрытие и точность")
    conf_rows = []
    for thr in CONF_STEPS:
        m = poly_conf_a >= thr
        if not m.any():
            continue
        conf_rows.append((thr, float(m.mean()), float((poly_pred[m] == poly_true_a[m]).mean())))
        print(f"    >={thr:.1f}  покрытие {conf_rows[-1][1]:.2f}  точность {conf_rows[-1][2]:.3f}")

    # --- 2. Сравнение с бустингом ------------------------------------------
    acc_lgbm = None
    lgbm_of: dict[str, str] = {}
    if use_lgbm:
        try:
            import lightgbm as lgb
            from sklearn.model_selection import StratifiedGroupKFold
        except ImportError:
            print("\nlightgbm или sklearn недоступны, сравнение пропущено")
        else:
            cols = [c for c in seasons.columns if c not in ("pid", "year", "crop", "curve")]
            classes = sorted(seasons["crop"].unique())
            cmap = {c: i for i, c in enumerate(classes)}
            y = seasons["crop"].map(cmap).to_numpy()
            groups = seasons["pid"].to_numpy()
            oof = np.zeros((len(seasons), len(classes)))
            splitter = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
            for tr_i, te_i in splitter.split(seasons[cols], y, groups):
                model = lgb.LGBMClassifier(
                    objective="multiclass", num_class=len(classes), n_estimators=300,
                    learning_rate=0.05, num_leaves=15, min_child_samples=20,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    class_weight="balanced", verbose=-1, random_state=42,
                )
                model.fit(seasons.iloc[tr_i][cols], y[tr_i])
                oof[te_i] = model.predict_proba(seasons.iloc[te_i][cols])
            agg = pd.DataFrame(oof, columns=classes)
            agg["pid"] = groups
            poly_p = agg.groupby("pid")[classes].mean()
            pred_l = np.array([classes[i] for i in poly_p.values.argmax(1)])
            true_l = labels.reindex(poly_p.index).to_numpy()
            lgbm_of = dict(zip(poly_p.index.astype(str), pred_l))
            acc_lgbm = float((pred_l == true_l).mean())
            grp_l = float((np.array([CROP_GROUP.get(c, c) for c in pred_l])
                           == np.array([CROP_GROUP.get(c, c) for c in true_l])).mean())
            print(f"\nградиентный бустинг на тех же признаках: "
                  f"по полям {acc_lgbm:.4f}, по группам {grp_l:.4f}")
            # Калибровка бустинга проверяется отдельно и обязательно: модель
            # обучена с выравниванием весов классов и без калибровки выдаёт
            # уверенность 0,95 там, где верна в трёх случаях из четырёх. Если
            # интерфейс показывает это число пользователю, оно должно что-то
            # значить, поэтому таблица «покрытие против точности» идёт в отчёт.
            conf_l = poly_p.values.max(1)
            print("  порог уверенности бустинга: покрытие и точность")
            for thr in CONF_STEPS:
                m = conf_l >= thr
                if m.any():
                    print(f"    >={thr:.1f}  покрытие {float(m.mean()):.2f}  "
                          f"точность {float((pred_l[m] == true_l[m]).mean()):.3f}")

    # --- 3. Главное: выигрыш в норме ----------------------------------------
    # Поле без своей истории получает норму по культуре. Вопрос: насколько эта
    # норма ближе к настоящей кривой поля, чем норма «в среднем по всем».
    print("\nвыигрыш в норме (RMSE нормы против собственной кривой поля):")
    det_of = dict(zip(poly_ids, poly_pred))
    order = ["культура не учтена", "названа бустингом", "подобрана по кривой", "известна точно"]
    res: dict[str, list[float]] = {k: [] for k in order}
    for pid, (own_mean, _s, _l, _a, crop) in curves.items():
        proto = _crop_prototypes(curves, exclude=pid)
        glob = _global_prototype(curves, exclude=pid)
        ref = own_mean[SEASON_GRID - 1]
        if crop in proto:
            res["известна точно"].append(_rmse(proto[crop], ref))
        det_crop = det_of.get(pid)
        if det_crop in proto:
            res["подобрана по кривой"].append(_rmse(proto[det_crop], ref))
        lg_crop = lgbm_of.get(pid)
        if lg_crop in proto:
            res["названа бустингом"].append(_rmse(proto[lg_crop], ref))
        res["культура не учтена"].append(_rmse(glob, ref))

    summary = {}
    for key in order:
        if not res[key]:
            continue
        v = np.array(res[key], dtype=float)
        v = v[np.isfinite(v)]
        summary[key] = float(v.mean())
        print(f"  {key:<26} {v.mean():.4f}   медиана {np.median(v):.4f}  n={len(v)}")
    ceiling = summary["культура не учтена"] - summary["известна точно"]
    for key in ("подобрана по кривой", "названа бустингом"):
        if key not in summary or not ceiling:
            continue
        gain = summary["культура не учтена"] - summary[key]
        print(f"  {key:<22} отыгрывает {gain:+.4f} из {ceiling:+.4f} "
              f"({100 * gain / ceiling:.0f} % доступного)")

    return {
        "acc_season": acc_season, "acc_poly": acc_poly, "acc_group": acc_group,
        "base_poly": base_poly, "base_group": base_group, "acc_lgbm": acc_lgbm,
        "norm_rmse": summary, "conf_curve": conf_rows,
    }


def train(path=MODEL_PATH) -> dict:
    """Обучает классификатор названия культуры на всех размеченных полях.

    Честность результата обеспечивает не эта функция, а run(): здесь модель
    учится на всём, что есть, потому что в бою ей достанется поле, которого в
    наборе не было вовсе. Оценка её точности берётся из кросс-валидации run(),
    и путать эти два числа нельзя — на обучающих данных модель почти безошибочна.
    """
    import pickle

    import lightgbm as lgb

    from src.core.crop_profile import FEATURES

    seasons = season_table(load_labelled())
    classes = sorted(seasons["crop"].unique())
    cmap = {c: i for i, c in enumerate(classes)}
    y = seasons["crop"].map(cmap).to_numpy()
    X = seasons[list(FEATURES)]
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(classes), n_estimators=300,
        learning_rate=0.05, num_leaves=15, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        # Классы перекошены вчетверо (48 полей пшеницы против 6 подсолнечника),
        # и без выравнивания весов модель просто перестаёт называть редкие.
        class_weight="balanced", verbose=-1, random_state=42,
    )
    model.fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"booster": model.booster_, "classes": classes,
                     "features": tuple(FEATURES)}, f, protocol=4)
    acc_train = float((model.predict(X) == y).mean())
    print(f"обучено на {len(X)} сезонах, классов {len(classes)}, "
          f"точность на обучении {acc_train:.4f} (не путать с честной из run())")
    print(f"сохранено: {path}")
    return {"classes": classes, "n": len(X), "acc_train": acc_train}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Определение культуры по кривой (E19)")
    ap.add_argument("--train", action="store_true",
                    help="обучить классификатор и сохранить в models/")
    ap.add_argument("--no-lgbm", action="store_true", help="без сравнения с бустингом")
    args = ap.parse_args(argv)
    if args.train:
        train()
        return 0
    run(use_lgbm=not args.no_lgbm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
