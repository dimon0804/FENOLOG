"""E05. LightGBM поверх предсказаний остальных методов.

Гипотеза. Ни один метод восстановления не лучший везде. На разрыве в один-два дня
выигрывает слабое сглаживание, на разрыве в два месяца — климатологический якорь,
а суточная поправка соседних полей помогает только там, где соседей достаточно.
Все эти условия наблюдаемы до предсказания: длина разрыва, плотность окна, разброс
нормы, число соседей в пролёте. Значит поверх методов можно поставить арбитра,
который по обстановке взвесит их сам. Это классический стекинг, и целевая у него
та же, что и метрика, — сам primary_ndvi при функции потерь MSE, без суррогатов.

Два инженерных решения, на которых всё держится.

1. ПРИЗНАКИ — ВСЕ КОЛОНКИ base_preds, а не фиксированный список. Над проектом
   работают несколько человек, реестр методов пополняется на ходу. Надстройка,
   зашитая на конкретные ключи, устарела бы к вечеру; эта усиливается сама.

2. ОБУЧАЮЩАЯ ВЫБОРКА РАСШИРЯЕТСЯ ЗА СЧЁТ train_dataset. Оцениваемых точек всего
   три тысячи — для бустинга со ста шестьюдесятью признаками мало. Поверх строк
   train_dataset строится второй контрольный набор теми же шаблонами (см.
   features.build_extra_points), несколько независимых розыгрышей. Это дало
   больше, чем любой перебор гиперпараметров: 0.0671 без расширения против
   0.0637 с семью репликами при прочих равных.

Утечки нет. Дополнительные точки строятся поверх УЖЕ замаскированной таблицы, а
при обучении на каждом фолде из них берутся только полигоны обучающей части
GroupKFold — поле целиком либо в обучении, либо в контроле.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import features as FT
from src.ml import holdout as H
from src.ml.dataset import PolygonView
from src.ml.registry import BaseMethod, register

MODEL_DIR = Path("models")
# Кэш признаков расширенного набора. Пересобирается сам, когда меняется протокол
# или список базовых методов; файл можно удалить в любой момент.
CACHE_PATH = MODEL_DIR / "_e05_feature_cache.pkl"
FINAL_MODEL_PATH = MODEL_DIR / "e05_lgbm.pkl"

# Сколько независимых розыгрышей расширенного набора строить. Каждый добавляет
# около 3250 точек. Отдача убывает: 0→1 даёт -0.0012, 1→3 ещё -0.0009,
# 3→5 -0.0002, 5→7 ноль. Пять — точка, где кривая ложится на полку, а прогон
# ещё не разорителен: каждая реплика стоит прогона всех базовых методов.
N_REPLICATES = int(os.environ.get("E05_REPLICATES", "5"))

# Подбор гиперпараметров почти ничего не меняет (весь разброс по сетке уложился
# в 0.0006), поэтому взята умеренно ёмкая конфигурация с сильной L2 и ранней
# остановкой. Глубина ограничена явно: точек мало, лес переобучается мгновенно.
PARAMS = dict(
    objective="l2",
    learning_rate=0.02,
    num_leaves=63,
    max_depth=8,
    min_data_in_leaf=40,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=30.0,
    verbose=-1,
    n_jobs=4,
    seed=42,
)
MAX_ROUNDS = 3000
EARLY_STOP = 100


# --------------------------------------------------------------------------- #
# Общий для всех фолдов и вариантов набор признаков
# --------------------------------------------------------------------------- #

class _Bundle:
    """Признаки основных и дополнительных точек, посчитанные один раз на прогон.

    Собирать их заново на каждом из пяти фолдов было бы вчетверо дороже самой
    валидации: тяжёлая часть — не обучение, а прогон восьми десятков базовых
    методов по дополнительным точкам.
    """

    def __init__(self, X, key_index, extra, feats, crop_map):
        self.X = X                  # признаки основных точек, в порядке context["points"]
        self.key_index = key_index  # (polygon_id, ord_day) -> строка X
        self.extra = extra          # список (X2, y2, g2)
        self.feats = feats
        self.crop_map = crop_map    # кодировка культуры, обязана дожить до инференса


_BUNDLES: dict[int, _Bundle] = {}


def _fingerprint(points, base_preds: pd.DataFrame, n_rep: int) -> str:
    """Отпечаток протокола: набор точек, набор базовых методов, число реплик.

    Меняется от любой правки holdout.py (сдвигаются точки и их truth) и от
    появления нового метода в реестре — ровно тогда кэш и обязан протухнуть.
    """
    h = hashlib.sha1()
    h.update(",".join(sorted(base_preds.columns)).encode())
    h.update(f"|{len(points)}|{n_rep}|".encode())
    h.update(np.array([p.truth for p in points], dtype=np.float64).tobytes())
    h.update(np.array([p.ord_day for p in points], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _build_bundle(views, points, context, n_rep: int) -> _Bundle:
    base_preds: pd.DataFrame = context["base_preds"]
    raw = context["raw"]
    masked = context["df"]

    weather = FT.WeatherGroups(raw)
    crops = sorted({v.crop_type for v in views.values() if v.crop_type})
    crop_map = {c: i for i, c in enumerate(crops)}

    fb = FT.FeatureBuilder(views, weather=weather, crop_map=crop_map)
    X = fb.transform(points, base_preds)
    key_index = {(p.polygon_id, int(p.ord_day)): i for i, p in enumerate(points)}
    feats = list(X.columns)

    fp = _fingerprint(points, base_preds, n_rep)
    extra: list[tuple[pd.DataFrame, np.ndarray, np.ndarray]] = []
    cached = _load_cache(fp)
    if cached is not None:
        extra = cached
    elif n_rep > 0:
        templates = H.extract_templates(raw)
        for pts, v2, m2 in FT.build_extra_points(masked, templates, n_replicates=n_rep):
            bp2 = FT.base_preds_for(pts, v2, m2, raw, base_preds.columns)
            X2 = FT.FeatureBuilder(v2, weather=weather, crop_map=crop_map).transform(pts, bp2)
            extra.append((X2[feats].astype(np.float32),
                          np.array([p.truth for p in pts]),
                          np.array([p.polygon_id for p in pts])))
        _save_cache(fp, extra)

    return _Bundle(X, key_index, extra, feats, crop_map)


def _load_cache(fp: str):
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            blob = pickle.load(f)
        return blob["extra"] if blob.get("fingerprint") == fp else None
    except Exception:
        return None


def _save_cache(fp: str, extra) -> None:
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"fingerprint": fp, "extra": extra}, f, protocol=4)
    except Exception:
        pass


def get_bundle(views, points, context, n_rep: int = N_REPLICATES) -> _Bundle:
    """Ключ кэша — сам объект base_preds: он создаётся один раз на прогон
    валидации и копируется по фолдам как есть."""
    key = id(context["base_preds"])
    b = _BUNDLES.get(key)
    if b is None:
        b = _build_bundle(views, points, context, n_rep)
        _BUNDLES[key] = b
    return b


# --------------------------------------------------------------------------- #
# Метод
# --------------------------------------------------------------------------- #

class _LGBMStack(BaseMethod):
    """Бустинг-арбитр над остальными методами.

    anchor — ключ опорного метода. Если задан, модель учится на ПОПРАВКЕ к нему
    (truth − anchor), а предсказание собирается как anchor + поправка. На
    зашумлённых данных такая постановка иногда устойчивее прямой, но здесь она
    проиграла: замер обеих в отчёте exp_e05.md.
    """

    needs_fit = True

    def __init__(self, anchor: str | None = None, use_extra: bool = True,
                 n_replicates: int = N_REPLICATES, params: dict | None = None):
        self.anchor = anchor
        self.use_extra = use_extra
        self.n_replicates = n_replicates if use_extra else 0
        self.params = dict(PARAMS, **(params or {}))
        self.model = None
        self.bundle: _Bundle | None = None
        self.fallback: str | None = None
        # Всё, что нужно предсказанию и переживает обучение: порядок колонок,
        # кодировка культуры, имя опорной колонки. Без них predict_external
        # не воспроизведёт ту же матрицу признаков, на которой училась модель.
        self.feats: list[str] = []
        self.crop_map: dict[str, int] = {}
        self.anchor_col: str | None = None
        self._ext_builder = None

    # -- обучение ----------------------------------------------------------- #

    def fit(self, views: dict[str, PolygonView], points: list, context: dict) -> None:
        import lightgbm as lgb

        bundle = get_bundle(views, points, context, self.n_replicates)
        self.bundle = bundle
        self.feats = list(bundle.feats)
        self.crop_map = dict(bundle.crop_map)
        self._ext_builder = None
        mask = context["train_mask"]
        groups = np.array([p.polygon_id for p in points])
        truth = np.array([p.truth for p in points], dtype=float)

        anchor_col = f"p_{self.anchor}" if self.anchor else None
        if anchor_col and anchor_col not in bundle.feats:
            anchor_col = None
        self.fallback = anchor_col or ("p_mean" if "p_mean" in bundle.feats else None)

        Xs = [bundle.X.loc[mask, bundle.feats]]
        ys = [truth[mask] - (bundle.X[anchor_col].to_numpy()[mask] if anchor_col else 0.0)]
        gs = [groups[mask]]

        # Дополнительные точки берутся только с полигонов обучающей части фолда:
        # иначе поле контрольной части попало бы в обучение через свой же 2015 год.
        if self.use_extra:
            train_pol = set(groups[mask])
            for X2, y2, g2 in bundle.extra:
                m = np.isin(g2, list(train_pol))
                if not m.any():
                    continue
                Xs.append(X2[m])
                ys.append(y2[m] - (X2[anchor_col].to_numpy()[m] if anchor_col else 0.0))
                gs.append(g2[m])

        Xtr = pd.concat(Xs, ignore_index=True)[bundle.feats]
        ytr = np.concatenate(ys)
        gtr = np.concatenate(gs)

        # Ранняя остановка по внутреннему разрезу, тоже по полигонам: случайный
        # разрез по строкам оставил бы соседние дни одного поля по обе стороны
        # и число деревьев вышло бы завышенным.
        upol = np.unique(gtr)
        rs = np.random.RandomState(0)
        rs.shuffle(upol)
        hold = set(upol[: max(1, len(upol) // 5)])
        vm = np.isin(gtr, list(hold))
        cat = ["crop_type"] if "crop_type" in bundle.feats else []
        ds_tr = lgb.Dataset(Xtr[~vm], ytr[~vm], categorical_feature=cat)
        ds_va = lgb.Dataset(Xtr[vm], ytr[vm], reference=ds_tr)
        probe = lgb.train(self.params, ds_tr, num_boost_round=MAX_ROUNDS,
                          valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        best = probe.best_iteration or MAX_ROUNDS

        # Найденным числом деревьев добучаемся на всей обучающей части фолда:
        # отдавать пятую часть полигонов только под остановку расточительно.
        ds_all = lgb.Dataset(Xtr, ytr, categorical_feature=cat)
        self.model = lgb.train(self.params, ds_all, num_boost_round=best)
        self.anchor_col = anchor_col

    # -- предсказание -------------------------------------------------------- #

    def predict_points(self, points: list, views, context: dict) -> np.ndarray:
        """Быстрый путь валидации: признаки берутся из кэша прогона по ключу.

        Точки, которых в кэше нет, уходят в predict_external и считаются честно.
        Раньше они получали медиану опорного метода, и на боевом прогоне туда
        уходили ВСЕ 3112 контрольных точек: их в локальном наборе нет по
        построению. Молчаливая деградация до константы — худший из возможных
        отказов, поэтому теперь непопадание в кэш просто дороже, а не хуже.
        """
        bundle = self.bundle or get_bundle(views, context["points"], context, self.n_replicates)
        rows = np.array([bundle.key_index.get((p.polygon_id, int(p.ord_day)), -1) for p in points])
        out = np.full(len(points), np.nan)
        ok = rows >= 0
        if ok.any():
            Xq = bundle.X.iloc[rows[ok]][bundle.feats]
            pred = self.model.predict(Xq)
            if self.anchor_col:
                pred = pred + Xq[self.anchor_col].to_numpy()
            out[ok] = pred
        if (~ok).any():
            miss = [p for p, m in zip(points, ~ok) if m]
            out[~ok] = self.predict_external(miss, views, context)
        return np.clip(out, 0.0, 1.0)

    def predict_external(self, points: list, views, context: dict) -> np.ndarray:
        """Предсказание на точках, которых не было в контрольном наборе.

        От predict_points отличается тем, что таблица признаков строится здесь
        же по переданным points и views, а не ищется в кэше прогона валидации.
        Реплики расширенного набора не нужны: они существуют только ради
        обучения. Ни контрольный набор, ни truth не читаются — ни один признак
        их не требует, поэтому путь годится для боевого инференса.

        Расстояния до соседей пересчитываются здесь заново по known_ord самих
        views, а поля left_dist/right_dist/month/truth переданных точек
        игнорируются. Так вызывающая сторона не может незаметно передать
        геометрию, посчитанную по другому набору наблюдений, — а это
        единственное место, где такая ошибка не упала бы, а тихо испортила
        предсказание.
        """
        if self.model is None:
            raise RuntimeError("модель не обучена: сначала fit(...) или load_model()")
        if not points:
            return np.empty(0, dtype=float)

        norm = _normalise_points(points, views)
        X = self._external_builder(views, context).transform(norm, context["base_preds"])

        # Порядок и состав колонок берём из обучения. Реестр методов между
        # обучением и инференсом меняться не должен, но если это случилось —
        # лучше громко сказать, чем предсказывать по сдвинутым признакам.
        missing = [c for c in self.feats if c not in X.columns]
        if missing:
            print(f"[e05] ВНИМАНИЕ: при инференсе не собрано {len(missing)} признаков "
                  f"из {len(self.feats)}, первые: {missing[:5]}")
            for c in missing:
                X[c] = np.nan
        Xq = X[self.feats]
        pred = self.model.predict(Xq)
        if self.anchor_col:
            pred = pred + Xq[self.anchor_col].to_numpy()
        return np.clip(pred, 0.0, 1.0)

    def _external_builder(self, views, context) -> FT.FeatureBuilder:
        """Сборщик признаков для боевых точек, с кэшем на экземпляре.

        Погодные группы и суточная поправка соседей считаются по всему набору
        и стоят несколько секунд — пересчитывать их на каждый вызов не нужно.
        Кодировка культуры обязана совпасть с обучением, поэтому crop_map
        берётся сохранённый, а не пересобранный по этим views.
        """
        if self._ext_builder is None:
            weather = FT.WeatherGroups(context.get("raw", context["df"]))
            self._ext_builder = FT.FeatureBuilder(views, weather=weather,
                                                  crop_map=self.crop_map)
        return self._ext_builder

    # -- важность признаков --------------------------------------------------- #

    def importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        return pd.Series(self.model.feature_importance("gain"),
                         index=self.model.feature_name()).sort_values(ascending=False)


@register("e05_lgbm", "LightGBM поверх всех методов, + train", experiment="E05",
          tags=("stack",))
class LGBMStack(_LGBMStack):
    """Рабочая конфигурация: прямая цель, расширенная обучающая выборка."""

    def __init__(self):
        super().__init__(anchor=None, use_extra=True)


# Варианты-ablation держатся под флагом: каждый добавляет полминуты к общему
# прогону валидации, которым пользуются все участники, а нужны они только для
# таблицы сравнения в отчёте.
if os.environ.get("E05_ABLATION"):

    @register("e05_noextra", "LightGBM без расширения выборки", experiment="E05")
    class LGBMNoExtra(_LGBMStack):
        def __init__(self):
            super().__init__(anchor=None, use_extra=False)

    @register("e05_res_sib", "LightGBM, поправка к sibit10, + train", experiment="E05")
    class LGBMResSib(_LGBMStack):
        def __init__(self):
            super().__init__(anchor="sibit10", use_extra=True)

    @register("e05_res_whit", "LightGBM, поправка к whit1000, + train", experiment="E05")
    class LGBMResWhit(_LGBMStack):
        def __init__(self):
            super().__init__(anchor="whit1000", use_extra=True)

    @register("e05_res_sib_noextra", "LightGBM, поправка к sibit10, без расширения",
              experiment="E05")
    class LGBMResSibNoExtra(_LGBMStack):
        def __init__(self):
            super().__init__(anchor="sibit10", use_extra=False)


# --------------------------------------------------------------------------- #
# Инференс сохранённой моделью
# --------------------------------------------------------------------------- #

def _normalise_points(points: list, views: dict[str, PolygonView]) -> list:
    """Пересобирает точки, считая расстояния до соседей по самим views.

    Именно так их считает протокол валидации (holdout._neighbour_distances),
    и именно так их обязан считать инференс: геометрия разрыва — признак, а не
    метаданные, и посчитанная по другому набору наблюдений она сдвинет
    предсказание, ничего не сломав по дороге.

    truth не читается и не заполняется: на боевых точках его нет.
    """
    out = [None] * len(points)
    by_polygon: dict[str, list[int]] = {}
    for i, p in enumerate(points):
        by_polygon.setdefault(p.polygon_id, []).append(i)
    for pid, idx in by_polygon.items():
        ords = np.array([int(points[i].ord_day) for i in idx], dtype=np.int64)
        known = views[pid].known_ord if pid in views else np.array([], dtype=np.int64)
        if len(known):
            left, right = H._neighbour_distances(ords, known)
        else:
            left = right = np.full(len(ords), -1, dtype=np.int64)
        months = pd.to_datetime([pd.Timestamp.fromordinal(int(o)) for o in ords]).month
        for k, i in enumerate(idx):
            out[i] = H.HoldoutPoint(polygon_id=pid, ord_day=int(ords[k]),
                                    truth=float("nan"),
                                    left_dist=int(left[k]), right_dist=int(right[k]),
                                    month=int(months[k]))
    return out


def save_model(method: "_LGBMStack", path: Path = FINAL_MODEL_PATH) -> Path:
    """Кладёт обученную модель на диск в том виде, который читает from_saved.

    Раньше файл `models/e05_lgbm.pkl` не создавался ни одной командой
    репозитория — он появился однажды и дальше жил сам по себе. Для критерия
    воспроизводимости это дыра: артефакт нельзя пересобрать. Теперь можно,
    вызовом `python -m src.cli.batch_infer --retrain --save-model`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "booster": method.model,
        "feats": list(method.feats),
        "crop_map": dict(method.crop_map or {}),
        "anchor": method.anchor,
        "anchor_col": method.anchor_col,
        "params": dict(method.params),
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f, protocol=4)
    return path


def from_saved(path: Path = FINAL_MODEL_PATH) -> "_LGBMStack":
    """Готовый к predict_external метод из models/e05_lgbm.pkl, без обучения."""
    blob = load_model(path)
    m = _LGBMStack(anchor=blob.get("anchor"), use_extra=False)
    m.model = blob["booster"]
    m.feats = list(blob["feats"])
    m.crop_map = dict(blob.get("crop_map") or {})
    m.anchor_col = blob.get("anchor_col")
    return m

def load_model(path: Path = FINAL_MODEL_PATH):
    """Загружает финальную модель, обученную на всех точках сразу.

    Возвращает словарь: booster, feats (порядок колонок), params, crop_map.
    Порядок колонок обязателен — LightGBM сверяет имена, но не порядок, а
    признаки собираются из base_preds, чей порядок зависит от реестра.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
