"""Климатическая норма NDVI по типу культуры.

Зачем нужна. Собственная норма полигона считается по его же истории и всегда
точнее, но существует она не всегда: полю, у которого есть только текущий сезон,
норму по себе построить не из чего, а без нормы нет ни z-оценки, ни статуса, ни
разговора об угнетении — продукт на таком поле просто молчит. Норма по культуре
закрывает эту дыру грубым, но осмысленным резервом: озимая пшеница везде растёт
примерно по одному календарю.

Что важно понимать про её точность. Разброс полей внутри одной культуры велик:
медианная корреляция кривых внутри культуры 0,949 против 0,855 между культурами,
а RMSE между двумя случайными полями озимой пшеницы 0,108 — больше, чем ошибка
самого восстановления. Поэтому норма по культуре — резерв на безрыбье, а не
замена собственной климатологии. Это измерено, а не предположено, числа и
методика в reports/exp_e04.md.

Как усредняются кривые. Поля одной культуры различаются уровнем NDVI (сезонный
уровень от 0,27 до 0,62), поэтому сырое усреднение рискует размазать форму.
Реализованы три схемы, выбор между ними сделан по числу (leave-one-polygon-out):
    raw      — усреднить сырые кривые;
    centered — вычесть у каждого поля его собственный сезонный уровень (по умолчанию);
    scaled   — дополнительно поделить на собственный размах.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Окно по дню года, ±8 дней. Ровно то окно, которым организаторы считали
# ndvi_climatology_mean (восстановлено по обучающему набору с MAE 0,0016),
# поэтому норма по культуре и норма по полигону живут в одной шкале сглаживания
# и их можно смешивать, не пересчитывая одну в другую.
DOY_WINDOW = 8
# Минимум сезонов, начиная с которого кривая поля годится в опору для нормы культуры
MIN_YEARS = 3
# Минимум наблюдений в окне, иначе точка кривой считается неопределённой
MIN_OBS = 3
# Границы вегетационного сезона: за его пределами наблюдений в наборе нет вообще
SEASON_DOY = (91, 305)
# Пол для разброса: нулевой разброс превращает z-оценку в бесконечность
MIN_SPREAD = 0.02
# Сколько дней сезона поле обязано покрывать, чтобы войти в норму культуры
MIN_SEASON_DAYS = 30

_DOYS = np.arange(1, 367)


def _circular_window(doy_obs: np.ndarray, target: int, window: int) -> np.ndarray:
    """Маска окна по дню года с замыканием через Новый год."""
    diff = np.abs(doy_obs - target)
    diff = np.minimum(diff, 366 - diff)
    return diff <= window


def polygon_doy_curve(
    doy: np.ndarray, values: np.ndarray, window: int = DOY_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """Кривая одного поля по дню года: (среднее, разброс) на сетке doy 1..366.

    Leave-one-out по годам здесь намеренно нет. LOO нужен там, где норму
    сравнивают с наблюдением того же сезона; здесь строится «портрет поля
    целиком», а защита от подглядывания обеспечивается тем, что в норму своей
    культуры поле не входит вовсе — проверка идёт leave-one-polygon-out.
    """
    mean = np.full(366, np.nan)
    std = np.full(366, np.nan)
    for i, d in enumerate(_DOYS):
        m = _circular_window(doy, d, window)
        if int(m.sum()) < MIN_OBS:
            continue
        sub = values[m]
        mean[i] = sub.mean()
        std[i] = sub.std(ddof=1)
    return mean, std


class CropClimatology:
    """Норма NDVI по типу культуры: среднее и разброс на каждый день года.

    Использование:
        clim = CropClimatology().fit(df)
        if clim.has(crop):
            mu, sigma = clim.norm(crop, doy_array)
    """

    def __init__(self, scheme: str = "centered", window: int = DOY_WINDOW):
        if scheme not in ("raw", "centered", "scaled"):
            raise ValueError(f"неизвестная схема усреднения: {scheme!r}")
        self.scheme = scheme
        self.window = window
        self._mean: dict[str, np.ndarray] = {}     # культура -> 366 значений
        self._spread: dict[str, np.ndarray] = {}
        self._n_fields: dict[str, int] = {}        # сколько полей вошло в норму

    # ---------------------------------------------------------------- обучение

    def fit(self, df: pd.DataFrame) -> "CropClimatology":
        """Строит норму по каждой культуре, используя только поля с историей.

        df должен содержать anon_polygon_id, crop_type, primary_ndvi и либо date,
        либо готовые _doy/_year. Строки без primary_ndvi игнорируются — а значит,
        скрытые протоколом валидации точки в норму не попадают автоматически,
        отдельной защиты от утечки не требуется.
        """
        work = df.loc[df["primary_ndvi"].notna(), :].copy()
        if "_doy" not in work.columns or "_year" not in work.columns:
            dates = pd.to_datetime(work["date"])
            work["_doy"] = dates.dt.dayofyear
            work["_year"] = dates.dt.year

        season = np.zeros(366, dtype=bool)
        season[SEASON_DOY[0] - 1 : SEASON_DOY[1]] = True

        # Шаг 1. Портрет каждого поля, у которого хватает истории.
        curves: dict[str, tuple] = {}
        crop_of: dict[str, str] = {}
        for pid, g in work.groupby("anon_polygon_id", sort=False):
            if g["_year"].nunique() < MIN_YEARS:
                continue
            crop = g["crop_type"].dropna()
            if crop.empty:
                continue
            mean, std = polygon_doy_curve(
                g["_doy"].to_numpy(dtype=int),
                g["primary_ndvi"].to_numpy(dtype=float),
                self.window,
            )
            filled = np.isfinite(mean) & season
            if int(filled.sum()) < MIN_SEASON_DAYS:
                continue
            level = float(np.nanmean(mean[filled]))
            amp = float(np.nanstd(mean[filled]))
            curves[str(pid)] = (mean, std, level, max(amp, 1e-6))
            crop_of[str(pid)] = str(crop.iloc[0])

        # Шаг 2. Усреднение портретов внутри культуры выбранной схемой.
        by_crop: dict[str, list[str]] = {}
        for pid, crop in crop_of.items():
            by_crop.setdefault(crop, []).append(pid)

        for crop, pids in by_crop.items():
            mean, spread = self._average(curves, pids)
            self._mean[crop] = mean
            self._spread[crop] = spread
            self._n_fields[crop] = len(pids)
        return self

    def _average(self, curves: dict, pids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Усредняет кривые полей одной культуры выбранной схемой.

        Возвращает норму в абсолютной шкале NDVI. Для восстановления по якорю
        важна только форма: постоянный сдвиг нормы сокращается при вычитании и
        обратном прибавлении. Но для z-оценки нужен абсолютный уровень, поэтому
        уровень возвращается — средний по полям культуры.
        """
        M = np.vstack([curves[p][0] for p in pids])            # кривые полей
        S = np.vstack([curves[p][1] for p in pids])            # внутриполевой разброс
        levels = np.array([curves[p][2] for p in pids])[:, None]
        amps = np.array([curves[p][3] for p in pids])[:, None]

        # Зимние дни года пусты у всех полей — nanmean по пустому срезу законно
        # возвращает NaN, предупреждение о нём только зашумляет вывод валидации.
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.scheme == "raw":
                mean = np.nanmean(M, axis=0)
            elif self.scheme == "centered":
                mean = np.nanmean(M - levels, axis=0) + float(levels.mean())
            else:  # scaled
                mean = np.nanmean((M - levels) / amps, axis=0) * float(amps.mean()) + float(levels.mean())

            # Разброс нормы складывается из двух независимых источников:
            # межгодовой изменчивости внутри поля и расхождения самих полей.
            within = np.nanmean(S ** 2, axis=0)
            between = np.nanvar(M - levels, axis=0)

        spread = np.sqrt(np.nan_to_num(within, nan=0.0) + np.nan_to_num(between, nan=0.0))
        spread = np.where(np.isfinite(mean), np.clip(spread, MIN_SPREAD, None), np.nan)
        return mean, spread

    # ------------------------------------------------------------- применение

    def has(self, crop_type: str | None) -> bool:
        """Есть ли норма для такой культуры."""
        return crop_type is not None and str(crop_type) in self._mean

    def norm(self, crop_type: str | None, doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает (среднее, разброс) нормы на указанные дни года.

        NaN, если культуры нет в норме — вызывающий код обязан это проверять
        либо через has(), либо по NaN в ответе.
        """
        doy = np.atleast_1d(np.asarray(doy))
        if not self.has(crop_type):
            nan = np.full(doy.shape, np.nan, dtype=float)
            return nan, nan.copy()
        idx = np.clip(doy.astype(int), 1, 366) - 1
        return self._mean[str(crop_type)][idx], self._spread[str(crop_type)][idx]

    @property
    def crops(self) -> list[str]:
        return sorted(self._mean)

    def n_fields(self, crop_type: str) -> int:
        """Сколько полей стоит за нормой культуры — мера доверия к ней."""
        return self._n_fields.get(str(crop_type), 0)

    # ------------------------------------------------------------ сериализация

    def save(self, path) -> None:
        """Сохраняет норму в JSON: файл мелкий, читается глазами и версионируется."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scheme": self.scheme,
            "window": self.window,
            "crops": {
                crop: {
                    "mean": [None if not np.isfinite(v) else round(float(v), 6) for v in self._mean[crop]],
                    "spread": [None if not np.isfinite(v) else round(float(v), 6) for v in self._spread[crop]],
                    "n_fields": self._n_fields[crop],
                }
                for crop in self._mean
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CropClimatology":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(scheme=payload.get("scheme", "centered"), window=payload.get("window", DOY_WINDOW))
        for crop, blob in payload["crops"].items():
            obj._mean[crop] = np.array([np.nan if v is None else v for v in blob["mean"]], dtype=float)
            obj._spread[crop] = np.array([np.nan if v is None else v for v in blob["spread"]], dtype=float)
            obj._n_fields[crop] = int(blob["n_fields"])
        return obj
