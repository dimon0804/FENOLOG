"""Загрузка данных и представление ряда одного полигона.

Два источника:
    data/private_features.csv — тестовый набор, 78 полигонов, есть is_synthetic_gap
    data/train_dataset.csv    — обучающий набор, 39 полигонов, есть ndvi_zscore и status

Пары «полигон + дата» этих файлов не пересекаются ни разу: организаторы разрезали
одну и ту же посуточную сетку на две части. Поэтому наблюдения из train — это
не «другие данные», а дополнительные точки того же ряда, и подмешивать их в
восстановление законно. Управляется флагом use_train, чтобы вклад можно было
замерить, а не принять на веру.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
TEST_PATH = DATA_DIR / "private_features.csv"
TRAIN_PATH = DATA_DIR / "train_dataset.csv"

# Что известно про контрольную строку в реальном тесте: только эти три поля.
# Всё остальное замаскировано, включая погоду, климатологию, doy и year.
VISIBLE_IN_CONTROL = ("anon_polygon_id", "date", "crop_type", "is_synthetic_gap")

# Признаки, которые надо занулить у скрытых точек локальной валидации,
# иначе валидация окажется честнее реальности и все выводы поедут.
MASKABLE = (
    "s2_ndvi", "s2_evi", "s2_ndwi",
    "landsat_ndvi", "landsat_evi", "landsat_ndwi",
    "modis_ndvi", "modis_evi",
    "era5_temp_c", "era5_precip_mm",
    "year", "primary_ndvi", "doy",
    "ndvi_climatology_mean", "ndvi_climatology_std",
    "ndvi_zscore", "n_reference_years", "status",
)


@dataclass
class PolygonView:
    """Всё, что метод восстановления имеет право видеть по одному полигону.

    frame        — полный посуточный кусок таблицы, у скрытых строк признаки уже стёрты
    known_ord    — дни (date.toordinal) наблюдений, которые методу видны
    known_values — значения primary_ndvi в этих днях
    Разделение на frame и known_* сделано ради скорости: numpy-массивы читаются
    в горячем цикле, а frame нужен только тем методам, которым нужна погода.
    """

    polygon_id: str
    crop_type: str | None
    frame: pd.DataFrame
    known_ord: np.ndarray
    known_values: np.ndarray
    known_source: np.ndarray | None = None   # "test" / "train" для каждого наблюдения
    meta: dict = field(default_factory=dict)

    def window_stats(self, center_ord: int, half_width: int = 30) -> tuple[int, float]:
        """Сколько известных наблюдений в окне вокруг даты и их среднее.

        Нужно как признак для бустинга: чем плотнее окно, тем надёжнее сглаживание.
        """
        mask = np.abs(self.known_ord - center_ord) <= half_width
        n = int(mask.sum())
        return n, (float(self.known_values[mask].mean()) if n else float("nan"))


def load_frame(path: str | Path, kind: str) -> pd.DataFrame:
    """Читает CSV и приводит типы. kind: 'test' или 'train'."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["_ord"] = df["date"].map(pd.Timestamp.toordinal)
    df["_month"] = df["date"].dt.month
    df["_doy"] = df["date"].dt.dayofyear
    df["_year"] = df["date"].dt.year
    df["_source"] = kind
    if "is_synthetic_gap" not in df.columns:
        df["is_synthetic_gap"] = False
    df["is_synthetic_gap"] = df["is_synthetic_gap"].fillna(False).astype(bool)
    return df


def load_all(use_train: bool = True, test_path=TEST_PATH, train_path=TRAIN_PATH) -> pd.DataFrame:
    """Собирает единую таблицу. Строки train добавляются как дополнительные даты."""
    test = load_frame(test_path, "test")
    if not use_train or not Path(train_path).exists():
        return test.sort_values(["anon_polygon_id", "_ord"]).reset_index(drop=True)

    train = load_frame(train_path, "train")
    # Колонки выравниваем по тесту: в train есть ndvi_zscore и status, их
    # оставляем — они пригодятся доменному ядру, но не восстановлению.
    combined = pd.concat([test, train], ignore_index=True, sort=False)
    combined["is_synthetic_gap"] = combined["is_synthetic_gap"].fillna(False).astype(bool)
    return combined.sort_values(["anon_polygon_id", "_ord"]).reset_index(drop=True)


def mask_rows(df: pd.DataFrame, hidden_index: np.ndarray) -> pd.DataFrame:
    """Стирает у выбранных строк всё, кроме id, даты и культуры.

    Ровно то же самое организаторы сделали с контрольными точками, поэтому
    локальная валидация меряет ту же задачу, а не более лёгкую.
    """
    out = df.copy()
    cols = [c for c in MASKABLE if c in out.columns]
    out.loc[hidden_index, cols] = np.nan
    return out


def build_views(df: pd.DataFrame) -> dict[str, PolygonView]:
    """Разбивает таблицу на представления по полигонам."""
    views: dict[str, PolygonView] = {}
    for polygon_id, g in df.groupby("anon_polygon_id", sort=False):
        g = g.sort_values("_ord")
        known = g["primary_ndvi"].notna().to_numpy()
        crop = g["crop_type"].dropna()
        views[polygon_id] = PolygonView(
            polygon_id=str(polygon_id),
            crop_type=str(crop.iloc[0]) if len(crop) else None,
            frame=g,
            known_ord=g.loc[known, "_ord"].to_numpy(dtype=np.int64),
            known_values=g.loc[known, "primary_ndvi"].to_numpy(dtype=float),
            known_source=g.loc[known, "_source"].to_numpy(),
        )
    return views
