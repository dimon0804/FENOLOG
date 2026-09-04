"""Точка входа доменного ядра: превращает собранные данные в готовый анализ.

Это единственная функция, которую вызывает API. Слой провайдеров ничего не знает
про то, как считается норма и как ищутся аномалии, а ядро ничего не знает про то,
откуда взялись наблюдения.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.contracts import AnalysisResult, SeriesInput, SeriesPoint
from src.core.anomaly import build_periods
from src.core.climatology import fit_climatology, has_enough_history, zscore
from src.core.restore import restore_on_grid

# Шаг выдачи итогового ряда наружу: посуточно график слишком тяжёлый для фронтенда
OUTPUT_STEP_DAYS = 1


def analyze(inp: SeriesInput, output_step: int = OUTPUT_STEP_DAYS) -> AnalysisResult:
    """Собирает ряд, восстанавливает пропуски, считает норму и находит аномалии."""
    obs = [o for o in inp.observations if o.ndvi is not None and np.isfinite(o.ndvi)]
    if not obs:
        return AnalysisResult(
            polygon_id=inp.polygon_id,
            meta={"n_obs": 0, "error": "нет ни одного пригодного наблюдения"},
        )

    df = pd.DataFrame(
        [{"date": pd.Timestamp(o.date), "ndvi": float(o.ndvi), "source": o.source} for o in obs]
    ).sort_values("date")
    # Если на одну дату пришло несколько сенсоров, берём медиану
    df = df.groupby("date", as_index=False).agg(ndvi=("ndvi", "median"), source=("source", "first"))

    t_days = df["date"].map(pd.Timestamp.toordinal).to_numpy()
    grid, restored = restore_on_grid(t_days, df["ndvi"].to_numpy())
    grid_dates = [date.fromordinal(int(d)) for d in grid]

    # Климатическая норма считается по восстановленному ряду: он покрывает всю сетку
    clim_source = pd.Series(restored, index=pd.to_datetime(grid_dates))
    enough = has_enough_history(pd.Series(pd.to_datetime(grid_dates)))
    clim = fit_climatology(clim_source.index.to_series(), clim_source) if enough else None

    doys = np.array([d.timetuple().tm_yday for d in grid_dates])
    if clim is not None:
        clim_mean = clim["mean"].reindex(doys).to_numpy()
        clim_std = clim["std"].reindex(doys).to_numpy()
        z = zscore(restored, clim_mean, clim_std)
        clim_years = int(np.nanmax(clim["n_years"].to_numpy())) if len(clim) else 0
    else:
        clim_mean = np.full(len(grid), np.nan)
        clim_std = np.full(len(grid), np.nan)
        z = np.full(len(grid), np.nan)
        clim_years = 0

    observed_map = dict(zip(df["date"].dt.date, df["ndvi"]))
    source_map = dict(zip(df["date"].dt.date, df["source"]))

    series: list[SeriesPoint] = []
    for i in range(0, len(grid_dates), output_step):
        d = grid_dates[i]
        has_obs = d in observed_map
        series.append(
            SeriesPoint(
                date=d,
                observed=float(observed_map[d]) if has_obs else None,
                restored=round(float(restored[i]), 4),
                climatology_mean=None if np.isnan(clim_mean[i]) else round(float(clim_mean[i]), 4),
                climatology_std=None if np.isnan(clim_std[i]) else round(float(clim_std[i]), 4),
                zscore=None if np.isnan(z[i]) else round(float(z[i]), 2),
                is_restored=not has_obs,
                source=source_map.get(d),
            )
        )

    anomalies = build_periods(grid_dates, z, inp.weather)

    return AnalysisResult(
        polygon_id=inp.polygon_id,
        series=series,
        anomalies=anomalies,
        meta={
            "n_obs": len(df),
            "sources": sorted(set(df["source"].dropna())),
            "first_date": str(grid_dates[0]),
            "last_date": str(grid_dates[-1]),
            "climatology_years": clim_years,
            "has_climatology": clim is not None,
            "crop_type": inp.crop_type,
        },
    )
