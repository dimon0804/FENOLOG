"""Графики для исследовательского отчёта и защиты.

    python notebooks/figures.py

Шесть картинок, каждая отвечает на один вопрос и читается без пояснений.
Складываются в reports/figures/, подписи к ним — в reports/figures/README.md.

    fig1_ndvi_restoration.png   как выглядит восстановленный ряд
    fig2_daily_correction.png   почему работает суточная поправка (главная идея)
    fig3_experiment_progress.png путь от baseline организаторов к финальной конфигурации
    fig4_rmse_by_gap.png        где именно выигрыш — разрез по длине разрыва
    fig5_holdout_geometry.png   контрольный набор геометрически совпадает с реальным
    fig6_anomaly_case.png       найденный период угнетения с версией причины

Откуда берутся числа.
  1, 2, 6 — считаются здесь же по данным: реестр методов (src/ml/registry.py),
           остатки соседей (src/ml/m_e06_sibling.py), доменное ядро (src/core/analyze.py).
  3, 5    — перенесены из reports/experiments.md как есть, не пересчитываются.
  4       — разрез по бинам от команды
           `python -m src.ml.validate --no-save --only mean2 whit1000 final`
           (зерно 42, тот же протокол, что и весь журнал). В журнале этой строки нет:
           опубликованные там разрезы сняты до починки протокола и с финальной
           конфигурацией несопоставимы.

Оформление: всё по-русски, DejaVu Sans (кириллица есть по умолчанию), 150 dpi,
ширина около 10 дюймов, спокойная палитра, различимая в чёрно-белой печати —
серии разводятся не только цветом, но и формой маркера, штриховкой и типом линии.

Скрипт устойчив к отсутствию данных: каждая картинка строится независимо, падение
одной не мешает остальным, отсутствующий полигон приводит к пропуску, а не к ошибке.
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows: кириллица в консоли
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Запуск как `python notebooks/figures.py` — корень проекта в путь импорта
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports" / "figures"

# --------------------------------------------------------------------------
# Оформление
# --------------------------------------------------------------------------

# Палитра намеренно скучная: две ступени синего, серый фон и один тёплый акцент.
# В чёрно-белой печати они превращаются в четыре разные плотности серого, поэтому
# серии дополнительно разводятся маркером, штриховкой и типом линии.
INK = "#16202b"        # почти чёрный — основной текст и главная линия
DARK = "#22405c"       # тёмно-синий — «наш» метод
MID = "#6d92b4"        # средний синий — промежуточные конфигурации
PALE = "#c3ced8"       # светло-серый — фон, вспомогательные серии
ACCENT = "#a84b23"     # тёмная охра — контрольные точки, найденные аномалии
GREY = "#7d8892"       # серый — норма, вспомогательные подписи

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9aa4ad",
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _save(fig, name: str) -> Path:
    """Сохраняет картинку и печатает путь. Одна картинка — один файл."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  сохранено: {path.relative_to(ROOT)}")
    return path


def _value_grid(ax, axis: str = "y") -> None:
    """Бледная сетка ПОД данными — только там, где надо считывать величину."""
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#e2e6ea", linewidth=0.8)


def _mask_far(grid: np.ndarray, known_ord: np.ndarray, values: np.ndarray,
              max_gap: int = 30) -> np.ndarray:
    """Разрывает кривую там, где до ближайшего наблюдения дальше max_gap дней.

    Без этого сглаживание рисует плавную дугу через зимний перерыв, когда съёмок
    нет вовсе, и график обещает знание, которого у метода нет.
    """
    if len(known_ord) == 0:
        return np.full(len(grid), np.nan)
    dist = np.abs(grid[:, None] - known_ord[None, :]).min(axis=1)
    out = values.astype(float).copy()
    out[dist > max_gap] = np.nan
    return out


def _dates(ords: np.ndarray) -> np.ndarray:
    return np.array([pd.Timestamp.fromordinal(int(o)) for o in ords])


def _ru(value: str | float, digits: int = 4) -> str:
    """Число по-русски: десятичная запятая и настоящий знак минус."""
    text = value if isinstance(value, str) else f"{value:.{digits}f}"
    return text.replace(".", ",").replace("-", "−")


def _comma_axis(ax, axis: str = "y", digits: int = 2) -> None:
    """Десятичная запятая на делениях оси — подписи тоже оцениваются."""
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _pos: _ru(v, digits))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


RU_MONTHS = ["янв", "фев", "мар", "апр", "май", "июн",
             "июл", "авг", "сен", "окт", "ноя", "дек"]


def _ru_month_axis(ax) -> None:
    """Месяцы по оси дат подписываются по-русски: локаль matplotlib тут не помощник."""
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _pos: RU_MONTHS[mdates.num2date(v).month - 1]
        )
    )


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    """Русское склонение числительного: 1 поле, 2 поля, 5 полей."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


# --------------------------------------------------------------------------
# График 1. Ряд NDVI с восстановленными разрывами
# --------------------------------------------------------------------------

@dataclass
class _Target:
    """Минимальная точка для методов реестра: им нужны только эти четыре поля."""

    polygon_id: str
    ord_day: int
    left_dist: int
    right_dist: int


def _neighbour_dists(grid: np.ndarray, known_ord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Расстояния до ближайшего известного значения слева и справа."""
    if len(known_ord) == 0:
        return np.full(len(grid), 999), np.full(len(grid), 999)
    pos = np.searchsorted(known_ord, grid, side="left")
    left = np.where(pos > 0, grid - known_ord[np.maximum(pos - 1, 0)], 999)
    right = np.where(pos < len(known_ord), known_ord[np.minimum(pos, len(known_ord) - 1)] - grid, 999)
    return left.astype(int), right.astype(int)


def _norm_by_doy(frame: pd.DataFrame) -> np.ndarray | None:
    """Климатическая норма полигона, свёрнутая по дню года (индекс 1..366).

    Колонка ndvi_climatology_mean у контрольных строк замаскирована, поэтому норма
    собирается по всем годам сразу и потом читается на нужные дни года.
    """
    if "ndvi_climatology_mean" not in frame.columns:
        return None
    sub = frame[["_doy", "ndvi_climatology_mean"]].dropna()
    if sub.empty:
        return None
    by_doy = sub.groupby("_doy")["ndvi_climatology_mean"].mean()
    curve = pd.Series(np.nan, index=range(1, 367), dtype=float)
    curve.loc[by_doy.index] = by_doy.to_numpy()
    return curve.interpolate(limit_direction="both").to_numpy()


def _pick_season(views) -> tuple[str, int] | None:
    """Выбирает поле и сезон для главной картинки — по данным, а не вписанным в код.

    Нужны три вещи сразу: длинная история (чтобы норма была осмысленной), много
    контрольных точек (иначе показывать нечего) и хотя бы один длинный разрыв
    (иначе не видно, ради чего всё затевалось). Отсюда счёт: число контрольных
    точек плюс половина самого длинного разрыва сезона.
    """
    best, best_score = None, -1.0
    for pid, view in views.items():
        f = view.frame
        if f.loc[f["primary_ndvi"].notna(), "_year"].nunique() < 10:
            continue
        for year, gaps in f[f["is_synthetic_gap"]].groupby("_year"):
            obs = f[(f["_year"] == year) & f["primary_ndvi"].notna()]
            if len(obs) < 45:
                continue
            ords = np.sort(obs["_ord"].to_numpy())
            max_gap = int(np.diff(ords).max()) if len(ords) > 1 else 0
            score = len(gaps) + max_gap / 2.0
            if score > best_score:
                best, best_score = (pid, int(year)), score
    return best


def figure_1_restoration(views, df) -> None:
    """Главная картинка проекта: что метод делает с рядом одного поля."""
    from src.core.restore import restore_on_grid
    from src.ml.m_e06_sibling import cleaned_series
    from src.ml.registry import REGISTRY

    picked = _pick_season(views)
    if picked is None:
        print("  график 1 пропущен: нет полигона с длинной историей и контрольными точками")
        return
    best, season_year = picked

    view = views[best]
    frame = view.frame

    # Кривая — сглаживание Уиттекера по очищенному ряду: ровно то, на чём стоит
    # финальная конфигурация (λ = 500 после снятия общей суточной помехи).
    clean, _, _ = cleaned_series(views, lam=1000.0, corr_power=3.0)
    known_ord, known_clean = clean.get(best, (view.known_ord, view.known_values))
    grid, curve = restore_on_grid(known_ord, known_clean, lam=500.0, mix=1.0)
    # Метод и сам подрезает выдачу к физическому диапазону NDVI, кривая рисуется так же
    curve = np.clip(curve, 0.0, 1.0)

    norm_curve = _norm_by_doy(frame)
    doy_of = {int(o): int(d) for o, d in zip(frame["_ord"], frame["_doy"])}

    fig, (ax_top, ax) = plt.subplots(
        2, 1, figsize=(10.0, 7.4), height_ratios=[1.0, 1.55], constrained_layout=True
    )

    # --- верхняя панель: вся история, чтобы был виден масштаб данных ---
    line_all = _mask_far(grid, known_ord, curve)
    ax_top.plot(_dates(grid), line_all, color=DARK, linewidth=1.0)
    ax_top.plot(_dates(view.known_ord), view.known_values, ".", color=PALE, markersize=2.0, zorder=1)
    lo = pd.Timestamp(f"{season_year}-03-15")
    hi = pd.Timestamp(f"{season_year}-11-15")
    ax_top.axvspan(lo, hi, color=ACCENT, alpha=0.13, zorder=0)
    ax_top.annotate(
        f"сезон {season_year} — внизу крупно",
        xy=(lo, 0.94), xycoords=("data", "axes fraction"),
        xytext=(-6, 0), textcoords="offset points",
        ha="right", va="center", fontsize=11, color=ACCENT,
    )
    n_years = frame.loc[frame["primary_ndvi"].notna(), "_year"].nunique()
    ax_top.set_ylabel("NDVI")
    ax_top.set_title(
        f"Восстановление ряда NDVI: {best}, {view.crop_type or 'культура не указана'}, "
        f"{n_years} сезонов наблюдений"
    )
    ax_top.set_ylim(0, 1.02)

    # --- нижняя панель: один сезон крупно ---
    lo_o, hi_o = int(lo.toordinal()), int(hi.toordinal())
    sel = (grid >= lo_o) & (grid <= hi_o)
    season_grid = grid[sel]
    ax.plot(_dates(season_grid), _mask_far(season_grid, known_ord, curve[sel]),
            color=DARK, linewidth=2.2, zorder=3, label="восстановленная кривая")

    obs = frame[frame["primary_ndvi"].notna() & (frame["_ord"] >= lo_o) & (frame["_ord"] <= hi_o)]
    ax.plot(obs["date"], obs["primary_ndvi"], "o", markersize=5.5,
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.1,
            linestyle="none", zorder=4, label="спутниковые наблюдения")

    # Контрольные точки: значение замаскировано организаторами, метод его достраивает
    gaps = frame[frame["is_synthetic_gap"] & (frame["_ord"] >= lo_o) & (frame["_ord"] <= hi_o)]
    if len(gaps):
        ords = gaps["_ord"].to_numpy(dtype=np.int64)
        left, right = _neighbour_dists(ords, view.known_ord)
        targets = [_Target(best, int(o), int(l), int(r)) for o, l, r in zip(ords, left, right)]
        pred = REGISTRY["final"].factory().predict_points(targets, views, {})
        ax.plot(gaps["date"], pred, "D", markersize=7.5, color=ACCENT,
                markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=5,
                label=f"контрольные точки, значение скрыто ({len(gaps)} шт.)")

    if norm_curve is not None:
        doys = np.array([doy_of.get(int(o), 1) for o in season_grid])
        ax.plot(_dates(season_grid), norm_curve[np.clip(doys, 1, 366) - 1],
                linestyle=(0, (5, 3)), color=GREY, linewidth=1.8, zorder=2,
                label="климатическая норма поля")

    ax.set_ylabel("NDVI")
    ax.set_xlabel(f"дата, сезон {season_year}")
    # Верх шкалы — по данным плюс место под легенду: пустая половина графика
    # съедает разрешение там, где как раз и надо разглядеть отдельные точки.
    top = float(np.nanmax(np.concatenate([
        obs["primary_ndvi"].to_numpy(dtype=float),
        curve[sel][np.isfinite(curve[sel])],
    ])))
    ax.set_ylim(0, min(1.0, top * 1.42))
    ax.legend(loc="upper right", ncol=1)
    _ru_month_axis(ax)
    _comma_axis(ax, "y", 1)
    _comma_axis(ax_top, "y", 1)
    ax.set_title(f"Сезон {season_year} крупно: наблюдения, восстановленная кривая и контрольные точки",
                 fontsize=12, pad=8)

    _save(fig, "fig1_ndvi_restoration.png")


# --------------------------------------------------------------------------
# График 2. Почему работает суточная поправка
# --------------------------------------------------------------------------

def figure_2_daily_correction(views) -> None:
    """Ключевая картинка: шум наблюдения общий для полей, снятых одним пролётом."""
    from src.ml.m_e06_sibling import daily_correction, residual_table

    table, _ = residual_table(views, lam=1000.0)
    if table.empty:
        print("  график 2 пропущен: таблица остатков пуста")
        return
    arr = table.to_numpy()
    seen = np.isfinite(arr)

    # День для левой панели выбирается по данным: наибольшее по модулю среднее
    # остатков среди дней, когда снималось хотя бы 45 полей из 78.
    n_fields = seen.sum(axis=1)
    mean_res = np.divide(np.where(seen, arr, 0.0).sum(axis=1), np.maximum(n_fields, 1))
    ok = n_fields >= 45
    if not ok.any():
        ok = n_fields >= max(5, int(n_fields.max() * 0.6))
    row = int(np.flatnonzero(ok)[np.argmax(np.abs(mean_res[ok]))])
    day = pd.Timestamp.fromordinal(int(table.index[row]))
    values = np.sort(arr[row][seen[row]])
    shift = float(mean_res[row])
    same_side = int((np.sign(values) == np.sign(shift)).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.8), width_ratios=[1.0, 1.0],
                                   constrained_layout=True)

    # --- левая панель: один день, все поля ---
    x = np.arange(len(values))
    ax1.vlines(x, 0, values, color=PALE, linewidth=1.4, zorder=1)
    ax1.plot(x, values, "o", markersize=5.5, color=DARK, markeredgecolor="white",
             markeredgewidth=0.7, linestyle="none", zorder=3)
    ax1.axhline(0.0, color=INK, linewidth=1.2, zorder=2)
    ax1.axhline(shift, color=ACCENT, linewidth=2.0, linestyle=(0, (5, 3)), zorder=4)
    ax1.annotate(
        _ru(f"общая суточная поправка {shift:+.3f}"),
        xy=(0, shift), xytext=(0, 15), textcoords="offset points",
        ha="left", va="bottom", color=ACCENT, fontsize=11,
    )
    ax1.set_xlabel(f"поля, снятые {day.strftime('%d.%m.%Y')} ({len(values)} шт.)")
    ax1.set_ylabel("остаток от своей сглаженной кривой")
    ax1.set_title(
        f"Один день: {same_side} {_plural(same_side, ('поле', 'поля', 'полей'))} "
        f"из {len(values)}\nошибаются в одну сторону",
        fontsize=12, pad=8,
    )
    _value_grid(ax1)
    _comma_axis(ax1, "y", 1)
    ax1.set_xticks([])

    # --- правая панель: распределение остатков до и после снятия поправки ---
    # Равные веса соседей — базовая версия E06: именно её остаточный разброс
    # (0,056) назван в отчёте новой оценкой физического потолка.
    days, corr = daily_correction(views, lam=1000.0, corr_power=0.0)
    c_flat = pd.DataFrame(corr, index=days).reindex(index=table.index,
                                                    columns=table.columns).to_numpy()
    days_w, corr_w = daily_correction(views, lam=1000.0, corr_power=3.0)
    c_w = pd.DataFrame(corr_w, index=days_w).reindex(index=table.index,
                                                     columns=table.columns).to_numpy()

    before = arr[seen]
    after = (arr - c_flat)[seen]
    after_w = (arr - c_w)[seen]
    sd_b, sd_a, sd_w = float(np.std(before)), float(np.std(after)), float(np.std(after_w))

    bins = np.linspace(-0.28, 0.28, 65)
    ax2.hist(before, bins=bins, density=True, color=PALE, edgecolor=GREY, linewidth=0.8,
             label=_ru(f"до поправки, σ = {sd_b:.4f}"))
    ax2.hist(after, bins=bins, density=True, histtype="step", color=DARK, linewidth=2.2,
             label=_ru(f"после поправки, σ = {sd_a:.4f}"))
    ax2.axvline(0.0, color=INK, linewidth=1.0)
    ax2.set_xlabel("остаток NDVI")
    ax2.set_ylabel("плотность")
    ax2.set_ylim(0, 14.5)
    ax2.set_title("Все даты: снятие общей помехи\nсужает распределение остатков", fontsize=12, pad=8)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99))
    ax2.annotate(
        _ru(f"со взвешиванием соседей\nпо корреляции: σ = {sd_w:.4f}"),
        xy=(0.99, 0.78), xycoords="axes fraction", ha="right", va="top",
        fontsize=10.5, color=GREY,
    )
    _value_grid(ax2)
    _comma_axis(ax2, "x", 1)

    fig.suptitle("Суточная помеха общая для всех полей, поэтому её можно вычесть",
                 fontsize=14, fontweight="bold")
    _save(fig, "fig2_daily_correction.png")


# --------------------------------------------------------------------------
# График 3. Прогресс по экспериментам
# --------------------------------------------------------------------------

# Числа перенесены из reports/experiments.md (итоговая таблица E08 и сводка
# в CLAUDE.md). Не пересчитываются: единственный источник истины — журнал.
PROGRESS = [
    # (подпись, RMSE, отклонён ли эксперимент)
    #
    # Числа — по действующему протоколу валидации. Раньше первые два брались из
    # чернового протокола (0,0915 и 0,0858), где геометрия скрытых точек не
    # совпадала с настоящей. Смешивать их с остальными нельзя: на графике это
    # давало завышенный прирост.
    ("Baseline организаторов:\nсреднее двух соседей", 0.0888, False),
    ("Первая конфигурация проекта:\nсмесь 50/50", 0.0834, False),
    ("E01. Уиттекер λ = 1000", 0.0794, False),
    ("E03. Побинный подбор λ", 0.0793, True),
    ("E06. Суточная поправка\nпо соседним полям", 0.0694, False),
    ("E04. Климатологический якорь", 0.0657, False),
    ("E07. Регрессия по соседям", 0.0655, True),
    ("E02b. Очистка ряда", 0.0651, False),
    ("E08. Сборка без обучения", 0.0642, False),
    ("E05. Бустинг над методами\n(финальная конфигурация)", 0.0596, False),
]
# Оценка остаточного шума наблюдения после снятия общей суточной составляющей
NOISE_FLOOR = 0.055


def figure_3_progress() -> None:
    """Путь от baseline организаторов к финальной конфигурации, включая тупики."""
    labels = [row[0] for row in PROGRESS]
    values = [row[1] for row in PROGRESS]
    rejected = [row[2] for row in PROGRESS]

    fig, ax = plt.subplots(figsize=(10.0, 5.8), constrained_layout=True)
    y = np.arange(len(values))[::-1]  # сверху хуже, снизу лучше — взгляд идёт вниз

    for yi, val, label, rej in zip(y, values, labels, rejected):
        is_final = val == min(values)
        color = DARK if is_final else (MID if not rej else "white")
        ax.barh(yi, val, height=0.66, color=color,
                edgecolor=INK if rej else "none",
                linewidth=1.1 if rej else 0,
                hatch="///" if rej else None, zorder=3)
        ax.text(val + 0.0012, yi, _ru(val),
                va="center", ha="left", fontsize=11,
                fontweight="bold" if is_final else "normal",
                color=INK, zorder=4)

    ax.axvline(NOISE_FLOOR, color=ACCENT, linewidth=2.0, linestyle=(0, (5, 3)), zorder=5)
    # Подпись вынесена выше верхнего столбца: поверх данных она бы их закрыла
    ax.set_ylim(-0.65, len(values) - 1 + 1.05)
    ax.annotate(
        "оценка остаточного шума наблюдения ≈ 0,055",
        xy=(NOISE_FLOOR, len(values) - 1 + 0.6), xytext=(5, 0), textcoords="offset points",
        ha="left", va="center", color=ACCENT, fontsize=11,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("RMSE на локальной валидации, 2 999 точек, зерно 42")
    ax.set_xlim(0.045, 0.099)
    # Прирост считается из самих данных: раньше в заголовке стояло «−30 %»
    # текстом, и после пересчёта чисел он остался неверным.
    gain = (values[0] - min(values)) / values[0] * 100
    ax.set_title(
        f"Что дал каждый эксперимент: −{gain:.0f} % RMSE к baseline организаторов")
    _value_grid(ax, axis="x")
    _comma_axis(ax, "x", 2)

    ax.legend(
        handles=[
            Patch(facecolor=MID, label="принято в работу"),
            Patch(facecolor="white", edgecolor=INK, hatch="///", label="эксперимент отклонён"),
            Patch(facecolor=DARK, label="финальная конфигурация"),
        ],
        loc="lower right", ncol=1,
    )
    _save(fig, "fig3_experiment_progress.png")


# --------------------------------------------------------------------------
# График 4. RMSE по длине разрыва
# --------------------------------------------------------------------------

GAP_LABELS = ["1–2", "3–4", "5–7", "8–13", "14–29", "30–61", "62–181"]
# Разрез снят командой
#   python -m src.ml.validate --no-save --only mean2 whit1000 final
# на зерне 42 — тот же протокол, что и весь журнал экспериментов.
GAP_RMSE = {
    "Baseline: среднее двух соседей": [0.0740, 0.0759, 0.0839, 0.0894, 0.0975, 0.1004, 0.1442],
    "Уиттекер λ = 1000": [0.0611, 0.0716, 0.0757, 0.0768, 0.0883, 0.0855, 0.1274],
    "Финальная конфигурация": [0.0471, 0.0601, 0.0631, 0.0592, 0.0757, 0.0748, 0.0894],
}


def figure_4_gap_breakdown() -> None:
    """Где именно выигрыш: на всех длинах разрыва, максимум — на самых длинных."""
    fig, ax = plt.subplots(figsize=(10.0, 5.2), constrained_layout=True)
    x = np.arange(len(GAP_LABELS))
    width = 0.26
    styles = [
        (PALE, GREY, "..", None),
        (MID, "none", None, None),
        (DARK, "none", None, None),
    ]
    for i, ((name, vals), (face, edge, hatch, _)) in enumerate(zip(GAP_RMSE.items(), styles)):
        ax.bar(x + (i - 1) * width, vals, width=width, label=name,
               color=face, edgecolor=edge if edge != "none" else "none",
               linewidth=0.9, hatch=hatch, zorder=3)

    # Подписи только у крайних бинов: иначе двадцать одно число превращает
    # график в таблицу и перестаёт читаться за десять секунд.
    for i, (name, vals) in enumerate(GAP_RMSE.items()):
        for j in (0, len(GAP_LABELS) - 1):
            ax.text(x[j] + (i - 1) * width, vals[j] + 0.002, _ru(vals[j], 3),
                    ha="center", va="bottom", fontsize=9.5, color=INK, rotation=90)

    gain = (1 - GAP_RMSE["Финальная конфигурация"][-1] / GAP_RMSE["Baseline: среднее двух соседей"][-1]) * 100
    ax.annotate(
        f"на самых длинных разрывах выигрыш {gain:.0f} %",
        xy=(x[-1] + 0.45, 0.180), ha="right", va="bottom", fontsize=11, color=ACCENT,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(GAP_LABELS)
    ax.set_xlabel("длина разрыва, дней")
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, 0.196)
    ax.set_yticks(np.arange(0.0, 0.181, 0.02))
    ax.set_title("Выигрыш есть во всех бинах длины разрыва, а не только в среднем")
    ax.legend(loc="upper left")
    _value_grid(ax)
    _comma_axis(ax, "y", 2)
    _save(fig, "fig4_rmse_by_gap.png")


# --------------------------------------------------------------------------
# График 5. Совпадение геометрии контрольного набора
# --------------------------------------------------------------------------

# Таблица из reports/experiments.md, раздел «Протокол локальной валидации».
GEOMETRY_LABELS = ["1–2", "3–4", "5–7", "8–13", "14–29", "30–61", "62–181", "182+"]
GEOMETRY_REAL = [6.6, 17.7, 27.3, 27.5, 15.8, 1.4, 3.2, 0.4]
GEOMETRY_LOCAL = [6.5, 17.8, 27.4, 27.6, 15.6, 1.4, 3.2, 0.4]


def figure_5_geometry() -> None:
    """Доказательство корректности протокола: локальный набор не легче реального."""
    fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    x = np.arange(len(GEOMETRY_LABELS))
    width = 0.38

    ax.bar(x - width / 2, GEOMETRY_REAL, width=width, color=PALE, edgecolor=GREY,
           linewidth=0.9, hatch="..", label="реальные контрольные точки (3 112 шт.)", zorder=3)
    ax.bar(x + width / 2, GEOMETRY_LOCAL, width=width, color=DARK,
           label="локальная валидация (2 999 шт.)", zorder=3)

    for xi, (a, b) in enumerate(zip(GEOMETRY_REAL, GEOMETRY_LOCAL)):
        ax.text(xi - width / 2, a + 0.4, _ru(a, 1),
                ha="center", va="bottom", fontsize=10, color=INK)
        ax.text(xi + width / 2, b + 0.4, _ru(b, 1),
                ha="center", va="bottom", fontsize=10, color=INK)

    worst = max(abs(a - b) for a, b in zip(GEOMETRY_REAL, GEOMETRY_LOCAL))
    ax.annotate(
        f"наибольшее расхождение\nпо бину — {_ru(worst, 1)} процентного пункта",
        xy=(0.99, 0.55), xycoords="axes fraction", ha="right", va="top",
        fontsize=11, color=ACCENT,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(GEOMETRY_LABELS)
    ax.set_xlabel("длина разрыва, дней")
    ax.set_ylabel("доля точек, %")
    ax.set_ylim(0, 33)
    ax.set_title("Контрольный набор геометрически неотличим от набора организаторов")
    ax.legend(loc="upper right")
    _value_grid(ax)
    _save(fig, "fig5_holdout_geometry.png")


# --------------------------------------------------------------------------
# График 6. Найденный период угнетения с объяснением
# --------------------------------------------------------------------------

# Случай 1 из reports/anomaly_examples.md: засуха с полным согласием с разметкой.
CASE_POLYGON = "AOI-0043"
CASE_YEAR = 2019
CAUSE_TITLES = {
    "drought": "дефицит влаги",
    "heat": "температурный стресс",
    "cold": "затяжной холод",
    "excess_water": "переувлажнение",
    "abrupt": "резкое событие",
    "harvest": "уборка или скашивание",
    "not_weather": "причина не погодная",
    "unknown": "версия не определена",
}
SEVERITY_TITLES = {
    "critical": "критическая аномалия",
    "suppression": "угнетение биомассы",
    "normal": "штатное развитие",
}


def figure_6_anomaly_case(df) -> None:
    """Конкретный найденный период угнетения и погода, объясняющая версию причины."""
    from src.contracts import Observation, SeriesInput, WeatherPoint
    from src.core.analyze import analyze

    g = df[df["anon_polygon_id"] == CASE_POLYGON]
    # Разбор в отчёте сделан по обучающему набору (2010-2024), воспроизводим его
    if "_source" in g.columns and (g["_source"] == "train").any():
        g = g[g["_source"] == "train"]
    if g.empty:
        print(f"  график 6 пропущен: полигона {CASE_POLYGON} нет в данных")
        return

    crop = g["crop_type"].dropna()
    crop = str(crop.iloc[0]) if len(crop) else None
    obs = [
        Observation(date=r.date.date(), ndvi=float(r.primary_ndvi))
        for r in g[g["primary_ndvi"].notna()].itertuples()
    ]
    weather = [
        WeatherPoint(
            date=r.date.date(),
            temp_c=None if pd.isna(r.era5_temp_c) else float(r.era5_temp_c),
            precip_mm=None if pd.isna(r.era5_precip_mm) else float(r.era5_precip_mm),
        )
        for r in g.itertuples()
    ]
    result = analyze(SeriesInput(polygon_id=CASE_POLYGON, observations=obs,
                                 weather=weather, crop_type=crop))

    periods = [a for a in result.anomalies if a.start.year == CASE_YEAR]
    if not periods:
        print(f"  график 6 пропущен: у {CASE_POLYGON} нет периода в {CASE_YEAR} году")
        return
    period = max(periods, key=lambda a: a.duration_days)

    series = pd.DataFrame(
        [
            {"date": pd.Timestamp(p.date), "observed": p.observed, "restored": p.restored,
             "norm": p.climatology_mean, "std": p.climatology_std}
            for p in result.series
        ]
    )
    lo = pd.Timestamp(period.start) - pd.Timedelta(days=35)
    hi = pd.Timestamp(period.end) + pd.Timedelta(days=35)
    s = series[(series["date"] >= lo) & (series["date"] <= hi)]

    wf = g[["date", "era5_temp_c", "era5_precip_mm"]].dropna(how="all", subset=["era5_temp_c", "era5_precip_mm"])
    wf = wf[(wf["date"] >= lo) & (wf["date"] <= hi)]

    fig, (ax, axw) = plt.subplots(2, 1, figsize=(10.0, 7.0), height_ratios=[2.3, 1.0],
                                  sharex=True, constrained_layout=True)

    # --- верхняя панель: ряд, норма и найденный период ---
    if s["norm"].notna().any():
        ax.fill_between(s["date"], s["norm"] - s["std"], s["norm"] + s["std"],
                        color=PALE, alpha=0.55, linewidth=0, zorder=1)
        ax.plot(s["date"], s["norm"], linestyle=(0, (5, 3)), color=GREY, linewidth=1.8,
                zorder=3, label="климатическая норма ± 1 σ")
    ax.axvspan(pd.Timestamp(period.start), pd.Timestamp(period.end),
               color=ACCENT, alpha=0.14, zorder=0)
    ax.plot(s["date"], s["restored"], color=DARK, linewidth=2.2, zorder=4,
            label="восстановленный ряд NDVI")
    seen = s[s["observed"].notna()]
    ax.plot(seen["date"], seen["observed"], "o", markersize=4.5, markerfacecolor="white",
            markeredgecolor=INK, markeredgewidth=1.0, linestyle="none", zorder=5,
            label="спутниковые наблюдения")

    severity = SEVERITY_TITLES.get(period.severity, period.severity)
    cause = CAUSE_TITLES.get(period.cause, period.cause)
    ax.set_ylabel("NDVI")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{CASE_POLYGON}, {crop or 'культура не указана'}: "
        f"{severity}, {period.start.strftime('%d.%m')} – {period.end.strftime('%d.%m.%Y')}, "
        f"{period.duration_days} дней"
    )

    # Карточка с вердиктом ставится левее найденного периода: там ряд ещё
    # ровный и место свободно, а поверх данных подпись их бы закрыла.
    evidence = period.evidence or {}
    card = [
        severity.upper(),
        _ru(f"min z = {period.min_zscore:.2f}, средний z = {period.mean_zscore:.2f}"),
        f"версия: {cause}",
        _ru(f"уверенность {period.cause_confidence:.2f}"),
    ]
    if "precip_30d_mm" in evidence and "precip_30d_norm_mm" in evidence:
        card.append(
            "осадки за 30 дней {:.0f} мм при норме {:.0f} мм —\nэто {:.0f} % нормы".format(
                evidence["precip_30d_mm"], evidence["precip_30d_norm_mm"],
                100 * float(evidence.get("precip_ratio", 0.0)))
        )
    ax.annotate(
        "\n".join(card),
        xy=(0.01, 0.97), xycoords="axes fraction", ha="left", va="top",
        fontsize=11, color=ACCENT, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=ACCENT,
                  linewidth=0.9, alpha=0.92),
    )
    ax.legend(loc="upper right", ncol=1)
    _comma_axis(ax, "y", 1)

    # --- нижняя панель: погода, из которой ядро вывело версию причины ---
    if len(wf):
        axw.bar(wf["date"], wf["era5_precip_mm"].fillna(0.0), width=1.0,
                color=MID, linewidth=0, zorder=3, label="осадки, мм/сут")
        axw.set_ylabel("осадки, мм", color=MID)
        axw.tick_params(axis="y", colors=MID)
        axt = axw.twinx()
        axt.spines["right"].set_visible(True)
        axt.plot(wf["date"], wf["era5_temp_c"], color=ACCENT, linewidth=1.6, zorder=4,
                 label="температура, °C")
        axt.set_ylabel("температура, °C", color=ACCENT)
        axt.tick_params(axis="y", colors=ACCENT)
        axw.axvspan(pd.Timestamp(period.start), pd.Timestamp(period.end),
                    color=ACCENT, alpha=0.14, zorder=0)
        axw.legend(handles=[
            Patch(facecolor=MID, label="осадки, мм/сут"),
            Line2D([0], [0], color=ACCENT, linewidth=1.6, label="температура, °C"),
        ], loc="upper left", ncol=2)
    axw.set_xlabel(f"дата, {CASE_YEAR} год")
    _ru_month_axis(axw)
    axw.set_title("Погода за тот же интервал: версию причины видно глазами",
                  fontsize=12, pad=6)

    _save(fig, "fig6_anomaly_case.png")


# --------------------------------------------------------------------------

def main() -> None:
    from src.ml.dataset import build_views, load_all
    from src.ml.registry import discover

    discover()
    print("Загрузка данных…")
    df = load_all(use_train=True)
    views = build_views(df)
    print(f"  {len(df)} строк, {len(views)} полигонов")

    # Каждая картинка строится независимо: падение одной не должно останавливать
    # остальные. Хакатон, данные могут быть неполными, а отчёт нужен целиком.
    tasks = [
        ("график 1 — восстановление ряда", lambda: figure_1_restoration(views, df)),
        ("график 2 — суточная поправка", lambda: figure_2_daily_correction(views)),
        ("график 3 — прогресс по экспериментам", figure_3_progress),
        ("график 4 — RMSE по длине разрыва", figure_4_gap_breakdown),
        ("график 5 — геометрия контрольного набора", figure_5_geometry),
        ("график 6 — период угнетения", lambda: figure_6_anomaly_case(df)),
    ]
    failed = 0
    for title, fn in tasks:
        print(title)
        try:
            fn()
        except Exception:  # noqa: BLE001 — отчёт важнее одной картинки
            failed += 1
            print(f"  ОШИБКА, картинка пропущена:\n{traceback.format_exc()}")
    print(f"Готово. Ошибок: {failed}. Каталог: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
