"""Графики для клиентского PDF-отчёта.

Читатель отчёта — фермер, агроном, оценщик банка или страховой, а не автор
модели. Поэтому здесь нет ни одного термина из внутренней кухни: на подписях
осей и в легендах «зелёная масса», «норма», «разброс», а z-оценки, RMSE и
сигмы остаются в коде. Расшифровка NDVI даётся ровно один раз — на главном
графике истории, дальше он уже не нужен.

Каждая функция возвращает готовый PNG в виде ``bytes`` и не имеет права упасть:
PDF собирается по расписанию, и лучше отдать честную заглушку «данных
недостаточно», чем сорвать весь документ из-за одного графика.
"""

from __future__ import annotations

import io
from datetime import date, datetime

import matplotlib

# Бэкенд Agg выбирается до импорта pyplot: сервис крутится в контейнере без
# дисплея, и любой оконный бэкенд там просто не инициализируется.
matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

__all__ = [
    "chart_series",
    "chart_last_season",
    "chart_forecast",
    "chart_score",
    "chart_seasons",
]

# --------------------------------------------------------------------------
# Палитра и общий стиль
# --------------------------------------------------------------------------

# Цвета повторяют веб-интерфейс один в один: человек смотрит карточку поля на
# сайте, потом скачивает PDF, и оба документа должны выглядеть как один продукт.
GREEN = "#4e9b36"
GREEN_DARK = "#2f6b2a"
RED = "#d4342a"
ORANGE = "#e08a20"
TEXT = "#101010"
MUTED = "#6f6f6f"
GRID = "#e6e6e6"
WHITE = "#ffffff"

# rcParams применяются точечно через rc_context, а не глобально: модуль может
# импортироваться внутри чужого процесса (API-воркер), и портить ему настройки
# matplotlib мы не хотим.
_RC = {
    # DejaVu Sans — единственный шрифт, который гарантированно есть в поставке
    # matplotlib и при этом содержит кириллицу. На Arial полагаться нельзя:
    # в linux-контейнере его нет, и все подписи превратятся в квадраты.
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "axes.unicode_minus": True,
}

_MM_PER_INCH = 25.4

# Сокращения месяцев вшиты руками: locale-зависимое форматирование дат в
# контейнере отдаёт английские названия, а отчёт русскоязычный.
_RU_MONTHS = {
    1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "май", 6: "июн",
    7: "июл", 8: "авг", 9: "сен", 10: "окт", 11: "ноя", 12: "дек",
}

# Причина угнетения приходит из детектора машинным ключом — на графике вместо
# него должно стоять слово, понятное без словаря.
_CAUSE_RU = {
    "drought": "засуха",
    "heat": "жара",
    "frost": "заморозки",
    "waterlogging": "переувлажнение",
    "excess_rain": "переувлажнение",
    "abrupt": "резкий спад",
    "non_weather": "причина не в погоде",
    "unknown": "причина не ясна",
}

_SEVERITY_RU = {
    "critical": "Сильное угнетение",
    "suppression": "Умеренное угнетение",
}

_SEVERITY_COLOR = {
    "critical": RED,
    "suppression": ORANGE,
}

# Последний рубеж обороны: минимальный валидный PNG 1×1 белый пиксель. Нужен
# только на случай, если даже отрисовка заглушки не удалась.
_BLANK_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08"
    b"\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?"
    b"\x00\x05\xfe\x02\xfe\xdc\xccY\xe7\x00\x00\x00\x00IEND\xaeB`\x82"
)


# --------------------------------------------------------------------------
# Мелкие помощники
# --------------------------------------------------------------------------

def _mm(value: float) -> float:
    """Миллиметры в дюймы — matplotlib умеет только дюймы, а вёрстка PDF в мм."""
    return float(value) / _MM_PER_INCH


def _parse_date(value) -> date | None:
    """Мягкий разбор даты: битую строку пропускаем, а не роняем весь график."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _num(value) -> float:
    """Любое «нечисло» (None, строка, NaN) превращаем в NaN.

    Так пропуски сами собой становятся разрывами линии, и не нужно городить
    отдельные маски для каждой серии.
    """
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _new_figure(width_mm: float, height_mm: float):
    """Создаёт фигуру с оформлением отчёта: без рамки, с горизонтальной сеткой."""
    fig, ax = plt.subplots(figsize=(_mm(width_mm), _mm(height_mm)))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    # Верхняя и правая рамки — визуальный шум, из-за них график выглядит как
    # научная иллюстрация, а не как страница клиентского документа.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)  # сетка под данными, иначе она режет линии
    ax.tick_params(length=0, pad=3)
    return fig, ax


def _render(fig, dpi: int) -> bytes:
    """Фигура -> PNG в памяти. Файлы на диск не пишем: PDF собирается из байтов."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


def _placeholder(width_mm: float, height_mm: float, dpi: int,
                 detail: str = "") -> bytes:
    """Честная заглушка вместо графика.

    Пустое место на странице читается как ошибка вёрстки, поэтому пишем прямым
    текстом, что данных не хватило.
    """
    try:
        with plt.rc_context(_RC):
            fig, ax = _new_figure(width_mm, height_mm)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            # Рамку оставляем: она держит размер картинки при обрезке по
            # bbox_inches="tight" и заодно показывает, что место под график
            # предусмотрено, а не потерялось при вёрстке.
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(GRID)
            text = "Данных для графика недостаточно"
            if detail:
                text += "\n" + detail
            ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED,
                    fontsize=9, linespacing=1.6)
            return _render(fig, dpi)
    except Exception:
        plt.close("all")
        return _BLANK_PNG


def _guard(builder, result, width_mm, height_mm, dpi, detail="") -> bytes:
    """Единая страховка: любое падение отрисовки превращается в заглушку.

    Отчёт формируется на живых данных из внешних источников, где регулярно
    встречаются пустые ряды и отсутствующие поля. Ронять сборку PDF из-за
    одной картинки недопустимо, поэтому исключения гасятся здесь.
    """
    try:
        with plt.rc_context(_RC):
            return builder(result or {}, width_mm, height_mm, dpi)
    except Exception:
        plt.close("all")  # незакрытая фигура утекла бы вместе с исключением
        return _placeholder(width_mm, height_mm, dpi, detail)


# --------------------------------------------------------------------------
# Ось времени
# --------------------------------------------------------------------------

def _apply_time_axis(ax, first: date, last: date) -> None:
    """Подписи дат по-русски и с плотностью, подходящей длине периода.

    Пять лет истории и три месяца прогноза требуют разной детализации: в первом
    случае хватает годов, во втором нужны числа месяца.
    """
    span = max((last - first).days, 1)

    if span > 900:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        fmt = lambda d: str(d.year)  # noqa: E731
    elif span > 400:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fmt = lambda d: f"{_RU_MONTHS[d.month]}\n{d.year}"  # noqa: E731
    elif span > 150:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        fmt = lambda d: _RU_MONTHS[d.month]  # noqa: E731
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 15)))
        fmt = lambda d: f"{d.day} {_RU_MONTHS[d.month]}"  # noqa: E731

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _pos: fmt(mdates.num2date(v)))
    )


def _legend_below(ax, ncol: int = 3) -> None:
    """Легенда под графиком.

    Внутри рамки она неизбежно накрывает данные: кривая NDVI занимает всю
    ширину, свободного угла не остаётся.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=ncol, frameon=False, handlelength=1.6, columnspacing=1.4,
              handletextpad=0.6, labelcolor=TEXT)


# --------------------------------------------------------------------------
# Разбор входных данных
# --------------------------------------------------------------------------

def _extract_series(result: dict) -> dict | None:
    """Ряд наблюдений -> набор numpy-массивов одной длины.

    Даты сразу переводятся в числа matplotlib: так одинаково работают и линии,
    и заливки, и вертикальные полосы аномалий, без сюрпризов с конвертерами
    единиц измерения.
    """
    raw = result.get("series")
    if not isinstance(raw, list) or len(raw) < 2:
        return None

    rows = []
    for point in raw:
        if not isinstance(point, dict):
            continue
        day = _parse_date(point.get("date"))
        if day is None:
            continue
        rows.append((day, point))
    if len(rows) < 2:
        return None

    rows.sort(key=lambda item: item[0])  # источники не гарантируют порядок
    dates = [item[0] for item in rows]
    points = [item[1] for item in rows]

    mean = np.array([_num(p.get("climatology_mean")) for p in points])
    std = np.array([_num(p.get("climatology_std")) for p in points])

    return {
        "dates": dates,
        "x": np.array(mdates.date2num(dates), dtype=float),
        "restored": np.array([_num(p.get("restored")) for p in points]),
        "observed": np.array([_num(p.get("observed")) for p in points]),
        "clim_mean": mean,
        "clim_low": mean - std,
        "clim_high": mean + std,
    }


def _anomaly_spans(result: dict) -> list[dict]:
    """Периоды угнетения в виде готовых к отрисовке отрезков."""
    spans = []
    for item in result.get("anomalies") or []:
        if not isinstance(item, dict):
            continue
        start = _parse_date(item.get("start"))
        end = _parse_date(item.get("end"))
        if start is None or end is None:
            continue
        severity = item.get("severity") if item.get("severity") in _SEVERITY_COLOR else "suppression"
        spans.append({
            "start": start,
            "end": end,
            "severity": severity,
            "color": _SEVERITY_COLOR[severity],
            "cause": _CAUSE_RU.get(item.get("cause"), "причина не ясна"),
            "days": item.get("duration_days") or (end - start).days + 1,
        })
    return spans


def _plot_norm_and_series(ax, data: dict, *, line_width: float,
                          marker_size: float, ylabel: str) -> None:
    """Общий слой обоих графиков истории: норма, коридор, кривая, снимки.

    Порядок отрисовки задаёт читаемость: сначала фон (коридор нормы), затем
    пунктир нормы, сверху фактическая кривая и точки снимков.
    """
    x = data["x"]

    ax.fill_between(x, data["clim_low"], data["clim_high"], color=GREEN,
                    alpha=0.13, linewidth=0,
                    label="Обычный разброс для этого поля")
    ax.plot(x, data["clim_mean"], color=MUTED, linestyle="--", linewidth=0.9,
            label="Норма (среднее за прошлые годы)")
    ax.plot(x, data["restored"], color=GREEN, linewidth=line_width,
            solid_capstyle="round", label="Зелёная масса поля")

    # Реальные снимки показываем отдельно: между ними значения восстановлены
    # моделью, и человек вправе видеть, где факт, а где расчёт.
    observed = data["observed"]
    seen = np.isfinite(observed)
    if seen.any():
        ax.scatter(x[seen], observed[seen], s=marker_size, color=GREEN_DARK,
                   zorder=5, linewidths=0, label="Даты реальных снимков")

    ax.set_ylabel(ylabel, color=TEXT, linespacing=1.5)


def _ylimits(*arrays) -> tuple[float, float]:
    """Границы по вертикали по всем видимым слоям, с запасом на подписи."""
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low = float(values.min())
    high = float(values.max())
    if high - low < 0.05:  # почти константный ряд иначе схлопнется в полоску
        low, high = low - 0.05, high + 0.05
    pad = (high - low) * 0.08
    return max(low - pad, -0.05), min(high + pad, 1.15)


# --------------------------------------------------------------------------
# 1. Вся история наблюдений
# --------------------------------------------------------------------------

def _build_series(result: dict, width_mm: float, height_mm: float, dpi: int) -> bytes:
    data = _extract_series(result)
    if data is None:
        return _placeholder(width_mm, height_mm, dpi,
                            "по этому полю нет ряда наблюдений")

    fig, ax = _new_figure(width_mm, height_mm)
    _plot_norm_and_series(
        ax, data, line_width=1.1, marker_size=4.5,
        # Единственное место в отчёте, где раскрывается аббревиатура NDVI:
        # дальше по документу читателю хватает слов «зелёная масса».
        ylabel="Зелёная масса поля\n(NDVI: 0 — голая земля, 1 — густая зелень)",
    )

    # Полосы аномалий рисуем после линий, но с малой прозрачностью — они должны
    # читаться как подсветка фона, а не перекрывать данные.
    shown = set()
    for span in _anomaly_spans(result):
        ax.axvspan(mdates.date2num(span["start"]), mdates.date2num(span["end"]),
                   color=span["color"], alpha=0.14, linewidth=0, zorder=0)
        shown.add(span["severity"])

    handles, labels = ax.get_legend_handles_labels()
    for severity in ("critical", "suppression"):
        if severity in shown:
            handles.append(Patch(facecolor=_SEVERITY_COLOR[severity], alpha=0.35,
                                 linewidth=0))
            labels.append(_SEVERITY_RU[severity])
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, frameon=False, handlelength=1.6, columnspacing=1.4,
              handletextpad=0.6, labelcolor=TEXT)

    ax.set_xlim(data["x"][0], data["x"][-1])
    ax.set_ylim(*_ylimits(data["restored"], data["observed"], data["clim_low"],
                          data["clim_high"]))
    _apply_time_axis(ax, data["dates"][0], data["dates"][-1])
    return _render(fig, dpi)


def chart_series(result: dict, width_mm: float = 170, height_mm: float = 80,
                 dpi: int = 200) -> bytes:
    """Вся доступная история поля: факт, норма и периоды угнетения. PNG."""
    return _guard(_build_series, result, width_mm, height_mm, dpi,
                  "по этому полю нет ряда наблюдений")


# --------------------------------------------------------------------------
# 2. Последний сезон крупным планом
# --------------------------------------------------------------------------

def _season_window(dates: list[date]) -> tuple[date, date] | None:
    """Границы последнего вегетационного сезона (апрель — октябрь).

    Берём последний год, в котором на сезон приходится хоть сколько-то
    наблюдений: если ряд обрывается в январе, показывать пустой график этого
    года бессмысленно, полезнее предыдущий сезон целиком.
    """
    years = sorted({d.year for d in dates}, reverse=True)
    for year in years:
        start = date(year, 4, 1)
        end = date(year, 10, 31)
        inside = [d for d in dates if start <= d <= end]
        if len(inside) >= 20:
            # Окно подрезаем по фактическим данным, чтобы справа не висел
            # пустой хвост до конца октября у ещё не законченного сезона.
            return max(start, min(inside)), min(end, max(inside))
    return None


def _build_last_season(result: dict, width_mm: float, height_mm: float, dpi: int) -> bytes:
    data = _extract_series(result)
    if data is None:
        return _placeholder(width_mm, height_mm, dpi,
                            "по этому полю нет ряда наблюдений")

    window = _season_window(data["dates"])
    if window is None:
        return _placeholder(width_mm, height_mm, dpi,
                            "в последнем сезоне слишком мало наблюдений")
    start, end = window

    mask = np.array([start <= d <= end for d in data["dates"]])
    part = {key: (value[mask] if isinstance(value, np.ndarray)
                  else [d for d, keep in zip(value, mask) if keep])
            for key, value in data.items()}

    fig, ax = _new_figure(width_mm, height_mm)
    # Линия и точки толще, чем на пятилетнем графике: здесь всего один сезон,
    # и место позволяет показать форму кривой подробно.
    _plot_norm_and_series(ax, part, line_width=1.8, marker_size=13,
                          ylabel="Зелёная масса поля")

    spans = [s for s in _anomaly_spans(result)
             if s["end"] >= start and s["start"] <= end]

    low, high = _ylimits(part["restored"], part["observed"], part["clim_low"],
                         part["clim_high"])
    if spans:
        # Запас сверху нужен только под подписи аномалий. Когда угнетений не
        # было, пустая полоса над кривой лишь мельчит сам график.
        high = min(high + (high - low) * 0.30, 1.25)
    ax.set_ylim(low, high)
    x_left, x_right = mdates.date2num(start), mdates.date2num(end)
    ax.set_xlim(x_left, x_right)

    label_transform = blended_transform_factory(ax.transData, ax.transAxes)
    shown = set()
    for index, span in enumerate(spans):
        left = mdates.date2num(max(span["start"], start))
        right = mdates.date2num(min(span["end"], end))
        ax.axvspan(left, right, color=span["color"], alpha=0.16, linewidth=0,
                   zorder=0)
        shown.add(span["severity"])
        # Подписи чередуем по высоте: соседние периоды угнетения часто идут
        # подряд, и на одном уровне их тексты слиплись бы.
        level = (0.965, 0.885, 0.805)[index % 3]
        # У краёв сезона подпись прижимаем к границе графика, иначе она уезжает
        # за рамку и налезает на подписи оси.
        center = (left + right) / 2
        share = (center - x_left) / max(x_right - x_left, 1e-9)
        if share < 0.16:
            anchor_x, align = x_left, "left"
        elif share > 0.84:
            anchor_x, align = x_right, "right"
        else:
            anchor_x, align = center, "center"
        ax.text(anchor_x, level, f"{span['days']} дн. · {span['cause']}",
                transform=label_transform, ha=align, va="top",
                fontsize=6.5, color=span["color"])

    handles, labels = ax.get_legend_handles_labels()
    for severity in ("critical", "suppression"):
        if severity in shown:
            handles.append(Patch(facecolor=_SEVERITY_COLOR[severity], alpha=0.35,
                                 linewidth=0))
            labels.append(_SEVERITY_RU[severity])
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, handlelength=1.6, columnspacing=1.4,
              handletextpad=0.6, labelcolor=TEXT)

    ax.set_xlabel(f"Сезон {start.year} года", color=MUTED, labelpad=2)
    _apply_time_axis(ax, start, end)
    return _render(fig, dpi)


def chart_last_season(result: dict, width_mm: float = 170, height_mm: float = 80,
                      dpi: int = 200) -> bytes:
    """Последний вегетационный сезон крупно, с подписанными аномалиями. PNG."""
    return _guard(_build_last_season, result, width_mm, height_mm, dpi,
                  "в последнем сезоне слишком мало наблюдений")


# --------------------------------------------------------------------------
# 3. Прогноз на ближайший месяц
# --------------------------------------------------------------------------

def _build_forecast(result: dict, width_mm: float, height_mm: float, dpi: int) -> bytes:
    meta = result.get("meta") or {}
    forecast = meta.get("forecast") or {}

    dates = [_parse_date(d) for d in (forecast.get("dates") or [])]
    dates = [d for d in dates if d is not None]
    values = [_num(v) for v in (forecast.get("ndvi") or [])]
    if len(dates) < 2 or len(values) < len(dates):
        return _placeholder(width_mm, height_mm, dpi, "прогноз не построен")

    values = np.array(values[:len(dates)])
    low = np.array([_num(v) for v in (forecast.get("low") or [])][:len(dates)])
    high = np.array([_num(v) for v in (forecast.get("high") or [])][:len(dates)])
    clim = np.array([_num(v) for v in (forecast.get("clim") or [])][:len(dates)])
    # Коридор рисуем только если он пришёл целиком: половинчатая заливка
    # выглядит как ошибка расчёта.
    has_band = low.size == len(dates) and high.size == len(dates)
    has_clim = clim.size == len(dates)

    anchor = _parse_date(forecast.get("anchor_date")) or dates[0]
    fig, ax = _new_figure(width_mm, height_mm)

    # Хвост факта нужен для масштаба: без него прогноз висит в воздухе и по
    # нему невозможно понять, продолжает он тренд или ломает его.
    data = _extract_series(result)
    tail_x = tail_y = None
    tail_start = anchor
    if data is not None:
        keep = np.array([(anchor - d).days <= 60 and d <= anchor
                         for d in data["dates"]])
        if keep.sum() >= 2:
            tail_x = data["x"][keep]
            tail_y = data["restored"][keep]
            tail_clim = data["clim_mean"][keep]
            tail_start = min(d for d, k in zip(data["dates"], keep) if k)
            ax.plot(tail_x, tail_clim, color=MUTED, linestyle="--",
                    linewidth=0.9, label="Норма (среднее за прошлые годы)")
            ax.plot(tail_x, tail_y, color=GREEN, linewidth=1.8,
                    solid_capstyle="round", label="Что было на самом деле")

    fx = np.array(mdates.date2num(dates), dtype=float)
    if has_band:
        ax.fill_between(fx, low, high, color=GREEN, alpha=0.16, linewidth=0,
                        label="Возможный разброс прогноза")
    if has_clim:
        ax.plot(fx, clim, color=MUTED, linestyle="--", linewidth=0.9,
                label=None if tail_x is not None else "Норма (среднее за прошлые годы)")
    ax.plot(fx, values, color=GREEN_DARK, linewidth=1.8, linestyle="-",
            dashes=(4, 2), label="Прогноз")

    # Стык факта и прогноза — главный ориентир на графике, поэтому линия
    # подписана словом, а не оставлена на догадку читателя.
    ax.axvline(mdates.date2num(anchor), color=MUTED, linewidth=0.9,
               linestyle=":", zorder=1)
    label_transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(mdates.date2num(anchor), 1.02, "  сегодня", transform=label_transform,
            ha="left", va="bottom", fontsize=7, color=MUTED)

    layers = [values]
    if has_band:
        layers += [low, high]
    if has_clim:
        layers.append(clim)
    if tail_y is not None:
        layers.append(tail_y)
    ax.set_ylim(*_ylimits(*layers))
    ax.set_xlim(mdates.date2num(tail_start), fx[-1])
    ax.set_ylabel("Зелёная масса поля", color=TEXT)
    _apply_time_axis(ax, tail_start, dates[-1])
    # Две колонки, а не четыре: подписи здесь длинные, в одну строку они
    # вылезают за ширину графика.
    _legend_below(ax, ncol=2)
    return _render(fig, dpi)


def chart_forecast(result: dict, width_mm: float = 170, height_mm: float = 70,
                   dpi: int = 200) -> bytes:
    """Хвост факта и прогноз на месяц вперёд с коридором. PNG."""
    return _guard(_build_forecast, result, width_mm, height_mm, dpi,
                  "прогноз не построен")


# --------------------------------------------------------------------------
# 4. Из чего сложился балл поля
# --------------------------------------------------------------------------

# Порядок фиксирован: сверху то, что чаще всего объясняет итоговую оценку.
_COMPONENTS = (
    ("stability", "Ровность по годам"),
    ("stress", "Отсутствие стресса"),
    ("productivity", "Продуктивность"),
    ("trend", "Многолетний тренд"),
)


def _component_color(value: float) -> str:
    """Светофор по значению: так столбик читается ещё до чтения цифры."""
    if value >= 60:
        return GREEN
    if value >= 35:
        return ORANGE
    return RED


def _build_score(result: dict, width_mm: float, height_mm: float, dpi: int) -> bytes:
    meta = result.get("meta") or {}
    components = ((meta.get("score") or {}).get("components")) or {}

    rows = [(title, _num(components.get(key)))
            for key, title in _COMPONENTS if key in components]
    rows = [(title, value) for title, value in rows if np.isfinite(value)]
    if not rows:
        return _placeholder(width_mm, height_mm, dpi, "оценка поля не рассчитана")

    fig, ax = _new_figure(width_mm, height_mm)
    ax.grid(False)  # у горизонтальных столбиков сетка по Y бессмысленна
    positions = np.arange(len(rows))[::-1]  # первый пункт списка — сверху

    for pos, (_title, value) in zip(positions, rows):
        # Серая «дорожка» до 100 показывает, сколько до максимума не хватило:
        # без неё столбик в 26 баллов не с чем сравнить.
        ax.barh(pos, 100, height=0.55, color=GRID, linewidth=0, zorder=1)
        ax.barh(pos, max(value, 0.8), height=0.55, color=_component_color(value),
                linewidth=0, zorder=2)
        # Подпись внутри столбика, если он достаточно длинный, иначе справа от
        # него — на коротких столбиках белый текст просто не поместится.
        if value >= 22:
            ax.text(value - 2.5, pos, f"{value:.0f}", ha="right", va="center",
                    fontsize=8, color=WHITE, zorder=3)
        else:
            ax.text(value + 2.5, pos, f"{value:.0f}", ha="left", va="center",
                    fontsize=8, color=TEXT, zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels([title for title, _ in rows], color=TEXT, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Баллы из 100", color=MUTED, labelpad=2)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    return _render(fig, dpi)


def chart_score(result: dict, width_mm: float = 80, height_mm: float = 60,
                dpi: int = 200) -> bytes:
    """Четыре составляющие итогового балла поля. PNG."""
    return _guard(_build_score, result, width_mm, height_mm, dpi,
                  "оценка поля не рассчитана")


# --------------------------------------------------------------------------
# 5. Сравнение сезонов между собой
# --------------------------------------------------------------------------

def _season_is_complete(season: dict, result: dict) -> bool:
    """Закончен ли сезон.

    Флаг ``complete`` есть не во всех версиях расчёта, поэтому при его
    отсутствии смотрим, докуда дотянулись данные: сезон текущего года,
    оборвавшийся в сентябре, честнее пометить как неполный.
    """
    if isinstance(season.get("complete"), bool):
        return season["complete"]
    coverage = _num(season.get("coverage"))
    if np.isfinite(coverage):
        return coverage >= 0.9
    year = season.get("year")
    last = _parse_date((result.get("meta") or {}).get("last_date"))
    if not isinstance(year, int) or last is None:
        return True
    # Вегетационный сезон закрывается концом октября: если ряд по этому году
    # обрывается раньше, накопленная биомасса заведомо занижена.
    return last >= date(year, 10, 25) or last.year > year


def _build_seasons(result: dict, width_mm: float, height_mm: float, dpi: int) -> bytes:
    meta = result.get("meta") or {}
    seasons = ((meta.get("score") or {}).get("seasons")) or []

    rows = []
    for season in seasons:
        if not isinstance(season, dict):
            continue
        integral = _num(season.get("integral"))
        if not np.isfinite(integral):
            continue
        rows.append({
            "year": season.get("year"),
            "integral": integral,
            "vs_norm": _num(season.get("vs_norm")),
            "complete": _season_is_complete(season, result),
        })
    if not rows:
        return _placeholder(width_mm, height_mm, dpi, "сезоны не разобраны")

    rows.sort(key=lambda item: (item["year"] is None, item["year"]))

    fig, ax = _new_figure(width_mm, height_mm)
    ax.grid(False)

    # Уровень нормы восстанавливаем из отклонения: vs_norm = (факт − норма)/норма.
    # Отдельного поля с нормой в результате нет, а линия «сколько должно быть»
    # для читателя важнее любых цифр на оси.
    norms = [item["integral"] / (1 + item["vs_norm"])
             for item in rows
             if np.isfinite(item["vs_norm"]) and item["vs_norm"] > -0.95]
    norm = float(np.median(norms)) if norms else None

    top = max(item["integral"] for item in rows)
    if norm:
        top = max(top, norm)
    ax.set_ylim(0, top * 1.28)  # запас сверху под подписи отклонений

    xs = np.arange(len(rows))
    for x, item in zip(xs, rows):
        deviation = item["vs_norm"]
        positive = (not np.isfinite(deviation)) or deviation >= 0
        color = GREEN if positive else RED
        # Незаконченный сезон рисуем штриховкой и полупрозрачной заливкой:
        # сравнивать его с полными годами напрямую нельзя.
        ax.bar(x, item["integral"], width=0.6, color=color, linewidth=0,
               alpha=0.35 if not item["complete"] else 1.0,
               hatch="//" if not item["complete"] else None,
               edgecolor=color, zorder=2)
        if np.isfinite(deviation):
            ax.text(x, item["integral"] + top * 0.03,
                    f"{deviation * 100:+.0f} %", ha="center", va="bottom",
                    fontsize=7.5, color=color)

    if norm:
        ax.axhline(norm, color=MUTED, linestyle="--", linewidth=0.9, zorder=3)
        ax.text(len(rows) - 0.45, norm, " норма", ha="left", va="center",
                fontsize=7, color=MUTED)

    ax.set_xticks(xs)
    ax.set_xticklabels([
        f"{item['year']}\nнеполный" if not item["complete"] else str(item["year"])
        for item in rows
    ], color=TEXT, fontsize=7.5)
    # Абсолютное значение накопленной биомассы читателю ничего не говорит,
    # смысл несут только сравнение столбиков между собой и проценты сверху.
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylabel("Зелёная масса,\nнакопленная за сезон", color=TEXT,
                  linespacing=1.4)

    legend = [
        Patch(facecolor=GREEN, linewidth=0, label="Лучше нормы"),
        Patch(facecolor=RED, linewidth=0, label="Хуже нормы"),
    ]
    if any(not item["complete"] for item in rows):
        legend.append(Patch(facecolor=GREEN, alpha=0.35, hatch="//",
                            edgecolor=GREEN, linewidth=0,
                            label="Сезон ещё не закончился"))
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False, handlelength=1.6, columnspacing=1.4,
              handletextpad=0.6, labelcolor=TEXT)
    return _render(fig, dpi)


def chart_seasons(result: dict, width_mm: float = 170, height_mm: float = 65,
                  dpi: int = 200) -> bytes:
    """Сравнение сезонов: накопленная биомасса и отклонение от нормы. PNG."""
    return _guard(_build_seasons, result, width_mm, height_mm, dpi,
                  "сезоны не разобраны")
