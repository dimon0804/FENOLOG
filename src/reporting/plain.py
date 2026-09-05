"""Перевод внутренних метрик анализа на язык клиентского отчёта.

Отчёт читает не аналитик, а фермер, агроном, оценщик банка или страховой. Для
них «z-оценка −2,99» — это шум, поэтому в PDF не должно попасть ни одного
внутреннего термина: ни z, ни сигмы, ни NDVI без расшифровки, ни RMSE.

Правила, которым подчинён весь модуль:

1. Ничего не выдумывать. Если нормы нет — так и написать, а не подставить ноль.
   Если уверенность в причине низкая — назвать это версией, а не фактом.
2. Не противоречить самому себе. Числа для фразы берутся из того же evidence,
   на котором ядро выбрало причину, поэтому «осадков 55 процентов от нормы» и
   «засухи не было» не могут оказаться в одном абзаце: формулировка причины
   строится из тех же чисел, а не отдельно от них.
3. Работать на неполных данных. Любая функция при пустом result возвращает
   осмысленную заглушку и не бросает исключение — PDF собирается всегда.

Модуль ничего не считает заново: он только пересказывает то, что уже посчитало
ядро (`src/core/`). Любая новая цифра здесь была бы вторым источником истины.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

# ============================================================================
# Словари форм и уровней
# ============================================================================

# Месяцы в родительном падеже: «2 мая», а не «2 май».
_MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)

# Названия причин — заголовком карточки, поэтому с большой буквы и без деталей.
_CAUSE_TITLE = {
    "drought": "Засуха",
    "heat": "Жара",
    "excess_water": "Переувлажнение",
    "abrupt": "Резкое событие на поле",
    "non_weather": "Погода ни при чём",
    "unknown": "Причина не установлена",
}

# Классы риска из скоринга. Дублируем словами, чтобы в отчёте буква «D» не
# осталась без перевода: для читателя буква сама по себе ничего не значит.
_GRADE_RISK = {
    "A": "риск существенно ниже среднего",
    "B": "риск ниже среднего",
    "C": "риск средний",
    "D": "риск выше среднего",
    "E": "риск высокий",
}

# Подписи составляющих балла. Ключи приходят из scoring.py.
_COMPONENT_LABEL = {
    "stability": "Ровность по годам",
    "stress": "Сезоны без провалов",
    "productivity": "Уровень зелёной массы",
    "trend": "Направление за годы",
}


# ============================================================================
# Числа, даты, склонения
# ============================================================================


def _cap(text: str) -> str:
    """Заглавная только первая буква. str.capitalize() гасит остальные слова."""
    text = str(text or "")
    return text[:1].upper() + text[1:]


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова по числу: 1 день, 2 дня, 5 дней, 21 день."""
    n = abs(int(n))
    if 11 <= n % 100 <= 14:
        return many
    last = n % 10
    if last == 1:
        return one
    if last in (2, 3, 4):
        return few
    return many


def _days(n: int) -> str:
    """«34 дня», «1 день», «5 дней»."""
    n = int(n)
    return f"{n} {_plural(n, 'день', 'дня', 'дней')}"


def _years(n: int) -> str:
    n = int(n)
    return f"{n} {_plural(n, 'год', 'года', 'лет')}"


def _num(x: float, digits: int = 1) -> str:
    """Число русской записью: разделитель дробной части — запятая."""
    if x is None:
        return "—"
    text = f"{x:.{digits}f}"
    if digits > 0:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", ",")


def _pct(x: float) -> str:
    """Доля 0.214 -> «21 процент». Только число со словом, без «выше/ниже»."""
    n = int(round(abs(x) * 100))
    return f"{n} {_plural(n, 'процент', 'процента', 'процентов')}"


def _pct_vs_norm(x: float) -> str:
    """Доля 0.214 -> «на 21 процент выше нормы», −0.231 -> «ниже нормы»."""
    if abs(x) < 0.02:
        return "на уровне нормы"
    return f"на {_pct(x)} {'выше' if x > 0 else 'ниже'} нормы"


def _mm(x: float, unit: bool = True) -> str:
    """Осадки: до целых миллиметров, дробные доли для читателя бессмысленны.

    `unit=False` — для второго числа в паре «18 миллиметров вместо обычных 63»:
    повторять единицу измерения дважды в одной фразе незачем, а согласовать её
    падеж после «вместо обычных» без этого не выходит.
    """
    n = int(round(x))
    if not unit:
        return str(n)
    return f"{n} {_plural(n, 'миллиметр', 'миллиметра', 'миллиметров')}"


def _degrees(x: float) -> str:
    """«на 6,6 градуса», «на 5 градусов» — форма зависит от целости числа."""
    v = abs(x)
    if abs(v - round(v)) < 0.05:
        n = int(round(v))
        return f"{n} {_plural(n, 'градус', 'градуса', 'градусов')}"
    return f"{_num(v)} градуса"


def _to_date(value) -> date | None:
    """Дата из ISO-строки или объекта date. На мусоре возвращает None."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _fmt_date(value, with_year: bool = True) -> str:
    """«2 мая 2022» либо «2 мая» без года."""
    d = _to_date(value)
    if d is None:
        return "дата неизвестна"
    text = f"{d.day} {_MONTHS[d.month - 1]}"
    return f"{text} {d.year}" if with_year else text


def _fmt_range(start, end) -> str:
    """Диапазон дат по-русски.

    Внутри одного месяца год и месяц не повторяются («2–5 мая 2022»), внутри
    одного года не повторяется год («2 мая — 5 июня 2022»). Это не украшение:
    повтор месяца в короткой карточке читается как опечатка.
    """
    a, b = _to_date(start), _to_date(end)
    if a is None or b is None:
        return _fmt_date(a or b)
    if a == b:
        return _fmt_date(a)
    if a.year == b.year and a.month == b.month:
        return f"{a.day}–{b.day} {_MONTHS[a.month - 1]} {a.year}"
    if a.year == b.year:
        return f"{_fmt_date(a, with_year=False)} — {_fmt_date(b)}"
    return f"{_fmt_date(a)} — {_fmt_date(b)}"


def _fmt_doy(doy: int) -> str:
    """День года -> «25 мая». Год берётся невисокосный: ±1 день роли не играет."""
    try:
        d = date(2001, 1, 1) + timedelta(days=int(doy) - 1)
    except (TypeError, ValueError, OverflowError):
        return "неизвестная дата"
    return _fmt_date(d, with_year=False)


def _span_words(first, last) -> str:
    """Длина истории словами: «5 лет и 5 месяцев»."""
    a, b = _to_date(first), _to_date(last)
    if a is None or b is None:
        return ""
    months = (b.year - a.year) * 12 + (b.month - a.month)
    if b.day < a.day:
        months -= 1
    months = max(months, 0)
    y, m = divmod(months, 12)
    parts = []
    if y:
        parts.append(_years(y))
    if m:
        parts.append(f"{m} {_plural(m, 'месяц', 'месяца', 'месяцев')}")
    return " и ".join(parts) if parts else "меньше месяца"


# ============================================================================
# Перевод внутренних шкал в слова
# ============================================================================


def _depth_words(z: float | None) -> str:
    """z-оценка -> насколько сильно поле отклонилось вниз.

    Пороги те же, что в contracts.py (−1 и −2), но третья ступень (−3) введена
    только для текста: между «чуть ниже двух сигм» и «минус пять» для читателя
    разница есть, а слова были бы одни и те же.
    """
    if z is None:
        return "сравнить не с чем"
    if z >= -1:
        return "в пределах обычного"
    if z >= -2:
        return "ниже обычного"
    if z >= -3:
        return "заметно ниже обычного"
    return "намного ниже обычного"


def _confidence_words(c: float | None, subject: str = "уверенность") -> str:
    """Уверенность в долях -> три градации словами."""
    if c is None:
        return f"{subject} не оценивалась"
    if c >= 0.7:
        return f"{subject} высокая"
    if c >= 0.4:
        return f"{subject} средняя"
    return f"{subject} низкая"


def _hedge(confidence: float | None) -> str:
    """Приставка к утверждению о причине.

    Ниже 0,4 причина — версия, выше 0,7 её можно называть прямо. Между ними
    осторожное «скорее всего». Это требование к честности отчёта: банк не должен
    прочитать догадку как установленный факт.
    """
    c = confidence if confidence is not None else 0.0
    if c >= 0.7:
        return ""
    if c >= 0.4:
        return "скорее всего, "
    return "похоже, что "


def _ratio_words(ratio: float | None) -> str:
    """Отношение к норме -> слова. 0.28 -> «28 процентов от обычного»."""
    if ratio is None:
        return ""
    if ratio < 1:
        return f"{_pct(ratio)} от нормы"
    if ratio >= 1.8:
        return f"почти в {_num(ratio)} раза больше обычного"
    if ratio >= 1.35:
        return "примерно в полтора раза больше обычного"
    return f"на {_pct(ratio - 1)} больше обычного"


def _safe(value, default=None):
    """Число или default: в JSON поле может прийти строкой, None или отсутствовать."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _meta(result: dict) -> dict:
    m = (result or {}).get("meta")
    return m if isinstance(m, dict) else {}


def _series(result: dict) -> list:
    s = (result or {}).get("series")
    return s if isinstance(s, list) else []


def _anomalies(result: dict) -> list:
    a = (result or {}).get("anomalies")
    return a if isinstance(a, list) else []


def _has_norm(result: dict) -> bool:
    """Есть ли с чем сравнивать. «none» и отсутствие флага — нормы нет."""
    meta = _meta(result)
    if meta.get("climatology_source") == "none":
        return False
    if meta.get("has_climatology") is False:
        return False
    return bool(meta.get("has_climatology")) or meta.get("climatology_source") in ("polygon", "crop")


def _recent_state(result: dict, window_days: int = 30) -> dict:
    """Состояние поля на последних снимках.

    Берём хвост ряда, а не одну последнюю точку: одна точка на восстановленном
    ряду легко оказывается выбросом, а решение о тоне всего отчёта принимается
    именно здесь.
    """
    series = _series(result)
    out = {"z": None, "days": 0, "observed": 0, "last": None}
    dates = [_to_date(p.get("date")) for p in series if isinstance(p, dict)]
    dates = [d for d in dates if d is not None]
    if not dates:
        return out
    last = max(dates)
    out["last"] = last
    edge = last - timedelta(days=window_days)
    zs, observed = [], 0
    for p in series:
        if not isinstance(p, dict):
            continue
        d = _to_date(p.get("date"))
        if d is None or d < edge:
            continue
        z = _safe(p.get("zscore"))
        if z is not None:
            zs.append(z)
        if p.get("observed") is not None:
            observed += 1
    out["days"] = len(zs)
    out["observed"] = observed
    if zs:
        out["z"] = sum(zs) / len(zs)
    return out


# ============================================================================
# Вердикт
# ============================================================================


def verdict(result: dict) -> dict:
    """Первый экран отчёта: что с полем сейчас и на чём основан вывод."""
    meta = _meta(result)
    series = _series(result)
    if not series:
        return {
            "tone": "nodata",
            "title": "Данных по полю нет",
            "text": (
                "Собрать спутниковые снимки по этому контуру не удалось, поэтому "
                "оценить состояние поля нечем. Обычно так бывает, если контур "
                "очень мал, лежит вне зоны съёмки или период наблюдений пуст."
            ),
        }
    if not _has_norm(result):
        return {
            "tone": "nodata",
            "title": "Сравнивать не с чем",
            "text": (
                "Снимки по полю есть, но истории для сравнения не набралось: "
                "мы видим, сколько на поле зелёной массы, и не можем сказать, "
                "много это или мало именно для него. Оценка появится после "
                "того, как накопится хотя бы пара полных сезонов наблюдений."
            ),
        }

    state = _recent_state(result)
    z = state["z"]
    anomalies = _anomalies(result)
    score = meta.get("score") if isinstance(meta.get("score"), dict) else {}
    grade = str(score.get("grade") or "")
    score_value = _safe(score.get("score"))

    # Тон складывается из двух вещей: как поле выглядит сейчас и что было
    # раньше. История сама по себе никогда не красит отчёт в красный — поле,
    # которое сегодня в норме, «плохим» назвать нельзя, но и промолчать о
    # провальных сезонах для банка нельзя тоже, поэтому получается «watch».
    if z is None:
        tone_now = "nodata"
    elif z < -2:
        tone_now = "bad"
    elif z < -1:
        tone_now = "watch"
    else:
        tone_now = "ok"
    tone_hist = "watch" if grade in ("D", "E") or len(anomalies) >= 3 else "ok"
    order = {"nodata": 0, "ok": 1, "watch": 2, "bad": 3}
    tone = tone_now if order[tone_now] >= order[tone_hist] else tone_hist
    if tone_now == "nodata":
        tone = tone_hist

    last_words = _fmt_date(state["last"])
    parts = []

    if z is None:
        parts.append(f"Последние снимки — по {last_words} года, но сравнить их с нормой не вышло.")
    elif z >= -1:
        parts.append(
            f"На последних снимках (по {last_words} года) поле выглядит обычно для "
            f"этого времени года: зелёной массы примерно столько, сколько на нём бывает в среднем."
        )
    elif z >= -2:
        parts.append(
            f"На последних снимках (по {last_words} года) поле развивается хуже "
            f"обычного: зелёной массы заметно меньше, чем бывает на нём в это время года."
        )
    else:
        parts.append(
            f"На последних снимках (по {last_words} года) поле в плохом состоянии: "
            f"зелёной массы намного меньше, чем бывает на нём в это время года."
        )

    if state["days"]:
        obs = state["observed"]
        if obs:
            parts.append(
                f"Вывод опирается на последний месяц: настоящими снимками закрыто "
                f"{_days(obs)} из {state['days']}, остальные достроены расчётом."
            )
        else:
            parts.append(
                f"Чистых снимков за последний месяц не было, значения на эти "
                f"{_days(state['days'])} достроены по соседним датам."
            )

    if anomalies:
        worst = min(anomalies, key=lambda a: _safe(a.get("min_zscore"), 0.0))
        longest = max(anomalies, key=lambda a: _safe(a.get("duration_days"), 0.0))
        n = len(anomalies)
        parts.append(
            f"За всю историю наблюдений нашлось {n} "
            f"{_plural(n, 'период', 'периода', 'периодов')}, когда поле развивалось "
            f"хуже обычного; самый долгий длился {_days(_safe(longest.get('duration_days'), 0))} "
            f"и начался {_fmt_date(longest.get('start'))} года, самый глубокий — "
            f"{_fmt_range(worst.get('start'), worst.get('end'))}."
        )
    else:
        parts.append("Периодов, когда поле развивалось хуже обычного, за историю наблюдений не нашлось.")

    if score_value is not None and grade in _GRADE_RISK:
        n = int(round(score_value))
        parts.append(
            f"Итоговая оценка поля — {n} {_plural(n, 'балл', 'балла', 'баллов')} "
            f"из 100, {_GRADE_RISK[grade]}."
        )

    if tone == "ok":
        title = "Поле в норме"
    elif tone_now == "ok" and tone == "watch":
        title = "Сейчас норма, история неровная"
    elif tone == "watch":
        title = "Поле развивается хуже обычного"
    elif tone == "bad":
        title = "Поле в плохом состоянии"
    else:
        title = "Оценить поле не удалось"

    return {"tone": tone, "title": title, "text": " ".join(parts)}


# ============================================================================
# Карточка поля
# ============================================================================


def field_summary(result: dict, polygon: dict | None = None) -> list[tuple[str, str]]:
    """Пары «подпись — значение» для шапки отчёта.

    Строки, которых нет в данных, не выводятся вовсе: пустая строка «Площадь: —»
    в документе для банка выглядит как потерянные данные.
    """
    meta = _meta(result)
    polygon = polygon if isinstance(polygon, dict) else {}
    rows: list[tuple[str, str]] = []

    name = str(polygon.get("name") or "").strip()
    if name:
        rows.append(("Поле", name))

    # Культура. Показывается вместе с тем, откуда она взялась: «озимая пшеница»
    # из карточки поля и «озимая пшеница, определена сервисом» — это разные по
    # надёжности сведения, и в документе для банка разница существенна.
    crop = polygon.get("crop_type") or meta.get("crop_type")
    crop = str(crop).strip() if crop else ""
    detection = meta.get("crop_detection") or {}
    if not crop:
        rows.append(("Культура", "не указана и не определена"))
    elif meta.get("crop_source") == "detected":
        conf = _safe(detection.get("confidence"))
        suffix = f", определена сервисом по форме кривой (уверенность {_num(conf, 2)})" \
            if conf is not None else ", определена сервисом по форме кривой"
        rows.append(("Культура", crop + suffix))
    else:
        rows.append(("Культура", crop))

    area = _safe(polygon.get("area_ha"))
    if area is not None:
        rows.append(("Площадь", f"{_num(area, 2)} га"))

    center = polygon.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        lon, lat = _safe(center[0]), _safe(center[1])
        if lon is not None and lat is not None:
            ns = "с. ш." if lat >= 0 else "ю. ш."
            ew = "в. д." if lon >= 0 else "з. д."
            rows.append(("Центр поля", f"{_num(abs(lat), 4)}° {ns}, {_num(abs(lon), 4)}° {ew}"))

    first, last = meta.get("first_date"), meta.get("last_date")
    if first or last:
        span = _span_words(first, last)
        value = _fmt_range(first, last)
        if span:
            value += f" ({span})"
        rows.append(("Период наблюдений", value))

    n_obs = _safe(meta.get("n_obs"))
    if n_obs is not None:
        n = int(n_obs)
        rows.append(("Спутниковых снимков", str(n)))

    sources = meta.get("sources")
    if isinstance(sources, dict) and sources:
        names = {"s2": "Sentinel-2", "landsat": "Landsat", "modis": "MODIS"}
        parts = [
            f"{names.get(k, k)} — {int(v)}"
            for k, v in sorted(sources.items(), key=lambda kv: -_safe(kv[1], 0))
            if _safe(v, 0) > 0
        ]
        if parts:
            rows.append(("С каких спутников", ", ".join(parts)))

    days_between = None
    first_d, last_d = _to_date(first), _to_date(last)
    if first_d and last_d and n_obs:
        days_between = (last_d - first_d).days
    if days_between and days_between > 0 and n_obs:
        step = days_between / n_obs
        rows.append(("Как часто поле видно", f"в среднем раз в {_days(round(step))}"))

    clim = meta.get("climatology_source")
    years = _safe(meta.get("climatology_years"))
    if clim == "polygon":
        value = "собственная история этого поля"
        if years:
            value += f" за {_years(int(years))}"
    elif clim == "crop":
        value = "усреднённая норма по культуре — грубее, чем по самому полю"
    else:
        value = "нормы нет, сравнивать не с чем"
    rows.append(("Норма для сравнения", value))

    src = str(polygon.get("source") or "").strip()
    if src:
        rows.append(("Откуда контур", {
            "drawn": "нарисован вручную на карте",
            "osm": "взят из OpenStreetMap",
        }.get(src, src)))

    return rows


# ============================================================================
# Карточка аномалии
# ============================================================================


def _cause_text(anomaly: dict) -> tuple[str, str]:
    """Объяснение периода и совет обычными словами.

    Числа для фразы берутся из того же `evidence`, по которому ядро выбрало
    причину. Поэтому текст физически не может сказать «дождей было мало» там,
    где ядро увидело переувлажнение: источник чисел один.
    """
    cause = str(anomaly.get("cause") or "unknown")
    ev = anomaly.get("evidence") if isinstance(anomaly.get("evidence"), dict) else {}
    conf = _safe(anomaly.get("cause_confidence"), 0.0)
    hedge = _hedge(conf)
    duration = int(_safe(anomaly.get("duration_days"), 0) or 0)

    precip = _safe(ev.get("precip_30d_mm"))
    precip_norm = _safe(ev.get("precip_30d_norm_mm"))
    ratio = _safe(ev.get("precip_ratio"))
    t_anom = _safe(ev.get("temp_anomaly_c"))
    t_mean = _safe(ev.get("temp_mean_c"))
    t_norm = _safe(ev.get("temp_norm_c"))

    def rain_phrase() -> str:
        if precip is None or precip_norm is None:
            return ""
        tail = f" — это {_ratio_words(ratio)}" if ratio is not None else ""
        return (f"За месяц перед этим на поле выпало {_mm(precip)} дождя "
                f"вместо обычных {_mm(precip_norm, unit=False)}{tail}.")

    if cause == "drought":
        # Температуру упоминаем только при заметном отклонении и обязательно
        # с правильным знаком: «холоднее обычного» рядом с выводом «не хватало
        # влаги» не противоречие, но читателю нужно объяснить, что засуха
        # бывает и без жары. Иначе абзац выглядит как ошибка расчёта.
        if t_anom is None or abs(t_anom) < 1.5:
            temp = ""
        elif t_anom > 0:
            temp = f"Вдобавок было теплее обычного на {_degrees(t_anom)}, влага уходила быстрее."
        else:
            temp = f"Жары при этом не было: держалось даже холоднее обычного на {_degrees(t_anom)}."
        text = " ".join(x for x in [
            rain_phrase(),
            temp,
            _cap(f"{hedge}полю не хватало именно влаги.") if hedge else "Полю не хватало именно влаги.",
        ] if x)
        advice = (
            "Сверьте период с записями по полю: был ли полив, как выглядели посевы. "
            "Если поле застраховано от засухи, это тот эпизод, который стоит подтвердить на месте."
        )
        return text, advice

    if cause == "heat":
        heat_bits = []
        if t_mean is not None and t_norm is not None:
            heat_bits.append(
                f"Средняя температура в эти дни была {_degrees(t_mean)} при обычных {_num(t_norm)}."
            )
        elif t_anom is not None:
            heat_bits.append(f"Было теплее обычного на {_degrees(t_anom)}.")
        if ratio is not None and ratio < 0.9:
            heat_bits.append(f"Дождя при этом выпало {_ratio_words(ratio)}.")
        heat_bits.append(
            _cap(f"{hedge}растения страдали от жары.") if hedge else "Растения страдали от жары."
        )
        advice = "Проверьте по журналу работ, совпал ли этот период с цветением или наливом — тогда потери заметнее всего."
        return " ".join(heat_bits), advice

    if cause == "excess_water":
        bits = [rain_phrase()]
        if t_anom is not None and t_anom < -1.5:
            bits.append(f"Было холоднее обычного на {_degrees(t_anom)}, вода уходила медленно.")
        bits.append(
            _cap(f"{hedge}на поле застаивалась вода, и растения развивались хуже.")
            if hedge else "На поле застаивалась вода, и растения развивались хуже."
        )
        advice = "Осмотрите низкие места поля и состояние водоотводных канав — переувлажнение обычно повторяется из года в год на одних и тех же участках."
        return " ".join(x for x in bits if x), advice

    if cause == "abrupt":
        drop = _safe(ev.get("z_drop_10d"))
        bits = ["Зелёная масса упала резко, за считаные дни, а не постепенно — так растения сами не вянут."]
        window = ev.get("harvest_window")
        start_doy = _safe(ev.get("start_doy"))
        inside = None
        if isinstance(window, (list, tuple)) and len(window) == 2 and start_doy is not None:
            lo, hi = _safe(window[0]), _safe(window[1])
            if lo is not None and hi is not None:
                inside = lo <= start_doy <= hi
                if inside:
                    bits.append(
                        f"Срок совпадает с обычным окном уборки в этих местах — примерно "
                        f"с {_fmt_doy(lo)} по {_fmt_doy(hi)}, поэтому больше всего это похоже на уборку."
                    )
                else:
                    # Вывод меняется на противоположный, а не дополняется: сказать
                    # «окно уборки другое» и тут же «похоже на уборку» — это ровно
                    # то противоречие, из-за которого читатель перестаёт верить отчёту.
                    bits.append(
                        f"Но на плановую уборку по срокам не похоже: обычно здесь убирают "
                        f"примерно с {_fmt_doy(lo)} по {_fmt_doy(hi)}, а спад пришёлся на "
                        f"{_fmt_date(anomaly.get('start'), with_year=False)}."
                    )
        if inside is False:
            bits.append(_cap(
                f"{hedge}это разовое событие на поле: потрава, палы, повреждение или проход техники."
                if hedge else "Это разовое событие на поле: потрава, палы, повреждение или проход техники."
            ))
        elif inside is None:
            bits.append("Так выглядит либо уборка, либо разовое повреждение: град, потрава, палы, техника.")
        if drop is not None and drop >= 2:
            bits.append("Падение очень глубокое, само по себе поле так не теряет зелень.")
        advice = "Уточните по журналу работ, что было на поле в эти дни. От ответа зависит, считать это потерей или плановой уборкой."
        return " ".join(bits), advice

    if cause == "non_weather":
        bits = []
        if ratio is not None and precip is not None and precip_norm is not None:
            if ratio >= 1.15:
                bits.append(
                    f"Дождей выпало даже больше обычного: {_mm(precip)} вместо {_mm(precip_norm, unit=False)}."
                )
            elif ratio <= 0.85:
                # Формулировка «мало дождя, но засухи нет» — самое опасное место
                # отчёта: без оговорки она читается как противоречие. Поэтому
                # сразу объясняем, почему недобора осадков здесь не хватает.
                bits.append(
                    f"Дождей было меньше обычного — {_mm(precip)} вместо {_mm(precip_norm, unit=False)}, — "
                    f"но одного этого мало, чтобы объяснить спад: в прошлые годы поле переносило "
                    f"такую погоду без потерь."
                )
            else:
                bits.append(
                    f"Осадки держались около нормы: {_mm(precip)} при обычных {_mm(precip_norm, unit=False)}."
                )
        if t_anom is not None and abs(t_anom) < 1.5:
            bits.append("Температура тоже была обычной для этого времени года.")
        elif t_anom is not None:
            direction = "теплее" if t_anom > 0 else "холоднее"
            bits.append(
                f"Было {direction} обычного на {_degrees(t_anom)} — для такого спада этого мало."
            )
        if duration >= 60:
            bits.append(
                "Поле держалось ниже нормы почти весь сезон при обычной погоде. "
                "Чаще всего так выглядит пар, поздний сев или смена культуры."
            )
        else:
            bits.append(
                "Погода этот период не объясняет. Обычно за таким стоит то, что делали на самом "
                "поле: сев не в срок, смена культуры, обработка или потрава."
            )
        if conf < 0.4:
            bits.append("Это версия, а не установленный факт: подтвердить её можно только по данным хозяйства.")
        advice = "Спросите в хозяйстве, что происходило на поле в эти дни и было ли оно вообще засеяно."
        return " ".join(bits), advice

    bits = ["Причину этого периода по имеющимся данным определить не удалось: "
            "ни погода, ни характер спада на объяснение не тянут."]
    if ratio is not None and precip is not None and precip_norm is not None:
        bits.append(f"Осадков за месяц {_mm(precip)} при обычных {_mm(precip_norm, unit=False)}.")
    advice = "Стоит посмотреть журнал полевых работ за эти даты — по спутнику причина не читается."
    return " ".join(bits), advice


def anomaly_card(anomaly: dict) -> dict:
    """Одна карточка периода угнетения для таблицы аномалий в отчёте."""
    if not isinstance(anomaly, dict) or not anomaly:
        return {
            "when": "период не указан", "duration": "", "depth": "сравнить не с чем",
            "cause": "Причина не установлена", "confidence": "уверенность не оценивалась",
            "text": "Данных по этому периоду нет.", "advice": "", "tone": "watch",
        }

    severity = str(anomaly.get("severity") or "")
    min_z = _safe(anomaly.get("min_zscore"))
    duration = int(_safe(anomaly.get("duration_days"), 0) or 0)
    conf = _safe(anomaly.get("cause_confidence"))
    cause = str(anomaly.get("cause") or "unknown")
    text, advice = _cause_text(anomaly)

    return {
        "when": _fmt_range(anomaly.get("start"), anomaly.get("end")),
        "duration": _days(duration) if duration else "",
        "depth": _depth_words(min_z),
        "cause": _CAUSE_TITLE.get(cause, _CAUSE_TITLE["unknown"]),
        "confidence": _confidence_words(conf),
        "text": text,
        "advice": advice,
        # Красным красим только критические периоды: «угнетение» — это повод
        # присмотреться, а не потеря, и заливать им половину отчёта нельзя.
        "tone": "bad" if severity == "critical" else "watch",
    }


# ============================================================================
# Балл поля
# ============================================================================


def _component_note(key: str, value: float) -> str:
    """Одна строка пояснения к составляющей балла — что мерили и как читать число."""
    v = int(round(value))
    if key == "stability":
        level = "сезоны похожи друг на друга" if v >= 65 else (
            "сезоны заметно разные" if v >= 40 else "сезоны сильно расходятся между собой")
        return f"Насколько похожи сезоны друг на друга: {level}."
    if key == "stress":
        level = "таких дней почти не бывает" if v >= 65 else (
            "такие периоды случаются регулярно" if v >= 40 else "таких дней много")
        return f"Сколько дней за сезон поле проводит хуже обычного: {level}."
    if key == "productivity":
        level = "набирает много" if v >= 65 else (
            "набирает средне" if v >= 40 else "набирает мало")
        return f"Сколько зелёной массы поле набирает за сезон: {level}."
    if key == "trend":
        level = "год от года поле идёт вверх" if v >= 65 else (
            "заметного движения нет" if v >= 40 else "год от года поле слабеет")
        return f"Куда движется поле за все годы наблюдений: {level}."
    return "Составляющая итогового балла."


def _season_words(verdict_text: str) -> str:
    """Готовая оценка сезона из ядра -> язык фермера.

    Ядро отдаёт короткий вердикт («недобор биомассы»), но это язык эксперта.
    Незнакомые формулировки не пропускаем в отчёт вовсе: лучше промолчать,
    чем показать читателю внутренний термин.
    """
    text = str(verdict_text or "").strip().lower()
    if not text:
        return ""
    base = {
        "провальный сезон": "сезон провальный",
        "недобор биомассы": "зелёной массы заметно меньше обычного",
        "в пределах нормы": "всё шло как обычно",
        "выше нормы": "зелёной массы больше обычного",
    }
    head = text.split(",")[0].strip()
    if head not in base:
        return ""
    out = base[head]
    if "критическ" in text:
        out += ", с глубоким провалом внутри сезона"
    return out


def _plain_flag(flag: str) -> str:
    """Оговорки скоринга — на человеческий язык.

    Флаги ядра написаны для разработчика и содержат «интеграл NDVI» и «две
    сигмы». Известные переписываем, неизвестные пропускаем в отчёт как есть:
    молча потерять предупреждение хуже, чем показать его сухим языком.
    """
    text = str(flag or "")
    low = text.lower()
    if "тип культуры не задан" in low:
        return ("Культура поля не указана, поэтому уровень зелёной массы сравнивался "
                "с усреднённой шкалой, а не с нормой конкретной культуры.")
    if "интеграл ndvi" in low:
        return ("Балл опирается на зелёную массу, накопленную за сезон по снимкам. "
                "Это косвенная оценка, а не измеренная урожайность в центнерах с гектара.")
    if "порог z" in low or "сигм" in low:
        return ("Порог «ниже нормы» подстраивается под собственный разброс поля: на ровном "
                "поле его пробивает даже небольшое отклонение, на неровном не пробивает и крупное. "
                "Поэтому главный вес в балле отдан недобору зелёной массы за сезон, а не числу плохих дней.")
    if "сезон" in low and ("мало" in low or "два" in low):
        return _cap(text)
    return _cap(text) if text else ""


def score_block(result: dict) -> dict:
    """Раздел «Оценка поля»: балл, из чего он сложился и как его читать."""
    meta = _meta(result)
    score = meta.get("score") if isinstance(meta.get("score"), dict) else {}
    if not score or _safe(score.get("score")) is None:
        return {
            "score": None, "grade": "", "headline": "Балл поля не рассчитан",
            "text": ("Для оценки нужна история хотя бы за пару сезонов и построенная по ней "
                     "норма. По этому полю их не набралось, поэтому балл не выставлен: "
                     "поставить его наугад было бы хуже, чем не ставить вовсе."),
            "components": [], "caveats": [],
        }

    value = int(round(_safe(score.get("score"), 0)))
    grade = str(score.get("grade") or "")
    risk = _GRADE_RISK.get(grade, "класс риска не определён")
    headline = f"{_cap(risk)}: {value} {_plural(value, 'балл', 'балла', 'баллов')} из 100"

    seasons = score.get("seasons") if isinstance(score.get("seasons"), list) else []
    yield_vs = _safe(score.get("yield_vs_norm"))
    stress_days = _safe(score.get("stress_days_per_season"))
    confidence = _safe(score.get("confidence"))

    parts = [
        f"Балл {value} из 100 — это {risk}. "
        f"Шкала общая для всех полей и не зависит от того, какие поля оценивали рядом."
    ]
    if seasons:
        n = len(seasons)
        parts.append(f"Оценка собрана по {n} {_plural(n, 'сезону', 'сезонам', 'сезонам')} наблюдений.")
        worst = min(seasons, key=lambda s: _safe(s.get("vs_norm"), 0.0))
        worst_vs = _safe(worst.get("vs_norm"))
        if worst_vs is not None and worst_vs < -0.1:
            worst_days = _safe(worst.get("stress_days"))
            tail = ""
            if worst_days and worst_days >= 1:
                tail += f", и {_days(round(worst_days))} поле провело хуже обычного"
            if "критическ" in str(worst.get("verdict") or "").lower():
                tail += "; внутри сезона был глубокий обвал"
            parts.append(
                f"Худшим был {int(_safe(worst.get('year'), 0))} год: зелёной массы за сезон "
                f"{_pct_vs_norm(worst_vs)}{tail}."
            )
    if yield_vs is not None:
        # yield_vs_norm в ядре — это ПОСЛЕДНИЙ сезон, а не среднее по годам.
        # Назвать его средним значило бы соврать в самом заметном месте отчёта.
        last_season = seasons[-1] if seasons else {}
        year = _safe(last_season.get("year"))
        when = f"В последнем сезоне ({int(year)} год) " if year else "В последнем сезоне "
        note = ""
        # «complete» появился в ядре позже фикстур: если ключа нет — про полноту
        # сезона молчим, а не додумываем её.
        if last_season.get("complete") is False:
            note = " Сезон ещё не закончился, поэтому итог может измениться."
        words = _season_words(last_season.get("verdict"))
        tail = f" — {words}" if words else ""
        parts.append(f"{when}поле набрало зелёной массы {_pct_vs_norm(yield_vs)}{tail}.{note}")
    if stress_days is not None and stress_days >= 1:
        parts.append(f"В среднем за сезон поле проводит хуже обычного около {_days(round(stress_days))}.")
    parts.append(
        "Для банка или страховой это значит вот что: чем ниже балл, тем сильнее "
        "урожай на этом поле пляшет от года к году, и тем осторожнее стоит "
        "закладывать верхнюю границу сбора."
    )

    components = []
    raw = score.get("components") if isinstance(score.get("components"), dict) else {}
    for key in ("stability", "stress", "productivity", "trend"):
        if key in raw:
            v = _safe(raw.get(key))
            if v is None:
                continue
            components.append((_COMPONENT_LABEL.get(key, key), int(round(v)), _component_note(key, v)))

    caveats_list = []
    if confidence is not None:
        caveats_list.append(
            f"{_cap(_confidence_words(confidence, 'достоверность самой оценки'))}: "
            f"чем короче история поля и чем реже снимки, тем она ниже."
        )
    for flag in (score.get("flags") or []):
        plain = _plain_flag(flag)
        if plain:
            caveats_list.append(plain)

    return {
        "score": value,
        "grade": grade,
        "headline": headline,
        "text": " ".join(parts),
        "components": components,
        "caveats": caveats_list,
    }


# ============================================================================
# Прогноз
# ============================================================================


def forecast_block(result: dict) -> dict:
    """Раздел «Что будет дальше»: короткий прогноз с честными оговорками."""
    meta = _meta(result)
    fc = meta.get("forecast") if isinstance(meta.get("forecast"), dict) else {}
    if not fc or not fc.get("dates"):
        return {
            "headline": "Прогноз не строился",
            "text": ("Чтобы заглянуть вперёд, нужна норма поля и свежий снимок. "
                     "Здесь чего-то из этого не хватило, поэтому прогноза нет."),
            "confidence_words": "уверенность не оценивалась",
            "tone": "nodata",
        }

    horizon = int(_safe(fc.get("horizon_days"), 0) or 0)
    risk = str(fc.get("risk") or "")
    anchor = fc.get("anchor_date")
    stale = int(_safe(fc.get("stale_days"), 0) or 0)
    deviation = _safe(fc.get("expected_deviation_mean"), _safe(fc.get("expected_deviation")))
    confidence = _safe(fc.get("confidence"))
    dates = fc.get("dates") or []
    last_day = dates[-1] if isinstance(dates, list) and dates else None

    tone = {"low": "ok", "moderate": "watch", "high": "bad"}.get(risk, "watch")
    headline = {
        "low": f"Ближайшие {_days(horizon)}: поле должно остаться около нормы",
        "moderate": f"Ближайшие {_days(horizon)}: поле может уйти ниже нормы",
        "high": f"Ближайшие {_days(horizon)}: ожидается заметный недобор",
    }.get(risk, f"Прогноз на ближайшие {_days(horizon)}")

    parts = []
    if last_day:
        # Год не повторяем дважды в одной фразе, но и не теряем его совсем:
        # прогноз на границе года («от 20 декабря до 19 января») без года врёт.
        a, b = _to_date(anchor), _to_date(last_day)
        if a and b and a.year == b.year:
            parts.append(
                f"Прогноз построен от снимка на {_fmt_date(a, with_year=False)} "
                f"и доходит до {_fmt_date(b)} года."
            )
        else:
            parts.append(
                f"Прогноз построен от снимка на {_fmt_date(anchor)} года "
                f"и доходит до {_fmt_date(last_day)} года."
            )
    if deviation is not None:
        if abs(deviation) < 0.03:
            parts.append("Ожидаемое отклонение от нормы небольшое — в пределах обычного разброса поля.")
        else:
            direction = "выше" if deviation > 0 else "ниже"
            parts.append(f"Ожидается, что поле будет держаться {direction} нормы весь этот срок.")
    if stale > 0:
        parts.append(
            f"Последний чистый снимок был {_days(stale)} назад, поэтому отсчёт идёт от него, "
            f"а не от сегодняшнего дня."
        )
    parts.append(
        "Прогноз опирается на норму поля и его нынешнее отклонение от неё; "
        "будущей погоды он не знает. Ливень, сухая неделя или выход техники в поле его отменяют."
    )

    return {
        "headline": headline,
        "text": " ".join(parts),
        "confidence_words": _confidence_words(confidence),
        "tone": tone,
    }


# ============================================================================
# Справочные разделы
# ============================================================================


def peers_block(result: dict) -> dict | None:
    """Поле на фоне соседей — понятным языком. None, если сравнивать было не с чем.

    Раздел отвечает на первый вопрос любого агронома и любого страховщика: это у
    меня одного или у всех. Ответ на него меняет решение целиком, поэтому он
    вынесен отдельным блоком, а не спрятан в пояснении к периоду.
    """
    meta = _meta(result)
    peers = meta.get("peers")
    if not isinstance(peers, dict) or not peers.get("peers_same_group"):
        return None

    total = int(_safe(peers.get("peers_total"), 0) or 0)
    same = int(_safe(peers.get("peers_same_group"), 0) or 0)
    lines: list[str] = []

    rank = peers.get("rank") or {}
    place, of = _safe(rank.get("place")), _safe(rank.get("of"))
    delta = _safe(rank.get("delta_pct"))
    headline = "Поле на фоне соседей"
    tone = "ok"
    if place is not None and of is not None:
        headline = f"{int(place)} место из {int(of)} среди соседних полей"
    if delta is not None:
        if delta >= 5:
            lines.append(
                f"За сезон поле набрало биомассы на {_pct(abs(delta) / 100)} больше, "
                f"чем в среднем соседние поля той же группы."
            )
        elif delta <= -5:
            tone = "warn"
            lines.append(
                f"За сезон поле набрало биомассы на {_pct(abs(delta) / 100)} меньше, "
                f"чем в среднем соседние поля той же группы. Это не приговор — "
                f"разница может объясняться сортом, сроком сева или почвой, — но "
                f"повод посмотреть, чем соседи отличаются."
            )
        else:
            lines.append(
                "За сезон поле набрало примерно столько же биомассы, сколько соседние "
                "поля той же группы: заметной разницы нет."
            )

    other = total - same
    if other > 0:
        lines.append(
            f"Сравнение шло только с {same} полями своей группы. Ещё {other} соседних "
            f"полей заняты другой культурой, и в сравнение они не брались: у другой "
            f"культуры другой календарь развития, и разница с ней говорила бы о "
            f"культуре, а не о состоянии поля."
        )

    district = [p for p in (peers.get("periods") or []) if p.get("scope") == "район"]
    local = [p for p in (peers.get("periods") or []) if p.get("scope") == "поле"]
    if district:
        lines.append(
            f"Просадок, которые случились по всему району: {len(district)}. В такие "
            f"периоды вместе с этим полем проседали и соседние — причину стоит искать "
            f"в погоде, а не в поле."
        )
    if local:
        tone = "warn" if tone == "ok" else tone
        lines.append(
            f"Просадок, которых у соседей не было: {len(local)}. Погода легла на район "
            f"одинаково, а просело только это поле — значит дело в нём самом: "
            f"агротехника, семена, техника, вредители."
        )

    if not lines:
        return None
    return {"headline": headline, "tone": tone, "lines": lines,
            "peers_total": total, "peers_same_group": same}


def glossary() -> list[tuple[str, str]]:
    """Словарик в конце отчёта: термины, которые всё же пришлось оставить."""
    return [
        ("Зелёная масса (NDVI)",
         "Спутник видит, сколько на поле живой зелени, и сводит это к одному числу. "
         "Голая земля даёт около нуля, густой посев в разгар роста — около единицы. "
         "Это не урожай в центнерах, а показатель того, насколько поле зелёное в этот день."),
        ("Климатическая норма поля",
         "Средняя зелёная масса этого же поля на тот же день года за прошлые сезоны. "
         "Норма своя у каждого поля: то, что нормально для пастбища, для пшеницы мало."),
        ("Коридор нормы",
         "Полоса вокруг нормы, в которую поле попадает в обычный год. Пока линия поля "
         "идёт внутри полосы, всё штатно; выход вниз за её пределы и есть повод для разбора."),
        ("Период угнетения",
         "Отрезок в несколько дней или недель, когда поле держалось ниже коридора нормы. "
         "Короткие провалы на день-два не считаются — их даёт облачность и шум съёмки."),
        ("Критическая аномалия",
         "Тот же период угнетения, но глубокий: поле ушло вниз намного дальше обычного "
         "разброса. Именно такие периоды разбираются по погоде и попадают в оценку риска."),
        ("Восстановление ряда",
         "Спутник видит поле не каждый день: мешают облака и расписание пролётов. "
         "Пропущенные дни достраиваются по соседним снимкам и по норме поля, поэтому "
         "линия на графике сплошная, хотя точек съёмки меньше."),
        ("Балл поля",
         "Число от 0 до 100 и буква от A до E. Балл собирает ровность сезонов, число плохих "
         "дней, уровень зелёной массы и многолетнее направление. Чем ниже балл, тем сильнее "
         "результат поля пляшет от года к году."),
        ("Прогноз на месяц",
         "Продолжение линии поля вперёд от последнего снимка: куда оно пойдёт, если "
         "погода будет обычной. Будущей погоды прогноз не знает."),
    ]


def how_it_works() -> list[tuple[str, str]]:
    """«Как это считается» — без единой формулы, но без вранья об устройстве."""
    return [
        ("1. Берём контур поля",
         "Контур либо рисуется на карте, либо подтягивается из открытой карты OpenStreetMap. "
         "По контуру считается площадь и центр поля — к центру потом привязывается погода."),
        ("2. Собираем снимки",
         "Со спутников Sentinel-2, Landsat и MODIS за все доступные годы. Каждый снимок "
         "обрезается по контуру, кадры с облаками и тенями отбрасываются. Остаётся "
         "одно число на дату: сколько на поле зелёной массы."),
        ("3. Достраиваем пропуски",
         "Между снимками остаются дыры от нескольких дней до нескольких недель. Они "
         "заполняются по соседним датам, по типичному ходу сезона и по соседним полям, "
         "снятым тем же пролётом. Достроенные дни в отчёте отмечены отдельно."),
        ("4. Строим норму поля",
         "По прошлым сезонам считается, сколько зелёной массы бывает на этом поле в каждый "
         "день года, и насколько сильно значение обычно гуляет. Текущий год в свою норму "
         "не входит, иначе поле сравнивалось бы само с собой."),
        ("5. Ищем периоды хуже обычного",
         "Дни, когда поле держалось ниже своего коридора нормы, собираются в периоды. "
         "На каждый период поднимается погода — температура и осадки за предшествующий "
         "месяц — и проверяется, объясняет ли она провал. Если не объясняет, так и пишем."),
        ("6. Считаем балл и прогноз",
         "Сезоны сравниваются между собой и с нормой, из этого складывается балл поля. "
         "Отдельно строится прогноз на месяц вперёд от последнего снимка."),
    ]


def data_sources(result: dict) -> list[tuple[str, str]]:
    """Откуда взяты данные и что именно взято — с фактическими объёмами."""
    meta = _meta(result)
    sources = meta.get("sources") if isinstance(meta.get("sources"), dict) else {}
    rows: list[tuple[str, str]] = []

    def count(key: str) -> int:
        return int(_safe(sources.get(key), 0) or 0)

    if count("s2"):
        n = count("s2")
        rows.append(("Sentinel-2, Европейское космическое агентство",
                     f"{n} {_plural(n, 'снимок', 'снимка', 'снимков')}. Основной источник: "
                     f"разрешение 10 метров, съёмка раз в несколько дней."))
    if count("landsat"):
        n = count("landsat")
        rows.append(("Landsat 8 и 9, NASA и Геологическая служба США",
                     f"{n} {_plural(n, 'снимок', 'снимка', 'снимков')}. Подстраховка на даты, "
                     f"когда Sentinel-2 поле не снял."))
    if count("modis"):
        n = count("modis")
        rows.append(("MODIS, NASA",
                     f"{n} {_plural(n, 'значение', 'значения', 'значений')}. Грубее по деталям, "
                     f"но снимает почти ежедневно и удлиняет историю поля."))

    rows.append(("Microsoft Planetary Computer",
                 "Открытый каталог, через который получены все спутниковые снимки. "
                 "Данные бесплатные и общедоступные."))

    weather_days = _safe(meta.get("collected_weather_days"))
    if weather_days:
        n = int(weather_days)
        rows.append(("Open-Meteo, архив погоды ERA5",
                     f"{n} {_plural(n, 'сутки', 'суток', 'суток')} температуры и осадков по центру "
                     f"поля. По ним объясняются периоды, когда поле развивалось хуже обычного."))
    else:
        rows.append(("Open-Meteo, архив погоды ERA5",
                     "Температура и осадки по центру поля. По ним объясняются периоды, "
                     "когда поле развивалось хуже обычного."))

    rows.append(("OpenStreetMap",
                 "Контуры сельхозполей: по ним поле можно выбрать на карте, не обводя вручную."))
    return rows


# ============================================================================
# Ограничения по конкретному полю
# ============================================================================


def caveats(result: dict) -> list[str]:
    """Честные ограничения именно этого расчёта.

    Раздел обязательный: отчёт уходит в банк или страховую, и умолчание о том,
    что история поля короткая или снимков мало, там читается как обман.
    """
    meta = _meta(result)
    series = _series(result)
    out: list[str] = []

    if not series:
        return ["По этому полю не собрано ни одного снимка, поэтому всё, что выше, "
                "к нему не относится."]

    first, last = _to_date(meta.get("first_date")), _to_date(meta.get("last_date"))
    years = _safe(meta.get("climatology_years"))
    if first and last:
        seasons = last.year - first.year + 1
        if seasons <= 2 or (years is not None and years <= 2):
            out.append(
                f"История поля короткая — всего {_span_words(first, last)}. Норма, "
                f"построенная на паре сезонов, сама по себе шаткая: год-другой наблюдений "
                f"могут заметно сдвинуть все выводы."
            )

    # Определённая сервисом культура — это предположение, и оно влияет на окно
    # уборки и на эталон продуктивности. Умолчать о нём в документе для банка
    # нельзя: получится, что сервис выдал свою догадку за данные хозяйства.
    detection = meta.get("crop_detection") or {}
    if meta.get("crop_source") == "detected" and detection.get("detected"):
        out.append(
            f"Культура «{detection['detected']}» не была указана — сервис определил её "
            f"сам, по форме сезонной кривой. Проверено это определение на 78 полях: "
            f"конкретная культура угадывается примерно в трёх случаях из четырёх, "
            f"крупная группа (озимые против яровых) — в девяти из десяти. От культуры "
            f"зависят ожидаемое окно уборки и эталон продуктивности, поэтому, если "
            f"культура вам известна, впишите её — расчёт станет точнее."
        )
    if detection.get("conflict"):
        c = detection["conflict"]
        out.append(
            f"Заявленная культура «{c.get('declared')}» расходится с тем, что видно "
            f"в данных: поле развивается как «{c.get('observed_group')}». Чаще всего "
            f"это означает, что сменился севооборот, а карточка поля осталась "
            f"старой. Пока расхождение не выяснено, к выводам про продуктивность "
            f"и сроки стоит относиться с осторожностью."
        )

    clim = meta.get("climatology_source")
    if clim == "crop":
        note = meta.get("climatology_note")
        out.append(
            "Норма для сравнения взята не по самому полю, а усреднённая по культуре. "
            "Она грубее: поле может закономерно отличаться от средней по культуре, "
            "и это будет засчитано как отклонение."
            + (f" {note}" if note and "подобран по форме кривой" in str(note) else "")
        )
    elif clim == "none" or meta.get("has_climatology") is False:
        out.append(
            "Нормы для этого поля построить не удалось — сравнивать текущие значения "
            "не с чем. Всё, что сказано про «хуже обычного», к этому полю неприменимо."
        )

    n_obs = _safe(meta.get("n_obs"))
    if n_obs is not None and first and last:
        days = max((last - first).days, 1)
        per_year = n_obs / max(days / 365.0, 0.5)
        if n_obs < 30:
            out.append(
                f"Снимков мало — всего {int(n_obs)} за весь период. Короткие события "
                f"поле между съёмками могло просто не показать."
            )
        elif per_year < 20:
            out.append(
                f"Съёмка редкая: в среднем около {int(round(per_year))} снимков в год. "
                f"Короткий провал длиной в неделю мог остаться незамеченным."
            )

    restored = sum(1 for p in series if isinstance(p, dict) and p.get("observed") is None)
    if series and restored / len(series) > 0.5:
        share = restored / len(series)
        out.append(
            f"Линия на графике сплошная, но реальные снимки есть не на каждый день: "
            f"{_pct(share)} дней достроено расчётом по соседним датам и норме поля. "
            f"Отдельная точка может ошибаться сильнее, чем период целиком."
        )

    siblings = meta.get("siblings") if isinstance(meta.get("siblings"), dict) else {}
    if siblings and not siblings.get("applied"):
        # Причина из ядра написана для разработчика («бюджет времени исчерпан»).
        # Знакомые формулировки переводим, незнакомую просто не показываем: в
        # отчёте для банка внутренняя телеметрия выглядит как сбой.
        reason = str(siblings.get("reason") or "").lower()
        if "врем" in reason or "бюджет" in reason:
            tail = " На их сбор не хватило времени."
        elif "нет" in reason or "мало" in reason or "не наш" in reason:
            tail = " Подходящих полей рядом не нашлось."
        else:
            tail = ""
        out.append(
            "Поправка по соседним полям не применялась." + tail +
            " Она убирает общую помеху от условий съёмки; без неё отдельные дни "
            "чуть шумнее, но на выводы по периодам это почти не влияет."
        )

    if meta.get("crop_type") in (None, "", "unknown"):
        out.append(
            "Культура поля не указана. Сроки сева и уборки мы не знаем, поэтому резкий "
            "спад в конце сезона можно перепутать с плановой уборкой."
        )

    failures = meta.get("failures")
    if isinstance(failures, list) and failures:
        n = len(failures)
        out.append(
            f"При сборе данных {n} {_plural(n, 'источник ответил', 'источника ответили', 'источников ответили')} "
            f"ошибкой, часть снимков могла не попасть в расчёт."
        )

    fc = meta.get("forecast") if isinstance(meta.get("forecast"), dict) else {}
    stale = int(_safe(fc.get("stale_days"), 0) or 0)
    if stale >= 14:
        out.append(
            f"Свежих снимков нет уже {_days(stale)}: над полем держалась облачность "
            f"или не было пролёта. Чем дольше этот разрыв, тем менее актуален прогноз."
        )

    out.append(
        "И общее: спутник видит зелёную массу, а не урожай. Всё, что в отчёте сказано "
        "про уровень поля, — косвенная оценка по снимкам, а не замер в поле и не "
        "цифра из бункера комбайна."
    )
    return out
