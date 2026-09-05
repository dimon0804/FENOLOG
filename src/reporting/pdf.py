"""Вёрстка клиентского PDF-отчёта по одному полю.

Кому адресован документ. Не эксперту. Его открывает фермер, агроном хозяйства,
оценщик банка или страховой — человек, который не обязан знать, что такое
z-оценка и вегетационный индекс. Поэтому весь текст сюда приходит уже
переведённым на человеческий язык из `plain.py`, а числа показываются графиками
из `charts.py`. Этот модуль отвечает только за то, как всё разложено по бумаге.

Почему reportlab, а не HTML в PDF. Печатный документ должен собираться на
сервере без браузера и без системных библиотек вроде GTK, иначе он не соберётся
в контейнере. Reportlab — чистый Python и даёт полный контроль над полосой.

Почему шрифт берётся у matplotlib. DejaVu Sans лежит внутри пакета matplotlib,
который у нас и так стоит ради графиков. Значит кириллица в PDF будет всегда, на
любой машине, и не придётся надеяться, что в системе найдётся Arial.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Палитра — ровно та же, что в веб-интерфейсе, чтобы отчёт и экран читались
# как один продукт, а не как две разные программы.
# --------------------------------------------------------------------------- #
GREEN = colors.HexColor("#4e9b36")
GREEN_DEEP = colors.HexColor("#2f6b2a")
GREEN_SOFT = colors.HexColor("#eaf3e6")
CRIT = colors.HexColor("#d4342a")
CRIT_SOFT = colors.HexColor("#fbe9e7")
WARN = colors.HexColor("#e08a20")
WARN_SOFT = colors.HexColor("#fdf2e2")
INK = colors.HexColor("#101010")
SOFT = colors.HexColor("#6f6f6f")
LINE = colors.HexColor("#e6e6e6")
CANVAS = colors.HexColor("#f7f7f7")
WHITE = colors.white

# Цвет и заливка по тону вердикта. Тон приходит из plain.py и означает состояние
# поля: всё в порядке, стоит присмотреться, есть проблема, данных не хватает.
TONES = {
    "ok": (GREEN, GREEN_SOFT),
    "watch": (WARN, WARN_SOFT),
    "bad": (CRIT, CRIT_SOFT),
    "nodata": (SOFT, CANVAS),
}

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

FONT = "Fenolog"
FONT_BOLD = "Fenolog-Bold"


def _register_fonts() -> None:
    """Подключает шрифт с кириллицей. Вызывается один раз при первой сборке."""
    if FONT in pdfmetrics.getRegisteredFontNames():
        return
    try:
        import matplotlib

        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont(FONT, str(ttf / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont(FONT_BOLD, str(ttf / "DejaVuSans-Bold.ttf")))
        # Без этой строки тег <b> внутри абзаца молча не работает: reportlab не
        # знает, какое начертание парное к обычному, и подставляет то же самое.
        pdfmetrics.registerFontFamily(
            FONT, normal=FONT, bold=FONT_BOLD, italic=FONT, boldItalic=FONT_BOLD)
    except Exception as exc:  # noqa: BLE001
        # Без кириллического шрифта документ всё равно должен собраться: лучше
        # отчёт со встроенным Helvetica, чем пятисотая ошибка у пользователя.
        log.warning("pdf: не удалось подключить DejaVu (%s), беру Helvetica", exc)
        pdfmetrics.registerFontFamily("Helvetica")


def _styles() -> dict[str, ParagraphStyle]:
    """Набор стилей абзацев. Собирается после регистрации шрифта."""
    base = FONT if FONT in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold = FONT_BOLD if FONT_BOLD in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

    def st(name, size, leading, color=INK, font=base, space_after=0, **kw):
        return ParagraphStyle(
            name, fontName=font, fontSize=size, leading=leading, textColor=color,
            alignment=TA_LEFT, spaceAfter=space_after, **kw,
        )

    return {
        "base": base,
        "bold": bold,
        "h1": st("h1", 22, 27, INK, bold, 4),
        "h2": st("h2", 15, 20, INK, bold, 6),
        "h3": st("h3", 11.5, 15, INK, bold, 3),
        "body": st("body", 9.8, 14.5, INK, space_after=6),
        "small": st("small", 8.6, 12.5, SOFT),
        "tiny": st("tiny", 7.6, 10.5, SOFT),
        "lead": st("lead", 11.5, 17, INK, space_after=6),
        "cover_title": st("cover_title", 30, 35, INK, bold, 6),
        "cover_sub": st("cover_sub", 12, 17, SOFT),
        "verdict_title": st("verdict_title", 17, 22, INK, bold, 4),
        "score_num": st("score_num", 44, 48, INK, bold),
        "white_small": st("white_small", 8.6, 12, WHITE),
    }


class _Doc(BaseDocTemplate):
    """Документ с фирменной шапкой и подвалом на каждой странице."""

    def __init__(self, buf, title: str, subtitle: str, **kw):
        super().__init__(buf, pagesize=A4, title=title, author="Фенолог",
                         leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN + 12 * mm, bottomMargin=MARGIN + 6 * mm, **kw)
        self._title = title
        self._subtitle = subtitle
        frame = Frame(MARGIN, self.bottomMargin, CONTENT_W,
                      PAGE_H - self.topMargin - self.bottomMargin, id="main",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canv, doc) -> None:
        """Шапка и подвал. На титуле шапку не рисуем — там своя обложка."""
        canv.saveState()
        base = FONT if FONT in pdfmetrics.getRegisteredFontNames() else "Helvetica"
        bold = FONT_BOLD if FONT_BOLD in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

        if doc.page > 1:
            canv.setFillColor(GREEN)
            canv.rect(0, PAGE_H - 8 * mm, PAGE_W, 8 * mm, stroke=0, fill=1)
            canv.setFillColor(WHITE)
            canv.setFont(bold, 8.5)
            canv.drawString(MARGIN, PAGE_H - 5.6 * mm, "ФЕНОЛОГ")
            canv.setFont(base, 8.5)
            canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 5.6 * mm, self._subtitle[:70])

        canv.setStrokeColor(LINE)
        canv.setLineWidth(0.5)
        canv.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
        canv.setFillColor(SOFT)
        canv.setFont(base, 7.6)
        canv.drawString(MARGIN, 10 * mm,
                        "Фенолог — мониторинг вегетационной динамики по спутниковым снимкам")
        canv.drawRightString(PAGE_W - MARGIN, 10 * mm, f"с. {doc.page}")
        canv.restoreState()


def _png(data: bytes, width_mm_: float, height_mm_: float) -> Image | Spacer:
    """Готовая картинка в поток. Пустые данные не должны рвать сборку."""
    if not data:
        return Spacer(1, 2 * mm)
    img = Image(io.BytesIO(data))
    # Держим пропорции исходника, но вписываем в отведённую ширину полосы.
    ratio = img.imageHeight / max(img.imageWidth, 1)
    img.drawWidth = width_mm_ * mm
    img.drawHeight = min(width_mm_ * ratio, height_mm_) * mm
    img.hAlign = "LEFT"
    return img


def _card_inner_width(pad: float = 5 * mm) -> float:
    """Сколько места остаётся под содержимое внутри карточки.

    Три миллиметра съедает цветная полоса слева, ещё по `pad` — отступы. Считать
    это на глаз в каждом месте нельзя: таблица шире полосы молча вылезает за
    правый край карточки, и на глаз в коде это не видно, только на бумаге.
    """
    return CONTENT_W - 3 * mm - 2 * pad


def _card(flowables: list, accent, background, s: dict, pad: float = 5 * mm) -> Table:
    """Карточка с цветной полосой слева. Основной приём вёрстки этого отчёта."""
    inner = Table([[flowables]], colWidths=[CONTENT_W - 3 * mm - 2 * pad])
    inner.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    outer = Table([["", inner]], colWidths=[3 * mm, CONTENT_W - 3 * mm])
    outer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), accent),
        ("BACKGROUND", (1, 0), (1, 0), background),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING", (1, 0), (1, 0), pad),
        ("RIGHTPADDING", (1, 0), (1, 0), pad),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
    ]))
    return outer


def _kv_table(rows: list[tuple[str, str]], s: dict, col: float = 55 * mm,
              total: float | None = None) -> Table:
    """Таблица «подпись — значение». Подпись серая, значение чёрное.

    `total` — доступная ширина. Внутри карточки полоса уже, чем полоса страницы,
    и без явной ширины разделительные линии вылезают за правый край карточки.
    """
    total = CONTENT_W if total is None else total
    data = [[Paragraph(str(k), s["small"]), Paragraph(str(v), s["body"])] for k, v in rows]
    t = Table(data, colWidths=[col, total - col])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
    ]))
    return t


def _section(title: str, image, s: dict, intro: str | None = None,
             caption: str | None = None) -> KeepTogether:
    """Заголовок вместе со своим графиком одним неразрывным блоком.

    Иначе reportlab разрывает их по границе полосы, и читатель видит заголовок
    без картинки внизу страницы — самый заметный дефект вёрстки из возможных.
    """
    block: list = [Paragraph(title, s["h2"])]
    if intro:
        block.append(Paragraph(intro, s["body"]))
    block.append(image)
    if caption:
        block.append(Paragraph(caption, s["small"]))
    return KeepTogether(block)


def build_pdf(result: dict, polygon: dict | None = None) -> bytes:
    """Собирает клиентский отчёт по результату анализа. Возвращает PDF байтами.

    result  — результат анализа (см. AnalysisResult, сериализованный в JSON)
    polygon — запись участка из хранилища: имя, площадь, культура, центр

    Функция обязана отдать документ при любых входных данных: если анализ пустой,
    отчёт всё равно соберётся и честно скажет, что данных не хватило.
    """
    from src.reporting import charts, plain

    _register_fonts()
    s = _styles()
    polygon = polygon or {}
    meta = result.get("meta") or {}

    name = polygon.get("name") or result.get("polygon_id") or "Участок"
    today = datetime.now().strftime("%d.%m.%Y")

    buf = io.BytesIO()
    doc = _Doc(buf, title=f"Фенолог — отчёт по участку «{name}»", subtitle=str(name))
    story: list = []

    def h2(text: str) -> None:
        story.append(Paragraph(text, s["h2"]))

    # ------------------------------------------------------------------ #
    # Страница 1. Обложка и главный вывод
    # ------------------------------------------------------------------ #
    v = plain.verdict(result)
    accent, background = TONES.get(v.get("tone", "nodata"), TONES["nodata"])

    story.append(Paragraph("ФЕНОЛОГ", ParagraphStyle(
        "brand", fontName=s["bold"], fontSize=11, leading=14,
        textColor=GREEN, spaceAfter=2)))
    story.append(Paragraph("Отчёт о состоянии поля", s["cover_title"]))
    story.append(Paragraph(str(name), s["cover_sub"]))
    story.append(Spacer(1, 8 * mm))

    story.append(_card([
        Paragraph(v.get("title", "Состояние поля"), s["verdict_title"]),
        Paragraph(v.get("text", ""), s["body"]),
    ], accent, background, s))
    story.append(Spacer(1, 7 * mm))

    h2("Участок")
    story.append(_kv_table(plain.field_summary(result, polygon), s))
    story.append(Spacer(1, 7 * mm))

    # Короткий путь, когда считать нечего. Без него документ раскладывался на
    # девять полос, восемь из которых почти пустые: заголовки разделов,
    # заглушки графиков и словарь терминов к отсутствующим данным. Читателю от
    # такого документа хуже, чем от одностраничного, — он выглядит сломанным.
    if not (result.get("series") or []):
        story.append(_card([
            Paragraph("Почему отчёт короткий", s["h3"]),
            Paragraph(
                "По этому участку нет ни одного пригодного спутникового наблюдения, "
                "поэтому ни ряд, ни норму, ни периоды угнетения построить не на чем. "
                "Чаще всего причина одна из трёх: контур слишком мал для разрешения "
                "снимка, выбранный период закрыт сплошной облачностью, либо источник "
                "снимков был недоступен в момент сбора.", s["body"]),
            Paragraph(
                "Что делать: запустите разбор ещё раз чуть позже или расширьте период "
                "наблюдений. Если контур меньше половины гектара, снимки Sentinel-2 по "
                "нему усредняются по нескольким пикселям и результат будет ненадёжен "
                "даже при удачном сборе.", s["body"]),
        ], SOFT, CANVAS, s))
        story.append(Spacer(1, 6 * mm))
        h2("Откуда сервис берёт данные")
        story.append(_kv_table(plain.data_sources(result), s, col=60 * mm))
        doc.build(story)
        return buf.getvalue()

    story.append(PageBreak())
    story.append(_section(
        "Как поле развивалось в последнем сезоне",
        _png(charts.chart_last_season(result, height_mm=95), 170, 95), s,
        caption="Сплошная линия — сколько живой зелёной массы спутник видел на поле. "
                "Пунктир и светлая полоса — как это же поле выглядело в те же дни в "
                "прошлые годы, то есть привычная для него норма. Пока линия идёт "
                "внутри полосы, поле развивается обычно."))
    story.append(Spacer(1, 6 * mm))

    # ------------------------------------------------------------------ #
    # Вся история наблюдений
    # ------------------------------------------------------------------ #
    story.append(_section(
        "Вся история наблюдений",
        _png(charts.chart_series(result, height_mm=92), 170, 92), s,
        intro="Здесь видно поведение поля за все сезоны, которые удалось собрать. "
              "Каждый год повторяется одна и та же волна: рост весной, максимум в "
              "начале лета, спад после уборки. Отклонения от этой волны и есть то, "
              "что сервис ищет."))
    story.append(Spacer(1, 5 * mm))

    h2("Что означают события на графике")
    for term, definition in plain.glossary():
        story.append(Paragraph(f"<b>{term}</b> — {definition}", s["body"]))

    story.append(PageBreak())

    # ------------------------------------------------------------------ #
    # Страница 3. События сезона
    # ------------------------------------------------------------------ #
    anomalies = result.get("anomalies") or []
    h2("Что происходило с полем")
    if not anomalies:
        story.append(_card([
            Paragraph("Отклонений не найдено", s["h3"]),
            Paragraph("За весь период наблюдений поле не выходило за пределы своей "
                      "обычной нормы. Это хороший результат.", s["body"]),
        ], GREEN, GREEN_SOFT, s))
    else:
        story.append(Paragraph(
            f"Найдено периодов, когда поле развивалось хуже обычного: {len(anomalies)}. "
            "Для каждого сервис пытается назвать причину по данным о погоде — "
            "и честно говорит, когда уверенности мало.", s["body"]))
        story.append(Spacer(1, 2 * mm))

        for a in anomalies:
            card = plain.anomaly_card(a)
            acc, bg = TONES.get(card.get("tone", "watch"), TONES["watch"])
            # Шапка карточки строкой, а не таблицей: периодов бывает восемь и
            # больше, и таблица из четырёх строк на каждый раздувала раздел на
            # три полосы. Всё то же самое читается одной строкой.
            facts = " · ".join(x for x in (
                card.get("duration"), card.get("depth")) if x)
            body = [
                Paragraph(card.get("when", ""), s["h3"]),
                Paragraph(facts, s["small"]),
                Spacer(1, 2 * mm),
                Paragraph(card.get("text", ""), s["body"]),
                Paragraph(
                    f"<b>Причина:</b> {card.get('cause', '')} — {card.get('confidence', '')}",
                    s["small"]),
            ]
            if card.get("advice"):
                body.append(Spacer(1, 1.5 * mm))
                body.append(Paragraph(f"<b>Что имеет смысл сделать.</b> {card['advice']}",
                                      s["small"]))
            story.append(KeepTogether(_card(body, acc, bg, s, pad=4 * mm)))
            story.append(Spacer(1, 3.5 * mm))

    story.append(PageBreak())

    # ------------------------------------------------------------------ #
    # Страница 4. Оценка поля как объекта риска
    # ------------------------------------------------------------------ #
    sc = plain.score_block(result)
    h2("Оценка поля")
    story.append(Paragraph(
        "Этот раздел нужен, когда поле рассматривают как объект: при страховании, "
        "залоге или покупке. Балл собран из четырёх свойств и снижен там, где данных "
        "мало — завышать оценку из-за короткой истории было бы нечестно.", s["body"]))

    score_value = sc.get("score")
    has_score = isinstance(score_value, (int, float))
    grade_cell = Table([
        [Paragraph(str(score_value) if has_score else "—", s["score_num"])],
        [Paragraph(f"класс {sc.get('grade')}" if sc.get("grade") else "балл не выставлен",
                   s["small"])],
    ], colWidths=[32 * mm])
    grade_cell.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    inner = _card_inner_width(5 * mm)
    head = Table([[grade_cell, [
        Paragraph(sc.get("headline", ""), s["h3"]),
        Paragraph(sc.get("text", ""), s["body"]),
    ]]], colWidths=[36 * mm, inner - 36 * mm])
    head.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    # Полосу красим по самому баллу, а не по тону вердикта: рядом с числом 53
    # красная полоса из-за сегодняшнего состояния поля читалась бы как оценка
    # этого числа, хотя балл считается за все сезоны сразу.
    if not has_score:
        score_accent = SOFT
    elif score_value >= 70:
        score_accent = GREEN
    elif score_value >= 50:
        score_accent = WARN
    else:
        score_accent = CRIT
    story.append(_card([head], score_accent, CANVAS, s, pad=5 * mm))
    story.append(Spacer(1, 5 * mm))

    if has_score:
        story.append(_section(
            "Из чего собран балл",
            _png(charts.chart_score(result, width_mm=170, height_mm=52), 170, 52), s))
        for label, value, hint in sc.get("components", []):
            story.append(Paragraph(f"<b>{label} — {value} из 100.</b> {hint}", s["small"]))
        story.append(Spacer(1, 5 * mm))

    # Таблица по сезонам приходит из ядра отдельно от балла и бывает заполнена,
    # даже когда балл выставить не удалось. Но если пуста и она — рисовать
    # заглушку не нужно: словами это уже сказано в карточке выше.
    seasons = ((meta.get("score") or {}).get("seasons") or [])
    if seasons:
        # Разрыв нужен только когда выше есть блок компонент балла: иначе раздел
        # начинался бы с пустой полосы.
        if has_score:
            story.append(PageBreak())

        story.append(_section(
            "Как поле вело себя по сезонам",
            _png(charts.chart_seasons(result, height_mm=78), 170, 78), s,
            intro="Столбик — сколько зелёной массы поле набрало за сезон целиком. "
                  "Это не урожай в центнерах, а его косвенная мера: чем выше "
                  "столбик, тем больше поле работало за лето."))

    if sc.get("caveats"):
        story.append(Spacer(1, 5 * mm))
        story.append(_card(
            [Paragraph("Что снижает точность этой оценки", s["h3"])]
            + [Paragraph(f"— {c}", s["small"]) for c in sc["caveats"]],
            SOFT, CANVAS, s, pad=4 * mm))

    story.append(PageBreak())

    # ------------------------------------------------------------------ #
    # Страница 5. Прогноз
    # ------------------------------------------------------------------ #
    fb = plain.forecast_block(result)
    f_acc, f_bg = TONES.get(fb.get("tone", "nodata"), TONES["nodata"])
    h2("Что будет дальше")
    story.append(_card([
        Paragraph(fb.get("headline", "Прогноз"), s["h3"]),
        Paragraph(fb.get("text", ""), s["body"]),
        Paragraph(fb.get("confidence_words", ""), s["small"]),
    ], f_acc, f_bg, s, pad=4 * mm))
    story.append(Spacer(1, 5 * mm))
    story.append(_section(
        "Как это выглядит на графике",
        _png(charts.chart_forecast(result, height_mm=80), 170, 80), s,
        caption="Затенённая область — не ошибка расчёта, а честный разброс: внутри "
                "неё значение окажется с высокой вероятностью. Чем дальше от "
                "сегодняшнего дня, тем шире область — так и должно быть."))

    story.append(PageBreak())

    # ------------------------------------------------------------------ #
    # Страница 6. Откуда взялись выводы
    # ------------------------------------------------------------------ #
    h2("Как это считается")
    story.append(Paragraph(
        "Ниже — весь путь от снимка до вывода, без формул. Мы показываем его, "
        "чтобы отчёту можно было верить: каждый шаг проверяем и можем повторить.",
        s["body"]))
    for step_title, step_text in plain.how_it_works():
        story.append(Paragraph(step_title, s["h3"]))
        story.append(Paragraph(step_text, s["body"]))

    story.append(Spacer(1, 4 * mm))
    h2("Откуда берутся данные")
    story.append(_kv_table(plain.data_sources(result), s, col=60 * mm))

    limits = plain.caveats(result)
    if limits:
        story.append(Spacer(1, 5 * mm))
        h2("Границы применимости")
        story.append(Paragraph(
            "Мы обязаны сказать, чего этот отчёт не может. Ни один из пунктов ниже "
            "не отменяет выводов — но их стоит учитывать при решении.", s["body"]))
        for c in limits:
            story.append(Paragraph(f"— {c}", s["body"]))

    story.append(Spacer(1, 6 * mm))
    story.append(_card([
        Paragraph(f"Отчёт сформирован {today}", s["h3"]),
        Paragraph(
            "Документ носит информационный характер и построен на открытых спутниковых "
            "и метеорологических данных. Он не заменяет обследование поля на месте и не "
            "является отчётом об оценке в смысле законодательства об оценочной "
            "деятельности.", s["small"]),
    ], SOFT, CANVAS, s, pad=4 * mm))

    doc.build(story)
    return buf.getvalue()


def build_pdf_safe(result: dict, polygon: dict | None = None) -> bytes:
    """То же, но никогда не падает: при сбое отдаёт короткий документ с извинением.

    Отчёт скачивают из интерфейса одной кнопкой. Пятисотая ошибка вместо файла —
    худшее, что там может произойти, поэтому сбой вёрстки должен деградировать
    в честный однополосный документ, а не в отказ.
    """
    try:
        return build_pdf(result, polygon)
    except Exception as exc:  # noqa: BLE001
        log.exception("pdf: сборка отчёта не удалась")
        _register_fonts()
        s = _styles()
        buf = io.BytesIO()
        doc = _Doc(buf, title="Фенолог — отчёт", subtitle="отчёт")
        doc.build([
            Paragraph("Отчёт сформировать не удалось", s["h1"]),
            Paragraph(
                "Данные по участку собраны, но при сборке документа произошла ошибка. "
                "Результаты анализа доступны в интерфейсе сервиса. "
                f"Техническая причина: {type(exc).__name__}.", s["body"]),
        ])
        return buf.getvalue()
