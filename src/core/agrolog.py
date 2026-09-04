"""Журнал полевых работ и связь агротехники с найденными просадками NDVI.

Зачем модуль нужен. У сервиса есть версия причины «причина не погодная»: все
погодные пороги проверены, ни один не сработал, значит смотреть надо агротехнику.
На этом объяснение обрывалось. Журнал работ замыкает его: агроном вводит, что и
когда он делал на поле, а сервис сам связывает работу с просадкой — «гербицид
внесён за девять дней до начала просадки, вероятна фитотоксичность» или
«в периоде стоит уборка, падение плановое».

Что здесь принципиально. Правила связывания — не догадки, а агрономия:

  * системный гербицид или инсектицид проявляется ожогом и торможением роста
    на третий-четырнадцатый день после обработки, раньше — не успевает,
    позже — уже отпустило или причина другая;
  * уборка и обработка почвы роняют индекс по плану: это не беда, а работа,
    и тревогу с такого периода надо снимать, а не поднимать;
  * подкормка отзывается приростом биомассы за одну-две недели. Если работа
    была, а отклика нет — это сам по себе диагноз: влаги не хватило для
    усвоения либо на поле есть помеха посерьёзнее;
  * полив во время «засухи» опровергает версию дефицита влаги.

Молчание тоже часть логики: пустой журнал или отсутствие работ рядом с периодом
дают пустую строку и нулевую поправку. Дописывать к объяснению нечего — значит
дописывать не надо.

Модуль наружу не торчит: его вызывает ядро при сборке объяснения причины.
Ввод — ручной, а значит грязный: разбор терпит неполные строки, чужие названия
видов работ и три формата даты, а на битой строке не падает, а пропускает её
с сообщением.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from src.contracts import (
    CAUSE_ABRUPT,
    CAUSE_DROUGHT,
    CAUSE_UNKNOWN,
)
from src.core.anomaly import (
    HARVEST_WINDOW_BY_CROP,
    HARVEST_WINDOW_DEFAULT,
    CAUSE_HARVEST,
    CAUSE_NON_WEATHER,
)

LOG = logging.getLogger(__name__)

# --- Справочник видов работ --------------------------------------------------
# Значения — по-русски и в том виде, в каком их увидит агроном в выпадающем
# списке. Коды латиницей не заводим специально: журнал заполняет человек, а не
# смежная система, и лишний слой перевода тут только мешает.
KIND_SOWING = "сев"
KIND_FERTILIZER = "удобрение"
KIND_HERBICIDE = "гербицид"
KIND_FUNGICIDE = "фунгицид"
KIND_INSECTICIDE = "инсектицид"
KIND_IRRIGATION = "полив"
KIND_TILLAGE = "обработка почвы"
KIND_HARVEST = "уборка"
KIND_OTHER = "иное"

KNOWN_KINDS = (
    KIND_SOWING,
    KIND_FERTILIZER,
    KIND_HERBICIDE,
    KIND_FUNGICIDE,
    KIND_INSECTICIDE,
    KIND_IRRIGATION,
    KIND_TILLAGE,
    KIND_HARVEST,
    KIND_OTHER,
)

# Как узнать вид работы в том, что написал человек. Список, а не словарь, потому
# что порядок проверки важен: «предпосевная культивация» содержит и «посев»,
# и «культивац», а по смыслу это обработка почвы, поэтому почва стоит выше сева.
# «Обработка от вредителей» проверяется раньше «обработки почвы» по той же причине.
_KIND_HINTS: list[tuple[str, tuple[str, ...]]] = [
    (KIND_IRRIGATION, ("полив", "орошен", "дождеван", "irrigation")),
    (KIND_HARVEST, ("уборк", "жатв", "косов", "скашив", "обмолот", "harvest")),
    (KIND_HERBICIDE, ("гербицид", "herbicid", "прополк")),
    (KIND_FUNGICIDE, ("фунгицид", "fungicid", "протравлив")),
    (KIND_INSECTICIDE, ("инсектицид", "insecticid", "вредител", "клоп", "тля")),
    (
        KIND_FERTILIZER,
        (
            "удобрен", "подкорм", "селитр", "карбамид", "аммофос", "кас-", "кас ",
            "нитроаммофос", "сульфоаммофос", "азот", "fertiliz",
        ),
    ),
    (
        KIND_TILLAGE,
        (
            "обработка почв", "вспашк", "пахот", "дисков", "культивац", "боронов",
            "лущен", "глубокорыхл", "прикатыван", "tillage",
        ),
    ),
    (KIND_SOWING, ("сев", "посев", "посадк", "sowing")),
]

# --- Окна связывания работы с просадкой --------------------------------------
# Фитотоксичность и ожог от пестицида видны не сразу: первые двое суток растение
# ещё «держит удар», а после двух недель угнетение от разрешённой дозы уже снято.
# Всё, что раньше или позже, связывать с обработкой нельзя — это будет совпадение.
PHYTOTOX_MIN_LAG_DAYS = 3
PHYTOTOX_MAX_LAG_DAYS = 14
# Подкормка отзывается приростом биомассы за одну-две недели, максимум три:
# азот должен раствориться, дойти до корня и уйти в лист. Раньше недели отклика
# не ждут и его отсутствие ни о чём не говорит, позже трёх недель эффект уже
# размазан по другим факторам.
FERT_MIN_LAG_DAYS = 7
FERT_MAX_LAG_DAYS = 21
# Полив засчитываем против версии засухи, если он был внутри периода или прямо
# перед ним: запас влаги в почве после полива держится около недели.
IRRIGATION_LAG_DAYS = 7
# Насколько далеко назад от начала периода вообще имеет смысл смотреть журнал.
DEFAULT_LOOKBACK_DAYS = 30

# --- Поправки к уверенности --------------------------------------------------
# Знак поправки читается однозначно: журнал либо подтверждает версию, либо её
# снимает. Величина — по силе свидетельства: прямая запись о работе внутри
# периода весит больше, чем совпадение сроков с точностью до недели.
#
# Уборка в журнале — самое сильное свидетельство в модуле: это не догадка по
# форме кривой, а факт, введённый человеком. Поэтому версия «уборка» получает
# крупный плюс, а тревожная версия «резкое событие» — ещё более крупный минус:
# ложная тревога стоит дороже пропущенной, лента событий, полная плановых работ,
# обесценивает сервис целиком.
CONF_HARVEST_CONFIRM = 0.25       # версия «уборка» подтверждена журналом
CONF_ABRUPT_TO_PLANNED = -0.35    # «резкое событие» оказалось плановой уборкой
CONF_HARVEST_OVER_WEATHER = -0.20  # погодная версия проиграла записи об уборке
CONF_TILLAGE_ABRUPT = -0.30       # падение объясняется обработкой почвы
CONF_TILLAGE_OTHER = -0.15
# Фитотоксичность — единственное правило с заметным плюсом к «не погодной»
# версии: она была утверждением без содержания («ищите на поле»), а с журналом
# у неё появляется конкретный виновник и конкретное действие.
CONF_PHYTOTOX_NON_WEATHER = 0.15
CONF_PHYTOTOX_ABRUPT = 0.10       # обработка объясняет и одномоментность падения
# Подкормка без отклика — свидетельство слабое и косвенное: оно ничего не
# доказывает, а только поддерживает уже выбранную версию. Отсюда символические
# пять сотых: текст здесь ценнее поправки.
CONF_FERT_NO_RESPONSE = 0.05
# Полив против засухи — свидетельство прямое и опровергающее, поэтому вес
# сопоставим с уборкой, только со знаком минус.
CONF_IRRIGATION_VS_DROUGHT = -0.25
# Уверенность не должна ни обнуляться, ни доходить до единицы: и то и другое —
# обещание, которого сервис по одному журналу дать не может.
CONF_MIN, CONF_MAX = 0.05, 0.95

# Форматы дат, которые встречаются в ручном вводе: ISO из формы, точки из Excel,
# слэши из выгрузок. Двузначный год — последним, чтобы не перехватывать чужое.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%Y.%m.%d",
    "%d.%m.%y",
)

# Названия колонок, которые считаем одним и тем же полем. Пользователь заводит
# таблицу сам, поэтому «дата», «начало» и «date_from» должны работать одинаково.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "date_from": ("date_from", "date", "start", "дата", "дата_начала", "начало", "с"),
    "date_to": ("date_to", "end", "finish", "дата_окончания", "окончание", "конец", "по"),
    "kind": ("kind", "type", "вид", "тип", "вид_работ", "вид_работы", "операция"),
    "title": ("title", "name", "work", "название", "работа", "что", "препарат", "описание"),
    "note": ("note", "comment", "комментарий", "примечание", "заметка"),
}


@dataclass
class AgroEvent:
    """Одна запись из журнала работ по полю."""

    date_from: date
    date_to: date | None          # None = однодневная работа
    kind: str                     # см. справочник видов работ выше
    title: str                    # что именно: «аммиачная селитра 100 кг/га»
    note: str = ""                # свободный комментарий агронома

    @property
    def end(self) -> date:
        """Дата окончания работы; для однодневной совпадает с началом."""
        return self.date_to or self.date_from


# --- Разбор журнала ----------------------------------------------------------


def normalize_kind(raw: str | None) -> str:
    """Приводит написанное человеком к справочнику видов работ.

    Неизвестное — не ошибка: работа всё равно попадёт в журнал как KIND_OTHER
    и будет видна пользователю. Потерять запись хуже, чем не суметь её
    классифицировать.
    """
    if not raw:
        return KIND_OTHER
    text = str(raw).strip().lower().replace("ё", "е")
    if text in KNOWN_KINDS:
        return text
    for kind, hints in _KIND_HINTS:
        if any(hint in text for hint in hints):
            return kind
    return KIND_OTHER


def _parse_date(value) -> date | None:
    """Дата из чего угодно разумного; None, если разобрать не удалось."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    # ISO с временем ("2025-04-12T00:00:00") — самый частый гость из выгрузок
    head = text.split("T")[0].split(" ")[0]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(head, fmt).date()
        except ValueError:
            continue
    return None


def _pick(row: dict, field: str) -> object | None:
    """Значение поля по любому из его синонимов в заголовке таблицы."""
    for alias in _FIELD_ALIASES[field]:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _row_to_event(raw_row: dict, where: str) -> AgroEvent | None:
    """Строка таблицы или словаря -> AgroEvent. Битая строка -> None и сообщение."""
    # Заголовки нормализуем: регистр и пробелы в названиях колонок — не смысловая
    # разница, а особенность того, кто набирал таблицу.
    row = {
        str(k).strip().lower().replace(" ", "_"): v
        for k, v in raw_row.items()
        if k is not None
    }
    date_from = _parse_date(_pick(row, "date_from"))
    if date_from is None:
        LOG.warning("Журнал работ, %s: пропущена запись без разборчивой даты: %r", where, raw_row)
        return None
    date_to = _parse_date(_pick(row, "date_to"))
    if date_to is not None and date_to < date_from:
        # Перепутанные местами границы — типичная опечатка ручного ввода.
        # Молча меняем местами: смысл записи от этого не страдает.
        LOG.warning("Журнал работ, %s: даты переставлены местами, поправлено: %r", where, raw_row)
        date_from, date_to = date_to, date_from
    if date_to == date_from:
        date_to = None

    title_raw = _pick(row, "title")
    kind_raw = _pick(row, "kind")
    # Вид работ пытаемся угадать и по названию тоже: агроном чаще пишет
    # «внесли селитру», чем аккуратно выбирает вид из справочника.
    kind = normalize_kind(kind_raw)
    if kind == KIND_OTHER and title_raw:
        kind = normalize_kind(title_raw)
    if kind_raw and kind == KIND_OTHER:
        LOG.warning("Журнал работ, %s: вид работ %r неизвестен, записан как «иное»", where, kind_raw)

    note_raw = _pick(row, "note")
    return AgroEvent(
        date_from=date_from,
        date_to=date_to,
        kind=kind,
        title=str(title_raw).strip() if title_raw else str(kind_raw or kind).strip(),
        note=str(note_raw).strip() if note_raw else "",
    )


def _read_csv_rows(path: Path) -> list[dict]:
    """Строки CSV. Разделитель определяем сами: Excel в русской локали ставит «;»."""
    text = path.read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text.splitlines() else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


def load_events(source) -> list[AgroEvent]:
    """Читает журнал: путь к CSV или JSON, либо готовый список словарей.

    Терпимо относится к неполным данным: нет даты окончания — однодневная работа,
    неизвестный вид — KIND_OTHER, битая строка — пропускается с сообщением, а не
    роняет разбор. Пользователь вводит это руками, значит ошибётся.
    """
    if source is None:
        return []

    rows: list[dict] = []
    where = "список"

    if isinstance(source, (list, tuple)):
        events: list[AgroEvent] = []
        for i, item in enumerate(source):
            if isinstance(item, AgroEvent):
                events.append(item)
            elif isinstance(item, dict):
                ev = _row_to_event(item, "запись %d" % (i + 1))
                if ev is not None:
                    events.append(ev)
            else:
                LOG.warning("Журнал работ: запись %d непонятного типа, пропущена: %r", i + 1, item)
        return sorted(events, key=lambda e: (e.date_from, e.kind))

    if isinstance(source, (str, Path)):
        text = str(source).strip()
        path = Path(source)
        if path.exists():
            where = path.name
            try:
                if path.suffix.lower() == ".json":
                    rows = _as_rows(json.loads(path.read_text(encoding="utf-8-sig")))
                else:
                    rows = _read_csv_rows(path)
            except Exception as exc:  # разбор файла целиком не должен ронять анализ
                LOG.warning("Журнал работ: файл %s прочитать не удалось (%s), журнал пуст", path, exc)
                return []
        elif text.startswith("[") or text.startswith("{"):
            # Иногда журнал приходит не файлом, а телом запроса из веб-формы
            where = "JSON-строка"
            try:
                rows = _as_rows(json.loads(text))
            except Exception as exc:
                LOG.warning("Журнал работ: JSON разобрать не удалось (%s), журнал пуст", exc)
                return []
        else:
            LOG.warning("Журнал работ: файла %s нет, журнал считаем пустым", source)
            return []
    else:
        LOG.warning("Журнал работ: источник типа %s не поддержан", type(source).__name__)
        return []

    events = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            LOG.warning("Журнал работ, %s: строка %d не таблица, пропущена", where, i + 1)
            continue
        ev = _row_to_event(row, "%s, строка %d" % (where, i + 1))
        if ev is not None:
            events.append(ev)
    return sorted(events, key=lambda e: (e.date_from, e.kind))


def _as_rows(payload) -> list[dict]:
    """JSON бывает и списком записей, и объектом с ключом events — принимаем оба."""
    if isinstance(payload, dict):
        for key in ("events", "journal", "работы", "записи"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


# --- Выборка работ рядом с периодом ------------------------------------------


def events_near(
    events: list[AgroEvent],
    start: date,
    end: date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> list[AgroEvent]:
    """Работы, которые могли повлиять на период: попавшие в него или бывшие незадолго до.

    Назад смотрим шире, чем вперёд: работа после конца просадки объяснить её
    не может (причина не бывает позже следствия), а работа за три недели до
    начала — вполне.
    """
    if not events:
        return []
    window_start = start - timedelta(days=max(0, lookback_days))
    near = [e for e in events if e.end >= window_start and e.date_from <= end]
    return sorted(near, key=lambda e: (e.date_from, e.kind))


def _overlaps(event: AgroEvent, start: date, end: date) -> bool:
    """Работа пересекается с периодом хотя бы одним днём."""
    return event.date_from <= end and event.end >= start


def _lag_before(event: AgroEvent, start: date) -> int | None:
    """Сколько дней прошло от конца работы до начала просадки; None — если работа не раньше."""
    if event.end >= start:
        return None
    return (start - event.end).days


def _fmt(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def _fmt_span(event: AgroEvent) -> str:
    """Дата работы человеческим языком: одна дата или интервал."""
    if event.date_to and event.date_to != event.date_from:
        return "с %s по %s" % (_fmt(event.date_from), _fmt(event.date_to))
    return _fmt(event.date_from)


def _label(event: AgroEvent) -> str:
    """Как назвать работу в тексте: название, если оно есть, иначе вид работ.

    Вид работ к названию не приклеиваем: он и так назван в самой фразе
    («стоит уборка», «проведена обработка»), а «уборка: уборка прямым
    комбайнированием» читается как машинный вывод, а не как речь.
    """
    title = (event.title or "").strip()
    return title if title else event.kind


def _harvest_timing_note(event: AgroEvent, crop_type: str | None) -> str:
    """Сверяет дату уборки из журнала с характерным окном уборки культуры.

    Единственное место, где культура работает не на формулировку, а на смысл:
    уборка на месяц раньше срока — это либо опечатка в журнале, либо аварийная
    уборка погибшего посева, и в обоих случаях об этом стоит сказать вслух.
    """
    key = (crop_type or "").strip().lower()
    lo, hi = HARVEST_WINDOW_BY_CROP.get(key, HARVEST_WINDOW_DEFAULT)
    doy = event.date_from.timetuple().tm_yday
    if lo <= doy <= hi:
        return ""
    crop_name = "культуры «%s»" % crop_type if crop_type else "этого региона"
    return (
        " Правда, день года %d выходит за характерное окно уборки %s (%d-%d): "
        "проверьте дату в журнале, а если она верна — это могла быть аварийная "
        "уборка." % (doy, crop_name, lo, hi)
    )


def _brief(event: AgroEvent) -> dict:
    """Компактное представление работы для словаря свидетельств."""
    out = {"date_from": event.date_from.isoformat(), "kind": event.kind, "title": event.title}
    if event.date_to:
        out["date_to"] = event.date_to.isoformat()
    if event.note:
        out["note"] = event.note
    return out


# --- Дополнение к объяснению причины -----------------------------------------


def explain_with_agro(
    cause: str,
    confidence: float,
    start: date,
    end: date,
    events: list[AgroEvent],
    crop_type: str | None = None,
) -> tuple[str, float, dict]:
    """Дополнение к объяснению причины с учётом журнала работ.

    Возвращает (текст-дополнение или пустая строка, поправка к уверенности, свидетельства).
    Текст должен читаться как продолжение уже готовой фразы, а не дублировать её.

    Правила проверяются не подряд, а по силе свидетельства. Запись об уборке или
    обработке почвы внутри периода объясняет просадку целиком и делает разговор
    о фитотоксичности и подкормках бессмысленным, поэтому она обрывает разбор.
    Остальные правила складываются: гербицид за неделю до просадки и подкормка
    без отклика — это два разных наблюдения, и агроному нужны оба.

    `crop_type` пока участвует только в формулировке (уборка «озимой пшеницы»
    читается лучше, чем просто «уборка»). Место под культуро-зависимые окна
    отклика оставлено осознанно: сроки проявления фитотоксичности у пропашных
    и зерновых различаются, но проверить это на выданных данных нечем.
    """
    if not events:
        # Журнал пуст — молчим. Пустая строка и ноль означают «ядро, ничего
        # не меняй»: объяснение остаётся ровно таким, каким его собрала погода.
        return "", 0.0, {}

    near = events_near(events, start, end, DEFAULT_LOOKBACK_DAYS)
    evidence: dict = {"agro_events_total": len(events), "agro_events_near": len(near)}
    if not near:
        return "", 0.0, evidence

    parts: list[str] = []
    rules: list[str] = []
    delta = 0.0

    inside = [e for e in near if _overlaps(e, start, end)]

    # --- Правило 1. Уборка внутри периода: тревога снимается полностью ---
    harvest = next((e for e in inside if e.kind == KIND_HARVEST), None)
    if harvest is not None:
        timing = _harvest_timing_note(harvest, crop_type)
        # Уборка не в срок — свидетельство слабее: либо в журнале опечатка, либо
        # посев убирали аварийно, и тогда тревогу снимать целиком нельзя.
        # Поправку в этом случае урезаем, а не отменяем: запись всё-таки есть.
        off_window = 0.6 if timing else 1.0
        if cause == CAUSE_HARVEST:
            delta = CONF_HARVEST_CONFIRM
            text = (
                "Журнал работ это подтверждает: %s, %s. Версия опирается уже не на форму "
                "кривой, а на запись агронома.%s"
                % (_label(harvest), _fmt_span(harvest), timing)
            )
        elif cause == CAUSE_ABRUPT:
            delta = CONF_ABRUPT_TO_PLANNED
            if timing:
                # Уборка есть, но не в срок: падение объяснено, тревога — нет.
                text = (
                    "Падение объясняется записью в журнале работ: на эти дни стоит "
                    "уборка — %s, %s.%s"
                    % (_label(harvest), _fmt_span(harvest), timing)
                )
            else:
                text = (
                    "Но тревоги период не требует: в журнале работ на эти дни стоит "
                    "уборка — %s, %s. Падение индекса подтверждено журналом работ: это "
                    "плановая работа, а не повреждение посева."
                    % (_label(harvest), _fmt_span(harvest))
                )
        else:
            delta = CONF_HARVEST_OVER_WEATHER
            text = (
                "При этом внутри периода в журнале работ стоит уборка — %s, %s. Скорее "
                "всего, индекс упал именно из-за неё, а не по погодной причине: это "
                "плановая работа.%s" % (_label(harvest), _fmt_span(harvest), timing)
            )
        evidence["agro_rules"] = ["harvest_in_period"]
        evidence["agro_events"] = [_brief(harvest)]
        # Подсказка ядру: версию причины разумно переписать на «уборку».
        # Саму причину модуль не меняет — это зона attribute_cause.
        evidence["agro_suggest_cause"] = CAUSE_HARVEST
        evidence["agro_harvest_in_crop_window"] = not timing
        evidence["agro_conf_delta"] = _clamp_delta(confidence, delta * off_window)
        return text, evidence["agro_conf_delta"], evidence

    # --- Правило 2. Обработка почвы внутри периода: индекс падает по плану ---
    tillage = next((e for e in inside if e.kind == KIND_TILLAGE), None)
    if tillage is not None:
        delta = CONF_TILLAGE_ABRUPT if cause == CAUSE_ABRUPT else CONF_TILLAGE_OTHER
        text = (
            "Внутри периода в журнале работ стоит обработка почвы — %s, %s. После неё "
            "растительный покров снимается механически, и низкий индекс — ожидаемый "
            "результат работы, а не признак угнетения посева."
            % (_label(tillage), _fmt_span(tillage))
        )
        evidence["agro_rules"] = ["tillage_in_period"]
        evidence["agro_events"] = [_brief(tillage)]
        evidence["agro_conf_delta"] = _clamp_delta(confidence, delta)
        return text, evidence["agro_conf_delta"], evidence

    used: list[AgroEvent] = []

    # --- Правило 3. Пестицид за 3-14 дней до начала просадки: фитотоксичность ---
    pest = [
        (e, _lag_before(e, start))
        for e in near
        if e.kind in (KIND_HERBICIDE, KIND_INSECTICIDE)
    ]
    pest = [
        (e, lag)
        for e, lag in pest
        if lag is not None and PHYTOTOX_MIN_LAG_DAYS <= lag <= PHYTOTOX_MAX_LAG_DAYS
    ]
    if pest:
        # Если обработок несколько, берём ближайшую к началу просадки: срок
        # проявления у неё самый правдоподобный.
        event, lag = min(pest, key=lambda pair: pair[1])
        if cause in (CAUSE_NON_WEATHER, CAUSE_UNKNOWN):
            delta += CONF_PHYTOTOX_NON_WEATHER
            lead = "Журнал работ даёт конкретную зацепку"
        elif cause == CAUSE_ABRUPT:
            delta += CONF_PHYTOTOX_ABRUPT
            lead = "Журнал работ объясняет и одномоментность падения"
        else:
            # Погода уже объяснила период. Обработка тогда не причина, а
            # отягчающий фактор, и уверенность в погодной версии она не меняет.
            lead = "Дополнительно к погоде"
        parts.append(
            "%s: за %d дней до начала просадки проведена обработка — %s, %s. Это ровно тот "
            "срок, на котором проявляется фитотоксичность или ожог — осмотрите листовой "
            "аппарат и проверьте дозу, фазу и условия внесения."
            % (lead, lag, _label(event), _fmt_span(event))
        )
        rules.append("phytotoxicity")
        used.append(event)

    # --- Правило 4. Подкормка за 7-21 день до начала, а отклика нет ---
    fert = [(e, _lag_before(e, start)) for e in near if e.kind == KIND_FERTILIZER]
    fert = [
        (e, lag)
        for e, lag in fert
        if lag is not None and FERT_MIN_LAG_DAYS <= lag <= FERT_MAX_LAG_DAYS
    ]
    if fert:
        event, lag = min(fert, key=lambda pair: pair[1])
        if cause in (CAUSE_DROUGHT, CAUSE_NON_WEATHER):
            delta += CONF_FERT_NO_RESPONSE
        parts.append(
            "Отдельно стоит отметить: подкормка внесена за %d дней до начала просадки "
            "— %s, %s, — а отклика в индексе нет. Обычно подкормка отзывается приростом "
            "за одну-две недели. Значит либо она не усвоилась из-за нехватки влаги, "
            "либо на поле есть другая помеха — болезни, вредители, состояние семян."
            % (lag, _label(event), _fmt_span(event))
        )
        rules.append("fertilizer_no_response")
        used.append(event)

    # --- Правило 5. Полив во время «засухи»: версия дефицита влаги слабеет ---
    if cause == CAUSE_DROUGHT:
        irrigation = [
            e
            for e in near
            if e.kind == KIND_IRRIGATION
            and (
                _overlaps(e, start, end)
                or (_lag_before(e, start) or 999) <= IRRIGATION_LAG_DAYS
            )
        ]
        if irrigation:
            event = irrigation[-1]
            delta += CONF_IRRIGATION_VS_DROUGHT
            parts.append(
                "Однако по журналу работ поле в это время поливали — %s, %s. Влагу посеву "
                "давали, поэтому версия дефицита влаги слабеет: ищите причину среди "
                "прочего — качество и равномерность полива, засоление, болезни корня."
                % (_label(event), _fmt_span(event))
            )
            rules.append("irrigation_vs_drought")
            used.append(event)

    if not parts:
        # Работы рядом были, но ни одна не связывается с просадкой по срокам.
        # Перечислять их в объяснении — засорять текст: агроном и так знает,
        # что он делал. В свидетельствах они при этом остаются.
        evidence["agro_rules"] = []
        return "", 0.0, evidence

    evidence["agro_rules"] = rules
    evidence["agro_events"] = [_brief(e) for e in used]
    evidence["agro_conf_delta"] = _clamp_delta(confidence, delta)
    return " ".join(parts), evidence["agro_conf_delta"], evidence


def _clamp_delta(confidence: float, delta: float) -> float:
    """Поправка, урезанная так, чтобы итоговая уверенность осталась в [0.05; 0.95].

    Возвращаем именно поправку, а не новую уверенность: ядро складывает её со
    своей, и так видно, сколько именно добавил журнал работ.
    """
    try:
        base = float(confidence)
    except (TypeError, ValueError):
        base = 0.0
    target = min(CONF_MAX, max(CONF_MIN, base + delta))
    return round(target - base, 2)


# --- Пример журнала для демонстрации -----------------------------------------


def example_journal(polygon_id: str, year: int) -> list[AgroEvent]:
    """Правдоподобный журнал работ по озимой пшенице на юге России за сезон.

    Сезон озимой пшеницы начинается осенью предыдущего года, поэтому `year` —
    это год уборки: сев ложится на сентябрь `year - 1`. Календарь взят типовой
    для Ростовской области: сев в конце сентября, подкормка по мёрзлоталой почве
    в конце февраля и вторая в кущение, гербицид в апреле, фунгицид по флаг-листу
    в мае, уборка на переломе июня и июля, лущение стерни следом.
    """
    field_note = "поле %s" % polygon_id
    prev = year - 1
    return [
        AgroEvent(date(prev, 9, 15), date(prev, 9, 18), KIND_TILLAGE,
                  "предпосевная культивация на 6 см", field_note),
        AgroEvent(date(prev, 9, 26), date(prev, 9, 29), KIND_SOWING,
                  "озимая пшеница «Ермак», 220 кг/га", field_note),
        AgroEvent(date(prev, 10, 5), None, KIND_FERTILIZER,
                  "аммофос 100 кг/га при севе", field_note),
        AgroEvent(date(year, 2, 26), None, KIND_FERTILIZER,
                  "аммиачная селитра 150 кг/га по мёрзлоталой почве", field_note),
        AgroEvent(date(year, 3, 28), None, KIND_FERTILIZER,
                  "КАС-32, 80 кг/га в фазе кущения", field_note),
        AgroEvent(date(year, 4, 14), None, KIND_HERBICIDE,
                  "гербицид «Гранстар Про», 20 г/га, фаза кущения", field_note),
        AgroEvent(date(year, 5, 8), None, KIND_FUNGICIDE,
                  "фунгицид «Альто Супер», 0.5 л/га по флаг-листу", field_note),
        AgroEvent(date(year, 5, 22), None, KIND_INSECTICIDE,
                  "инсектицид против клопа вредная черепашка, 0.15 л/га", field_note),
        AgroEvent(date(year, 6, 28), date(year, 7, 4), KIND_HARVEST,
                  "уборка прямым комбайнированием, 42 ц/га", field_note),
        AgroEvent(date(year, 7, 16), date(year, 7, 18), KIND_TILLAGE,
                  "лущение стерни на 8 см", field_note),
    ]


# --- Демонстрация трёх сценариев ---------------------------------------------


def _demo() -> None:
    """Три сценария вживую: уборка в периоде, гербицид перед просадкой, пустой журнал."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    def show(title: str, cause: str, conf: float, base_text: str,
             start: date, end: date, events: list[AgroEvent]) -> None:
        add, delta, ev = explain_with_agro(cause, conf, start, end, events, "озимая пшеница")
        print("=" * 78)
        print(title)
        print("-" * 78)
        print("Период: %s - %s, версия «%s», уверенность %.2f" % (_fmt(start), _fmt(end), cause, conf))
        print()
        print("Было:  " + base_text)
        print()
        print("Стало: " + (base_text + " " + add if add else base_text))
        print()
        print("Поправка к уверенности: %+.2f  ->  %.2f" % (delta, round(conf + delta, 2)))
        print("Свидетельства: %s" % ev)
        print()

    journal = example_journal("AOI-0007", 2025)

    show(
        "Сценарий 1. Уборка попадает в найденный период — тревога снимается",
        CAUSE_ABRUPT, 0.6,
        "Резкое падение индекса на 2.4 стандартных отклонения за декаду. Похоже на "
        "одномоментное событие: уборку, потраву или механическое повреждение посева.",
        date(2025, 6, 30), date(2025, 7, 20), journal,
    )

    show(
        "Сценарий 2. Гербицид за неделю до просадки — версия фитотоксичности",
        CAUSE_NON_WEATHER, 0.5,
        "Осадков 41 мм при норме 62 мм (67 процентов), температура отличается от нормы "
        "на 0.1 градуса — ни засухи, ни жары, ни холода, ни переувлажнения за эти 26 дней "
        "не было. Погода отклонение не объясняет: проверяйте агротехнику, состояние "
        "семян, болезни и вредителей.",
        date(2025, 4, 22), date(2025, 5, 17), journal,
    )

    show(
        "Сценарий 3. Журнал пуст — поведение не меняется",
        CAUSE_NON_WEATHER, 0.5,
        "Осадков 41 мм при норме 62 мм (67 процентов), температура отличается от нормы "
        "на 0.1 градуса — ни засухи, ни жары, ни холода, ни переувлажнения за эти 26 дней "
        "не было. Погода отклонение не объясняет: проверяйте агротехнику, состояние "
        "семян, болезни и вредителей.",
        date(2025, 4, 22), date(2025, 5, 17), [],
    )


if __name__ == "__main__":
    _demo()
