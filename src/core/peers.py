"""Сравнение поля с соседними полями — с проверкой, что на них растёт.

Зачем это нужно продукту. Ответ «ваше поле просело на 1,8 сигмы» агроном ещё
может получить и без сервиса. А вот ответ на вопрос «это у меня одного или у
всех?» он без сервиса не получит никак, и стоит этот ответ дороже: районная
просадка означает погоду, страховое событие и разговор с соседями, а локальная
означает, что дело в самом поле — семена, техника, агротехника, вредители.
Различить их можно только сравнением, и сравнивать надо с соседями.

Почему нельзя сравнивать со всеми подряд. На соседнем поле может расти другая
культура, и тогда сравнивать нечего: у подсолнечника в июле пик, у озимой
пшеницы в июле стерня. Поле, сравненное с соседом другой культуры, получит
«отставание на 40 %» на ровном месте, и это будет чистая выдумка сервиса.

Поэтому здесь сначала определяется культура каждого соседа — по его же кривой,
тем же модулем crop_profile, — и в сравнение берутся только поля своей
фенологической группы. Соседи с другой культурой не выбрасываются молча: они
считаются и называются в ответе, потому что «рядом два поля с другой культурой»
это тоже сведения о районе.

Чем меряется просадка соседа. Своей истории у соседнего поля обычно нет: сервис
видит его первый раз и качает те же несколько сезонов, что и по целевому полю.
Поэтому z-оценка соседа считается по норме его собственной культуры — той самой
climatology по культуре, что уже есть в ядре. Без определения культуры этот шаг
был бы невозможен: норму не к чему было бы привязать.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from src.core.crop_profile import CROP_GROUP, GROUP_TITLE, season_curve

# Порог z, ниже которого поле считается просевшим. Тот же, что у класса
# «угнетение биомассы» в постановке задачи: два разных порога в одном
# продукте означали бы два разных ответа на один и тот же вопрос.
DEPRESSED_Z = -1.0
# Доля просевших соседей, начиная с которой явление называется районным.
# Половина — не порог, а именно большинство: при трёх соседях из пяти уже
# нельзя говорить «дело в вашем поле».
DISTRICT_SHARE = 0.5
# Минимум соседей своей группы, при котором сравнение вообще проводится.
# По двум полям вывод о районе делать нельзя, и честнее промолчать.
MIN_PEERS = 3


@dataclass
class PeerField:
    """Соседнее поле: ряд наблюдений и всё, что о нём известно."""
    peer_id: str
    dates: list[date] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    crop_type: str | None = None      # если известна из тегов OSM или от пользователя
    distance_km: float | None = None


def _plural(n: int, one: str, few: str, many: str) -> str:
    """Число со словом в верной форме: «1 поле», «2 поля», «5 полей»."""
    n10, n100 = abs(n) % 10, abs(n) % 100
    if n10 == 1 and n100 != 11:
        word = one
    elif 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        word = few
    else:
        word = many
    return f"{n} {word}"


def _group_of(crop: str | None) -> str | None:
    return CROP_GROUP.get(str(crop).strip().lower()) if crop else None


def _season_integral(dates, values, year: int) -> float | None:
    """Накопленная зелёная масса за сезон: площадь под кривой NDVI.

    Именно интеграл, а не средний или максимальный NDVI: пик описывает один
    удачный день, а интеграл — весь сезон, и именно он связан с урожаем.
    """
    curve = season_curve(dates, values, year=year)
    return None if curve is None else float(curve.mean())


def _period_z(dates, values, start: date, end: date, crop: str | None, clim) -> float | None:
    """Насколько поле просело за период относительно нормы своей культуры.

    Возвращает None, если наблюдений в периоде нет или норма для культуры
    недоступна: молчание здесь честнее нуля, который выглядит как «всё в норме».
    """
    if clim is None or not crop or not clim.has(crop):
        return None
    obs = [(d, v) for d, v in zip(dates, values)
           if v is not None and np.isfinite(v) and start <= d <= end]
    if len(obs) < 2:
        return None
    doys = np.array([d.timetuple().tm_yday for d, _ in obs], dtype=int)
    vals = np.array([v for _, v in obs], dtype=float)
    mean, std = clim.norm(crop, doys)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    ok = np.isfinite(mean) & np.isfinite(std) & (std > 1e-6)
    if not ok.any():
        return None
    return float(np.mean((vals[ok] - mean[ok]) / std[ok]))


def _detect(peer: PeerField, detector) -> tuple[str | None, str | None, str]:
    """Культура соседа: сначала то, что известно, потом определение по кривой."""
    if peer.crop_type:
        return peer.crop_type, _group_of(peer.crop_type), "известна"
    if detector is None:
        return None, None, "не определена"
    res = detector.predict_series(peer.dates, peer.values)
    if not res.get("scores"):
        return None, None, "не определена"
    return res["best_guess"], res.get("group"), "определена по кривой"


def compare_with_peers(
    dates: list[date],
    values: list[float],
    peers: list[PeerField],
    crop_type: str | None = None,
    anomalies=None,
    year: int | None = None,
    detector=None,
    climatology=None,
) -> dict:
    """Сравнивает поле с соседями своей группы и разбирает, где явление районное.

    dates/values — ряд целевого поля (наблюдения, не восстановленный ряд).
    peers        — соседние поля, уже собранные слоем провайдеров.
    crop_type    — культура целевого поля (заявленная или определённая).
    anomalies    — найденные периоды: по каждому проверяется, просели ли соседи.
    year         — сезон для сравнения урожайности; по умолчанию последний.

    Возвращает словарь для meta. Пустой разбор — штатный исход: соседей мало,
    ряды короткие, нормы нет. Сервис в этом случае просто не показывает блок.
    """
    if detector is None or climatology is None:
        from src.core.crop_profile import _models

        det, _cls = _models()
        detector = detector or det
        if climatology is None:
            from src.core.analyze import _autoload_crop_climatology

            climatology = _autoload_crop_climatology()

    out: dict = {"peers_total": len(peers), "peers_same_group": 0,
                 "crop_mix": {}, "rank": None, "periods": [], "verdict": ""}
    if not peers:
        out["verdict"] = "соседние поля не найдены, сравнение не проводилось"
        return out

    target_group = _group_of(crop_type)
    if target_group is None and detector is not None:
        res = detector.predict_series(dates, values)
        target_group = res.get("group")
    out["target_group"] = target_group

    same: list[tuple[PeerField, str | None]] = []
    for p in peers:
        p_crop, group, _how = _detect(p, detector)
        key = GROUP_TITLE.get(group, group) if group else "культура не определена"
        out["crop_mix"][key] = out["crop_mix"].get(key, 0) + 1
        # Сосед другой группы в сравнение не идёт: у него другой календарь, и
        # разница «мы против него» описывала бы разницу культур, а не состояние.
        if group is not None and group == target_group:
            same.append((p, p_crop))
    out["peers_same_group"] = len(same)

    if len(same) < MIN_PEERS:
        out["verdict"] = (
            f"рядом найдено {len(peers)} полей, но своей группы среди них "
            f"{len(same)} — для сравнения нужно хотя бы {MIN_PEERS}. "
            "Вывод о районе по такой выборке был бы выдумкой."
        )
        return out

    # --- Место поля среди соседей своей группы по накопленной массе ---------
    # Сезон для сравнения — последний ПОЛНЫЙ, а не последний вообще. Текущий год
    # почти всегда оборван на сегодняшнем дне, и накопленная за неполный сезон
    # масса несравнима с полными сезонами соседей: поле проиграло бы всем просто
    # потому, что сентябрь ещё не наступил.
    mine = None
    if year is not None:
        mine = _season_integral(dates, values, year)
    elif dates:
        for candidate in sorted({d.year for d in dates}, reverse=True):
            mine = _season_integral(dates, values, candidate)
            if mine is not None:
                year = candidate
                break
    peer_integrals = []
    for p, _crop in same:
        v = _season_integral(p.dates, p.values, year) if year else None
        if v is not None:
            peer_integrals.append((p.peer_id, v))

    if mine is not None and len(peer_integrals) >= MIN_PEERS:
        vals = np.array([v for _, v in peer_integrals], dtype=float)
        median = float(np.median(vals))
        place = int((vals > mine).sum()) + 1
        delta = (mine - median) / median * 100.0 if median else 0.0
        out["rank"] = {
            "place": place, "of": len(vals) + 1,
            "own_integral": round(mine, 4),
            "peer_median": round(median, 4),
            "delta_pct": round(delta, 1),
            "year": year,
        }

    # --- Районное явление или локальное ------------------------------------
    for a in (anomalies or []):
        start = a.start if hasattr(a, "start") else a["start"]
        end = a.end if hasattr(a, "end") else a["end"]
        hits, checked = 0, 0
        for p, p_crop in same:
            z = _period_z(p.dates, p.values, start, end, p_crop, climatology)
            if z is None:
                continue
            checked += 1
            hits += int(z <= DEPRESSED_Z)
        if checked < MIN_PEERS:
            continue
        share = hits / checked
        district = share >= DISTRICT_SHARE
        out["periods"].append({
            "start": str(start), "end": str(end),
            "peers_checked": checked, "peers_depressed": hits,
            "share": round(share, 2),
            "scope": "район" if district else "поле",
            # Факт и вывод разделены намеренно. Факт («просело 1 из 6») уместен
            # всегда. Вывод — нет: если период уже объяснён уборкой, фраза «дело
            # в самом поле, проверяйте вредителей» противоречит сама себе.
            # Решает, показывать ли вывод, вызывающая сторона: она знает версию
            # причины, а этот модуль о ней не осведомлён.
            "fact": f"Соседние поля той же группы: просело {hits} из {checked}.",
            "verdict": (
                "Явление районное — причину логично искать в погоде, а не в поле."
                if district else
                "Явление локальное: погода на район легла одинаково, значит дело "
                "в самом поле — агротехника, семена, техника, вредители."
            ),
        })

    # --- Итоговая фраза ----------------------------------------------------
    parts = []
    if out["rank"]:
        r = out["rank"]
        sign = "выше" if r["delta_pct"] >= 0 else "ниже"
        parts.append(
            f"по накопленной за сезон {r['year']} биомассе поле на {r['place']} месте "
            f"из {r['of']} среди полей своей группы в округе, "
            f"{abs(r['delta_pct']):.0f} % {sign} медианы соседей"
        )
    other = out["peers_total"] - out["peers_same_group"]
    if other > 0:
        parts.append(
            f"ещё {_plural(other, 'соседнее поле', 'соседних поля', 'соседних полей')} "
            f"в сравнение не {'взято' if other == 1 else 'взяты'}: на них другая "
            f"культура, и календарь развития у них свой"
        )
    district = [p for p in out["periods"] if p["scope"] == "район"]
    local = [p for p in out["periods"] if p["scope"] == "поле"]
    if district:
        parts.append(f"районных просадок за период: {len(district)}")
    if local:
        parts.append(f"просадок, которых у соседей не было: {len(local)}")
    out["verdict"] = "; ".join(parts) if parts else "сравнение не дало выводов"
    return out
