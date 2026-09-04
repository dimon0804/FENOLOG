"""Построение локального контрольного набора — сердце протокола валидации.

Задача: спрятать часть известных значений так, чтобы получившиеся точки были
геометрически неотличимы от реальных контрольных точек организаторов.
Наивное «спрятать 20 % случайных точек» этого не даёт: у случайной точки разрыв
почти всегда короткий, а в реальном наборе встречаются разрывы до 189 дней,
и именно на них методы расходятся сильнее всего.

Приём — маскирование по шаблону. Из реального теста снимается список троек
(расстояние до соседа слева, расстояние справа, месяц). Каждая такая тройка —
шаблон. Чтобы его воспроизвести, берётся случайное известное наблюдение нужного
месяца и вместе с ним прячутся все известные наблюдения, попавшие внутрь
интервала шаблона. Тогда у спрятанной точки соседи оказываются ровно там же, где
у реальной контрольной. Оценивается только центр шаблона; соседи, снесённые
заодно, из обучения выбывают, но в метрику не идут — иначе набор перекосило бы
в сторону коротких разрывов.

Зерно фиксировано (42), протокол детерминирован и воспроизводится командой
    python -m src.ml.validate
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 42
# Контрольные точки организаторов встречаются только в вегетационный сезон
SEASON_MONTHS = (4, 5, 6, 7, 8, 9, 10)
# Бины длины разрыва для отчётных таблиц (сумма расстояний до соседей слева и справа)
GAP_BINS = [0, 2, 4, 7, 13, 29, 61, 181, 10_000]
GAP_LABELS = ["1-2", "3-4", "5-7", "8-13", "14-29", "30-61", "62-181", "182+"]
# Сколько кандидатов просматривать, подбирая место под один шаблон
SCAN_LIMIT = 400


@dataclass
class HoldoutPoint:
    """Одна спрятанная точка локальной валидации."""

    polygon_id: str
    ord_day: int
    truth: float
    left_dist: int
    right_dist: int
    month: int


def _neighbour_distances(ords: np.ndarray, known_ords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой даты — расстояние в днях до ближайшего известного слева и справа.

    Отсутствие соседа с какой-то стороны кодируется -1: это экстраполяция,
    в реальном наборе таких точек меньше процента.
    """
    pos = np.searchsorted(known_ords, ords, side="left")
    left = np.full(len(ords), -1, dtype=np.int64)
    right = np.full(len(ords), -1, dtype=np.int64)
    has_left = pos > 0
    left[has_left] = ords[has_left] - known_ords[pos[has_left] - 1]
    has_right = pos < len(known_ords)
    right[has_right] = known_ords[pos[has_right]] - ords[has_right]
    return left, right


def extract_templates(test_df: pd.DataFrame) -> pd.DataFrame:
    """Снимает геометрию реальных контрольных точек: (слева, справа, месяц).

    Соседи ищутся по тому же набору наблюдений, который увидит метод в момент
    инференса: если в работу подмешан train_dataset, разрывы у реальных
    контрольных точек становятся короче, и шаблоны обязаны это учитывать.
    Иначе локальная валидация будет мерить задачу, которой уже нет.
    """
    rows = []
    for polygon_id, g in test_df.groupby("anon_polygon_id", sort=False):
        g = g.sort_values("_ord")
        known_ords = g.loc[g["primary_ndvi"].notna(), "_ord"].to_numpy(dtype=np.int64)
        targets = g[g["is_synthetic_gap"]]
        if targets.empty or len(known_ords) == 0:
            continue
        t_ords = targets["_ord"].to_numpy(dtype=np.int64)
        left, right = _neighbour_distances(t_ords, known_ords)
        rows.append(
            pd.DataFrame(
                {
                    "anon_polygon_id": polygon_id,
                    "left_dist": left,
                    "right_dist": right,
                    "month": targets["_month"].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def gap_bin(left: np.ndarray | int, right: np.ndarray | int) -> np.ndarray:
    """Длина разрыва = сумма расстояний до соседей. Односторонний случай удваивается."""
    left = np.atleast_1d(np.asarray(left))
    right = np.atleast_1d(np.asarray(right))
    span = np.where(left < 0, right * 2, np.where(right < 0, left * 2, left + right))
    return pd.cut(span, bins=GAP_BINS, labels=GAP_LABELS, right=True).astype(object)


def build_holdout(
    df: pd.DataFrame,
    templates: pd.DataFrame,
    hide_frac: float = 0.20,
    seed: int = SEED,
    n_scored: int = 3000,
) -> tuple[list[HoldoutPoint], np.ndarray]:
    """Прячет hide_frac известных значений по шаблонам реальных контрольных точек.

    Возвращает список оцениваемых точек и индексы строк таблицы, которые надо
    замаскировать целиком (центры шаблонов плюс снесённые заодно соседи).
    """
    rng = np.random.default_rng(seed)

    # Кандидаты — известные наблюдения. Месяц центра шаблона всегда лежит в
    # апреле-октябре, поэтому вне сезона ничего спрятано не будет автоматически.
    known_mask = df["primary_ndvi"].notna().to_numpy()
    budget = int(hide_frac * known_mask.sum())

    # Для каждого полигона держим отсортированные дни известных наблюдений
    # и флаг «уже спрятано». Всё в numpy — цикл по шаблонам горячий.
    per_polygon: dict[str, dict] = {}
    for polygon_id, g in df[known_mask].groupby("anon_polygon_id", sort=False):
        g = g.sort_values("_ord")
        per_polygon[str(polygon_id)] = {
            "pid": str(polygon_id),
            "ord": g["_ord"].to_numpy(dtype=np.int64),
            "row": g.index.to_numpy(),
            "value": g["primary_ndvi"].to_numpy(dtype=float),
            "month": g["_month"].to_numpy(dtype=np.int64),
            "src": g["_source"].to_numpy() if "_source" in g.columns else np.full(len(g), "test"),
            "hidden": np.zeros(len(g), dtype=bool),
            # Соседи уже собранных контрольных точек: их прятать нельзя, иначе
            # разрыв у ранее размещённой точки задним числом растянется
            "protected": np.zeros(len(g), dtype=bool),
        }

    # Пул кандидатов на роль центра шаблона, разложенный по месяцам. Тянем
    # равномерно по наблюдениям, а не по полигонам: иначе маленькие поля
    # выедаются первыми и геометрия перекашивается.
    # Центром шаблона может быть только строка тестового файла: именно такие
    # строки мы предсказываем на самом деле. Наблюдения из train участвуют как
    # соседи и могут быть снесены заодно, но в метрику не попадают никогда —
    # иначе замер вклада train сравнивал бы разные множества точек.
    pool: dict[int, list[tuple[str, int]]] = {m: [] for m in SEASON_MONTHS}
    for pid, state in per_polygon.items():
        for i, m in enumerate(state["month"]):
            if m in pool and state["src"][i] == "test":
                pool[int(m)].append((pid, i))
    for m in pool:
        rng.shuffle(pool[m])

    # Шаблоны с односторонним разрывом отбрасываем: воспроизвести край ряда,
    # не разрушив полигон целиком, нельзя, а таких точек меньше процента.
    tpl = templates[(templates["left_dist"] > 0) & (templates["right_dist"] > 0)]
    tpl = tpl.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    tpl_bin = gap_bin(tpl["left_dist"].to_numpy(), tpl["right_dist"].to_numpy())

    # Квота по бинам длины разрыва. Без неё короткие шаблоны, которым труднее
    # найти место, недобираются, длинные добираются легко, и локальный набор
    # снова оказывается тяжелее реального.
    shares = pd.Series(tpl_bin).value_counts(normalize=True)
    quota = {b: int(round(shares.get(b, 0.0) * n_scored)) for b in GAP_LABELS}

    points: list[HoldoutPoint] = []
    hidden_rows: list[int] = []
    hidden_count = 0
    attempt = 0
    max_attempts = 200 * len(tpl)
    cursor = {m: 0 for m in pool}

    while hidden_count < budget and attempt < max_attempts and sum(quota.values()) > 0:
        k = attempt % len(tpl)
        attempt += 1
        t = tpl.iloc[k]
        left_need, right_need, month = int(t.left_dist), int(t.right_dist), int(t.month)
        want_bin = tpl_bin[k]
        candidates = pool.get(month)
        if not candidates or quota.get(want_bin, 0) <= 0:
            continue

        # Сканируем пул месяца от текущего курсора, пока не найдём место, куда
        # шаблон ложится точь-в-точь. Курсор общий на месяц, поэтому кандидаты
        # не перебираются по кругу заново на каждом шаблоне.
        placed = None
        for _ in range(SCAN_LIMIT):
            cursor[month] = (cursor[month] + 1) % len(candidates)
            pid, i = candidates[cursor[month]]
            state = per_polygon[pid]
            if state["hidden"][i]:
                continue
            p = int(state["ord"][i])

            # Слева должен остаться живой сосед не ближе left_need дней, всё что
            # ближе — прячем вместе с центром. Справа симметрично.
            alive = ~state["hidden"]
            left_idx = np.flatnonzero(alive & (state["ord"] <= p - left_need))
            right_idx = np.flatnonzero(alive & (state["ord"] >= p + right_need))
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            jl, jr = int(left_idx[-1]), int(right_idx[0])
            got_left, got_right = p - int(state["ord"][jl]), int(state["ord"][jr]) - p

            # Ключевая проверка протокола: фактический разрыв обязан попасть в тот
            # же бин, что и шаблон. Без неё захват соседей растягивает разрывы.
            if gap_bin(got_left, got_right)[0] != want_bin:
                continue
            candidate_victims = np.flatnonzero(
                alive & (state["ord"] > state["ord"][jl]) & (state["ord"] < state["ord"][jr]))
            if state["protected"][candidate_victims].any():
                continue
            placed = (state, i, p, jl, jr, got_left, got_right, alive)
            break

        if placed is None:
            continue
        state, i, p, jl, jr, got_left, got_right, alive = placed

        victims = np.flatnonzero(alive & (state["ord"] > state["ord"][jl]) & (state["ord"] < state["ord"][jr]))
        state["hidden"][victims] = True
        hidden_rows.extend(state["row"][victims].tolist())
        hidden_count += len(victims)
        quota[want_bin] -= 1
        state["protected"][jl] = True
        state["protected"][jr] = True
        points.append(
            HoldoutPoint(
                polygon_id=state["pid"],
                ord_day=p,
                truth=float(state["value"][i]),
                left_dist=got_left,
                right_dist=got_right,
                month=month,
            )
        )

    # Пересчёт фактических расстояний после сборки набора.
    #
    # Расстояния, записанные в момент размещения шаблона, верны только на тот
    # момент. Более поздний шаблон может спрятать наблюдение, которое раньше было
    # засчитано соседом уже собранной точки — и тогда её реальный разрыв
    # оказывается длиннее записанного. Без этого пересчёта расходились 4,4 %
    # точек, в отдельных случаях на два порядка: записано 4 дня, фактически 168.
    # Ошибка не искажает саму RMSE, но портит разрез по бинам и признаки
    # «расстояние до соседа» у методов, которые на них опираются.
    for state in per_polygon.values():
        state["alive_ord"] = state["ord"][~state["hidden"]]
    for point in points:
        alive_ord = per_polygon[point.polygon_id]["alive_ord"]
        left, right = _neighbour_distances(np.array([point.ord_day], dtype=np.int64), alive_ord)
        point.left_dist, point.right_dist = int(left[0]), int(right[0])

    return points, np.array(sorted(set(hidden_rows)), dtype=np.int64)


def describe_holdout(points: list[HoldoutPoint], templates: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает распределение по длине разрыва: цель против того, что вышло."""
    got = pd.DataFrame(
        {
            "bin": gap_bin(
                np.array([p.left_dist for p in points]),
                np.array([p.right_dist for p in points]),
            )
        }
    )
    want = pd.DataFrame(
        {"bin": gap_bin(templates["left_dist"].to_numpy(), templates["right_dist"].to_numpy())}
    )
    table = pd.DataFrame(
        {
            "контрольные точки, %": want["bin"].value_counts(normalize=True) * 100,
            "локальная валидация, %": got["bin"].value_counts(normalize=True) * 100,
        }
    ).reindex(GAP_LABELS).fillna(0.0).round(1)
    table.index.name = "разрыв, дней"
    return table
