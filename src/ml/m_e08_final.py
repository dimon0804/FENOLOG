"""E08. Сборка финальной конфигурации из принятых экспериментов.

К этому моменту приняты три независимых приёма, и каждый мерился в своём стеке:

    E02b  очистка ряда: придавливание брака, мягкая медиана по календарному окну,
          веса наблюдений по сенсору, суточная поправка со взвешиванием соседей
          по корреляции остатков                                     0,0654
    E04   климатологический якорь на разрывах от 45 дней              0,0656
    E06b  суточная поправка со взвешиванием (входит в оба стека выше)

E02b и E04 бьют в разные места: первый снимает шум наблюдения на всех точках,
второй правит форму кривой на длинных разрывах, где интерполяция физически не
может знать про сезонный подъём. Значит они должны складываться — но это надо
проверить числом, а не принять на веру. Тут это и делается.

Сборка намеренно устроена через реестр, а не через наследование: оба метода
приходят из чужих модулей, и склеивать их внутренности значило бы завязаться на
детали, которые авторы вправе поменять. Здесь берутся только их предсказания.
"""
from __future__ import annotations

import numpy as np

from src.ml.registry import BaseMethod, REGISTRY, register

# Ключи принятых методов из чужих модулей
KEY_CLEAN = "e02s_best_f10"  # очистка ряда со взвешенной суточной поправкой
KEY_ANCHOR = "e04_file_g45"  # чистый якорь, включённый от 45 дней разрыва

# Порог длины разрыва, начиная с которого якорь имеет смысл (из E04)
GAP_MIN = 45


class _Combined(BaseMethod):
    """Очистка ряда везде плюс климатологический якорь на длинных разрывах.

    weight — доля якоря на тех точках, где он включён. Единица означает полную
    замену. E04 показал, что оптимум около половины и что полная замена не
    воспроизводится от зерна к зерну: якорь несёт свой сигнал, но и свою ошибку
    нормы, и на некоторых наборах вторая перевешивает первую.
    """

    def __init__(self, weight: float = 0.5, gap_min: int = GAP_MIN):
        self.weight = weight
        self.gap_min = gap_min

    def predict_points(self, points, views, context):
        base = REGISTRY[KEY_CLEAN].factory().predict_points(points, views, context)
        if self.weight <= 0.0:
            return base
        anchor = REGISTRY[KEY_ANCHOR].factory().predict_points(points, views, context)

        # Длина разрыва в том же смысле, в каком её считает E04: сумма расстояний
        # до ближайших известных наблюдений слева и справа
        span = np.array([p.left_dist + p.right_dist for p in points], dtype=float)
        use = span >= self.gap_min

        out = base.copy()
        out[use] = (1.0 - self.weight) * base[use] + self.weight * anchor[use]
        return np.clip(out, 0.0, 1.0)


@register("final", "Финальная конфигурация: очистка + якорь, доля 0,5", experiment="E08",
          tags=("final",))
class Final(_Combined):
    def __init__(self):
        super().__init__(weight=0.5)


@register("final_w03", "Финальная конфигурация, доля якоря 0,3", experiment="E08")
class FinalW03(_Combined):
    def __init__(self):
        super().__init__(weight=0.3)


@register("final_w07", "Финальная конфигурация, доля якоря 0,7", experiment="E08")
class FinalW07(_Combined):
    def __init__(self):
        super().__init__(weight=0.7)


@register("final_w00", "Контроль: только очистка, якорь выключен", experiment="E08")
class FinalW00(_Combined):
    """Должен совпасть с e02s_best до последнего знака — проверка обвязки."""

    def __init__(self):
        super().__init__(weight=0.0)


@register("final_g30", "Финальная конфигурация, порог разрыва 30 дней", experiment="E08")
class FinalG30(_Combined):
    def __init__(self):
        super().__init__(weight=0.5, gap_min=30)


@register("final_g60", "Финальная конфигурация, порог разрыва 60 дней", experiment="E08")
class FinalG60(_Combined):
    def __init__(self):
        super().__init__(weight=0.5, gap_min=60)
