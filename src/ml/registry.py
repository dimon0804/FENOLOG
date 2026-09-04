"""Реестр методов восстановления.

Зачем реестр, а не список в validate.py: над экспериментами работают несколько
человек одновременно. Каждый новый метод — отдельный файл `src/ml/m_*.py` с
декоратором @register. Общих файлов никто не правит, конфликтов при слиянии нет.

Метод — это объект с методом predict(view, target_ords) -> np.ndarray.
Если ему нужно обучение, он выставляет needs_fit = True и получает fit(...)
на полигонах обучающих фолдов. Простую функцию можно зарегистрировать как есть,
она будет обёрнута автоматически.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.ml.dataset import PolygonView


class BaseMethod:
    """Базовый класс метода восстановления."""

    #: True, если метод обучается на данных других полигонов (тогда нужен GroupKFold)
    needs_fit: bool = False

    def fit(self, views: dict[str, PolygonView], points: list, context: dict) -> None:
        """Обучение на полигонах обучающих фолдов. По умолчанию ничего не делает."""

    def predict(self, view: PolygonView, target_ords: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_points(self, points: list, views: dict[str, PolygonView], context: dict) -> np.ndarray:
        """Предсказание сразу по списку контрольных точек разных полигонов.

        Реализация по умолчанию группирует точки по полигонам и зовёт predict.
        Методы-надстройки (например бустинг поверх остальных) переопределяют её,
        потому что им нужна не только геометрия ряда, но и таблица признаков.
        """
        out = np.empty(len(points), dtype=float)
        by_polygon: dict[str, list[int]] = {}
        for i, p in enumerate(points):
            by_polygon.setdefault(p.polygon_id, []).append(i)
        for polygon_id, idx in by_polygon.items():
            ords = np.array([points[i].ord_day for i in idx], dtype=np.int64)
            out[np.array(idx)] = self.predict(views[polygon_id], ords)
        return out


class _FunctionMethod(BaseMethod):
    """Обёртка вокруг обычной функции без состояния."""

    def __init__(self, fn: Callable[[PolygonView, np.ndarray], np.ndarray]):
        self._fn = fn

    def predict(self, view: PolygonView, target_ords: np.ndarray) -> np.ndarray:
        return self._fn(view, target_ords)


@dataclass
class MethodSpec:
    name: str            # короткий ключ, он же колонка в таблице предсказаний
    title: str           # как метод называется в отчётной таблице
    experiment: str      # к какому эксперименту журнала относится
    factory: Callable[[], BaseMethod]
    tags: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, MethodSpec] = {}


def register(name: str, title: str, experiment: str = "", tags: tuple[str, ...] = ()):
    """Декоратор регистрации. Вешается на класс-наследник BaseMethod или на функцию."""

    def wrap(obj):
        if isinstance(obj, type) and issubclass(obj, BaseMethod):
            factory = obj
        else:
            factory = lambda fn=obj: _FunctionMethod(fn)  # noqa: E731
        if name in REGISTRY:
            raise ValueError(f"метод {name!r} уже зарегистрирован")
        REGISTRY[name] = MethodSpec(name=name, title=title, experiment=experiment,
                                    factory=factory, tags=tags)
        return obj

    return wrap


def discover() -> list[str]:
    """Подхватывает все модули src/ml/m_*.py. Порядок — алфавитный, он же порядок строк."""
    package_dir = Path(__file__).resolve().parent
    loaded = []
    for mod in sorted(pkgutil.iter_modules([str(package_dir)]), key=lambda m: m.name):
        if not mod.name.startswith("m_"):
            continue
        importlib.import_module(f"src.ml.{mod.name}")
        loaded.append(mod.name)
    return loaded
