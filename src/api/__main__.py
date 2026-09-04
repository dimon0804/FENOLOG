"""Точка запуска сервиса: `python -m src.api`.

Отдельный модуль нужен, чтобы способ запуска был один и тот же везде — в
README, в Dockerfile и на машине разработчика. Строку `uvicorn src.api.app:app`
с полудюжиной флагов иначе пришлось бы повторять в трёх местах и держать
синхронной.
"""
from __future__ import annotations

import argparse
import os
import sys


def _force_utf8() -> None:
    """Русские сообщения лога должны читаться и в консоли Windows.

    Без этого стандартный вывод берёт кодировку консоли (cp866/cp1251), и весь
    журнал сбора превращается в набор вопросительных знаков. В контейнере то же
    самое делает PYTHONIOENCODING, но на машине разработчика её никто не ставит.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def main() -> None:
    _force_utf8()
    parser = argparse.ArgumentParser(
        prog="python -m src.api",
        description="Веб-сервис «Фенолог»: мониторинг вегетационной динамики",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("FENOLOG_HOST", "0.0.0.0"),
        help="адрес прослушивания (по умолчанию 0.0.0.0 — иначе порт не виден из контейнера)",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("FENOLOG_PORT", "8000")),
        help="порт (по умолчанию 8000)",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="перезапуск при правке кода, только для разработки",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        # Один рабочий процесс намеренно: реестр задач и хранилище живут в памяти
        # процесса, и второй воркер не увидел бы задач первого.
        workers=1,
    )


if __name__ == "__main__":
    main()
