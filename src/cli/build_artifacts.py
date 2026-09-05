"""Пересборка артефактов модели из исходных данных.

    python -m src.cli.build_artifacts              # всё, что можно пересобрать
    python -m src.cli.build_artifacts --only climatology

Зачем нужно. Постановка требует, чтобы артефакты подключались по инструкции и
без ручных доработок. В репозитории лежат три файла — норма по культурам,
определитель культуры и обученный бустинг, — и до появления этой команды первый
из них нельзя было пересобрать ничем: код построения существовал, а точки входа
к нему не было. Артефакт, который нельзя воспроизвести, — это чёрный ящик, даже
если он лежит рядом с исходниками.

Что чем собирается:

    models/crop_climatology.json   этой командой
    models/crop_classifier.pkl     этой командой (обёртка над src.ml.crop_id)
    models/e05_lgbm.pkl            python -m src.cli.batch_infer --retrain --save-model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MODELS = Path(__file__).resolve().parents[2] / "models"


def build_climatology(out: Path) -> dict:
    """Норма по типам культур — для полей, у которых нет собственной истории.

    Считается по обоим выданным наборам сразу: чем больше полей одной культуры,
    тем устойчивее средняя кривая. Скрытые контрольные точки в норму не попадают
    сами собой — у них `primary_ndvi` пуст, а `fit` берёт только заполненные.
    """
    from src.core.crop_climatology import CropClimatology
    from src.ml.dataset import TEST_PATH, TRAIN_PATH, load_frame

    frames = []
    for path, kind in ((TEST_PATH, "test"), (TRAIN_PATH, "train")):
        if Path(path).exists():
            frames.append(load_frame(str(path), kind))
    if not frames:
        raise SystemExit("не найдены исходные наборы данных в data/")

    df = pd.concat(frames, ignore_index=True, sort=False)
    model = CropClimatology().fit(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    return {
        "файл": str(out),
        "культур": len(model.crops),
        "полей по культурам": {c: model.n_fields(c) for c in model.crops},
    }


def build_classifier(out: Path) -> dict:
    """Определитель культуры по форме сезонной кривой."""
    from src.ml import crop_id

    return crop_id.train(path=out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Пересборка артефактов модели")
    parser.add_argument("--only", choices=["climatology", "classifier"],
                        help="собрать только один артефакт")
    parser.add_argument("--models-dir", default=str(MODELS), help="куда складывать файлы")
    args = parser.parse_args()

    models = Path(args.models_dir)
    jobs = []
    if args.only in (None, "climatology"):
        jobs.append(("норма по культурам", build_climatology, models / "crop_climatology.json"))
    if args.only in (None, "classifier"):
        jobs.append(("определитель культуры", build_classifier, models / "crop_classifier.pkl"))

    for title, fn, path in jobs:
        print(f"Собираю: {title}")
        try:
            info = fn(path)
        except Exception as exc:  # noqa: BLE001 — один артефакт не должен ронять остальные
            print(f"  НЕ УДАЛОСЬ: {type(exc).__name__}: {exc}")
            continue
        for key, value in (info or {}).items():
            print(f"  {key}: {value}")
        print()

    print("Готово. Бустинг пересобирается отдельно:")
    print("  python -m src.cli.batch_infer --method e05_lgbm --retrain --save-model")


if __name__ == "__main__":
    main()
