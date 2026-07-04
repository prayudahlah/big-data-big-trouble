from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent.parent.parent


def load_train(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    train_dir = _HERE / data_dir / "train"
    records = []
    for label_dir in sorted(train_dir.iterdir()):
        if label_dir.is_dir():
            for img in sorted(label_dir.glob("*")):
                if img.is_file():
                    records.append({
                        "path": str(img.relative_to(_HERE)),
                        "label": label_dir.name,
                    })
    return pd.DataFrame(records)


def load_test(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    test_dir = _HERE / data_dir / "test"
    records = []
    for img in sorted(test_dir.glob("*")):
        if img.is_file():
            records.append({
                "path": str(img.relative_to(_HERE)),
                "image_id": img.stem,
            })
    return pd.DataFrame(records)


def load_submission_example(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    return pd.read_csv(_HERE / data_dir / "submission.csv")
