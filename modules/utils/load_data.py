from pathlib import Path

import gdown


def load_data(
    gdrive_id: str = "1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7",
    output_dir: str | Path = "../data/raw",
) -> None:
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Data already exists in {output_dir}, skipping download")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{gdrive_id}"
    gdown.download_folder(url, output=str(output_dir), quiet=False)
