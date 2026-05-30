"""Generate episodes.csv from the dataset directory."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_IDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
OUTPUT_CSV = PROJECT_ROOT / "data" / "episodes.csv"


def main():
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode_id", "file_name", "file_path", "class_id", "label"],
        )
        writer.writeheader()
        eid = 0
        for label_name in sorted(CLASS_IDS):
            class_id = CLASS_IDS[label_name]
            cat_dir = DATASET_DIR / label_name
            if not cat_dir.exists():
                continue
            for fname in sorted(cat_dir.glob("*.txt")):
                rel_path = str(cat_dir.relative_to(PROJECT_ROOT) / fname.name)
                writer.writerow({
                    "episode_id": eid,
                    "file_name": fname.name,
                    "file_path": rel_path,
                    "class_id": class_id,
                    "label": label_name,
                })
                eid += 1
    print(f"Wrote {eid} episodes -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
