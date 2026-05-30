"""Generate train/test split files from episodes.csv with stratified sampling."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASS_IDS, TEST_SPLIT_RATIO, SPLIT_RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPISODES_CSV = PROJECT_ROOT / "data" / "episodes.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def main():
    df = pd.read_csv(EPISODES_CSV)
    train_cats = [c for c in CLASS_IDS if c != "other"]
    train_mask = df["label"].isin(train_cats)
    train_df = df[train_mask]
    other_df = df[~train_mask]

    # Stratified split on training classes only
    _, test_from_train = train_test_split(
        train_df,
        test_size=TEST_SPLIT_RATIO,
        stratify=train_df["class_id"],
        random_state=SPLIT_RANDOM_SEED,
    )
    train_df_final = train_df.drop(test_from_train.index)
    test_df = pd.concat([test_from_train, other_df])

    # Write split files
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df_final["episode_id"].to_csv(SPLITS_DIR / "train.txt", index=False, header=False)
    test_df["episode_id"].to_csv(SPLITS_DIR / "test.txt", index=False, header=False)

    # Report stats
    print(f"Total episodes: {len(df)}")
    print(f"Train: {len(train_df_final)} episodes")
    print(f"Test: {len(test_df)} episodes (including {len(other_df)} 'other')")
    print(f"Test ratio: {len(test_df) / len(df):.2%}")

    # Stratification check
    print("\nPer-class split counts:")
    for cat in sorted(CLASS_IDS):
        class_df = df[df["label"] == cat]
        n_train = train_df_final[train_df_final["label"] == cat].shape[0]
        n_test = test_df[test_df["label"] == cat].shape[0]
        print(f"  {cat:15s}: train={n_train}  test={n_test}  total={n_train + n_test}")


if __name__ == "__main__":
    main()
