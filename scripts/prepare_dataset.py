import json
import random
from pathlib import Path

# Config
INPUT_PATH = Path("data/common_voice/en/metadata.json")
OUTPUT_BASE = Path("data")
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

# Load metadata
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

# Shuffle
random.seed(SEED)
random.shuffle(data)

# Split
n = len(data)
train_end = int(SPLITS["train"] * n)
val_end = train_end + int(SPLITS["val"] * n)

splits = {
    "train": data[:train_end],
    "val": data[train_end:val_end],
    "test": data[val_end:]
}

# Save
for split_name, samples in splits.items():
    out_path = OUTPUT_BASE / split_name / "metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {out_path}")

