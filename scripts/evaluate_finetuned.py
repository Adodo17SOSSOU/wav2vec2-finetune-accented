import os
import json
from pathlib import Path
from datasets import Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
from tqdm import tqdm
from evaluate import load

# Paths
DATA_DIR = Path("data/test")
CHECKPOINT_DIR = "models/checkpoints/checkpoint-3"

RESULTS_PATH = Path("results/fine_tuned_predictions.json")

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load metadata
with open(DATA_DIR / "metadata.json", "r") as f:
    samples = json.load(f)

# Metrics
wer_metric = load("wer")
cer_metric = load("cer")

# Inference and prediction
predictions = []

for sample in tqdm(samples, desc="Evaluating"):
    audio_path = sample["path"]
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, sr, 16000).squeeze()

    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(input_values=inputs.input_values.to(device)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    pred_text = processor.batch_decode(pred_ids)[0]

    predictions.append({
        "path": audio_path,
        "text": sample["text"],
        "prediction": pred_text
    })

# Save predictions
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(predictions, f, indent=2)

# Compute metrics
refs = [ex["text"] for ex in predictions]
hyps = [ex["prediction"] for ex in predictions]

wer = wer_metric.compute(predictions=hyps, references=refs)
cer = cer_metric.compute(predictions=hyps, references=refs)

print(f"WER: {wer:.3f}")
print(f"CER: {cer:.3f}")

from datetime import datetime
import pandas as pd

# === Leaderboard path ===
LEADERBOARD_PATH = Path("results/leaderboard.csv")
LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

# === Metadata about current run ===
checkpoint_name = CHECKPOINT_DIR.split("/")[-1]
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
entry = {
    "timestamp": timestamp,
    "checkpoint": checkpoint_name,
    "WER": round(wer, 4),
    "CER": round(cer, 4),
}

# === Append to leaderboard ===
if LEADERBOARD_PATH.exists():
    df = pd.read_csv(LEADERBOARD_PATH)
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
else:
    df = pd.DataFrame([entry])

df.to_csv(LEADERBOARD_PATH, index=False)
print(f"\n Leaderboard updated: {LEADERBOARD_PATH}")

