import os
import json
import sys
import re
import torch
import torchaudio
from pathlib import Path
from dataclasses import dataclass
from typing import Union
from datasets import DatasetDict, Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)

# Suppress broken stdout errors on some clusters
try:
    sys.stdout.flush()
except Exception:
    sys.stdout = open(os.devnull, 'w')

# ======= Config Paths =======
DATA_DIR = Path("data")
MODEL_DIR = Path("models/checkpoints")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ======= Load Wav2Vec2 Processor =======
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# ======= Text Normalization =======
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

# ======= Load Metadata JSON =======
def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def prepare_dataset(split):
    samples = load_json(DATA_DIR / split / "metadata.json")
    return Dataset.from_list(samples)

# ======= Preprocess Audio/Text =======
def preprocess(batch):
    waveform, sample_rate = torchaudio.load(batch["path"])
    if sample_rate != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)
    waveform = waveform.squeeze()

    # Extract input values
    batch["input_values"] = processor(waveform.numpy(), sampling_rate=16000).input_values[0]

    # Normalize and tokenize target
    norm_text = normalize_text(batch["text"])
    batch["labels"] = processor.tokenizer(norm_text).input_ids

    return batch

# ======= Load and Preprocess Dataset =======
data = DatasetDict({
    "train": prepare_dataset("train").map(preprocess),
    "validation": prepare_dataset("val").map(preprocess),
})

# ======= Load Model =======
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# ======= Data Collator =======
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.pad(
            {"input_values": input_values},
            padding=self.padding,
            return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                {"input_ids": labels},
                padding=self.padding,
                return_tensors="pt"
            )

        batch["labels"] = labels_batch["input_ids"]
        return batch

# ======= Training Arguments =======
training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    group_by_length=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    num_train_epochs=50,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=10,
    learning_rate=3e-4,
    warmup_steps=100,
    weight_decay=0.005,
    do_eval=True,
    disable_tqdm=True  # Prevent I/O issues in SLURM
)

# ======= Trainer Setup =======
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorCTCWithPadding(processor),
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    tokenizer=processor.feature_extractor
)

# ======= Training =======
trainer.train()

# ======= Save Final Model =======
model.save_pretrained(MODEL_DIR)
processor.save_pretrained(MODEL_DIR)
