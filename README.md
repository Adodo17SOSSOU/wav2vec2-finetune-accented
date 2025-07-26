# Fine-Tuning Wav2Vec2 for Accented English ASR

This project demonstrates how to fine-tune Facebook's `wav2vec2-base-960h` model on accented English speech data (Common Voice) using the Hugging Face Transformers and Datasets libraries.

---

## 📁 Project Structure

```
fine_tune_wav2vec2_accented/
├── data/
│   ├── common_voice/       # Raw data directory
│   ├── train/              # Metadata + WAV paths for training
│   ├── val/                # Validation set
│   └── test/               # Testing set (for evaluation)
│
├── logs/                  # SLURM output logs
├── models/
│   └── checkpoints/
│       └── checkpoint-3/   # Latest model checkpoint
│
├── results/
│   ├── fine_tuned_predictions.json  # Transcripts and references
│   ├── leaderboard.csv              # Logged metrics
│   └── plots/
│       └── wer_cer_bar.png          # Bar plot of WER/CER
│
├── scripts/
│   ├── fine_tune.py                # Training script
│   ├── evaluate_finetuned.py       # WER/CER + prediction generation
│   ├── prepare_dataset.py          # Metadata formatting
│   └── plot_leaderboard.py         # Visualization script
│
├── run_finetune.sh                 # SLURM job script
├── requirements.txt                # Conda/pip dependencies
└── README.md                       # This file
```

---

## 🚀 Setup

### 1. Create and activate conda environment

```bash
conda create -n asr_env python=3.10 -y
conda activate asr_env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🎯 Fine-Tuning

To train the model from scratch on the `train/` and `val/` splits:

```bash
bash run_finetune.sh
```

Logs will be saved in `logs/`. Checkpoints are stored in `models/checkpoints/`.

---

## 📈 Evaluation

After training completes:

```bash
python scripts/evaluate_finetuned.py
```

This will:

* Run inference on `data/test/`
* Save predictions to `results/fine_tuned_predictions.json`
* Compute and print **WER** and **CER**
* Log metrics to `leaderboard.csv`

---

## 📊 Visualization

To visualize evaluation metrics as a bar plot:

```bash
python scripts/plot_leaderboard.py
```

Output: `results/plots/wer_cer_bar.png`

---

## ✅ Outputs 

```
WER: 1.0278
CER: 0.8531 
```

![WER/CER Plot](results/plots/wer_cer_bar.png)

---

## 📌 Notes

* Make sure all audio is 16kHz mono.
* Use `prepare_dataset.py` to convert raw Common Voice TSVs into metadata.
* Models and processor are loaded from `facebook/wav2vec2-base-960h`.

---

## 📬 Contact

For any questions, feel free to reach out or open an issue.

Happy fine-tuning! 🎧
