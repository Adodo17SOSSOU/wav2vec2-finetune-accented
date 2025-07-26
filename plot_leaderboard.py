import matplotlib.pyplot as plt

# Replace these with your actual values

wer = 1.0278
cer = 0.8531

# Labels and values
metrics = ["WER", "CER"]
values = [wer, cer]

# Create bar plot
plt.figure(figsize=(5, 5))
bars = plt.bar(metrics, values)

# Annotate bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f"{height:.3f}", ha='center', va='bottom')

# Plot styling
plt.ylim(0, max(values) + 0.1)
plt.ylabel("Error Rate")
plt.title("WER and CER of Fine-Tuned Wav2Vec2")
plt.tight_layout()

# Save
plt.savefig("results/wer_cer_bar.png")
plt.close()

print(" Saved to results/wer_cer_bar.png")
