import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Load log dari modelling
LOG_FILE = "kriteria2/performance_log.json"

if not os.path.exists(LOG_FILE):
    raise FileNotFoundError(f"Log file tidak ditemukan di {LOG_FILE}")

with open(LOG_FILE, "r") as f:
    log_data = json.load(f)

# Tampilkan semua hasil
print("=== MONITORING METRIC ===")
for model, metrics in log_data.items():
    print(f"\nModel: {model}")
    for key, val in metrics.items():
        print(f"{key}: {val}")

# Simpan visualisasi akurasi
model_names = list(log_data.keys())
accuracies = [log_data[model]["accuracy"] for model in model_names]

plt.bar(model_names, accuracies, color='skyblue')
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.savefig("kriteria4/accuracy_monitoring.png")
plt.show()
