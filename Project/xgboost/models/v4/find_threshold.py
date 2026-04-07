import numpy as np
from sklearn.metrics import precision_recall_curve

probs = np.load("xgb_focal_val_probs.npy")
labels = np.load("xgb_focal_val_labels.npy")

prec, rec, thresholds = precision_recall_curve(labels, probs)

# Pick threshold where recall >= 0.75 and precision is maximized
target_recall = 0.75
# precision_recall_curve returns arrays in descending recall order
idx = np.searchsorted(-rec, -target_recall)  # first index where rec >= 0.75
best_threshold = float(thresholds[idx])

print(f"Threshold : {best_threshold:.6f}")
print(f"Precision : {prec[idx]:.4f}")
print(f"Recall    : {rec[idx]:.4f}")