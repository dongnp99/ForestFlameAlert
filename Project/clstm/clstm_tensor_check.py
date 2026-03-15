import numpy as np


FILE =  "../data/Daklak/final_inputs/clstm_data/clstm_tensor.npy"
FIRE_CHANNEL = -1   # fire là channel cuối

print("Loading tensor_norm...")

tensor = np.load(FILE, mmap_mode="r")

T, H, W, C = tensor.shape

print("\nTensor shape:", tensor.shape)

if FIRE_CHANNEL < 0:
    FIRE_CHANNEL = C + FIRE_CHANNEL

print("Fire channel:", FIRE_CHANNEL)

# =========================
# GLOBAL CHECK
# =========================

print("\n===== GLOBAL CHECK =====")

nan_count = 0
inf_count = 0

for c in range(C):

    ch = tensor[:,:,:,c]

    nan_count += np.isnan(ch).sum()
    inf_count += np.isinf(ch).sum()

print("Total NaN:", nan_count)
print("Total Inf:", inf_count)

# =========================
# CHANNEL STATS
# =========================

print("\n===== CHANNEL STATS =====")

for c in range(C):

    ch = tensor[:,:,:,c]

    mean = np.mean(ch)
    std = np.std(ch)
    mn = np.min(ch)
    mx = np.max(ch)

    print(f"\nChannel {c}")

    print("mean:", mean)
    print("std:", std)
    print("min:", mn)
    print("max:", mx)

# =========================
# FIRE LABEL CHECK
# =========================

print("\n===== FIRE LABEL CHECK =====")

fire = tensor[:,:,:,FIRE_CHANNEL]

print("min:", fire.min())
print("max:", fire.max())

unique = np.unique(fire)

print("unique values sample:", unique[:10])

fire_ratio = (fire > 0).mean()

print("fire pixel ratio:", fire_ratio)

if fire.min() < 0 or fire.max() > 1:
    print("WARNING: fire label corrupted")

# =========================
# EXTREME VALUE CHECK
# =========================

print("\n===== EXTREME VALUE CHECK =====")

for c in range(C):

    if c == FIRE_CHANNEL:
        continue

    ch = tensor[:,:,:,c]

    extreme = np.sum(np.abs(ch) > 10)

    if extreme > 0:
        print("Channel", c, "values >10:", extreme)

print("\nCHECK COMPLETE")
