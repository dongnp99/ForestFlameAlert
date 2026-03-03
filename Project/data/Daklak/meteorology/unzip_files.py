import os
import zipfile

RAW_DIR = "raw_data"
OUT_DIR = "unzipped_nc"

os.makedirs(OUT_DIR, exist_ok=True)

def is_zip(filepath):
    with open(filepath, "rb") as f:
        return f.read(4) == b"PK\x03\x04"

for fname in os.listdir(RAW_DIR):
    if not fname.lower().endswith(".nc"):
        continue

    src_path = os.path.join(RAW_DIR, fname)

    if not is_zip(src_path):
        print(f"SKIP (not zip): {fname}")
        continue

    base = fname.replace(".nc", "")  # era5_2015_01
    zip_path = os.path.join(RAW_DIR, base + ".zip")

    # 1️⃣ đổi đuôi nc → zip
    os.rename(src_path, zip_path)
    print(f"RENAMED: {fname} → {base}.zip")

    # 2️⃣ tạo thư mục theo tháng
    extract_dir = os.path.join(OUT_DIR, base)
    os.makedirs(extract_dir, exist_ok=True)

    # 3️⃣ giải nén
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # 4️⃣ tìm data_0.nc và đổi tên
    data_nc = os.path.join(extract_dir, "data_0.nc")

    if os.path.exists(data_nc):
        new_name = os.path.join(extract_dir, f"{base}.nc")
        os.rename(data_nc, new_name)
        print(f"  → data_0.nc → {base}.nc")
    else:
        print(f"⚠️  WARNING: data_0.nc not found in {base}")

print("✅ DONE: unzip ERA5 (data_0.nc handled)")
