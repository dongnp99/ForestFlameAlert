# ================================================================
# KIỂM TRA PYTORCH + CUDA — paste vào terminal hoặc jupyter
# ================================================================
import sys
print("Python    :", sys.version)

# 1. PyTorch có cài không
try:
    import torch
    print("PyTorch   :", torch.__version__)
except ImportError:
    print("PyTorch   : CHƯA CÀI — chạy: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    sys.exit()

# 2. CUDA có được build vào PyTorch không
print("CUDA built:", torch.cuda.is_available())
print("CUDA ver  :", torch.version.cuda)
print("cuDNN ver :", torch.backends.cudnn.version())

# 3. GPU có được nhận không
n_gpu = torch.cuda.device_count()
print("Số GPU    :", n_gpu)
for i in range(n_gpu):
    print(f"  GPU {i}  :", torch.cuda.get_device_name(i))
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"  VRAM    : {mem:.1f} GB")

# 4. Test thực sự tính toán trên GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    a = torch.randn(1000, 1000).to(device)
    b = torch.randn(1000, 1000).to(device)
    c = torch.matmul(a, b)
    print("GPU compute test: OK —", c.shape, "on", c.device)
else:
    print("GPU compute test: SKIP — không có GPU")

# 5. Kiểm tra các package cần thiết cho project
print("\nPackages cần thiết:")
packages = {
    "numpy"       : "numpy",
    "pandas"      : "pandas",
    "scikit-learn": "sklearn",
    "matplotlib"  : "matplotlib",
    "pyarrow"     : "pyarrow",   # đọc parquet
}
for name, mod in packages.items():
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {name:<14s}: OK  ({ver})")
    except ImportError:
        print(f"  {name:<14s}: CHƯA CÀI")