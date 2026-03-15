import torch
import torch.nn as nn


# ================================================================
# CẤU HÌNH MẶC ĐỊNH
# ================================================================
DEFAULT_CFG = {
    "seq_len":  7,
    "H":        137,
    "W":        138,
    "n_feat":   25,
    "filters":  [64, 32],
    "kernel":   3,
    "dropout":  0.3,
    "fl_gamma": 2.0,
    "fl_alpha": 0.75,   # tăng từ 0.25 → 0.75 để model học class fire tốt hơn
    "lr":       1e-3,
    "clipnorm": 1.0,
}


# ================================================================
# CONVLSTM CELL
# 1 bước thời gian: nhận x_t + (h_{t-1}, c_{t-1}) → (h_t, c_t)
# ================================================================
class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell — học spatial + temporal pattern cùng lúc.

    Thay vì nhân ma trận như LSTM thông thường,
    dùng convolution → giữ được spatial structure.

    Gates:
      i = sigmoid(conv(x, h))   — input gate
      f = sigmoid(conv(x, h))   — forget gate
      o = sigmoid(conv(x, h))   — output gate
      g = tanh(conv(x, h))      — cell gate
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2

        # Gộp 4 gates vào 1 conv → tối ưu tốc độ
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels * 4,    # i, f, o, g
            kernel_size,
            padding=pad,
            bias=True,
        )

    def forward(self, x, h, c):
        """
        Parameters
        ----------
        x : (B, C_in, H, W)      — input timestep hiện tại
        h : (B, C_hid, H, W)     — hidden state trước
        c : (B, C_hid, H, W)     — cell state trước

        Returns
        -------
        h_next : (B, C_hid, H, W)
        c_next : (B, C_hid, H, W)
        """
        combined = torch.cat([x, h], dim=1)       # (B, C_in+C_hid, H, W)
        gates    = self.conv(combined)             # (B, C_hid*4, H, W)

        i, f, o, g = gates.chunk(4, dim=1)        # mỗi cái (B, C_hid, H, W)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g                    # cập nhật cell state
        h_next = o * torch.tanh(c_next)           # cập nhật hidden state

        return h_next, c_next

    def init_hidden(self, batch_size: int, H: int, W: int,
                    device: torch.device):
        """Khởi tạo h_0, c_0 = zeros"""
        return (
            torch.zeros(batch_size, self.hidden_channels, H, W, device=device),
            torch.zeros(batch_size, self.hidden_channels, H, W, device=device),
        )


# ================================================================
# CONVLSTM LAYER
# Chạy ConvLSTMCell qua seq_len timesteps
# ================================================================
class ConvLSTMLayer(nn.Module):
    """
    Wrap ConvLSTMCell thành 1 layer xử lý cả sequence.

    return_sequences=True  → trả về (B, T, C_hid, H, W)
    return_sequences=False → trả về (B, C_hid, H, W)  — timestep cuối
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 kernel_size: int = 3, return_sequences: bool = True):
        super().__init__()
        self.cell             = ConvLSTMCell(in_channels, hidden_channels,
                                             kernel_size)
        self.hidden_channels  = hidden_channels
        self.return_sequences = return_sequences

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, T, C, H, W)  — input sequence

        Returns
        -------
        Nếu return_sequences=True  : (B, T, C_hid, H, W)
        Nếu return_sequences=False : (B, C_hid, H, W)
        """
        B, T, C, H, W = x.shape
        h, c = self.cell.init_hidden(B, H, W, x.device)

        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)    # x[:,t] = (B, C, H, W)
            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return torch.stack(outputs, dim=1)  # (B, T, C_hid, H, W)
        else:
            return h                             # (B, C_hid, H, W)


# ================================================================
# FOCAL LOSS
# Thêm pos_weight để xử lý imbalance 0.11% fire
# ================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss cho pixel-wise binary classification.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    gamma=2      : giảm trọng số mẫu dễ (fire=0 chiếm 99.89%)
    alpha=0.75   : tăng penalty cho class fire (positive)
    pos_weight   : nhân thêm trọng số cho BCE của class positive
                   với 0.11% fire, pos_weight ≈ 900
                   → model bị phạt nặng hơn khi bỏ sót fire
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 pos_weight: float = 1.0):
        super().__init__()
        self.gamma      = gamma
        self.alpha      = alpha
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        pred   : (B, 1, H, W) — xác suất sau sigmoid [0,1]
        target : (B, 1, H, W) — label 0/1
        """
        pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)

        # Weighted BCE: pos_weight làm nặng hơn loss khi bỏ sót fire
        bce = -(self.pos_weight * target * torch.log(pred) +
                (1 - target) * torch.log(1 - pred))

        p_t    = target * pred + (1 - target) * (1 - pred)
        weight = self.alpha * (1 - p_t) ** self.gamma
        loss   = weight * bce
        return loss.mean()


# ================================================================
# CONVLSTM WILDFIRE MODEL
# ================================================================
class ConvLSTMWildfire(nn.Module):
    """
    Kiến trúc:
      Input (B, T, C, H, W)
        → ConvLSTM #1 filters=64, return_sequences=True
        → BatchNorm3d + Dropout
        → ConvLSTM #2 filters=32, return_sequences=False
        → BatchNorm2d + SpatialDropout
        → Decoder Conv2D 32→16→1
        → Sigmoid
      Output (B, 1, H, W)
    """

    def __init__(self, cfg: dict = None):
        super().__init__()
        c = {**DEFAULT_CFG, **(cfg or {})}

        filters = c["filters"]   # [64, 32]
        k       = c["kernel"]    # 3
        drop    = c["dropout"]   # 0.3
        n_feat  = c["n_feat"]    # 25

        # ── ConvLSTM stack ───────────────────────────────────────
        self.convlstm1 = ConvLSTMLayer(
            in_channels=n_feat,
            hidden_channels=filters[0],
            kernel_size=k,
            return_sequences=True,    # giữ tất cả timestep
        )
        self.bn1      = nn.BatchNorm3d(filters[0])
        self.drop1    = nn.Dropout3d(drop)

        self.convlstm2 = ConvLSTMLayer(
            in_channels=filters[0],
            hidden_channels=filters[1],
            kernel_size=k,
            return_sequences=False,   # chỉ lấy timestep cuối
        )
        self.bn2      = nn.BatchNorm2d(filters[1])
        self.drop2    = nn.Dropout2d(drop)   # SpatialDropout trên 2D

        # ── Decoder ──────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Conv2d(filters[1], 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1,  kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, T, C, H, W)  — input sequence

        Returns
        -------
        out : (B, 1, H, W)  — xác suất cháy từng pixel
        """
        # ConvLSTM #1
        x = self.convlstm1(x)          # (B, T, 64, H, W)
        # BatchNorm3d expects (B, C, T, H, W) → cần permute
        x = x.permute(0, 2, 1, 3, 4)  # (B, 64, T, H, W)
        x = self.bn1(x)
        x = self.drop1(x)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, 64, H, W)

        # ConvLSTM #2
        x = self.convlstm2(x)          # (B, 32, H, W)
        x = self.bn2(x)
        x = self.drop2(x)

        # Decoder
        out = self.decoder(x)           # (B, 1, H, W)
        return out


# ================================================================
# HELPER: build model + loss + optimizer
# ================================================================
def build_model(cfg: dict = None, pos_weight: float = 1.0):
    """
    Trả về model, loss function, optimizer đã cấu hình.

    Parameters
    ----------
    cfg        : dict config, merge với DEFAULT_CFG
    pos_weight : float — tỷ lệ neg/pos từ training set
                 truyền vào từ compute_pos_weight() trong s4

    Returns
    -------
    model     : ConvLSTMWildfire
    criterion : FocalLoss
    optimizer : Adam
    """
    c = {**DEFAULT_CFG, **(cfg or {})}

    model     = ConvLSTMWildfire(c)
    criterion = FocalLoss(
        gamma      = c["fl_gamma"],
        alpha      = c["fl_alpha"],
        pos_weight = pos_weight,    # truyền pos_weight vào loss
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=c["lr"],
    )

    return model, criterion, optimizer


# ================================================================
# CHẠY TRỰC TIẾP ĐỂ KIỂM TRA
# ================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, criterion, optimizer = build_model(pos_weight=900.0)
    model = model.to(device)

    # Đếm params
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")

    # Forward pass với dummy input
    print("\nKiểm tra forward pass...")
    dummy_x = torch.randn(2, 7, 25, 137, 138).to(device)
    dummy_y = torch.randint(0, 2, (2, 1, 137, 138)).float().to(device)

    with torch.no_grad():
        out  = model(dummy_x)
        loss = criterion(out, dummy_y)

    print(f"  Input  : {dummy_x.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Range  : [{out.min():.4f}, {out.max():.4f}]")
    print(f"  Loss   : {loss.item():.4f}")

    # VRAM usage
    if device.type == "cuda":
        vram_used  = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n  VRAM used  : {vram_used:.2f} GB")
        print(f"  VRAM total : {vram_total:.2f} GB")
        print(f"  VRAM free  : {vram_total - vram_used:.2f} GB")

    print("\nForward pass: OK")