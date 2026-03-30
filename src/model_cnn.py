import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_kernel=(2, 2), dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            SEBlock(out_ch),

            nn.MaxPool2d(kernel_size=pool_kernel),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class AttentionPooling(nn.Module):
    """
    Input: (B, T, D)
    Output: pooled (B, D)
    """
    def __init__(self, dim: int):
        super().__init__()
        hidden = max(dim // 2, 64)
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, T, D)
        scores = self.attn(x)                  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled


class SER_CNN_Attention(nn.Module):
    """
    Input: (B, 3, 128, T)
    """
    def __init__(self, num_classes=6, dropout=0.35):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = ConvBlock(32, 64, pool_kernel=(2, 2), dropout=0.10)
        self.block2 = ConvBlock(64, 128, pool_kernel=(2, 2), dropout=0.15)
        self.block3 = ConvBlock(128, 256, pool_kernel=(2, 1), dropout=0.20)
        self.block4 = ConvBlock(256, 256, pool_kernel=(2, 1), dropout=0.20)

        # keep time, collapse frequency
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # temporal model
        self.bigru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn_pool = AttentionPooling(256)

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected (B, 3, 128, T), got {tuple(x.shape)}")

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)               # (B,256,F,T')

        x = self.freq_pool(x)            # (B,256,1,T')
        x = x.squeeze(2)                 # (B,256,T')
        x = x.transpose(1, 2)            # (B,T',256)

        x, _ = self.bigru(x)             # (B,T',256)
        x = self.attn_pool(x)            # (B,256)

        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SER_CNN_Attention()
    dummy = torch.randn(8, 3, 128, 256)
    out = model(dummy)
    print("input:", dummy.shape)
    print("output:", out.shape)