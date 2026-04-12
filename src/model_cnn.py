import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block:
    learns channel-wise importance weights.
    """
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
    """
    Basic CNN block:
    conv -> BN -> ReLU -> conv -> BN -> ReLU -> SE -> pooling -> dropout
    """
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


class AttentionStatsPooling(nn.Module):
    """
    Attention-based statistical pooling.

    Input:
        (B, T, D)

    Output:
        (B, 3D) = [attention pooled, mean pooled, std pooled]
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
        scores = self.attn(x)                     # (B, T, 1)
        weights = torch.softmax(scores, dim=1)   # attention weights
        attn_pool = torch.sum(x * weights, dim=1)
        mean_pool = torch.mean(x, dim=1)
        std_pool = torch.sqrt(torch.var(x, dim=1, unbiased=False) + 1e-5)
        return torch.cat([attn_pool, mean_pool, std_pool], dim=1)


class SER_CNN_Attention(nn.Module):
    """
    Final speech emotion recognition model.

    Input:
        (B, 3, 128, T)

    Pipeline:
        CNN feature extractor
        -> frequency pooling
        -> BiGRU temporal modeling
        -> attention/statistical pooling
        -> classifier
    """
    def __init__(self, num_classes=6, dropout=0.35):
        super().__init__()

        # Initial stem to expand 3-channel input into feature maps
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # CNN backbone
        self.block1 = ConvBlock(32, 64, pool_kernel=(2, 2), dropout=0.10)
        self.block2 = ConvBlock(64, 128, pool_kernel=(2, 2), dropout=0.15)
        self.block3 = ConvBlock(128, 256, pool_kernel=(2, 1), dropout=0.20)
        self.block4 = ConvBlock(256, 256, pool_kernel=(2, 1), dropout=0.20)

        # Collapse frequency dimension while keeping time dimension
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Bidirectional GRU for temporal modeling
        self.bigru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Attention-based pooling
        self.pool = AttentionStatsPooling(256)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(256 * 3),
            nn.Linear(256 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
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
        x = self.block4(x)               # (B, 256, F, T')

        x = self.freq_pool(x)            # (B, 256, 1, T')
        x = x.squeeze(2)                 # (B, 256, T')
        x = x.transpose(1, 2)            # (B, T', 256)

        x, _ = self.bigru(x)             # (B, T', 256)
        x = self.pool(x)                 # (B, 768)

        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SER_CNN_Attention()
    dummy = torch.randn(8, 3, 128, 300)
    out = model(dummy)
    print("input:", dummy.shape)
    print("output:", out.shape)