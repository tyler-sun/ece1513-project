import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 16)
        self.query = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False)
        )

    def forward(self, x):
        # x: (B, C, T)
        x_t = x.transpose(1, 2)          # (B, T, C)
        weights = self.query(x_t)        # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        out = torch.sum(x_t * weights, dim=1)   # (B, C)
        return out


class CNNEmotionRevised(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.attention = GlobalAttention(128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # expected: (B, 120, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)   # (B, 1, 120, T)
        elif x.dim() != 4:
            raise ValueError(f"Expected input with 3 or 4 dims, got {x.shape}")

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.freq_pool(x)    # (B, 128, 1, T_new)
        x = x.squeeze(2)         # (B, 128, T_new)

        x = self.attention(x)    # (B, 128)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = CNNEmotionRevised()
    dummy_input = torch.randn(8, 120, 200)
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")