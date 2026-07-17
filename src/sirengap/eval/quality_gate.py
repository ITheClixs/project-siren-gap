"""Task-referenced INR quality gate (protocol Ch4).

A fixed pixel-trained CNN must classify INR *renders* within 1.0 accuracy point
of its real-pixel accuracy. The CNN is trained once per dataset on real train
pixels and cached; the gate compares its accuracy on real vs rendered eval sets.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

GATE_MARGIN_PTS = 1.0


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, side: int, n_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * (side // 4) ** 2, 10),
        )
        self.in_ch, self.side = in_ch, side

    def forward(self, x: Tensor) -> Tensor:  # x: [N, P, C] flat in [-1,1]
        img = x.transpose(1, 2).reshape(-1, self.in_ch, self.side, self.side)
        return self.net(img)


def train_gate_cnn(
    images: Tensor,
    labels: Tensor,
    side: int,
    epochs: int = 2,
    batch: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    cache: Path | None = None,
) -> SmallCNN:
    model = SmallCNN(images.shape[2], side).to(device)
    if cache is not None and cache.exists():
        model.load_state_dict(torch.load(cache, map_location=device, weights_only=True))
        return model
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(images), generator=gen)
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            opt.zero_grad(set_to_none=True)
            out = model(images[idx].to(device))
            loss_fn(out, labels[idx].to(device)).backward()
            opt.step()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), cache)
    return model


@torch.no_grad()
def accuracy(model: SmallCNN, images: Tensor, labels: Tensor, device: str, batch: int = 512) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(images), batch):
        pred = model(images[i : i + batch].to(device)).argmax(dim=1).cpu()
        correct += int((pred == labels[i : i + batch]).sum())
    return 100.0 * correct / len(images)


def quality_gate(
    model: SmallCNN, real: Tensor, renders: Tensor, labels: Tensor, device: str
) -> dict:
    acc_real = accuracy(model, real, labels, device)
    acc_render = accuracy(model, renders, labels, device)
    return {
        "acc_real_pixels": acc_real,
        "acc_renders": acc_render,
        "gap_pts": acc_real - acc_render,
        "passes": bool(acc_real - acc_render <= GATE_MARGIN_PTS),
    }
