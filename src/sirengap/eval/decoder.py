"""Matched decoder and probes (protocol Part III common apparatus).

Decoder: MLP [D -> 1024 -> 512 -> 256 -> 10], GELU, dropout 0.1, AdamW 1e-3
cosine, <= 100 epochs, early stopping on the INR validation split. Also: linear
probe and kNN (k=10, cosine). Optional weight-space augmentation hook (rung W6):
a callable mapping (SirenParams minibatch, generator) -> SirenParams applied
fresh each step before flattening.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from sirengap.models.params import SirenParams

AugFn = Callable[[SirenParams, torch.Generator], SirenParams]


class MatchedMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [in_dim, 1024, 512, 256]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass(frozen=True)
class DecoderResult:
    test_acc: float
    val_acc: float
    epochs_ran: int
    extra_acc: dict[str, float] = field(default_factory=dict)


def _standardize(train: Tensor, *others: Tensor) -> tuple[Tensor, ...]:
    mu = train.mean(dim=0, keepdim=True)
    sd = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return tuple((t - mu) / sd for t in (train, *others))


@torch.no_grad()
def _acc(model: nn.Module, x: Tensor, y: Tensor, device: str, batch: int = 2048) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        pred = model(x[i : i + batch].to(device)).argmax(1).cpu()
        correct += int((pred == y[i : i + batch]).sum())
    return 100.0 * correct / len(x)


def train_matched_mlp(
    feats: dict[str, Tensor],
    labels: dict[str, Tensor],
    seed: int,
    device: str = "cpu",
    max_epochs: int = 100,
    patience: int = 10,
    batch: int = 512,
    lr: float = 1e-3,
    params_train: SirenParams | None = None,
    augment: AugFn | None = None,
    flatten: Callable[[SirenParams], Tensor] | None = None,
    extra_eval: dict[str, tuple[Tensor, Tensor]] | None = None,
) -> DecoderResult:
    """feats/labels keys: train/val/test. If augment is given, params_train must be
    the training-split SirenParams; each step applies a fresh group element to the
    minibatch and flattens (features are then standardized with the same stats).

    extra_eval maps a name to (features, labels) scored by the selected model under the
    *training* rung's standardization — the transfer evaluation used by rung X1."""
    torch.manual_seed(seed)
    x_tr, x_va, x_te = _standardize(feats["train"], feats["val"], feats["test"])
    mu = feats["train"].mean(dim=0, keepdim=True)
    sd = feats["train"].std(dim=0, keepdim=True).clamp_min(1e-6)
    y_tr, y_va, y_te = labels["train"], labels["val"], labels["test"]

    model = MatchedMLP(x_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    aug_gen = torch.Generator().manual_seed(seed + 7919)

    best_val, best_state, best_epoch = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            if augment is not None:
                assert params_train is not None and flatten is not None
                sub = _index_params(params_train, idx)
                xb = flatten(augment(sub, aug_gen))
                xb = (xb - mu) / sd
            else:
                xb = x_tr[idx]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb.to(device)), y_tr[idx].to(device)).backward()
            opt.step()
        sched.step()
        val_acc = _acc(model, x_va, y_va, device)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    extra = {
        name: _acc(model, (fx - mu) / sd, fy, device)
        for name, (fx, fy) in (extra_eval or {}).items()
    }
    return DecoderResult(
        test_acc=_acc(model, x_te, y_te, device),
        val_acc=best_val,
        epochs_ran=epoch + 1,
        extra_acc=extra,
    )


def _index_params(params: SirenParams, idx: Tensor) -> SirenParams:
    return SirenParams(
        hidden=tuple((w[idx], b[idx]) for w, b in params.hidden),
        w_out=params.w_out[idx],
        b_out=params.b_out[idx],
    )


def linear_probe(feats: dict[str, Tensor], labels: dict[str, Tensor], seed: int, device: str = "cpu") -> float:
    torch.manual_seed(seed)
    x_tr, x_te = _standardize(feats["train"], feats["test"])
    head = nn.Linear(x_tr.shape[1], 10).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(30):
        order = torch.randperm(len(x_tr), generator=gen)
        for i in range(0, len(order), 1024):
            idx = order[i : i + 1024]
            opt.zero_grad(set_to_none=True)
            loss_fn(head(x_tr[idx].to(device)), labels["train"][idx].to(device)).backward()
            opt.step()
    return _acc(head, x_te, labels["test"], device)


def knn_accuracy(feats: dict[str, Tensor], labels: dict[str, Tensor], k: int = 10) -> float:
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(feats["train"].numpy(), labels["train"].numpy())
    return 100.0 * float(clf.score(feats["test"].numpy(), labels["test"].numpy()))
