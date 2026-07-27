"""S1 ladder rungs: the feature maps decoded by the matched MLP (docs/prereg/S1.md).

Every rung is `corpus -> (features per split, labels per split)`, optionally carrying the
training-split SirenParams and an augmentation callable when the rung re-draws its input
during training. The decoder apparatus itself never changes across rungs — that is the
point of the ladder.

Corpora are loaded once and cached; the K-marginalized corpus (8 views x train images) is
loaded only when rung W7 is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch import Tensor

from sirengap.canon.calign import c_align
from sirengap.canon.csort import c_sort
from sirengap.canon.deep_invariants import encode_deep
from sirengap.data.images import load_dataset, spec_of
from sirengap.data.schema import load_corpus
from sirengap.eval.decoder import AugFn
from sirengap.fitting.batched import init_from_seeds, absorb_omega, make_coord_grid
from sirengap.models.forward import forward_canonical
from sirengap.models.params import SirenParams
from sirengap.symmetry.dinf import GroupElement, apply, random_element

SPLITS = ("train", "val", "test")
TEST_ID_OFFSET = 100000
N_PROBES = 256  # coordinates used by c_align's activation matching
AUG_MAX_WINDINGS = 1  # bounded family (close-read of 2402.04081); anchor A2 used the same


@dataclass(frozen=True)
class Rung:
    """One ladder rung, ready for the decoder."""

    name: str
    feats: dict[str, Tensor]
    labels: dict[str, Tensor]
    params_train: SirenParams | None = None
    augment: AugFn | None = None
    flatten: Callable[[SirenParams], Tensor] | None = None
    notes: str = ""

    @property
    def is_augmentation_bearing(self) -> bool:
        return self.augment is not None


@dataclass
class CorpusCache:
    """Lazily loads and slices the corpora of one dataset."""

    root: Path
    dataset: str
    _corpora: dict[str, tuple[SirenParams, pd.DataFrame]] = field(default_factory=dict)

    def corpus(self, protocol: str) -> tuple[SirenParams, pd.DataFrame]:
        if protocol not in self._corpora:
            self._corpora[protocol] = load_corpus(self.root / protocol)
        return self._corpora[protocol]

    def split_params(self, protocol: str) -> tuple[dict[str, SirenParams], dict[str, Tensor]]:
        params, meta = self.corpus(protocol)
        by_split, labels = {}, {}
        for split in SPLITS:
            mask = (meta["split"] == split).to_numpy()
            idx = torch.from_numpy(mask.nonzero()[0])
            by_split[split] = _index(params, idx)
            labels[split] = torch.from_numpy(meta.loc[mask, "label"].to_numpy().copy())
        return by_split, labels

    def images(self, protocol: str) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Real pixels for the same image ids, split the same way (rung P0)."""
        _, meta = self.corpus(protocol)
        x_train, _ = load_dataset(self.dataset, "train")
        x_test, _ = load_dataset(self.dataset, "test")
        out, labels = {}, {}
        for split in SPLITS:
            rows = meta[meta["split"] == split]
            ids = rows["image_id"].to_numpy()
            imgs = [
                x_test[i - TEST_ID_OFFSET] if i >= TEST_ID_OFFSET else x_train[i] for i in ids
            ]
            out[split] = torch.stack(imgs).flatten(1)
            labels[split] = torch.from_numpy(rows["label"].to_numpy().copy())
        return out, labels


def _index(params: SirenParams, idx: Tensor) -> SirenParams:
    return SirenParams(
        hidden=tuple((w[idx], b[idx]) for w, b in params.hidden),
        w_out=params.w_out[idx],
        b_out=params.b_out[idx],
    )


def _map_splits(by_split: dict[str, SirenParams], fn: Callable[[SirenParams], Tensor]) -> dict[str, Tensor]:
    return {split: fn(p) for split, p in by_split.items()}


def _chunked(fn: Callable[[SirenParams], Tensor], params: SirenParams, chunk: int = 4096) -> Tensor:
    """Apply a per-INR feature map in chunks (canonicalizers hold O(B n^2) intermediates)."""
    outs = [
        fn(_index(params, torch.arange(i, min(i + chunk, params.batch))))
        for i in range(0, params.batch, chunk)
    ]
    return torch.cat(outs)


def shared_init_template(params: SirenParams) -> SirenParams:
    """The corpus's shared initialization theta_0 (init_seed 0), as a batch-1 template.

    Data-independent by construction, and the frame W1 already lives in — so aligning
    P-random INRs to it is the like-for-like comparison the ladder wants.
    """
    widths = params.widths()
    in_dim = params.hidden[0][0].shape[2]
    out_dim = params.w_out.shape[1]
    return absorb_omega(init_from_seeds([0], in_dim, widths, out_dim))


def probe_coords(dataset: str, n_probes: int = N_PROBES) -> Tensor:
    """Deterministic subsample of the fit grid, used for activation matching."""
    side = spec_of(dataset).side
    grid = make_coord_grid(side, side)
    step = max(1, len(grid) // n_probes)
    return grid[::step][:n_probes]


def bounded_aug(sub: SirenParams, gen: torch.Generator) -> SirenParams:
    return apply(random_element(sub, gen, max_windings=AUG_MAX_WINDINGS), sub)


def render_features(params: SirenParams, dataset: str, device: str = "cpu") -> Tensor:
    """Oracle render of each INR on its fit grid, flattened (rung P1)."""
    side = spec_of(dataset).side
    coords = make_coord_grid(side, side, device=device)
    outs = []
    for i in range(0, params.batch, 2048):
        chunk = _index(params, torch.arange(i, min(i + 2048, params.batch))).to(device)
        with torch.no_grad():
            outs.append(forward_canonical(chunk, coords).clamp(-1, 1).flatten(1).cpu())
    return torch.cat(outs)


def build_frame(params: SirenParams, seed: int, size: int) -> list[GroupElement]:
    """`size` group elements drawn once and shared by every INR in the corpus (rung W9)."""
    one = _index(params, torch.zeros(1, dtype=torch.long))
    gen = torch.Generator().manual_seed(seed)
    return [random_element(one, gen, max_windings=AUG_MAX_WINDINGS) for _ in range(size)]


def _expand(g: GroupElement, batch: int) -> GroupElement:
    return GroupElement(
        d=tuple(t.expand(batch, -1) for t in g.d),
        j=tuple(t.expand(batch, -1) for t in g.j),
        perm=tuple(t.expand(batch, -1) for t in g.perm),
    )


def frame_average(params: SirenParams, frame: list[GroupElement]) -> Tensor:
    """Orbit average of the raw-weight features over a fixed frame (rung W9).

    The frame is drawn once per seed and shared by every INR — that is what makes this
    a frame rather than per-sample noise. Parity-odd coordinates average toward zero,
    which is exactly the cheap-invariantization behaviour the rung is meant to expose.
    """
    total = torch.zeros_like(params.flat())
    for g in frame:
        total = total + apply(_expand(g, params.batch), params).flat()
    return total / len(frame)
