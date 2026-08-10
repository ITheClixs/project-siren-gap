"""T20: the u-channel ablation (S14) that separates block width from colour information.

W12's (1,1) block is cat([sin b2, u]), so its width is 1+c and the output weights are the only
thing in it that scales with the output dimension. Two arms are needed to tell apart "the block
got narrower" from "colour information was removed", and this pins their construction.
"""

from __future__ import annotations

import torch

from sirengap.fitting.batched import absorb_omega, init_from_seeds
from sirengap.models.phasor import phasor_features


def _params(batch: int = 4, out_dim: int = 3):
    raw = init_from_seeds(list(range(batch)), in_dim=2, widths=(32, 32), out_dim=out_dim)
    return absorb_omega([t * 1.0 for t in raw])


def test_full_block_width_is_one_plus_c() -> None:
    for c in (1, 3):
        f = phasor_features(_params(out_dim=c))
        assert f["l2"][(1, 1)].shape[2] == 1 + c


def test_collapse_narrows_the_block_and_averages_the_channels() -> None:
    p = _params(out_dim=3)
    f = phasor_features(p, u_mode="mean")
    assert f["l2"][(1, 1)].shape[2] == 2, "collapse must leave sin(b) plus one channel"
    u = p.w_out.transpose(1, 2)
    assert torch.allclose(f["l2"][(1, 1)][:, :, -1], u.mean(dim=2), atol=1e-6)


def test_pad_keeps_the_block_width_while_carrying_the_same_information() -> None:
    p = _params(out_dim=3)
    pad = phasor_features(p, u_mode="mean_pad")
    coll = phasor_features(p, u_mode="mean")
    assert pad["l2"][(1, 1)].shape[2] == 4, "pad must match the full block width"
    # every padded channel is the same collapsed quantity: width without extra information
    chans = pad["l2"][(1, 1)][:, :, 1:]
    assert torch.allclose(chans[:, :, 0], chans[:, :, 1])
    assert torch.allclose(chans[:, :, 0], chans[:, :, 2])
    assert torch.allclose(chans[:, :, 0], coll["l2"][(1, 1)][:, :, -1])


def test_ablations_leave_every_other_block_untouched() -> None:
    p = _params(out_dim=3)
    base = phasor_features(p)
    for mode in ("mean", "mean_pad"):
        f = phasor_features(p, u_mode=mode)
        for blk in ((1, 0), (0, 1)):
            assert torch.equal(base["l2"][blk], f["l2"][blk])
        assert torch.equal(base["edge"], f["edge"])
        for blk in base["l1"]:
            assert torch.equal(base["l1"][blk], f["l1"][blk])


def test_the_neutral_block_still_sees_u_energy_in_every_mode() -> None:
    """(0,0) carries ||u||^2, which the ablation must not silently change."""
    p = _params(out_dim=3)
    base = phasor_features(p)
    for mode in ("mean", "mean_pad"):
        assert torch.equal(base["l2"][(0, 0)], phasor_features(p, u_mode=mode)["l2"][(0, 0)])


def test_default_is_unchanged() -> None:
    p = _params(out_dim=3)
    a, b = phasor_features(p), phasor_features(p, u_mode="full")
    for k in ("l1", "l2"):
        for blk in a[k]:
            assert torch.equal(a[k][blk], b[k][blk])
