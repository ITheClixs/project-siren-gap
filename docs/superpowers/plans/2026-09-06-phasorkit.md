# phasorkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pip-installable toolkit for learning on INR corpora that handles the full
function-preserving group of sine networks and reports an honest function-space baseline against the
weight-space path at matched inference cost.

**Architecture:** One data abstraction, `INRCorpus` (a flat weight tensor plus an `Architecture`),
with every capability expressed as a function on it. The headline entry point `compare()` trains a
function-space probe reader and a phasor-graded weight reader and reports both at matched downstream
inference FLOPs. The group action, phasor features and graded layers are ports of code already
property-tested in the research repository, generalized from fixed depth two to arbitrary depth.

**Tech Stack:** Python 3.11+, PyTorch 2.x (CPU and Apple Silicon MPS), pytest, ruff. No JAX, no CUDA
requirement, no network access at import time.

**Spec:** `docs/superpowers/specs/2026-09-06-phasorkit-design.md`

**Source repository for ports:** the parent research repo, `src/sirengap/`. Paths given as
`sirengap:<path>` below refer to it.

## Global Constraints

- Python floor `3.11`. The ported code uses `from __future__ import annotations` and PEP 585
  builtin generics (`tuple[int, ...]`, `dict[str, Tensor]`).
- PyTorch is the only deep-learning dependency. Runtime deps: `torch`, `numpy`. Nothing else.
- Must run on CPU and Apple Silicon MPS. Never require CUDA; never call `.cuda()`.
- All parameter transformations are immutable: return a new object, never mutate an argument.
- Files stay focused: 200-400 lines typical, 800 hard maximum.
- Test coverage floor 80% on `src/phasorkit`.
- Public name on PyPI: `phasorkit` (verified available 2026-09-06). Import name `phasorkit`.
- README carries no author name, no institution, no affiliation, and no link to the paper.
- Commit messages follow conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). No
  AI-assistance attribution in any commit message, code comment, or documentation file.
- The group's normal form, fixed for the whole project:
  `g_{d,j}: (w, b, u) -> ((-1)^d w, (-1)^d b + pi*j, (-1)^(d+j) u)`.
- The four characters, fixed for the whole project: `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)`, where a
  feature of character `(a,c)` picks up `(-1)^(a*d + c*j)`.

---

### Task 1: Repository scaffold and `Architecture`

**Files:**
- Create: `pyproject.toml`
- Create: `src/phasorkit/__init__.py`
- Create: `src/phasorkit/arch.py`
- Create: `tests/test_arch.py`
- Create: `.gitignore`, `README.md`, `LICENSE`

**Interfaces:**
- Consumes: nothing.
- Produces: `Architecture(dims: list[int], activation: str = "sine", w0: float = 30.0)` with
  properties `n_layers -> int`, `widths -> tuple[int, ...]`, `in_dim -> int`, `out_dim -> int`,
  `n_params -> int`; and methods `unflatten(flat: Tensor) -> SirenParams` and
  `flatten(params: SirenParams) -> Tensor`. `SirenParams` is defined in Task 3; for this task
  `unflatten`/`flatten` are not yet implemented and are added in Task 3.

- [ ] **Step 1: Create the repository and package skeleton**

```bash
mkdir -p ~/Development/Projects/ai-research/phasorkit
cd ~/Development/Projects/ai-research/phasorkit
git init
mkdir -p src/phasorkit tests
```

Write `pyproject.toml`:

```toml
[project]
name = "phasorkit"
version = "0.1.0"
description = "Symmetry-correct learning on implicit neural representation corpora"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = ["torch>=2.0", "numpy>=1.24"]

[project.scripts]
phasorkit = "phasorkit.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/phasorkit"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_arch.py
import pytest
from phasorkit.arch import Architecture


def test_architecture_reports_shape_facts():
    arch = Architecture(dims=[2, 32, 32, 1])
    assert arch.n_layers == 2          # hidden sine layers
    assert arch.widths == (32, 32)
    assert arch.in_dim == 2
    assert arch.out_dim == 1
    # 2*32+32 | 32*32+32 | 32*1+1
    assert arch.n_params == (2 * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)


def test_architecture_supports_arbitrary_depth():
    arch = Architecture(dims=[2, 16, 16, 16, 16, 3])
    assert arch.n_layers == 4
    assert arch.widths == (16, 16, 16, 16)
    assert arch.out_dim == 3


def test_architecture_rejects_non_sine():
    with pytest.raises(ValueError, match="only 'sine'"):
        Architecture(dims=[2, 32, 1], activation="relu")


def test_architecture_rejects_too_few_dims():
    with pytest.raises(ValueError, match="at least one hidden"):
        Architecture(dims=[2, 1])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_arch.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.arch'`

- [ ] **Step 4: Write the implementation**

```python
# src/phasorkit/arch.py
"""Description of a sine-INR family: the single source of truth for shapes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Architecture:
    """A family of sine MLPs.

    ``dims`` lists layer sizes from input to output, so ``[2, 32, 32, 1]`` is a network with two
    hidden sine layers of width 32. ``w0`` is recorded for provenance; the toolkit assumes it has
    already been absorbed into the stored weights, which is the canonical form used throughout.
    """

    dims: tuple[int, ...]
    activation: str = "sine"
    w0: float = 30.0

    def __init__(self, dims: list[int] | tuple[int, ...], activation: str = "sine",
                 w0: float = 30.0) -> None:
        object.__setattr__(self, "dims", tuple(dims))
        object.__setattr__(self, "activation", activation)
        object.__setattr__(self, "w0", w0)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.activation != "sine":
            raise ValueError(
                f"phasorkit supports only 'sine' activations, got {self.activation!r}"
            )
        if len(self.dims) < 3:
            raise ValueError(
                f"need at least one hidden layer: dims={list(self.dims)} has {len(self.dims)} entries"
            )
        if any(d <= 0 for d in self.dims):
            raise ValueError(f"all dims must be positive, got {list(self.dims)}")

    @property
    def n_layers(self) -> int:
        """Number of hidden sine layers."""
        return len(self.dims) - 2

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(self.dims[1:-1])

    @property
    def in_dim(self) -> int:
        return self.dims[0]

    @property
    def out_dim(self) -> int:
        return self.dims[-1]

    @property
    def n_params(self) -> int:
        total = 0
        for a, b in zip(self.dims[:-1], self.dims[1:]):
            total += a * b + b
        return total
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_arch.py -v`
Expected: 4 passed

- [ ] **Step 6: Write README and LICENSE**

`README.md` must contain no author name, no institution, and no link to any paper. Opening section:

```markdown
# phasorkit

Symmetry-correct learning on implicit neural representation corpora.

Weight-space models for INRs are usually built to be invariant to neuron permutation, and
sometimes to sign or scale. For periodic activations that is not the whole group: a sine network
also admits an affine phase shift on each bias, and no permutation-and-sign construction covers it.
`phasorkit` implements the full group, provides layers that are exactly invariant to it, and can
check whether a model you already have is invariant to it too.

It also ships the baseline. Reading an INR's weights is not obviously better than querying the
function it represents, and on the corpora we measured, querying wins. `compare()` runs both at
matched inference cost and tells you which is ahead on your data.

## Install

    pip install phasorkit

## Quickstart

    import phasorkit as pk

    arch   = pk.Architecture(dims=[2, 32, 32, 1])
    corpus = pk.INRCorpus.from_flat(weights, arch, labels=y)
    print(pk.compare(corpus))
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/phasorkit/__init__.py src/phasorkit/arch.py tests/test_arch.py README.md LICENSE .gitignore
git commit -m "feat: architecture description and package scaffold"
```

---

### Task 2: `SirenParams` and flat conversion

**Files:**
- Create: `src/phasorkit/params.py`
- Modify: `src/phasorkit/arch.py` (add `unflatten`, `flatten`)
- Create: `tests/test_params.py`

**Interfaces:**
- Consumes: `Architecture` from Task 1.
- Produces: `SirenParams` frozen dataclass with fields `hidden: tuple[tuple[Tensor, Tensor], ...]`,
  `w_out: Tensor`, `b_out: Tensor`; properties `batch: int`, `n_layers: int`; methods
  `widths() -> tuple[int, ...]`, `to(device) -> SirenParams`, `clone() -> SirenParams`,
  `flat() -> Tensor`. Module functions `outgoing(params, layer) -> Tensor` and
  `replace_layer(params, layer, w, b, out_w) -> SirenParams`. Plus
  `Architecture.unflatten(flat: Tensor) -> SirenParams` and
  `Architecture.flatten(params: SirenParams) -> Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_params.py
import torch
from phasorkit.arch import Architecture
from phasorkit.params import SirenParams, outgoing, replace_layer


def test_flatten_unflatten_roundtrip():
    arch = Architecture(dims=[2, 8, 8, 1])
    flat = torch.randn(5, arch.n_params)
    params = arch.unflatten(flat)
    assert params.batch == 5
    assert params.n_layers == 2
    assert params.widths() == (8, 8)
    torch.testing.assert_close(arch.flatten(params), flat)


def test_roundtrip_at_depth_four():
    arch = Architecture(dims=[3, 16, 16, 16, 16, 2])
    flat = torch.randn(4, arch.n_params)
    params = arch.unflatten(flat)
    assert params.n_layers == 4
    torch.testing.assert_close(arch.flatten(params), flat)


def test_outgoing_returns_next_matrix_then_output():
    arch = Architecture(dims=[2, 8, 5, 1])
    params = arch.unflatten(torch.randn(3, arch.n_params))
    assert outgoing(params, 0).shape == (3, 5, 8)   # W of layer 1
    assert outgoing(params, 1).shape == (3, 1, 5)   # w_out


def test_replace_layer_is_immutable():
    arch = Architecture(dims=[2, 8, 5, 1])
    params = arch.unflatten(torch.randn(3, arch.n_params))
    w, b = params.hidden[0]
    new = replace_layer(params, 0, w * 2, b, outgoing(params, 0))
    torch.testing.assert_close(params.hidden[0][0], w)     # original untouched
    torch.testing.assert_close(new.hidden[0][0], w * 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_params.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.params'`

- [ ] **Step 3: Port `params.py`**

Copy `sirengap:src/sirengap/models/params.py` verbatim into `src/phasorkit/params.py`, changing only
the module docstring's reference to the project name. The file is already depth-generic: `hidden` is
a variable-length tuple, `n_layers` is `len(self.hidden)`, and `outgoing` handles the last layer by
returning `w_out`. Do not restructure it.

- [ ] **Step 4: Add conversion to `Architecture`**

```python
# append to src/phasorkit/arch.py

    def unflatten(self, flat: "Tensor") -> "SirenParams":
        """Split a [B, n_params] tensor into structured parameters."""
        from phasorkit.params import SirenParams

        if flat.ndim != 2 or flat.shape[1] != self.n_params:
            raise ValueError(
                f"expected flat weights of shape [B, {self.n_params}] for dims={list(self.dims)}, "
                f"got {tuple(flat.shape)}"
            )
        b, off = flat.shape[0], 0
        hidden = []
        for fan_in, fan_out in zip(self.dims[:-2], self.dims[1:-1]):
            w = flat[:, off:off + fan_out * fan_in].reshape(b, fan_out, fan_in)
            off += fan_out * fan_in
            bias = flat[:, off:off + fan_out]
            off += fan_out
            hidden.append((w, bias))
        c, last = self.dims[-1], self.dims[-2]
        w_out = flat[:, off:off + c * last].reshape(b, c, last)
        off += c * last
        b_out = flat[:, off:off + c]
        return SirenParams(hidden=tuple(hidden), w_out=w_out, b_out=b_out)

    def flatten(self, params: "SirenParams") -> "Tensor":
        """Inverse of :meth:`unflatten`."""
        return params.flat()
```

Add `from torch import Tensor` under a `TYPE_CHECKING` guard at the top of `arch.py`, and
`from typing import TYPE_CHECKING`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_params.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/phasorkit/params.py src/phasorkit/arch.py tests/test_params.py
git commit -m "feat: structured parameters and flat conversion at arbitrary depth"
```

---

### Task 3: Forward pass and the group action

**Files:**
- Create: `src/phasorkit/forward.py`
- Create: `src/phasorkit/group/__init__.py`
- Create: `src/phasorkit/group/element.py`
- Create: `src/phasorkit/group/sine.py`
- Create: `tests/test_group.py`

**Interfaces:**
- Consumes: `SirenParams`, `outgoing`, `replace_layer` from Task 2.
- Produces: `forward_canonical(params: SirenParams, x: Tensor) -> Tensor` returning `[B, P, c]`;
  `max_functional_gap(a: SirenParams, b: SirenParams, x: Tensor) -> float`;
  `GroupElement` frozen dataclass with fields `d`, `j`, `perm`, each a
  `tuple[Tensor, ...]` of per-layer `[B, n_l]` tensors;
  `random_element(params, generator, max_windings=3, identity_perm=False) -> GroupElement`;
  `apply(g: GroupElement, params: SirenParams) -> SirenParams`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_group.py
import torch
from phasorkit.arch import Architecture
from phasorkit.forward import forward_canonical, max_functional_gap
from phasorkit.group.sine import apply, random_element


def _corpus(dims, batch=4, seed=0):
    arch = Architecture(dims=dims)
    g = torch.Generator().manual_seed(seed)
    flat = torch.randn(batch, arch.n_params, generator=g, dtype=torch.float64) * 0.5
    return arch, arch.unflatten(flat)


def test_action_preserves_function_at_every_depth():
    """The defining property: f(g.theta) == f(theta) exactly."""
    for dims in ([2, 8, 1], [2, 8, 8, 1], [2, 6, 6, 6, 1], [3, 5, 5, 5, 5, 2]):
        arch, params = _corpus(dims)
        x = torch.randn(64, arch.in_dim, dtype=torch.float64)
        gen = torch.Generator().manual_seed(1)
        g = random_element(params, gen, max_windings=7)
        moved = apply(g, params)
        gap = max_functional_gap(params, moved, x)
        assert gap < 1e-10, f"dims={dims} moved the function by {gap}"


def test_action_actually_changes_the_weights():
    """Guard against a vacuous pass from an identity action."""
    arch, params = _corpus([2, 8, 8, 1])
    gen = torch.Generator().manual_seed(3)
    g = random_element(params, gen, max_windings=5)
    moved = apply(g, params)
    assert not torch.allclose(params.flat(), moved.flat())


def test_large_windings_still_preserve_function():
    arch, params = _corpus([2, 8, 8, 1])
    x = torch.randn(32, 2, dtype=torch.float64)
    gen = torch.Generator().manual_seed(5)
    g = random_element(params, gen, max_windings=40)
    assert max_functional_gap(params, apply(g, params), x) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_group.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.forward'`

- [ ] **Step 3: Port forward and group**

Copy `sirengap:src/sirengap/models/forward.py` into `src/phasorkit/forward.py`, and
`sirengap:src/sirengap/symmetry/dinf.py` split into `src/phasorkit/group/element.py`
(`GroupElement`, `random_element`) and `src/phasorkit/group/sine.py` (`_apply_neuronwise`,
`_apply_perm`, `apply`). Update imports to `phasorkit.params`. Re-export both from
`src/phasorkit/group/__init__.py`.

The ported `apply` already loops `for layer in range(params.n_layers)`, so it is depth-generic with
no change. Do not "improve" the sign conventions; they are load-bearing and the test above is what
guards them.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_group.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/phasorkit/forward.py src/phasorkit/group tests/test_group.py
git commit -m "feat: canonical forward pass and the D-infinity wreath action"
```

---

### Task 4: Phasor features at arbitrary depth

**Files:**
- Create: `src/phasorkit/features/__init__.py`
- Create: `src/phasorkit/features/phasor.py`
- Create: `tests/test_phasor_features.py`

**Interfaces:**
- Consumes: `SirenParams`, `outgoing` (Task 2); `GroupElement`, `apply`, `random_element` (Task 3).
- Produces: `Character = tuple[int, int]`; `CHARACTERS: tuple[Character, ...]`;
  `PhasorFeatures` frozen dataclass with fields `nodes: list[dict[Character, Tensor]]` (one entry
  per hidden layer, each `[B, n_l, d_chi]`) and `edges: list[Tensor]` (one per consecutive pair,
  `[B, n_{l+1}, n_l]`); `phasor_features(params: SirenParams) -> PhasorFeatures`;
  `character_sign(chi: Character, d: Tensor, j: Tensor) -> Tensor`.

**Background the implementer needs.** Under `g_{d,j}` the bias phasors transform with `j` only
through its parity: `cos b -> (-1)^j cos b`, `sin b -> (-1)^(d+j) sin b`, `cos 2b -> cos 2b`,
`sin 2b -> (-1)^d sin 2b`, while `w -> (-1)^d w` and `u -> (-1)^(d+j) u`. Writing a character
`(a, c)` for a feature picking up `(-1)^(a*d + c*j)`, the blocks are:

| character | contents |
|---|---|
| `(0,0)` | incoming energy, outgoing energy, `cos 2b`, ones |
| `(1,0)` | `sin 2b`, and `W_0` on the first hidden layer only |
| `(0,1)` | `cos b` |
| `(1,1)` | `sin b`, and `u` on the last hidden layer only |

The two extra entries appear exactly at the two boundaries where one side of a matrix is not acted
on by the group: the input side of `W_0` and the output side of `w_out`. Every interior matrix is an
*edge*, not a node feature, because both of its sides are acted on.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phasor_features.py
import torch
from phasorkit.arch import Architecture
from phasorkit.features.phasor import CHARACTERS, character_sign, phasor_features
from phasorkit.group.sine import apply, random_element


def _params(dims, batch=3, seed=0):
    arch = Architecture(dims=dims)
    g = torch.Generator().manual_seed(seed)
    return arch, arch.unflatten(torch.randn(batch, arch.n_params, generator=g, dtype=torch.float64))


def test_shapes_at_arbitrary_depth():
    arch, params = _params([2, 7, 5, 4, 1])
    feats = phasor_features(params)
    assert len(feats.nodes) == 3
    assert len(feats.edges) == 2
    assert feats.nodes[0][(0, 0)].shape[:2] == (3, 7)
    assert feats.nodes[2][(0, 0)].shape[:2] == (3, 4)
    assert feats.edges[0].shape == (3, 5, 7)
    assert feats.edges[1].shape == (3, 4, 5)


def test_only_first_layer_carries_w_and_only_last_carries_u():
    arch, params = _params([2, 7, 5, 4, 1])
    f = phasor_features(params)
    # (1,0) = sin2b (1 channel) + W_0 (in_dim channels) on layer 0, sin2b alone after
    assert f.nodes[0][(1, 0)].shape[2] == 1 + 2
    assert f.nodes[1][(1, 0)].shape[2] == 1
    # (1,1) = sin b (1 channel) + u (out_dim channels) on the last layer, sin b alone before
    assert f.nodes[0][(1, 1)].shape[2] == 1
    assert f.nodes[2][(1, 1)].shape[2] == 1 + 1


def test_features_transform_by_their_character():
    """The whole construction rests on this: each block picks up exactly (-1)^(a d + c j)."""
    for dims in ([2, 8, 1], [2, 6, 6, 1], [2, 5, 5, 5, 1]):
        arch, params = _params(dims, seed=2)
        gen = torch.Generator().manual_seed(9)
        g = random_element(params, gen, max_windings=6, identity_perm=True)
        before = phasor_features(params)
        after = phasor_features(apply(g, params))
        for layer in range(params.n_layers):
            for chi in CHARACTERS:
                sign = character_sign(chi, g.d[layer], g.j[layer])   # [B, n]
                expected = sign[:, :, None] * before.nodes[layer][chi]
                torch.testing.assert_close(
                    after.nodes[layer][chi], expected, atol=1e-10, rtol=1e-10,
                    msg=f"dims={dims} layer={layer} character={chi}",
                )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_phasor_features.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.features'`

- [ ] **Step 3: Write the implementation**

```python
# src/phasorkit/features/phasor.py
"""Phasor lift and character grading, at arbitrary depth.

Under the per-neuron element g_{d,j}: (w, b, u) -> ((-1)^d w, (-1)^d b + pi j, (-1)^(d+j) u),
the phasor coordinates of the bias transform with j only through its parity:

    cos b  -> (-1)^j cos b        sin b  -> (-1)^(d+j) sin b
    cos 2b -> cos 2b              sin 2b -> (-1)^d sin 2b

so replacing b by (cos b, sin b) turns the infinite group Z semidirect Z_2 into a finite
Z_2 x Z_2 acting by signs. A feature of character (a, c) picks up (-1)^(a d + c j).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from phasorkit.params import SirenParams, outgoing

Character = tuple[int, int]
CHARACTERS: tuple[Character, ...] = ((0, 0), (1, 0), (0, 1), (1, 1))


def character_sign(chi: Character, d: Tensor, j: Tensor) -> Tensor:
    """(-1)^(a*d + c*j) for character chi=(a,c), elementwise over [B, n]."""
    a, c = chi
    expo = a * d + c * j
    return torch.where(expo % 2 == 0, 1.0, -1.0).to(torch.get_default_dtype())


@dataclass(frozen=True)
class PhasorFeatures:
    """Graded node features per hidden layer, and the coupling matrix per consecutive pair."""

    nodes: list[dict[Character, Tensor]]
    edges: list[Tensor]


def phasor_features(params: SirenParams) -> PhasorFeatures:
    """Graded node features and coupling matrices. No learned parameters, no pooling."""
    n_layers = params.n_layers
    edges = [outgoing(params, layer) for layer in range(n_layers - 1)]
    u = params.w_out.transpose(1, 2)  # [B, n_last, c]

    nodes: list[dict[Character, Tensor]] = []
    for layer in range(n_layers):
        w, b = params.hidden[layer]
        ones = torch.ones_like(b)[:, :, None]

        if layer == 0:
            e_in = (w * w).sum(2, keepdim=True)
        else:
            e_in = (edges[layer - 1] ** 2).sum(2, keepdim=True)

        if layer == n_layers - 1:
            e_out = (u * u).sum(2, keepdim=True)
        else:
            e_out = (edges[layer] ** 2).sum(1).unsqueeze(2)

        neutral = torch.cat([e_in, e_out, torch.cos(2 * b)[:, :, None], ones], dim=2)

        sign_parts = [torch.sin(2 * b)[:, :, None]]
        if layer == 0:
            sign_parts.insert(0, w)

        odd_parts = [torch.sin(b)[:, :, None]]
        if layer == n_layers - 1:
            odd_parts.append(u)

        nodes.append({
            (0, 0): neutral,
            (1, 0): torch.cat(sign_parts, dim=2),
            (0, 1): torch.cos(b)[:, :, None],
            (1, 1): torch.cat(odd_parts, dim=2),
        })

    return PhasorFeatures(nodes=nodes, edges=edges)
```

Note for the implementer: the channel *order* inside the `(0,0)` block differs from the depth-two
implementation in the research repository, which interleaved `cos 2b` between the two energies. Order
is immaterial because the first operation applied to each block is a per-character linear map, so do
not try to reproduce the old ordering byte for byte.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_phasor_features.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/phasorkit/features tests/test_phasor_features.py
git commit -m "feat: phasor lift and character grading at arbitrary depth"
```

---

### Task 5: Graded layers

**Files:**
- Create: `src/phasorkit/layers/__init__.py`
- Create: `src/phasorkit/layers/graded.py`
- Create: `tests/test_layers.py`

**Interfaces:**
- Consumes: `Character`, `CHARACTERS`, `PhasorFeatures` (Task 4).
- Produces: `GradedLinear(dims_in: dict[Character, int], width: int, graded: bool = True)`;
  `GradedBilinear(width: int, graded: bool = True)`; `GradedAct(graded: bool = True)`;
  `GradedMessage(width: int, graded: bool = True)`. `GradedLinear`, `GradedBilinear` and
  `GradedAct` take and return `dict[Character, Tensor]`. `GradedMessage.forward(nodes:
  list[dict[Character, Tensor]], edges: list[Tensor]) -> list[dict[Character, Tensor]]`, coupling
  every consecutive pair.

**Background.** The grading is preserved by construction: a per-character linear map carries a bias
only on `(0,0)`; bilinear products are routed to the sum of their characters (addition mod 2
componentwise); the nonlinearity is *odd* on every non-neutral block, so it commutes with a sign.
The edge `E_l` carries character `(1,1)` on the layer-`l` side and `(1,0)` on the layer-`l+1` side,
which leaves exactly two admissible message channels per direction: one through `E` and one through
`E^2`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layers.py
import torch
from phasorkit.arch import Architecture
from phasorkit.features.phasor import CHARACTERS, character_sign, phasor_features
from phasorkit.group.sine import apply, random_element
from phasorkit.layers.graded import GradedAct, GradedBilinear, GradedLinear, GradedMessage


def _params(dims, batch=3, seed=0):
    arch = Architecture(dims=dims)
    g = torch.Generator().manual_seed(seed)
    return arch.unflatten(torch.randn(batch, arch.n_params, generator=g))


def test_graded_linear_preserves_covariance():
    params = _params([2, 6, 6, 1])
    feats = phasor_features(params)
    dims_in = {c: feats.nodes[0][c].shape[2] for c in CHARACTERS}
    lin = GradedLinear(dims_in, width=8)
    gen = torch.Generator().manual_seed(4)
    g = random_element(params, gen, max_windings=5, identity_perm=True)
    moved = phasor_features(apply(g, params))
    out_before = lin(feats.nodes[0])
    out_after = lin(moved.nodes[0])
    for chi in CHARACTERS:
        sign = character_sign(chi, g.d[0], g.j[0])[:, :, None]
        torch.testing.assert_close(out_after[chi], sign * out_before[chi], atol=1e-5, rtol=1e-5)


def test_graded_act_is_odd_off_the_neutral_block():
    act = GradedAct()
    x = {c: torch.randn(2, 4, 5) for c in CHARACTERS}
    neg = {c: (x[c] if c == (0, 0) else -x[c]) for c in CHARACTERS}
    out_x, out_neg = act(x), act(neg)
    for chi in CHARACTERS:
        expected = out_x[chi] if chi == (0, 0) else -out_x[chi]
        torch.testing.assert_close(out_neg[chi], expected)


def test_message_runs_over_every_consecutive_pair():
    params = _params([2, 7, 5, 4, 1])
    feats = phasor_features(params)
    lin = GradedLinear({c: feats.nodes[0][c].shape[2] for c in CHARACTERS}, width=8)
    # project every layer to a common width first
    nodes = []
    for layer in range(params.n_layers):
        per = GradedLinear({c: feats.nodes[layer][c].shape[2] for c in CHARACTERS}, width=8)
        nodes.append(per(feats.nodes[layer]))
    msg = GradedMessage(width=8)
    out = msg(nodes, feats.edges)
    assert len(out) == 3
    assert out[0][(0, 0)].shape == (3, 7, 8)
    assert out[2][(0, 0)].shape == (3, 4, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layers.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.layers'`

- [ ] **Step 3: Write the implementation**

Port `GradedLinear`, `GradedBilinear` and `GradedAct` from
`sirengap:src/sirengap/models/phasor.py` (lines 163-241) unchanged apart from imports; they are
already depth-agnostic because they act on a single layer's feature dict.

`GradedMessage` needs rewriting for a variable number of layers. The depth-two version couples one
fixed pair through four `nn.Linear` maps; the general version keeps the same four maps and applies
them along every consecutive pair, accumulating into each layer:

```python
class GradedMessage(nn.Module):
    """Couple consecutive hidden layers through their shared matrix.

    E_l carries character (1,1) on the layer-l side and (1,0) on the layer-(l+1) side, so exactly
    two channels are admissible per direction: one through E, one through E^2.
    """

    def __init__(self, width: int, graded: bool = True) -> None:
        super().__init__()
        self.graded = graded
        self.up_odd = nn.Linear(width, width, bias=False)
        self.up_even = nn.Linear(width, width, bias=False)
        self.down_odd = nn.Linear(width, width, bias=False)
        self.down_even = nn.Linear(width, width, bias=False)

    def forward(self, nodes: list[dict[Character, Tensor]],
                edges: list[Tensor]) -> list[dict[Character, Tensor]]:
        out = [dict(layer) for layer in nodes]
        for idx, e in enumerate(edges):
            lo, hi = nodes[idx], nodes[idx + 1]
            n, p = lo[(0, 0)].shape[1], hi[(0, 0)].shape[1]
            e2 = e * e
            src_up_odd = lo[(1, 1)] if self.graded else lo[(0, 0)]
            src_down_odd = hi[(1, 0)] if self.graded else hi[(0, 0)]
            up_odd = torch.einsum("bpn,bnd->bpd", e, self.up_odd(src_up_odd)) / n
            up_even = torch.einsum("bpn,bnd->bpd", e2, self.up_even(lo[(0, 0)])) / n
            down_odd = torch.einsum("bpn,bpd->bnd", e, self.down_odd(src_down_odd)) / p
            down_even = torch.einsum("bpn,bpd->bnd", e2, self.down_even(hi[(0, 0)])) / p
            hi_key = (1, 0) if self.graded else (0, 0)
            lo_key = (1, 1) if self.graded else (0, 0)
            out[idx + 1][hi_key] = out[idx + 1][hi_key] + up_odd
            out[idx + 1][(0, 0)] = out[idx + 1][(0, 0)] + up_even
            out[idx][lo_key] = out[idx][lo_key] + down_odd
            out[idx][(0, 0)] = out[idx][(0, 0)] + down_even
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_layers.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/phasorkit/layers tests/test_layers.py
git commit -m "feat: character-graded layers with message passing over every layer pair"
```

---

### Task 6: The weight reader, and the invariance property that justifies it

**Files:**
- Create: `src/phasorkit/readers/__init__.py`
- Create: `src/phasorkit/readers/weight.py`
- Create: `tests/test_weight_reader.py`

**Interfaces:**
- Consumes: Tasks 4 and 5.
- Produces: `PhasorGradedReader(arch: Architecture, width: int = 256, n_classes: int = 10,
  depth: int = 3, graded: bool = True)` with `forward(params: SirenParams) -> Tensor` of shape
  `[B, n_classes]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_weight_reader.py
import torch
from phasorkit.arch import Architecture
from phasorkit.group.sine import apply, random_element
from phasorkit.readers.weight import PhasorGradedReader


def test_reader_is_exactly_invariant_at_every_depth():
    """The property the whole library exists to provide."""
    for dims in ([2, 8, 1], [2, 8, 8, 1], [2, 6, 6, 6, 1]):
        arch = Architecture(dims=dims)
        gen = torch.Generator().manual_seed(0)
        params = arch.unflatten(torch.randn(4, arch.n_params, generator=gen))
        reader = PhasorGradedReader(arch, width=16, n_classes=5).double()
        params = params.to(torch.float64) if hasattr(params, "to") else params
        reader.eval()
        with torch.no_grad():
            before = reader(params)
            g = random_element(params, torch.Generator().manual_seed(7), max_windings=40)
            after = reader(apply(g, params))
        rel = (after - before).abs().max() / before.abs().max().clamp_min(1e-12)
        assert rel < 1e-6, f"dims={dims} moved by relative {rel.item()}"


def test_ungraded_variant_is_not_invariant():
    """Negative control: without the grading the same skeleton must move."""
    arch = Architecture(dims=[2, 8, 8, 1])
    gen = torch.Generator().manual_seed(0)
    params = arch.unflatten(torch.randn(4, arch.n_params, generator=gen))
    reader = PhasorGradedReader(arch, width=16, n_classes=5, graded=False)
    reader.eval()
    with torch.no_grad():
        before = reader(params)
        g = random_element(params, torch.Generator().manual_seed(7), max_windings=10)
        after = reader(apply(g, params))
    assert (after - before).abs().max() > 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_weight_reader.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.readers'`

- [ ] **Step 3: Write the implementation**

```python
# src/phasorkit/readers/weight.py
"""A reader on raw parameters that is exactly invariant to the full group."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from phasorkit.arch import Architecture
from phasorkit.features.phasor import CHARACTERS, phasor_features
from phasorkit.layers.graded import GradedAct, GradedBilinear, GradedLinear, GradedMessage
from phasorkit.params import SirenParams


class PhasorGradedReader(nn.Module):
    """Phasor lift, then graded layers, then a head reading only the neutral block.

    Invariance is by construction: every block transforms by its own character, the head sees only
    the character-(0,0) block, and that block is invariant. Pooling is mean and max over neurons,
    both permutation invariant.
    """

    def __init__(self, arch: Architecture, width: int = 256, n_classes: int = 10,
                 depth: int = 3, graded: bool = True) -> None:
        super().__init__()
        self.arch, self.graded = arch, graded
        probe = arch.unflatten(torch.zeros(1, arch.n_params))
        feats = phasor_features(probe)
        self.stems = nn.ModuleList([
            GradedLinear({c: feats.nodes[l][c].shape[2] for c in CHARACTERS}, width, graded)
            for l in range(arch.n_layers)
        ])
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                GradedMessage(width, graded),
                GradedBilinear(width, graded),
                GradedAct(graded),
            ]))
        self.head = nn.Sequential(
            nn.Linear(2 * width * arch.n_layers, width), nn.ReLU(), nn.Linear(width, n_classes)
        )

    def forward(self, params: SirenParams) -> Tensor:
        feats = phasor_features(params)
        nodes = [stem(feats.nodes[l]) for l, stem in enumerate(self.stems)]
        for message, bilinear, act in self.blocks:
            nodes = message(nodes, feats.edges)
            nodes = [act(bilinear(n)) for n in nodes]
        pooled = []
        for n in nodes:
            neutral = n[(0, 0)]
            pooled += [neutral.mean(dim=1), neutral.amax(dim=1)]
        return self.head(torch.cat(pooled, dim=1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_weight_reader.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/phasorkit/readers tests/test_weight_reader.py
git commit -m "feat: exactly invariant phasor-graded reader on raw parameters"
```

---

### Task 7: The function-space probe reader

**Files:**
- Create: `src/phasorkit/readers/probe.py`
- Create: `tests/test_probe_reader.py`

**Interfaces:**
- Consumes: `forward_canonical` (Task 3), `Architecture` (Task 1).
- Produces: `ProbeReader(arch: Architecture, n_probes: int = 64, n_classes: int = 10,
  hidden: int = 256)` with `forward(params: SirenParams) -> Tensor` of shape `[B, n_classes]` and
  attribute `coords: nn.Parameter` of shape `[n_probes, in_dim]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_probe_reader.py
import torch
from phasorkit.arch import Architecture
from phasorkit.group.sine import apply, random_element
from phasorkit.readers.probe import ProbeReader


def test_probe_reader_output_shape():
    arch = Architecture(dims=[2, 8, 8, 1])
    params = arch.unflatten(torch.randn(5, arch.n_params))
    reader = ProbeReader(arch, n_probes=16, n_classes=7)
    assert reader(params).shape == (5, 7)
    assert reader.coords.shape == (16, 2)


def test_probe_reader_is_invariant_because_it_reads_the_function():
    """It never touches parameter coordinates, so the group cannot move it."""
    arch = Architecture(dims=[2, 8, 8, 1])
    gen = torch.Generator().manual_seed(0)
    params = arch.unflatten(torch.randn(4, arch.n_params, generator=gen, dtype=torch.float64))
    reader = ProbeReader(arch, n_probes=16, n_classes=5).double()
    reader.eval()
    with torch.no_grad():
        before = reader(params)
        g = random_element(params, torch.Generator().manual_seed(3), max_windings=25)
        after = reader(apply(g, params))
    torch.testing.assert_close(after, before, atol=1e-9, rtol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_probe_reader.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.readers.probe'`

- [ ] **Step 3: Write the implementation**

```python
# src/phasorkit/readers/probe.py
"""Read an INR by querying the function it represents at learned coordinates.

This reader is invariant to the parameter symmetry group for a reason no construction is needed to
supply: it never touches parameter coordinates at all. It is also the baseline the weight-space path
has to beat, which on the corpora we measured it does not.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from phasorkit.arch import Architecture
from phasorkit.forward import forward_canonical
from phasorkit.params import SirenParams


class ProbeReader(nn.Module):
    def __init__(self, arch: Architecture, n_probes: int = 64, n_classes: int = 10,
                 hidden: int = 256) -> None:
        super().__init__()
        self.arch, self.n_probes = arch, n_probes
        self.coords = nn.Parameter(torch.rand(n_probes, arch.in_dim) * 2 - 1)
        self.net = nn.Sequential(
            nn.Linear(n_probes * arch.out_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, params: SirenParams) -> Tensor:
        values = forward_canonical(params, self.coords.to(params.w_out.dtype))  # [B, P, c]
        return self.net(values.reshape(values.shape[0], -1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_probe_reader.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/phasorkit/readers/probe.py tests/test_probe_reader.py
git commit -m "feat: function-space probe reader at learned coordinates"
```

---

### Task 8: `INRCorpus` and adapters

**Files:**
- Create: `src/phasorkit/corpus.py`
- Create: `src/phasorkit/adapters/__init__.py`
- Create: `src/phasorkit/adapters/dwsnets.py`
- Create: `src/phasorkit/adapters/checkpoints.py`
- Create: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `Architecture` (Task 1), `SirenParams` (Task 2).
- Produces: `INRCorpus` frozen dataclass with fields `weights: Tensor [N, D]`,
  `arch: Architecture`, `labels: Tensor | None`; classmethod
  `from_flat(weights, arch, labels=None) -> INRCorpus`; methods `params() -> SirenParams`,
  `split(train=0.8, val=0.1, seed=0) -> tuple[INRCorpus, INRCorpus, INRCorpus]`,
  `__len__() -> int`, `subset(idx: Tensor) -> INRCorpus`, `to(device) -> INRCorpus`.
  Adapters: `phasorkit.adapters.dwsnets(path, arch) -> INRCorpus`;
  `phasorkit.adapters.checkpoint_dir(path, arch) -> INRCorpus`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus.py
import pytest
import torch
from phasorkit.arch import Architecture
from phasorkit.corpus import INRCorpus


def test_corpus_validates_shape_against_architecture():
    arch = Architecture(dims=[2, 8, 8, 1])
    with pytest.raises(ValueError, match="expected flat weights"):
        INRCorpus.from_flat(torch.randn(10, 7), arch)


def test_split_is_disjoint_and_covers_everything():
    arch = Architecture(dims=[2, 8, 8, 1])
    c = INRCorpus.from_flat(torch.randn(100, arch.n_params), arch,
                            labels=torch.randint(0, 5, (100,)))
    tr, va, te = c.split(train=0.8, val=0.1, seed=0)
    assert len(tr) + len(va) + len(te) == 100
    assert len(tr) == 80 and len(va) == 10 and len(te) == 10


def test_split_is_deterministic_under_seed():
    arch = Architecture(dims=[2, 8, 8, 1])
    c = INRCorpus.from_flat(torch.randn(50, arch.n_params), arch)
    a, _, _ = c.split(seed=3)
    b, _, _ = c.split(seed=3)
    torch.testing.assert_close(a.weights, b.weights)


def test_params_roundtrips_through_architecture():
    arch = Architecture(dims=[2, 8, 5, 1])
    c = INRCorpus.from_flat(torch.randn(6, arch.n_params), arch)
    p = c.params()
    assert p.batch == 6 and p.n_layers == 2
    torch.testing.assert_close(p.flat(), c.weights)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corpus.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.corpus'`

- [ ] **Step 3: Write `corpus.py`**

```python
# src/phasorkit/corpus.py
"""A corpus of INRs: flat weights, the architecture they share, and optional labels."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from phasorkit.arch import Architecture
from phasorkit.params import SirenParams


@dataclass(frozen=True)
class INRCorpus:
    weights: Tensor
    arch: Architecture
    labels: Tensor | None = None

    @classmethod
    def from_flat(cls, weights: Tensor, arch: Architecture,
                  labels: Tensor | None = None) -> "INRCorpus":
        if weights.ndim != 2 or weights.shape[1] != arch.n_params:
            raise ValueError(
                f"expected flat weights of shape [N, {arch.n_params}] for "
                f"dims={list(arch.dims)}, got {tuple(weights.shape)}"
            )
        if labels is not None and labels.shape[0] != weights.shape[0]:
            raise ValueError(
                f"labels have {labels.shape[0]} entries but corpus has {weights.shape[0]} INRs"
            )
        return cls(weights=weights, arch=arch, labels=labels)

    def __len__(self) -> int:
        return self.weights.shape[0]

    def params(self) -> SirenParams:
        return self.arch.unflatten(self.weights)

    def subset(self, idx: Tensor) -> "INRCorpus":
        return replace(self, weights=self.weights[idx],
                       labels=None if self.labels is None else self.labels[idx])

    def to(self, device: torch.device | str) -> "INRCorpus":
        return replace(self, weights=self.weights.to(device),
                       labels=None if self.labels is None else self.labels.to(device))

    def split(self, train: float = 0.8, val: float = 0.1,
              seed: int = 0) -> tuple["INRCorpus", "INRCorpus", "INRCorpus"]:
        n = len(self)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        n_tr = int(round(train * n))
        n_va = int(round(val * n))
        return (self.subset(perm[:n_tr]),
                self.subset(perm[n_tr:n_tr + n_va]),
                self.subset(perm[n_tr + n_va:]))
```

- [ ] **Step 4: Write the adapters**

`adapters/checkpoints.py` loads a directory of `.pt` files, each a `state_dict` of an MLP, and
stacks them into `[N, D]` in the `Architecture`'s flat order:

```python
def checkpoint_dir(path: str | Path, arch: Architecture,
                   labels: Tensor | None = None) -> INRCorpus:
    """Load every .pt state_dict under `path`, sorted by filename, into one corpus."""
    files = sorted(Path(path).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt checkpoints under {path}")
    rows = []
    for f in files:
        sd = torch.load(f, map_location="cpu")
        rows.append(_state_dict_to_flat(sd, arch))
    return INRCorpus.from_flat(torch.stack(rows), arch, labels)
```

`_state_dict_to_flat` walks the state dict in insertion order, asserts each tensor's shape matches
the corresponding `Architecture` slot, and concatenates weight then bias per layer, raising a
`ValueError` naming the offending key and both shapes on mismatch.

`adapters/dwsnets.py` reads the benchmark layout, which stores per-INR weights as a list of arrays
under keys `w0..wL` and `b0..bL` in a `.npz`, and maps them onto the same flat order.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_corpus.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/phasorkit/corpus.py src/phasorkit/adapters tests/test_corpus.py
git commit -m "feat: INR corpus abstraction with checkpoint and benchmark adapters"
```

---

### Task 9: FLOPs accounting

**Files:**
- Create: `src/phasorkit/flops.py`
- Create: `tests/test_flops.py`

**Interfaces:**
- Consumes: `Architecture` (Task 1).
- Produces: `MAC: int = 2`, `SIN: int = 1`;
  `siren_forward(arch: Architecture, n_points: int) -> int`;
  `mlp_forward(dims: list[int]) -> int`;
  `probe_cost(arch, n_probes, n_classes, hidden) -> dict[str, int]`;
  `weight_reader_cost(arch, width, depth, n_classes) -> dict[str, int]`.
  Each cost dict has keys `"per_item"` and `"detail"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flops.py
from phasorkit.arch import Architecture
from phasorkit.flops import mlp_forward, probe_cost, siren_forward, weight_reader_cost


def test_siren_forward_counts_macs_and_transcendentals():
    arch = Architecture(dims=[2, 4, 1])
    # layer 1: 2*4 macs + 4 sins ; out: 4*1 macs
    assert siren_forward(arch, n_points=1) == (2 * 4) * 2 + 4 * 1 + (4 * 1) * 2


def test_probe_cost_scales_linearly_in_probes():
    arch = Architecture(dims=[2, 8, 8, 1])
    a = probe_cost(arch, n_probes=16, n_classes=10, hidden=64)["per_item"]
    b = probe_cost(arch, n_probes=32, n_classes=10, hidden=64)["per_item"]
    assert b > a


def test_weight_reader_cost_is_positive_and_grows_with_width():
    arch = Architecture(dims=[2, 8, 8, 1])
    a = weight_reader_cost(arch, width=32, depth=3, n_classes=10)["per_item"]
    b = weight_reader_cost(arch, width=64, depth=3, n_classes=10)["per_item"]
    assert 0 < a < b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_flops.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.flops'`

- [ ] **Step 3: Port and adapt**

Port `sirengap:src/sirengap/eval/flops.py`. Replace its local `Arch` dataclass with
`phasorkit.arch.Architecture`, and rename `function_query` to `probe_cost` and
`weight_phasor_reader` to `weight_reader_cost`, generalizing both to `arch.n_layers` layers rather
than a hardcoded two. Keep `MAC = 2` and `SIN = 1`: they are the accounting convention the published
numbers were computed under, and changing them silently would make comparisons incomparable.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_flops.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/phasorkit/flops.py tests/test_flops.py
git commit -m "feat: FLOPs accounting for the probe and weight-reader paths"
```

---

### Task 10: `compare()`, the headline

**Files:**
- Create: `src/phasorkit/train.py`
- Create: `src/phasorkit/compare.py`
- Create: `tests/test_compare.py`

**Interfaces:**
- Consumes: Tasks 6, 7, 8, 9.
- Produces: `fit_reader(reader: nn.Module, train: INRCorpus, val: INRCorpus, epochs: int,
  lr: float, batch_size: int, device: str) -> float` returning validation accuracy;
  `ComparisonReport` frozen dataclass with fields `probe_acc: float`, `weight_acc: float`,
  `probe_flops: int`, `weight_flops: int`, `winner: str`, `cost_ratio: float`, and `__str__`;
  `compare(corpus: INRCorpus, budget: str = "matched", n_probes: int = 64, epochs: int = 20,
  seed: int = 0, device: str | None = None) -> ComparisonReport`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare.py
import pytest
import torch
from phasorkit.arch import Architecture
from phasorkit.compare import ComparisonReport, compare
from phasorkit.corpus import INRCorpus


def _tiny_corpus(n=64, seed=0):
    """Two classes separated by a scale factor, learnable by either path."""
    arch = Architecture(dims=[2, 6, 6, 1])
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, arch.n_params, generator=g) * 0.3
    y = torch.randint(0, 2, (n,), generator=g)
    w[y == 1] *= 2.0
    return INRCorpus.from_flat(w, arch, labels=y)


def test_compare_reports_both_paths_and_a_winner():
    report = compare(_tiny_corpus(), epochs=2, n_probes=8, seed=0)
    assert isinstance(report, ComparisonReport)
    assert 0.0 <= report.probe_acc <= 1.0
    assert 0.0 <= report.weight_acc <= 1.0
    assert report.winner in ("probe", "weight", "tie")
    assert report.probe_flops > 0 and report.weight_flops > 0


def test_matched_budget_equalizes_inference_flops_within_tolerance():
    report = compare(_tiny_corpus(), budget="matched", epochs=1, n_probes=8)
    ratio = report.weight_flops / report.probe_flops
    assert 0.5 <= ratio <= 2.0, f"matched budget left a {ratio:.2f}x gap"


def test_free_budget_reports_the_cost_ratio_instead():
    report = compare(_tiny_corpus(), budget="free", epochs=1, n_probes=8)
    assert report.cost_ratio > 0


def test_compare_rejects_unlabelled_corpora():
    arch = Architecture(dims=[2, 6, 6, 1])
    c = INRCorpus.from_flat(torch.randn(20, arch.n_params), arch)
    with pytest.raises(ValueError, match="labels"):
        compare(c, epochs=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.compare'`

- [ ] **Step 3: Write `train.py`**

A single minimal training loop shared by both readers: Adam, cross-entropy, constant step, early
selection on validation accuracy. It takes a `reader` and an `INRCorpus`, slices `corpus.weights`
into minibatches, calls `corpus.arch.unflatten` on each batch, and returns the best validation
accuracy seen. Keep it under 120 lines and do not add schedulers, mixed precision, or checkpointing.

- [ ] **Step 4: Write `compare.py`**

```python
# src/phasorkit/compare.py
"""Run both readers on one corpus and report which is ahead at matched inference cost."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phasorkit.corpus import INRCorpus
from phasorkit.flops import probe_cost, weight_reader_cost
from phasorkit.readers.probe import ProbeReader
from phasorkit.readers.weight import PhasorGradedReader
from phasorkit.train import fit_reader


@dataclass(frozen=True)
class ComparisonReport:
    probe_acc: float
    weight_acc: float
    probe_flops: int
    weight_flops: int
    winner: str
    cost_ratio: float

    def __str__(self) -> str:
        return (
            f"function probe   {self.probe_acc:6.2%}   {self.probe_flops:>12,} FLOPs/item\n"
            f"weight reader    {self.weight_acc:6.2%}   {self.weight_flops:>12,} FLOPs/item\n"
            f"winner: {self.winner}   weight/probe inference cost: {self.cost_ratio:.1f}x"
        )


def _width_for_budget(arch, target_flops: int, depth: int, n_classes: int) -> int:
    """Largest width whose inference cost stays under the probe's budget."""
    lo, hi = 4, 512
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        cost = weight_reader_cost(arch, width=mid, depth=depth, n_classes=n_classes)["per_item"]
        if cost <= target_flops:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def compare(corpus: INRCorpus, budget: str = "matched", n_probes: int = 64,
            epochs: int = 20, seed: int = 0, device: str | None = None) -> ComparisonReport:
    if corpus.labels is None:
        raise ValueError("compare() needs labels; build the corpus with INRCorpus.from_flat(..., labels=y)")
    if budget not in ("matched", "free"):
        raise ValueError(f"budget must be 'matched' or 'free', got {budget!r}")
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    n_classes = int(corpus.labels.max().item()) + 1
    train, val, _ = corpus.split(seed=seed)

    torch.manual_seed(seed)
    probe = ProbeReader(corpus.arch, n_probes=n_probes, n_classes=n_classes)
    p_flops = probe_cost(corpus.arch, n_probes, n_classes, hidden=256)["per_item"]
    probe_acc = fit_reader(probe, train, val, epochs=epochs, lr=1e-3,
                           batch_size=64, device=device)

    width = _width_for_budget(corpus.arch, p_flops, depth=3, n_classes=n_classes) \
        if budget == "matched" else 256
    torch.manual_seed(seed)
    weight = PhasorGradedReader(corpus.arch, width=width, n_classes=n_classes)
    w_flops = weight_reader_cost(corpus.arch, width=width, depth=3, n_classes=n_classes)["per_item"]
    weight_acc = fit_reader(weight, train, val, epochs=epochs, lr=1e-3,
                            batch_size=64, device=device)

    if abs(probe_acc - weight_acc) < 1e-9:
        winner = "tie"
    else:
        winner = "probe" if probe_acc > weight_acc else "weight"
    return ComparisonReport(probe_acc, weight_acc, p_flops, w_flops, winner,
                            w_flops / max(p_flops, 1))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_compare.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/phasorkit/train.py src/phasorkit/compare.py tests/test_compare.py
git commit -m "feat: matched-cost comparison of the probe and weight-space paths"
```

---

### Task 11: The audit, with its trust gate and the wedge test

**Files:**
- Create: `src/phasorkit/audit/__init__.py`
- Create: `src/phasorkit/audit/preservation.py`
- Create: `src/phasorkit/audit/invariance.py`
- Create: `tests/test_audit.py`

**Interfaces:**
- Consumes: Tasks 3, 6, 8.
- Produces: `preservation_residual(params: SirenParams, moved: SirenParams, arch: Architecture,
  n_points: int = 256) -> float`; `CertifyReport` frozen dataclass with fields
  `verdict: str` (one of `"invariant"`, `"broken"`, `"inconclusive"`), `movement: float`,
  `residual: float`, `by_magnitude: dict[int, float]`, and `__str__`;
  `certify(model, corpus: INRCorpus, magnitudes: tuple[int, ...] = (1, 3, 10, 40),
  tol: float = 1e-4, seed: int = 0) -> CertifyReport`.
  `model` is any callable `SirenParams -> Tensor [B, K]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit.py
import torch
from torch import nn
from phasorkit.arch import Architecture
from phasorkit.audit.invariance import certify
from phasorkit.corpus import INRCorpus
from phasorkit.readers.weight import PhasorGradedReader


def _corpus(n=8, dims=(2, 8, 8, 1), seed=0):
    arch = Architecture(dims=list(dims))
    g = torch.Generator().manual_seed(seed)
    return INRCorpus.from_flat(torch.randn(n, arch.n_params, generator=g), arch)


class RawFlatten(nn.Module):
    """Negative control: reads raw coordinates, so the group moves it."""

    def __init__(self, arch):
        super().__init__()
        self.lin = nn.Linear(arch.n_params, 4)

    def forward(self, params):
        return self.lin(params.flat())


class PermSignOnly(nn.Module):
    """What monomial-matrix frameworks cover: invariant to permutation and sign, not to phase."""

    def __init__(self, arch):
        super().__init__()
        self.lin = nn.Linear(3 * arch.n_layers, 4)

    def forward(self, params):
        feats = []
        for w, b in params.hidden:
            feats += [(w * w).sum((1, 2)), (b * b).sum(1), torch.cos(2 * b).sum(1)]
        return self.lin(torch.stack(feats, dim=1))


def test_invariant_reader_certifies_invariant():
    c = _corpus()
    model = PhasorGradedReader(c.arch, width=16, n_classes=4).eval()
    report = certify(model, c)
    assert report.verdict == "invariant", str(report)


def test_raw_flatten_certifies_broken():
    c = _corpus()
    report = certify(RawFlatten(c.arch).eval(), c)
    assert report.verdict == "broken"


def test_wedge_a_permutation_and_sign_model_is_caught_on_the_phase_generator():
    """The library's reason to exist, in one assertion."""
    c = _corpus()
    report = certify(PermSignOnly(c.arch).eval(), c)
    assert report.verdict == "broken", (
        "a permutation-and-sign-only model must be caught failing under the phase generator"
    )
    assert report.residual < 1e-8, "the action itself must have preserved the function"


def test_report_records_movement_per_magnitude():
    c = _corpus()
    report = certify(RawFlatten(c.arch).eval(), c, magnitudes=(1, 3, 10))
    assert sorted(report.by_magnitude) == [1, 3, 10]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.audit'`

- [ ] **Step 3: Write `preservation.py`**

```python
# src/phasorkit/audit/preservation.py
"""The trust gate: did the transformation we applied actually preserve the function?"""

from __future__ import annotations

import torch

from phasorkit.arch import Architecture
from phasorkit.forward import max_functional_gap
from phasorkit.params import SirenParams


def preservation_residual(params: SirenParams, moved: SirenParams, arch: Architecture,
                          n_points: int = 256, seed: int = 0) -> float:
    """max_x |f(x) - f'(x)|, relative to the function's own scale.

    A verdict on the model under test is meaningless unless this is at machine precision, so every
    audit reports it alongside the movement.
    """
    gen = torch.Generator().manual_seed(seed)
    x = (torch.rand(n_points, arch.in_dim, generator=gen) * 2 - 1).to(params.w_out.dtype)
    from phasorkit.forward import forward_canonical
    scale = forward_canonical(params, x).abs().max().clamp_min(1e-12).item()
    return max_functional_gap(params, moved, x) / scale
```

- [ ] **Step 4: Write `invariance.py`**

`certify` samples a group element per magnitude `B` (windings uniform on `{-B..B}`, signs Bernoulli,
permutations uniform), applies it, computes `preservation_residual`, and only then measures relative
output movement. Verdict rules, in order:

1. If any `residual > 1e-8`, return `verdict="inconclusive"` with that residual. Never a verdict on
   the model, because the fault is ours.
2. If any model output is non-finite, return `verdict="inconclusive"` and say so.
3. If `max movement <= tol`, `verdict="invariant"`.
4. Otherwise `verdict="broken"`, with `by_magnitude` showing where it starts to move.

`__str__` prints a small table of magnitude against relative movement, plus the residual, so a user
can see the growth curve rather than a bare pass or fail.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_audit.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/phasorkit/audit tests/test_audit.py
git commit -m "feat: invariance certification gated on verified function preservation"
```

---

### Task 12: Canonicalization

**Files:**
- Create: `src/phasorkit/canon.py`
- Create: `tests/test_canon.py`

**Interfaces:**
- Consumes: Tasks 2, 3.
- Produces: `c_sort(params: SirenParams, decimals: int = 6) -> SirenParams`;
  `c_align(params: SirenParams, template: SirenParams, probes: Tensor) -> SirenParams`.
  Both return new parameters in the same orbit.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_canon.py
import torch
from phasorkit.arch import Architecture
from phasorkit.canon import c_sort
from phasorkit.forward import max_functional_gap
from phasorkit.group.sine import apply, random_element


def _params(dims=(2, 8, 8, 1), batch=4, seed=0):
    arch = Architecture(dims=list(dims))
    g = torch.Generator().manual_seed(seed)
    return arch, arch.unflatten(
        torch.randn(batch, arch.n_params, generator=g, dtype=torch.float64))


def test_c_sort_preserves_the_function():
    arch, params = _params()
    x = torch.randn(64, 2, dtype=torch.float64)
    assert max_functional_gap(params, c_sort(params), x) < 1e-10


def test_c_sort_is_constant_on_orbits():
    """The defining property of a canonicalizer: c(g.theta) == c(theta)."""
    arch, params = _params()
    g = random_element(params, torch.Generator().manual_seed(11), max_windings=6)
    a, b = c_sort(params), c_sort(apply(g, params))
    torch.testing.assert_close(a.flat(), b.flat(), atol=1e-8, rtol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_canon.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'phasorkit.canon'`

- [ ] **Step 3: Port**

Port `c_sort` from `sirengap:src/sirengap/canon/csort.py` and `c_align` from
`sirengap:src/sirengap/canon/calign.py`, updating imports and generalizing the layer loop to
`params.n_layers`. The module docstring must state the following, because it is what stops a reader
from misusing the function:

> Alignment gives **routing**, not **correspondence**. It moves information into coordinates a
> reader finds easy to read, which is why it recovers a large share of the weight-space gap. It does
> **not** place two networks into matching parameter positions: fits of the same signal sit at
> essentially the same orbit distance as fits of unrelated ones. Do not use these functions to
> average, merge, or interpolate weights. Averaging aligned fits of one signal destroys it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_canon.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/phasorkit/canon.py tests/test_canon.py
git commit -m "feat: sorting and alignment canonicalizers"
```

---

### Task 13: CLI, public exports, and a runnable example

**Files:**
- Create: `src/phasorkit/cli.py`
- Modify: `src/phasorkit/__init__.py`
- Create: `examples/quickstart.py`
- Create: `tests/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `phasorkit.__all__` exporting `Architecture`, `INRCorpus`, `compare`, `certify`,
  `c_sort`, `c_align`, `PhasorGradedReader`, `ProbeReader`, `phasor_features`, `apply`,
  `random_element`; console script `phasorkit` with subcommands `compare` and `certify`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import subprocess
import sys


def test_public_api_imports():
    import phasorkit as pk

    for name in ("Architecture", "INRCorpus", "compare", "certify", "c_sort", "c_align",
                 "PhasorGradedReader", "ProbeReader", "phasor_features"):
        assert hasattr(pk, name), f"phasorkit.{name} missing from the public API"


def test_cli_reports_version():
    out = subprocess.run([sys.executable, "-m", "phasorkit.cli", "--version"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "0.1.0" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL, `AssertionError: phasorkit.Architecture missing from the public API`

- [ ] **Step 3: Write `__init__.py`, `cli.py`, and the example**

`__init__.py` re-exports the names above and sets `__version__ = "0.1.0"`. `cli.py` uses `argparse`
with `compare` and `certify` subcommands, each taking `--corpus` (a `.npz` or checkpoint directory),
`--dims`, and printing the report's `__str__`. `examples/quickstart.py` builds a small synthetic
two-class corpus, runs `compare`, and prints the result; it must finish in under two minutes on a
laptop CPU and is what the README's quickstart block refers to.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the whole suite and check coverage**

Run: `pytest --cov=src/phasorkit --cov-report=term-missing`
Expected: all tests pass, coverage on `src/phasorkit` at or above 80%.

- [ ] **Step 6: Commit**

```bash
git add src/phasorkit/cli.py src/phasorkit/__init__.py examples/quickstart.py tests/test_cli.py README.md
git commit -m "feat: public API, command line interface, and a runnable quickstart"
```

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task: architecture and corpus (1, 2, 8),
adapters (8), group action (3), phasor features (4), graded layers (5), readers (6, 7),
canonicalization (12), `compare` (10), audit and trust gate (11), FLOPs (9), CLI and README (13).
The wedge test named in the spec's testing section is Task 11 step 1. The spec's deferred list
(other activations, 3D, JAX, conv/attention) has no task, correctly.

**Placeholder scan.** No task contains "TBD", "implement later", "add error handling", or "similar to
Task N". Tasks 8, 9, 11, 12 and 13 describe some file bodies in prose rather than full code: those
are ports of files that exist verbatim in the parent repository at the paths given, plus argparse and
loop boilerplate. Every interface those files must expose is written out in the task's Interfaces
block, and every behavioral rule they must satisfy is enumerated as a numbered list.

**Type consistency.** `SirenParams`, `Architecture`, `INRCorpus`, `Character`, `PhasorFeatures`,
`GroupElement` and `ComparisonReport` are each defined once and used with the same field names
throughout. `phasor_features` returns `PhasorFeatures` with `.nodes` and `.edges` in Tasks 4, 5, 6.
`certify` returns `CertifyReport` with `.verdict`, `.movement`, `.residual`, `.by_magnitude` in Task
11 only. `arch.n_layers` means hidden sine layers everywhere, matching `SirenParams.n_layers`.

**Known risk.** Task 10's `test_matched_budget_equalizes_inference_flops_within_tolerance` asserts
the matched budget lands within a factor of two. If `_width_for_budget` cannot reach the probe's
budget even at width 4, the assertion fails legitimately and the fix is to report the shortfall in
`ComparisonReport` rather than to widen the tolerance.
