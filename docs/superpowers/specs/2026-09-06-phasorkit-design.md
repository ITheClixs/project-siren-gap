# phasorkit: a toolkit for learning on INR corpora

Design document. Status: approved, not yet implemented.
Supersedes `2026-09-02-orbitcheck-design.md`, whose certifier survives here as one module.

## The problem

A growing literature learns on the weights of implicit neural representations. Libraries exist for
it: `nfn` is pip-installable, DWSNets and ScaleGMN are on GitHub. Every one of them handles
permutation, and ScaleGMN adds scaling. None implements a phase action, so for periodic activations
they are provably incomplete: the function-preserving group of a sine network contains an affine
phase component that lies outside every monomial-matrix action, and no amount of permutation and
sign equivariance covers it.

That gap is not theoretical. Recent work on SIREN weight spaces identifies the native bias column
as a low-dimensional causal readout route for a trained reader, and treats it as an empirical
discovery with no symmetry theory attached. The bias column is exactly where the phase component
acts. The phasor lift is the principled treatment of that column, and it is the largest single term
in our reader's gain.

Separately, the field publishes weight-space accuracies without the obvious baseline. Querying the
same INR at a small number of learned coordinates is more accurate than every weight-space pipeline
we built and roughly two orders of magnitude cheaper at downstream inference. A practitioner
choosing weight space deserves to know that before they build.

`phasorkit` addresses both: correct symmetry handling for periodic weight spaces, and an honest
baseline that is one line of code.

## What it does

Given a corpus of INRs, the toolkit reads them two ways and tells the user which one wins on their
data at equal compute. Around that it exposes the symmetry machinery the weight-space path needs and
the audit tools to check that a model has the invariance it claims.

## Scope

**Architectures.** Sine MLPs at arbitrary depth. The group action is function-preserving at every
depth unconditionally, so the action and the phasor lift generalize; *completeness* is proved at one
and two hidden layers and is documented as asserted rather than proved beyond. This covers SIREN,
the public DWSNets and NFN benchmark corpora, and Implicit-Zoo's 2D split.

**Deferred:** other activations, 3D and NeRF, JAX, convolutional and attention architectures.

## Layout

```
phasorkit/
  arch.py            Architecture(dims, activation, w0); flat <-> structured conversion
  corpus.py          INRCorpus: weights [N, D] + Architecture + optional labels
  adapters/          DWSNets-NFN benchmark layout, Implicit-Zoo, raw checkpoint directories
  group/
    element.py       GroupElement, sampling, composition
    sine.py          the D_inf wr S_n action, batched, any depth
  features/
    phasor.py        phasor lift and character grading
  layers/            GradedLinear, GradedBilinear, GradedAct, GradedMessage, PhasorGradedReader
  readers/
    probe.py         function-space probe at learned coordinates
    weight.py        phasor-graded reader on raw parameters
  canon.py           c_align, c_sort
  compare.py         matched-FLOPs comparison
  audit/
    preservation.py  the trust gate: did the action preserve the function?
    invariance.py    how far did the model under test move?
  flops.py           FLOPs accounting for both paths
  cli.py             phasorkit compare | certify | canon
```

`group/sine.py`, `features/phasor.py`, `layers/` and `flops.py` are ports of code already written and
property-tested in the research repository, generalized from fixed depth two to arbitrary depth.

## Public API

```python
import phasorkit as pk

arch   = pk.Architecture(dims=[2, 32, 32, 1], activation="sine", w0=30.0)
corpus = pk.INRCorpus.from_flat(weights, arch, labels=y)
report = pk.compare(corpus, budget="matched")
print(report)
```

Three objects carry the design. `Architecture` describes the INR family and owns every shape
question. `INRCorpus` is a thin dataclass over a flat weight tensor plus an `Architecture`, with
adapters for the two corpus formats that actually exist in the wild. `compare` is the headline: it
trains the function probe and the weight reader at matched FLOPs and reports both accuracies, the
winner, and the inference cost ratio.

**What "matched" means.** The two paths are matched on **downstream inference FLOPs**, the cost of
classifying one held-out INR at deployment, which is the axis a practitioner actually pays on.
Training cost is reported alongside but is not the matching constraint, since the two paths train
very differently and matching there would compare nothing a user cares about. `budget="matched"`
equalizes inference FLOPs by sizing the weight reader to the probe's budget; `budget="free"` runs
both at their natural sizes and reports the cost ratio instead. The probe defaults to `K=64` learned
coordinates, which is the setting our measurements used, and `K` is a parameter.

Nothing else in the toolkit is privileged. `pk.canonicalize`, `pk.certify`, the group action and the
layers are all plain functions and modules on those same objects, so a user who wants only the
graded layers for their own pipeline imports those and ignores the rest.

## Data flow

```
corpus -> split -> probe path:  sample INR at K learned coords -> small MLP -> acc
                -> weight path: phasor lift -> graded layers -> head        -> acc
                -> matched-FLOPs Report
```

The group action sits orthogonal to that path and feeds three consumers: `canonicalize` as a feature
map, `audit` as a measurement, and augmentation, which is exported but documented as weak because
our own decomposition ladder puts bounded group augmentation at five to thirteen percent against
canonicalization's sixty-three to sixty-six.

## The trust model

The toolkit never reports a verdict on a transformation it has not verified. Every audit produces
two numbers and reports them together:

| quantity | meaning | expected |
|---|---|---|
| functional residual | how much the transformation changed the network's function | machine precision |
| output movement | how far the model under test drifted | zero if invariant |

The second is meaningful only when the first is negligible. If the applied action fails to preserve
the function, the audit returns `inconclusive` with the residual rather than accusing the model.
Without that gate a bug in the group action produces false accusations, and a certifier that can cry
wolf is worse than no certifier.

## Canonicalization, and a distinction worth preserving

`canon.py` ships in v1 even though the phrase "alignment gives routing, not correspondence" is what
killed two earlier product directions. The distinction is real and the docstring states it.

Exact alignment moves information into coordinates a reader finds easy to read. It does *not* place
two networks into matching parameter positions: same-image fits sit at orbit distance 0.279 against
0.280 for unrelated ones. For a **reader**, routing is not a defect, it is the mechanism, which is
why alignment recovers about two thirds of the gap. For **merging or compression** it is fatal, which
is why averaging eight aligned fits of one image gives 10.4 dB against 38.7 dB for a single fit.
Anything requiring correspondence stays out of this toolkit.

## Error handling

- Action not function-preserving: return `inconclusive` with the residual. Never a verdict.
- Shape or dimension mismatch against the `Architecture`: fail before compute, naming the expected
  shape.
- Non-finite model outputs: report them rather than silently emitting a large movement number.
- Dead or invisible neurons: note in the report, do not block, since the action is
  function-preserving regardless.

## Testing

Property tests are the spine.

- The group action preserves the function to machine precision, on randomly initialized and on
  fitted networks, at every supported depth.
- The phasor-graded reader is exactly invariant, verified out to winding $|j| = 40$, matching the
  existing test in the research repository.
- Positive control: a known-invariant reader certifies as invariant.
- Negative control: a raw flatten into a linear map certifies as broken.
- **The wedge test:** a model invariant to permutation and sign only, which is what the published
  monomial-matrix frameworks cover, must be caught failing on sine networks under the phase
  generator. If this assertion passes, the library has demonstrated its reason to exist.
- One end-to-end example that runs on a laptop in minutes, so the README's first code block is
  something a reader can execute.

Reproducing the published benchmark numbers is a separate validation exercise, not a gate on the
library.

## Distribution

A new public repository with its own name and README. No author or institution declarations, no link
to the paper, consistent with how the research repository is kept. Built to publishable quality in
this pass; the PyPI upload waits for explicit approval. `phasorkit` is available on PyPI as of
2026-09-06.

## Provenance

The group characterization, the completeness results at one and two hidden layers, the phasor
construction and its invariance proof, the FLOPs accounting and the audit methodology come from the
research in the parent repository. The toolkit is that research made executable by people who did
not do it.
