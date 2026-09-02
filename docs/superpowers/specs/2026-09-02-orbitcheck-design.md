# orbitcheck: an invariance certifier for weight-space models

Design document. Status: approved, not yet implemented.

## The problem

A growing literature builds models that consume neural network weights: DWSNets, NFN, graph
metanetworks, Monomial-NFN, ScaleGMN, universal neural functionals. Every one of them *asserts*
invariance to the weight-space symmetry group by construction. None of them *measures* it.

Assertion is not enough for two reasons. An architecture that is invariant in exact arithmetic can
lose it in floating point, and an architecture invariant to the group its authors had in mind can be
wildly non-invariant to the group the network actually has. The second failure is not hypothetical:
for periodic activations the function-preserving group contains an affine phase component that lies
outside every monomial-matrix action, so a framework built on monomial matrices is provably
incomplete there and will pass a model that a phase shift destroys.

`orbitcheck` measures what those papers assert.

## What it does

Given a model that reads weights and a description of the networks it reads, the tool applies exact
function-preserving transformations to those networks and measures how far the model's output moves.
An invariant model does not move. A model that moves is not invariant, and the sweep says under which
generator and at what magnitude it breaks.

## The trust model

The tool never reports a verdict on the strength of a transformation it has not verified.

Every audit produces two numbers and reports them together:

| quantity | meaning | expected |
|---|---|---|
| functional residual | how much the transformation changed the network's function | machine precision |
| output movement | how far the model under test drifted | zero if invariant |

The second number is meaningful only when the first is negligible. If the applied action fails to
preserve the function, the tool returns `inconclusive` and reports the residual rather than accusing
the model. Without this gate a bug in the group action produces false accusations, and a certifier
that can cry wolf is worse than no certifier.

The group action is function-preserving unconditionally, so the audit needs no genericity assumption
about the networks under test. Genericity is required for completeness and for canonicalisation,
neither of which this tool performs.

## Groups

The symmetry group depends on the activation, and getting it wrong is the failure the tool exists to
catch. Three families ship in v1.

| activation | group | status |
|---|---|---|
| ReLU | permutation, positive scaling | standard |
| odd (tanh and similar) | permutation, sign | standard |
| sine | `D_inf wr S_n`: permutation, sign, and the affine phase shift | complete; the phase component is outside every monomial-matrix action |

The sine case is the one the field currently gets wrong, and it is the wedge: the library is useful
across all three, and the case it uniquely gets right is the case published frameworks do not cover.

## Public API

```python
import orbitcheck as oc

arch   = oc.Architecture(dims=[2, 32, 32, 1], activation="sine")
report = oc.certify(model, arch, weights, magnitudes=(1, 3, 10, 40))
print(report)
```

The model under test is any callable `Tensor[B, D] -> Tensor[B, K]`, which is what a metanetwork is
at its boundary. A model with structured input is wrapped by the caller in a few lines.

`group_for(arch)` returns the group implied by the activation; passing `group=` overrides it, which
is how a user tests their model against a deliberately weaker group to see what that would have
missed.

`magnitudes` is the sweep axis and each group interprets it in its own units, so the sweep means the
same thing everywhere even though the number does not:

| group | magnitude means | a sweep looks like |
|---|---|---|
| sine | bound on the integer winding of the phase shift | `(1, 3, 10, 40)` |
| relu | bound on the log of the positive scale factor | `(0.1, 1.0, 3.0)` |
| odd | no continuous magnitude; sign and permutation are discrete | a single point, sampled repeatedly |

A group declares its own default sweep, so `certify()` without `magnitudes` does the right thing per
family and the argument exists for users who want to push further.

## Modules

Each unit has one purpose and a boundary that can be tested without its internals.

```
orbitcheck/
  arch.py            architecture spec, flat and structured conversion, probe grid
  params.py          WeightBatch: a batch of networks, from_flat / to_flat
  groups/
    base.py          Group protocol: sample(magnitude) -> Element; apply(Element, W)
    sine.py          D_inf wr S_n
    odd.py           sign and permutation
    relu.py          positive scaling and permutation
  audit/
    preservation.py  the trust gate: did the action change the function?
    invariance.py    output movement under sampled elements
    breakpoint.py    sweep magnitude, locate where invariance fails
  report.py          Report, pretty table, JSON, verdict
  cli.py             orbitcheck certify | sweep
```

`groups/sine.py` is a port of the group action already implemented and property-tested in the
research repository. `audit/invariance.py` generalises the audit script that measured the
phasor-graded reader at a relative logit movement of 3e-6 out to winding 40.

## The output that earns the install

A pass or fail is not useful on its own. The breakpoint sweep reports the growth curve of output
movement against transformation magnitude, which distinguishes a model that is invariant from one
that merely has not been pushed hard enough, and identifies which generator is responsible.

The reference behaviour, measured on the research corpora: an exactly invariant reader stays at
3e-6 relative movement out to winding 40, while the same architecture fed a raw rather than
phasor-lifted bias grows 6.2, then 1.7e2, then 2.4e4 across windings 3, 10 and 40. That shape is the
diagnosis.

## Error handling

The tool's job includes refusing to answer.

- Action not function-preserving: return `inconclusive` with the residual. Never a verdict.
- Shape or dimension mismatch against the architecture: fail immediately, before compute, naming the
  expected shape.
- Non-finite model outputs: report rather than silently emitting a large movement number.
- Dead or invisible neurons: note in the report, do not block, because the action is
  function-preserving regardless.

## Testing

- Property test per group: the action preserves the function to machine precision, on randomly
  initialised and on fitted networks.
- Positive control: a known-invariant reader certifies as invariant.
- Negative control: raw flatten into a linear map certifies as broken.
- **The wedge test:** a model invariant to permutation and sign only, which is what published
  monomial-matrix frameworks cover, must be caught failing on sine networks under the phase
  generator. If this test passes, the library has demonstrated its reason to exist in one assertion.

## Scope

**v1 ships:** three groups, MLP architectures, PyTorch, `certify()`, the breakpoint sweep, `Report`
with JSON and CLI exit codes, and the test suite above.

**Deliberately deferred.** The orbit-only intervention, which measures accuracy lost to group
scatter, requires a downstream task, labels and a trained reader. That is a large API surface for
something most users cannot run at audit time, whereas the invariance audit needs only weights and a
model. That self-containment is what makes the library installable, so the intervention waits for
v1.1.

Also out of scope: JAX, convolutional and attention architectures, any hosted service, the
interactive demo, and everything in the canonicalisation family. A spike measured what exact
alignment buys: 2 to 4 percent on corpus compression, and on averaging eight independent fits of one
image it reached 10.4 dB against 38.7 dB for a single fit, worse on every image tested. Exact
alignment routes information into readable coordinates; it does not place two networks in
corresponding parameter positions. Anything requiring correspondence stays out until that changes.

## Provenance

The group characterisation, the completeness result at one hidden layer, the exactness verification
and the audit methodology come from the research in this repository. The library is that research
made executable by people who did not do it.
