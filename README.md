# SIREN-GAP

### Research code: parameter symmetry and weight-space learning on sine-network INRs

[pre-registrations](docs/prereg/) · [claims ledger](docs/CLAIMS.md) · [lab notebook](docs/LAB_NOTEBOOK.md) · [prediction outcomes](docs/PREDICTION_OUTCOMES.csv) · [provenance](docs/PROVENANCE.md)

---

## What this is

An implicit neural representation (INR) stores a signal, such as an image, as the weights of a
small network fitted to it. A growing literature trains models that read those weights directly,
for example to classify the image an INR encodes.

That works well when every INR in a dataset is fitted from the **same** initialization, and badly
when each INR starts from its **own** random initialization, which is the realistic case. On MNIST
the difference is **80.4 accuracy points** for the same images, the same architecture and the same
reader. We call it the *weight-space perception gap*.

The usual explanation is parameter symmetry: many different weight vectors compute the same
function, and a reader that does not know this sees noise. This repository measures how much of
the gap that explanation actually covers, for SIREN networks (sine activations), and contains every
script, test, pre-registration and result behind the numbers below.

## Main findings

1. **The symmetry group.** For a sine network, flipping the sign of a neuron and shifting its bias
   by $\pi$ (with a sign change on its outgoing weights) both leave the function unchanged. These
   maps are known from prior work. Together they generate the infinite dihedral group $D_\infty$ per
   neuron, and $D_\infty \wr S_n$ per layer once neurons can be reordered. We prove that this is the
   *whole* group of function-preserving changes for generic networks with one hidden layer and with
   two, which is the depth every experiment here uses.

2. **Choosing a good representative helps a lot; the standard treatments do not.** Aligning each
   network to a fixed reference, an exact change that leaves every function untouched, recovers
   63%, 66% and 32% of the gap on MNIST, FashionMNIST and CIFAR-10. Augmentation with group
   elements, averaging over refits and frame averaging recover at most 13%.

3. **The group alone is enough to reproduce the gap.** Taking the shared-initialization corpus,
   keeping every network and its function fixed, and applying a random group element to each one
   costs **79.1 of the 80.4 points**. A factorial experiment splits that damage across the four
   kinds of change (Shapley values, MNIST):

   | change | share of the damage (points) |
   |---|---:|
   | reordering neurons | 44.2 |
   | sign flips | 29.9 |
   | $\pi$ bias shifts | 4.3 |
   | $2\pi$ bias windings | 0.4 |

   A scramble that *does* change the function (reordering first-layer neurons without rewiring the
   next layer) costs only 1.1 points, so the damage comes from the group, not from any reordering.
   Three independent random draws of the group agree to within 0.3 points.

4. **A reader that understands the group closes most of the gap.** Replacing each bias $b$ by
   $(\cos b, \sin b)$ turns the infinite group into a finite one that acts by sign changes, and a
   reader built around those signs is exactly invariant on the raw parameters. It recovers
   **0.917** of the gap at the same parameter count as the other readers, against 0.628 for the best
   alignment. On the published INR benchmarks it scores 93.51 (MNIST), 74.81 (FashionMNIST) and
   44.20 (CIFAR-10); among readers that use weights alone, that is third, third and first.

5. **Something besides symmetry remains, but not much.** The same invariant reader still loses 7.8
   points between the shared and the random corpus. Reading the INR as a function instead
   (querying it at 64 learned points) moves by only $-0.25$ points between the same two corpora, so
   the remaining loss is not explained by differences in fit quality.

6. **Reading the function beats reading the weights.** Querying an INR at 64 learned coordinates
   reaches 95.3% on MNIST for 1.6 MFLOP per network; the best weight pipeline without a learned
   invariant reader reaches 64.4% for 5.5 MFLOP, and the invariant reader costs about 100 times
   more than querying. At this scale, weight access loses on both accuracy and compute.

7. **Identifiability has no practical content at production width.** At width 2, a network fitted
   to a teacher's outputs recovers the teacher's parameters to six or seven significant figures. At width
   32 the optimizer leaves the correct orbit even when it starts on it.

## The decomposition ladder

Every row below applies a different *feature map* to the same corpora and decodes it with one
frozen classifier (an MLP with layers 1024, 512 and 256). Only the feature map changes. W1 and W3
are the raw weights of the shared-initialization and random-initialization corpora; the column
$f$ is the share of the gap between them that a row recovers,
$f = (\text{row} - \text{W3}) / (\text{W1} - \text{W3})$.

<!-- LADDER_TABLE:START -->
| rung | feature map | MNIST | $f$ | FashionMNIST | $f$ | CIFAR-10 | $f$ |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 | real pixels | 97.97 |  | 89.62 |  | 55.81 |  |
| P1 | oracle render of the fit | 97.59 |  | 89.44 |  | 56.23 |  |
| **W1** | raw weights, **shared init** | 94.36 |  | 82.97 |  | 44.29 |  |
| W2 | raw weights, shared init + SGD noise | 95.04 |  | 83.67 |  | 45.19 |  |
| **W3** | raw weights, **random init** | 13.92 |  | 12.66 |  | 12.64 |  |
| W4 † | $c_\text{sort}$, exact, template-free | 28.19 | 0.177 | 24.61 | 0.170 | 16.05 | 0.108 |
| **W5** † | $c_\text{align}$, exact, aligned to $\theta_0$ | 64.41 | 0.628 | 59.34 | 0.664 | 22.92 | 0.324 |
| W10 † | exact $L{=}2$ invariants | 35.54 | 0.269 | 42.77 | 0.428 | 29.54 | 0.534 |
| W6 † | bounded group augmentation | 18.12 | 0.054 | 14.86 | 0.032 | 16.57 | 0.128 |
| W7 † | $K$-marginalization ($K{=}8$) | 17.75 | 0.048 | 15.05 | 0.034 | 15.86 | 0.101 |
| W7-1/8 † | *control:* $K$ corpus, rows matched | 14.59 | 0.008 | 13.20 | 0.008 | 12.75 | 0.003 |
| W9 † | frame averaging, $R{=}64$ | 14.13 | 0.003 | 12.12 | -0.008 | 12.68 | 0.001 |
| W8 † | canonicalize, then augment | 10.27 | -0.045 | 10.20 | -0.035 | 10.65 | -0.063 |

† acts on the random-init corpus. W4, W5, W10 are **exactly** function-preserving. Chance = 10.
<!-- LADDER_TABLE:END -->

![the ladder](paper/figures/fig1_ladder.png)

**How to read it.** P0 and P1 show that fitting loses almost no information: a classifier on
renders of the fitted INRs matches one on real pixels. W1 against W3 is the gap. W4 and W5 change
only which member of each group orbit is shown to the classifier. W6, W7 and W9 are the usual ways
of dealing with symmetry, and they barely help. W10 is an exact invariant encoding.

Note that $f$ measures what one learning algorithm can recover, not a causal share of the gap. An
exact reframing creates no new information, but it can still move information into places a
classifier reads more easily. The causal question is answered by the group intervention in
finding 3, not by this table.

## How the corpora are built

One SIREN (two hidden layers of width 32) is fitted to each image, under four protocols that differ
in exactly one source of variation:

| protocol | initialization | optimizer noise | role |
|---|---|---|---|
| `P-shared-det` | one shared draw | none | no nuisance at all |
| `P-shared-stoch` | one shared draw | resampled | optimizer noise alone |
| `P-random` | independent per INR | resampled | the realistic setting |
| `P-random-K` | 8 independent draws per image | resampled | for averaging over refits |

Across MNIST, FashionMNIST and CIFAR-10 this is about 1.8 million fitted networks, all produced on
one laptop. A corpus is used only if a classifier trained on renders of its INRs matches one
trained on the real images, so no result can be explained by information lost in fitting.

**Pre-registration.** Every experiment's hypotheses, point predictions with 80% intervals, seed
counts and falsification rules were written down and frozen before the first result existed. Each
frozen document is identified by a hash in [docs/PREDICTION_LEDGER.csv](docs/PREDICTION_LEDGER.csv),
and every prediction is scored in [docs/PREDICTION_OUTCOMES.csv](docs/PREDICTION_OUTCOMES.csv),
including the ones that missed.

## Reproducing the results

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e .
make test                                                 # property tests T1 to T20

.venv/bin/python scripts/03_generate_inrbench.py ...      # fit a corpus (see the script's help)
.venv/bin/python scripts/04_quality_gate.py ...           # admission gate for a corpus
bash scripts/12_ladder_chain.sh                           # MNIST ladder
bash scripts/20_cifar_ladder.sh                           # CIFAR-10 ladder

.venv/bin/python scripts/37_orbit_intervention.py         # group intervention (S6)
.venv/bin/python scripts/75_s15_orbit_factorial.py        # factorial split of it (S15)
.venv/bin/python scripts/47_w12_phasor.py                 # the phasor-graded reader (W12)
.venv/bin/python scripts/52_w12_invariance_audit.py       # checks W12's invariance on fitted INRs
.venv/bin/python scripts/35_s5_pareto.py                  # function access against weight access
.venv/bin/python scripts/76_s16_function_nuisance.py      # function access on both corpora (S16)
.venv/bin/python scripts/73_absorb_omega_benchmark.py --dataset mnist   # published benchmark, canonical form
.venv/bin/python scripts/72_l2_proof_checks.py            # numerical checks for the depth-two proof

.venv/bin/python scripts/21_paper_figures.py              # every figure
.venv/bin/python scripts/22_paper_tables.py               # every table, including the one above
```

Scripts are numbered in the order the study ran them, and they can be resumed: a result that
already exists is skipped unless `--force` is given. The corpora themselves (about 14 GB) are not
in the repository; the scripts regenerate them from the public datasets.

## Repository layout

```
src/sirengap/   library: fitting, symmetry group, canonicalizers, invariants, readers, evaluation
tests/          property tests T1 to T20 (run on CPU)
scripts/        numbered entry points, one per step of the study
results/        the JSON results behind every number (fitted weights are not committed)
paper/figures/  figures, regenerated by scripts/21_paper_figures.py
docs/           pre-registrations, prediction ledger, lab notebook, claims ledger, proof memos
```

## Limitations

- **Fits are not fully converged.** Training every INR to a stationary point failed its own
  pre-registered check twice. A width-128 corpus that does interpolate shows the same pattern, but
  it changes width and convergence at the same time.
- **Scope.** Three image datasets, sine networks with two hidden layers, widths 32 and 128.
  Nothing here covers audio, 3-D fields, deeper networks or other activations.
- **The theorems are generic.** They hold for almost every network, but the exceptions include every
  network with rational weights, so they say nothing about one particular floating-point network.
  Depth three and beyond is a conjecture.
- **One device.** Everything ran on Apple MPS; a CUDA replication of one headline table is still
  owed.
- **The residual is reader-relative.** The 7.8 points that survive the invariant reader are that
  reader's shared-versus-random difference, not a measurement of a symmetry-free share of the gap.

## Documents

- [docs/CLAIMS.md](docs/CLAIMS.md): every claim, the artifact and script behind it, and its status.
- [docs/prereg/](docs/prereg/): the frozen pre-registrations, one per study (S1 to S17).
- [docs/LAB_NOTEBOOK.md](docs/LAB_NOTEBOOK.md): the append-only record of the study as it ran.
- [docs/PROVENANCE.md](docs/PROVENANCE.md): what is prior work and what is new, component by
  component.
- [docs/THINKING/proof-memos/](docs/THINKING/proof-memos/): the identifiability proofs in long form.
- [docs/OPEN_PROBLEMS.md](docs/OPEN_PROBLEMS.md) and [docs/RELATED_WORK.md](docs/RELATED_WORK.md).

## References

1. V. Sitzmann, J. Martel, A. Bergman, D. Lindell, G. Wetzstein. *Implicit Neural Representations
   with Periodic Activation Functions.* NeurIPS 2020. [arXiv:2006.09661](https://arxiv.org/abs/2006.09661)
2. A. Navon, A. Shamsian, I. Achituve, E. Fetaya, G. Chechik, H. Maron. *Equivariant Architectures
   for Learning in Deep Weight Spaces.* ICML 2023. [arXiv:2301.12780](https://arxiv.org/abs/2301.12780)
3. A. Zhou, K. Yang, K. Burns, A. Cardace, Y. Jiang, S. Sokota, J. Z. Kolter, C. Finn. *Permutation
   Equivariant Neural Functionals.* NeurIPS 2023. [arXiv:2302.14040](https://arxiv.org/abs/2302.14040)
4. M. Kofinas et al. *Graph Neural Networks for Learning Equivariant Representations of Neural
   Networks.* ICLR 2024. [arXiv:2403.12143](https://arxiv.org/abs/2403.12143)
5. I. Kalogeropoulos, G. Bouritsas, Y. Panagakis. *Scale Equivariant Graph Metanetworks.* NeurIPS
   2024. [arXiv:2406.10685](https://arxiv.org/abs/2406.10685)
6. H. Tran, T. Vo, T. Huu, T. M. Nguyen, N. Ho. *Monomial Matrix Group Equivariant Neural Functional
   Networks.* NeurIPS 2024. [arXiv:2409.11697](https://arxiv.org/abs/2409.11697)
7. A. Shamsian, A. Navon, D. W. Zhang, Y. Zhang, E. Fetaya, G. Chechik, H. Maron. *Improved
   Generalization of Weight Space Networks via Augmentations.* ICML 2024. [arXiv:2402.04081](https://arxiv.org/abs/2402.04081)
8. N. Dym, H. Lawrence, J. W. Siegel. *Equivariant Frames and the Impossibility of Continuous
   Canonicalization.* ICML 2024. [arXiv:2402.16077](https://arxiv.org/abs/2402.16077)
9. S. Papa, R. Valperga, D. Knigge, M. Kofinas, P. Lippe, J.-J. Sonke, E. Gavves. *How to Train
   Neural Field Representations: A Comprehensive Study and Benchmark.* CVPR 2024. [arXiv:2312.10531](https://arxiv.org/abs/2312.10531)
10. J. Kahana, E. Horwitz, I. Shuval, Y. Hoshen. *Deep Linear Probe Generators for Weight Space
    Learning.* ICLR 2025. [arXiv:2410.10811](https://arxiv.org/abs/2410.10811)
11. V. Vlačić, H. Bölcskei. *Affine Symmetries and Neural Network Identifiability.* Advances in
    Mathematics, 2021. [arXiv:2006.11727](https://arxiv.org/abs/2006.11727)

The full annotated bibliography is in [docs/RELATED_WORK.md](docs/RELATED_WORK.md).

---

License: [MIT](LICENSE). All computation ran on a single laptop (PyTorch, MPS and CPU).
