"""Project-wide constants (protocol Ch1 convention lock)."""

# SIREN frequency factor, absorbed into stored weights at dataset-creation time
# (see docs/THINKING/G0-theory-scoping.md §0). All symmetry/canon code assumes
# the canonical form h^l = sin(W^l h^{l-1} + b^l) with NO omega factor.
OMEGA_0: float = 30.0

# T1/T2 functional-preservation tolerance (fp32, protocol §0.4)
TOL_FUNC: float = 1e-4

# Numerical tolerance for canonicalizer idempotence (T3)
TOL_IDEMPOTENT: float = 1e-5
