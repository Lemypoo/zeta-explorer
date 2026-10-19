# zeta-explorer

[![tests](https://github.com/Lemypoo/zeta-explorer/actions/workflows/tests.yml/badge.svg)](https://github.com/Lemypoo/zeta-explorer/actions/workflows/tests.yml)

**The Riemann zeta function, computed from scratch and made explorable.**

![Z(t) on the critical line with its first 13 zeros](docs/z_function.png)

The Riemann Hypothesis says every non-trivial zero of ζ(s) lies on the line
Re(s) = 1/2. On that line ζ is complex, but rotating it by a phase θ(t) gives the
real-valued **Riemann-Siegel Z function**, whose sign changes are exactly the
zeros. zeta-explorer implements θ, Z, a zero finder and Gram points from the
formulas, and checks every result against mpmath.

> Early days: numerics core and the first plot are in. Domain-coloring plots of the
> critical strip, zero statistics, animations and a web explorer come next.

## Install

```bash
pip install "zeta-explorer[plot] @ git+https://github.com/Lemypoo/zeta-explorer"
```

## Quick start

```python
from zeta_explorer.riemann_siegel import theta, z_rs, z_exact
from zeta_explorer.zeros import find_zeros, gram_point

find_zeros(10, 30)   # [14.134725141734695, 21.022039638771552, 25.01085758014569]
len(find_zeros(10, 100))   # 29
gram_point(0)        # 17.845599540410863, where theta(t) = 0
z_rs(100.0), z_exact(100.0)   # fast approximation vs mpmath reference

from zeta_explorer.plot import plot_z
plot_z(0, 60).savefig("z.png")
```

## The math

**Theta** comes from Stirling's series for log Γ:

```
θ(t) = t/2·log(t/2π) − t/2 − π/8 + 1/(48t) + 7/(5760t³) + …
```

accurate to about 1e-13 for t ≥ 10.

**Riemann-Siegel formula.** With a = sqrt(t/2π), N = ⌊a⌋ and p = a − N:

```
Z(t) ≈ 2·Σ_{n=1..N} cos(θ(t) − t·log n)/sqrt(n)  +  (−1)^(N−1)·a^(−1/2)·C0(p)
C0(p) = cos(2π(p² − p − 1/16)) / cos(2πp)
```

Only about sqrt(t/2π) terms are needed, so at t = 10⁶ that's 399 terms. C0 has
removable singularities at p = 1/4 and 3/4 (limit 1/2 by L'Hôpital), which the code
handles explicitly. The leftover error is the next term, C1·(2π/t)^(3/4). Measured
from t = 20 to 5000, error ÷ (2π/t)^(3/4) stays flat at ≈ 0.030, matching theory,
and the tests assert that bound.

**Zeros** are sign changes of Z on a grid, refined with the Illinois method
(regula falsi that halves a stuck endpoint's weight). All 29 zeros below t = 100
match mpmath's `zetazero` to 1e-9, and the approximate Z finds the same 29.

**Gram points** solve θ(g_n) = nπ. θ is increasing and convex there, so Newton's
method started to the right of the root slides monotonically onto it.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
