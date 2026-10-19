# zeta-explorer devlog

## 2026-10-18: Kickoff, Riemann-Siegel numerics

- Scaffold: hatchling `src/` layout, ruff, GitHub Actions on Python 3.10 and 3.12.
  Dependencies: numpy, mpmath (reference values), matplotlib optional.
- `riemann_siegel.py`: theta from Stirling's series (plus its derivative), the
  C0 correction with its removable singularities handled, Z(t) from the
  Riemann-Siegel main sum + C0, and mpmath wrappers for reference values.
- `zeros.py`: sign-change scan + Illinois refinement, and Gram points via Newton.
- `plot.py`: Z(t) with zeros marked; rendered to `docs/z_function.png`.

Checks:

- theta matches mpmath's `siegeltheta` to 4e-13 from t = 10 to 1e5.
- All 29 zeros below t = 100 match `mpmath.zetazero` to 1e-9.
- Riemann-Siegel error / (2π/t)^(3/4) is flat at about 0.030 over [20, 5000],
  confirming the theoretical decay; the test uses 0.035 as the bound.
- Gram points g_0..g_2 match published values, and theta(g_n) = nπ up to n = 1e5.

Design notes:

- Web zeta visualizers are rare (mostly Wolfram or desktop apps), and implementing
  Riemann-Siegel from scratch is very doable, so that's the showpiece.
- First plot draft labeled the first three zeros, but the labels collided with
  the curve at every crossing; dropped them and listed the values in the README.

Next: higher Riemann-Siegel corrections C1 to C4, then mark
Gram points on the Z plot.

## 2026-10-18 (later): cleanup pass

Cut the module docstrings in `riemann_siegel.py` and `zeros.py` down to a line or
two, since the README already walks through the math, and dropped comments that
only restated the next line. The comments left in the Illinois loop and the Gram
point Newton step explain why those tricks work. Shortened two test names.

Also fixed a README mistake: Z at t = 10⁶ needs 398 terms, not 399
(floor(sqrt(10⁶/2π)) = floor(398.94)). No behavior changes, same 32 tests.
