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

## 2026-10-20: Riemann-Siegel C1 to C4, own zeta, Gram's law, Turing's method

Built:

- `riemann_siegel.py`: theta and Z now take arrays. The main sum builds a
  chunked (t, n) phase matrix and masks n > N(t) per row. Z gets the corrections
  C0 to C4 (`z_rs(t, corrections=4)` is the default). Each C_k is a Taylor
  polynomial in p - 1/2, computed at import from a 110-digit series division of
  Psi(1/2 + z) = -cos(2pi z^2 - 5pi/8) / cos(2pi z). This replaced the old C0
  with its L'Hopital special case.
- `zeta.py`: exact Bernoulli numbers, complex log-gamma (shift past |z| = 15,
  then Stirling), Euler-Maclaurin with Backlund's error bound, and the
  functional equation for Re s < 0 with chi assembled in log space.
- `gram.py`: sign-safe Z at many points, Gram's law, Gram blocks, a Rosser's
  rule checker that refines sampling until it finds enough sign changes, and
  summary stats.
- `turing.py`: Brent's form of Turing's method with Lehman's constants.
  `verify_zeros(T)` proves N(g_n) = n + 1 for a good Gram point g_n >= T.
- Plots: Gram points on Z (bad ones flagged) and the Riemann-Siegel error per
  correction order. Both rendered and checked by eye.

Checks:

- C1 to C4 match mpmath's numerical derivatives of Psi to 1e-15.
- max |error| t^((2k+3)/4) over [200, 2000]: 0.119, 0.052, 0.0090, 0.029, 0.015
  for k = 0..4, about the same over [20, 200]. Past t ~ 3000, C3 and C4 hit a
  ~3e-11 floor from double-precision phases (θ - t log n is ~1e5 rad).
- zeta vs mpmath: relative error < 1e-12 on 300 random points in
  |Re s| <= 30, |Im s| <= 200. zeta(1 - 2k) matches -B_2k/2k to 1e-13. The
  Backlund bound holds and is within 50x of the true error in every test case.
- Gram's law first fails at n = 126 (matches Hutchinson). Up to n = 100,000,
  88.9% of Gram points are good, and the block lengths are 78,694 / 9,445 / 778 /
  19 / 1 for lengths 1 to 5. No Rosser violations.
- verify_zeros(100000): N(g_138068) = 138,069, with K = 2 blocks, in 9 s. That
  matches mpmath's nzeros. Smaller heights are cross-checked in the tests.

Design notes:

- Ran the corrections through a Taylor series instead of closed-form
  derivatives because Psi^(12) by hand is hopeless and Psi is entire. The
  series division is ill-conditioned (poles of 1/cos at |z| = 1/4), so it
  needs extended precision once, at import; evaluating the polynomials is
  plain doubles.
- I couldn't find Gabcke's constants online to cite, so the README reports the
  measured constants only.
- Lehman's integral bound is stated for t > 168 (Trudgian notes the 168 is
  probably a misprint and smaller would do), so verify_zeros never starts below
  168.
- A sign of Z is only trusted from z_rs when |Z| > 1e-6, roughly 100x the
  C4 error at t = 200. Smaller values get recomputed with mpmath.
- Slip: the `plots` commit went in with a failing test (my shell chain checked
  the exit code of `tail`, not pytest). Fixed in the next commit, `fix`.

Next: functional equation demos (they now just need a plot since
chi exists), then domain coloring of the strip with a vectorized zeta, then
spacing statistics vs GUE.
