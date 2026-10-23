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

## 2026-10-22: zeta on arrays, phase portrait, continuation, zero spacings vs GUE

Built:

- `zeta.py`: `zeta`, `chi` and `loggamma` take arrays of any shape. Each point
  keeps its own N, the Dirichlet head is a chunked masked matrix, and the
  Euler-Maclaurin loop shrinks its index set as points meet their Backlund
  bound. Scalar calls go through the same code. Added `xi(s)`.
- `continuation.py`: partial sums of the Dirichlet series, the alternating eta
  series and the Euler-Maclaurin corrected sum.
- `zeros.py`: `illinois_many`, a vectorized Illinois method that retires each
  bracket once it converges (relative tolerance, since an absolute 1e-10 is
  below one ulp of t past 10⁶).
- `spacing.py`: `zeros_between(n_lo, n_hi)` brackets every zero between two good
  Gram points (refining Gram blocks until Rosser's count is met) and refines all
  brackets together; unfolding with θ/π; the GUE gap probability as a Fredholm
  determinant with Gauss-Legendre (Bornemann's method); the spacing density and
  CDF from a Chebyshev interpolant of E(s); Wigner surmise; Montgomery's pair
  correlation; an empirical pair correlation with no edge bias; KS distance.
- Plots: phase portrait of the strip, the spiral of ζ(1/2 + it), the three
  partial sums, and the spacing histogram with pair correlation. All four
  rendered and checked by eye; moved two legends and a label off the data.

Checks:

- Array zeta vs mpmath on 2,000 random points (|σ| ≤ 30, |t| ≤ 200): max
  relative error 2.9e-13. Array vs scalar calls agree to 1e-15 relative.
- ξ against mpmath to 1e-12, ξ(0) = ξ(1) = 1/2, ξ(2) = π/6, symmetry, real on
  the line.
- E(s) matches 1 − s + π²s⁴/36 − π⁴s⁶/675 for small s; 20 and 60 nodes agree to
  1e-12. p(s) agrees with 30-digit mpmath derivatives to about 3e-10, integrates
  to 1 with mean 1, variance 0.17999.
- 300 random 120×120 GUE matrices: KS distance to the sine-kernel CDF 0.0044,
  to Poisson 0.28.
- zeros_between(1000, 1060) gives zeros 1002 to 1061, matching `zetazero` to
  1e-8. Near n = 10⁷ it finds exactly 100,000 zeros between g_n and g_(n+100000)
  in 18 s.
- Spacings of 100,000 zeros at n = 10⁵, 10⁶, 10⁷: variance 0.1636, 0.1668,
  0.1693; KS distance to GUE 0.0174, 0.0138, 0.0110.
- Phase portrait grid (594,000 points): 1.05 s vectorized, 6.5 s for a loop over
  the old scalar zeta.

Design notes:

- The price of one code path: a single scalar zeta call went from about 11 µs to
  about 200 µs, mostly numpy overhead in the Euler-Maclaurin loop. Nothing hot
  calls zeta one point at a time, so I kept one implementation.
- E''(s) by Chebyshev differentiation loses about 6 digits near s = 0 (the
  derivative of a degree-80 series amplifies errors by deg² per order), which
  still leaves p(s) good to 1e-9. The CDF is clamped to 1 and the density to 0
  past s = 6, where E(6) ≈ 2e-20.
- The first draft used an absolute tolerance in the vectorized root finder and
  none of the brackets converged at t = 6·10⁵. Switched to relative.
- The zeros are more regular than GUE at these heights. The variance and KS
  numbers move toward GUE with height, matching the finite-height picture of
  Bogomolny and Keating (effective matrix size log(t/2π)/√(12Λ), about 3 at
  t = 5·10⁶). The dip in the pair correlation at 0.3 < u < 0.9 is about −0.013 at
  all three heights, so this range is too short to see it move.

Next: an animation sweeping σ across the strip, then the explicit formula
rebuilding π(x) from the zeros, then Lehmer pairs (the closest zeros found by
zeros_between are a natural place to start).

## 2026-10-23: explicit formula, Lehmer pairs, value distribution

Built:

- `spacing.first_zeros(n)`: the first n zeros, mpmath sign scan below the good
  Gram point g_79 and `zeros_between` above it, with a count check at the seam.
- `explicit.py`: complex Ei from scratch (series, E1 continued fraction with a
  depth that depends on |w|, asymptotic series past |w| = 40), li, Riemann's R
  from the Gram series, a sieve for π and ψ, Möbius, the tail integral by
  Gauss-Laguerre, and the explicit formulas for ψ(x), J(x) and π(x) from a list
  of zeros.
- `lehmer.py`: the Csordas-Smith-Varga sum g for every consecutive pair (exact
  over a window of neighbors and mirrored negative zeros, closed-form smooth
  density outside it), the Lehmer test Δ²g < 4/5, and the resulting lower bound
  on the de Bruijn-Newman constant.
- `values.py`: extrema of Z between zeros by vectorized golden section, running
  records, random samples of log|Z|, exact CUE cumulants from polygamma sums and
  the CUE density by Fourier inversion of the Keating-Snaith product.
- Plots: π(x) and ψ(x) − x rebuilt from 10, 100 and 1000 zeros, Lehmer's pair
  with a zoom, and the log|ζ| histogram against CUE and Selberg. All three
  rendered and checked by eye; the first drafts had a missing 1000-zero curve
  and colliding tick labels on the zoom, both fixed.

Checks:

- Ei vs mpmath on 2,851 random points: max relative error 1.5e-15. R(x) and li(x)
  to 1e-13, the tail integral to 1e-10 against mpmath quadrature.
- With 1000 zeros: π(x) within 0.033 on the half-integers up to 100.5 and rounds
  correctly on 1000.5 to 1099.5; ψ within 0.148. 100 zeros are 4 to 6 times worse.
  At a prime both formulas land on the midpoint of the jump.
- g against an independent formula from the Hadamard product of Ξ (Z', Z'', Z'''
  from mpmath at the zeros): relative error about 3e-6.
- 29 Lehmer pairs below t = 1000, matching Stopple (2017). Near t = 10⁶, 7,384 of
  114,660 pairs, against Stopple's 7,398 of 114,661. Lehmer's pair at 7005 gives
  λ = −7.112e-4, and the maximum of Z between its zeros sits at 7005.0818554, next
  to Stopple's ζ' zero at 7005.08185555. The van de Lune pair's published Δ gives
  −5.894e-9 against the published −5.895e-9.
- Z extrema have |Z'| < 1e-6 by mpmath; the record |Z| below 10⁵ is 28.216 at
  t = 77,403.722.
- CUE: cumulants vs mpmath polygamma to 1e-13; the density integrates to 1 and
  reproduces κ₂, κ₃; 20,000 Haar unitaries of size 12 fit the density within 0.015.
- log|Z| at T = 2π·e^N for N = 8 to 16 (200,000 samples each): variance 1.73 to
  2.09, CUE 1.83 to 2.18, Selberg 1.14 to 1.44.

Design notes:

- li(x^ρ) has to be Ei(ρ log x). The principal branch of li at the complex
  point x^ρ is wrong once γ log x > π, which is almost always.
- The continued fraction for E1 needs over 200 terms near the negative real
  axis at |z| = 10, and the series loses e^(0.3|w|) digits within π/4 of the
  positive real axis. Rather than paper over that wedge, `ei` raises there. The
  explicit formula only evaluates near the imaginary axis.
- The Lehmer bound is stated for zeros of H₀, which are 2γ. Plugging γ straight
  into the formula gives −1.78e-4 for Lehmer's pair instead of −7.11e-4. The
  1993 paper's van de Lune numbers (λ ≈ −Δ²/8 in x units) settled the
  convention. Stopple's paper writes the definition in γ, which is fine for Δ²g
  but not for λ.
- Negative zeros matter at the bottom: for the first pair they are 14% of g
  (0.0229 of 0.1644). They are summed exactly over the first 500 zeros.
- The zeta variance sits 0.07 to 0.10 below CUE at every height. That's
  presumably the arithmetic factor from small primes, which I haven't derived, so
  the README just reports it.
- An mpmath `zetazero(6709)` call ran for minutes, so the reference g uses
  `findroot` on `siegelz` started from our own zeros instead.

Next: an animation sweeping σ across the strip, then the 3D |ζ| landscape, then
a first static web page with precomputed zeros and the explicit-formula curves.

## 2026-10-23 (afternoon): fast multi-evaluation of Z, the 3D landscape, the σ sweep

Built:

- `multieval.py`: `nufft`, a type-1 nonuniform FFT by Gaussian gridding (2x
  oversampled grid, spreading width from the tolerance, deconvolution by
  e^(τk²)); `fft_size`, the next 5-smooth length; and `ZWindow`, a simplified
  Odlyzko-Schönhage evaluator. The window computes the Riemann-Siegel main sum on
  an evenly spaced grid with one NUFFT, interpolates between grid points with a
  Gaussian-windowed sinc, and adds the C0 to C4 remainder. `zeros_fast(n_lo, n_hi)`
  runs the Gram-block zero finder on top of it.
- `riemann_siegel.py`: split out `theta_tail` and `rs_remainder` so the window
  can reuse them.
- `spacing.py`: the Gram-block logic became `gram_block_zeros(g, n0, sample,
  refine)` so it works with any Z. It now handles Rosser violations by merging a
  short block with the previous block and then the following ones.
  `illinois_many` takes an absolute tolerance.
- Plots: `plot_landscape` (log|ζ| as a 3D surface), `sweep_sigma` (an animation
  of ζ(σ + it) as σ crosses the strip, saved as a GIF), and `plot_multieval`
  (accuracy and cost against the direct sum). All three rendered and checked by
  eye.

Checks:

- NUFFT against the direct sum: error below tol·Σ|c| for m = 64, 250, 1000 and
  tol = 1e-6, 1e-10, 1e-13.
- Window against mpmath `siegelz`: 2e-13, 8e-14, 3e-13, 2e-13, 1e-13 at
  t = 10⁴, 10⁶, 10⁸, 10¹⁰, 10¹². The direct sum at the same points: 1e-11, 3e-9,
  5e-7, 6e-5, 3.5e-3. Also exact on both sides of a jump in N at t = 2π·200².
- All 10⁴ zeros in Odlyzko's table for n = 10¹² + 1 to 10¹² + 10⁴: max difference
  2.8e-9, RMS 7e-10, in 2.5 s. The numbering matches the Gram-point count.
- Spacing statistics recomputed at n = 10⁷ with the window match the old row
  exactly (0.1693, 0.0110, −0.013), in 11 s instead of 18 s.
- Rosser violations: none in 100,000 Gram intervals from n = 10⁹, four from
  n = 10¹². A synthetic Z with a borrowed pair of zeros tests the merge, and one
  with a missing pair tests the error.

Benchmarks (this machine, 20,000 points per height, after the window is built):

| t | N | direct sum | window | window build |
|---|---|---|---|---|
| 10⁶ | 399 | 7.4 μs | 2.6 μs | 0.01 s |
| 10⁸ | 3,989 | 58 μs | 2.5 μs | 0.03 s |
| 10¹⁰ | 39,894 | 795 μs | 2.6 μs | 0.25 s |
| 10¹² | 206,393 | 7.7 ms | 2.6 μs | about 1.3 s |

Design notes:

- Precision was the bigger win. t·log n is ~10¹³ rad at t = 10¹², so a double
  phase is off by ~10⁻³. The window computes t₀ log n mod 2π and θ(t₀) once in
  mpmath and works in the offset u = t − t₀ from then on. θ(t₀ + u) − θ(t₀) is
  written with log1p so no term is large. Zeros and Gram points come back as
  offsets, since a double near 2.7·10¹¹ only resolves 6e-5.
- Interpolation error: with sampling at α = 2β, a sinc of bandwidth α and a
  Gaussian window with ε² = (α − β)/(Kδ), both the spectral leakage and the
  truncation are e^(−πK/4). K = 36 for 1e-12, and the bound
  e^(−9π) = 5e-13 sits just above the measured 1e-13 to 3e-13.
- Gaussian gridding needed w = 15 grid points on each side for 1e-13. The
  exponential-of-semicircle kernel FINUFFT uses would need fewer, but its
  Fourier transform has no closed form, and spreading is not the bottleneck
  here. The mpmath phases are: about 1 s of the 1.3 s build at 10¹².
- When N changes inside a window the grid uses the larger N and subtracts the
  last term directly where t is below 2πN². That beats splitting the window.
- zeros_fast's profile at 10¹² is now dominated by `np.polyval` in the
  degree-90 remainder polynomials, which get called on small arrays many times
  during root refinement.
- The first landscape drafts plotted |ζ| capped at 4 or 5. The pits at the zeros
  were hidden behind the front of the surface, and the cap made a flat yellow
  plateau. log₁₀|ζ| clipped to [−1.5, 1], with the zeros added to the t grid,
  fixed both. Viewing from higher up floated the zero markers off the surface,
  so the camera stays at 42°.
- The sweep uses σ = 0.5 + 0.4·sin³ so the animation pauses near the critical
  line; with a plain sine it rushed through σ = 1/2 in one frame.

Next: a static web page with precomputed data, then ζ in the browser with a
canvas domain coloring. Shrinking the remainder polynomials (degree 90 is far
more than doubles need) would speed up zeros_fast.
