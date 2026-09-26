# Known issues

What is known to be wrong in MultiSymplectic and not fixed yet. An entry leaves this file when its
fix merges, and the CHANGELOG entry of the fix names its ID.

### K1 · **`Pkg.test()` is red: 21 pass, 2 fail, 1 errors.**

- location: `test_full_multiplier_momentum_carry.jl:37`
- evidence: The package itself loads and the test
  environment resolves; the three failures are in the test files, which were written against a
  source tree that has since moved. They are recorded rather than papered over — no tolerance was
  widened and no assertion removed.
  - `test_full_multiplier_momentum_carry.jl:37` — `C.λ₀_x_coes == carried.λ₁_x_coes` after
    `copy_internal_variables!`; the carried multiplier arrives as all zeros.
  - `test_full_multiplier_momentum_carry.jl:40` — `Galerkin_Bspline_IntegratorCache` has no field
    `λ₁_carry_x_coes`. The test expects a separate carry slot that the struct does not define, so
    the failure is outside a `@test` and takes the rest of that file with it.
  - `test_fem_multisymplectic_wave.jl:61` — `err < 1e-2` against the exact solution, evaluated at
    `0.131` after integrating to `t = 200`.
- kind: defect
- found: 2026-08-31

### K2 · Stale bounds: `GeometricIntegratorsBase = "0.1.11"` against 0.6.4, `SimpleSolvers = "0.7.8"` against 0.13.2.

- location: —
- evidence: Only 10 of 31 dependencies are bounded at all.
- kind: defect
- found: 2026-08-31

### K3 · `Infiltrator` and `Plots` are hard dependencies of the package, and `test/Project.toml` pulls in `CairoMakie`, `Plots`, `BenchmarkTools` and `Revise` as well; a headless runner installs all of it to run five test files.

- location: `test/Project.toml`
- evidence: (`Gtk4`, `ProfileView` and `PProf` are `[weakdeps]` and are not.)
- kind: defect
- found: 2026-08-31

### K4 · `src/utils/common.jl` is one of two files in the tree that JuliaFormatter cannot process at all — four flat 64-element `Float64` literals, longest line 1624 characters, which take the Julia process down rather than raising a catchable error.

- location: `src/utils/common.jl`
- evidence: —
- kind: upstream
- found: 2026-09-07; Separately, four files are **not formatted and cannot be**, and each will
  block the pre-commit hook if ever staged:

### K5 · `src/integrator/NN_PDE_int.jl`, `NN_PDE_int_symbolic.jl` and `NN_PDE_LSGD_int.jl` fail for a different reason: JuliaFormatter's own output no longer parses.

- location: `src/integrator/NN_PDE_int.jl`
- evidence: It moves the closing
  `) where {IPMT,}` of a constructor signature behind the trailing `# hyperparameters for OGA2d`
  comment on the same line. It detects this itself and declines to write, so the three are
  unformatted rather than broken. Moving that comment onto its own line would fix them, but that
  is a content change.
- kind: upstream
- found: 2026-09-07; Separately, four files are **not formatted and cannot be**, and each will
  block the pre-commit hook if ever staged:
