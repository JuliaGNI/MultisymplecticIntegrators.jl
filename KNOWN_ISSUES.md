# Known issues

Defects found in review and recorded, not fixed. Each entry names its kind and its evidence.

### K1 · **Five broken marks are `@test_broken`, each with an issue.**

- location: `test/integrator/Galerkin_Bspline_int.jl:37`
- evidence: The package itself loads and
  the test environment resolves; the failures are in the test files, which were written against a
  source tree that has since moved. They are recorded rather than papered over — no tolerance was
  widened and no assertion removed.
  - `test/integrator/Galerkin_Bspline_int.jl:37` (#7) — `C.λ₀_x_coes == carried.λ₁_x_coes` after
    `copy_internal_variables!`; the carried multiplier arrives as all zeros.
  - `test/integrator/Galerkin_Bspline_int.jl:40` (#7) — `Galerkin_Bspline_IntegratorCache` has no
    field `λ₁_carry_x_coes`. The test expects a separate carry slot that the struct does not
    define. The two assertions at `:46` and `:47` that depend on it are broken as well.
  - `test/integrator/FEM_Multisymplectic_int.jl:61` (#2) — `err < 1e-2` against the exact
    solution, evaluated at `0.131` after integrating to `t = 200`.
- kind: defect
- found: 2026-08-31

### K2 · Stale bounds: `GeometricIntegratorsBase = "0.1.11"` against 0.6.4, `SimpleSolvers = "0.7.8"` against 0.13.2.

- location: —
- evidence: Only 10 of 31 dependencies are bounded at all.
- kind: defect
- found: 2026-08-31

### K3 · `Infiltrator` and `Plots` are hard dependencies of the package, and `test/Project.toml` pulls in `CairoMakie`, `Plots`, `BenchmarkTools` and `Revise` as well; a headless runner installs all of it to run six test files.

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

## KI-2 · dead code · unused test dependencies

No file under `test/` uses 20 of the 26 dependencies in `test/Project.toml`: every one except
Aqua, GeometricIntegratorsBase, GeometricSolutions, MultiSymplectic, SafeTestsets and Test. The
experiment scripts in `scripts/` use them, and `scripts/` has no environment of its own. Evidence:
`grep -rn "using\|import" test/` names only those six packages. Fix: remove the 20 from
`test/Project.toml`, or give `scripts/` its own environment. Found 2026-09-26.

## KI-3 · missing test · a broken mark that asserts nothing

`test/integrator/Galerkin_Bspline_int.jl:40` is
`@test_broken (C.λ₁_carry_x_coes .= 3 .* carried.λ₁_x_coes; true)`. It wraps a setup statement
that throws (issue #7: the cache has no field `λ₁_carry_x_coes`) so that lines 48–50 still run.
It adds one broken count that asserts nothing. Once #7 is fixed, it becomes a plain statement.
Found 2026-09-26.

## KI-4 · docs · the README and the manual say the package does not load

`README.md` (lines 8–12) and `docs/src/index.md` (lines 12–16) say that the package "does not
currently load" and has no `test/runtests.jl` entry point. The package loads, and
`test/runtests.jl` runs a green suite with broken marks. The manual page also cites a local file
path that no reader can open. Found 2026-09-26.
