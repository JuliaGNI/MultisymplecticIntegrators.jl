# Release Notes

All notable changes to MultiSymplectic.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually
changed, so that a compat-only bump can be told apart from a rename or a change in results.

This file was started on 2026-08-31 and deliberately holds no entries. Nothing has been
released yet — there are no tags, and `Project.toml` stands at `0.1.0` — so the development
history that predates this file is in `git log` alone. It is named as a gap rather than
reconstructed, because a changelog assembled after the fact loses exactly the reasoning that
makes it worth keeping.

## [Unreleased] — targeting 0.1.0

### New Features

- **Continuous integration.** The repository had no `.github` directory at all; it now carries the
  same `CI.yml`, `CompatHelper.yml`, `Documenter.yml` and `TagBot.yml` as every other repository in
  the tree. **CI is expected to be red** until the blockers under *Open Issues* are dealt with —
  that is the pre-existing state of the package, not a regression.
- **A documentation build.** `docs/` now holds `make.jl`, `Project.toml` and a two-page manual, so
  `Documenter` has something to build and the doctest job has an environment.
- **`test/runtests.jl`, so `Pkg.test()` runs something.** It includes the five files in `test/` that
  actually assert — the three multiplier momentum-carry tests, the bilinear FEM wave integrator and
  the space-time spline wave integrator. The other sixteen files there are numerical experiments
  that integrate for thousands of steps and write `.jld2` archives and figures; they stay
  hand-run. `test/integrator/SpaceTime_Spline_Wave_int.jl` states its assertions at top level, so
  the entry point supplies the `@testset` it lacks.
- **The test suite follows the shared convention.** `test/runtests.jl` runs each test file in its
  own `@safetestset`, in the `core` group. The five test files are renamed after the source file
  they test, under `test/integrator/`. `test/quality/aqua.jl` is new. The sixteen experiment
  scripts and the two batch scripts that held no test move from `test/` to `scripts/`. The four
  failing assertions of the full-multiplier test and the final-time error assertion of the FEM
  test are `@test_broken`, and four Aqua checks are `broken` (issues #2–#7).

### Bug Fixes

### Breaking Changes

- **`tests/` renamed to `test/`.** Neither `Pkg.test()` nor `julia-actions/julia-runtest` looks at
  `tests/`, so the ten scripts there were unreachable by any standard tool. Anything referring to
  the old path by name needs updating.
- **`[compat] julia = "1.10"` added.** There was no Julia bound at all. 1.10 is the LTS and the
  floor across the tree, and the CI matrix resolves its lower entry from this field.
- **`test/Manifest.toml` untracked.** The root one was already removed; this one survived the
  `tests/` → `test/` rename while `.gitignore` claimed to ignore it. The test environment now
  resolves from `test/Project.toml`, so anyone relying on the pinned versions it recorded will get
  a fresh resolve instead.

## Open Issues

Full detail, and the order to work in, in `~/Research/Tasks/Revive MultiSymplectic.md`. Recorded
2026-08-31:

- **Five failing assertions are `@test_broken`, each with an issue.** The package itself loads and
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
- Stale bounds: `GeometricIntegratorsBase = "0.1.11"` against 0.6.4, `SimpleSolvers = "0.7.8"`
  against 0.13.2. Only 10 of 31 dependencies are bounded at all.
- `Infiltrator` and `Plots` are hard dependencies of the package, and `test/Project.toml` pulls in
  `CairoMakie`, `Plots`, `BenchmarkTools` and `Revise` as well; a headless runner installs all of
  it to run five test files. (`Gtk4`, `ProfileView` and `PProf` are `[weakdeps]` and are not.)

Separately, four files are **not formatted and cannot be**, and each will block the pre-commit hook
if ever staged:

- `src/utils/common.jl` is one of two files in the tree that JuliaFormatter cannot process at all —
  four flat 64-element `Float64` literals, longest line 1624 characters, which take the Julia
  process down rather than raising a catchable error.
- `src/integrator/NN_PDE_int.jl`, `NN_PDE_int_symbolic.jl` and `NN_PDE_LSGD_int.jl` fail for a
  different reason: JuliaFormatter's own output no longer parses. It moves the closing
  `) where {IPMT,}` of a constructor signature behind the trailing `# hyperparameters for OGA2d`
  comment on the same line. It detects this itself and declines to write, so the three are
  unformatted rather than broken. Moving that comment onto its own line would fix them, but that
  is a content change.
