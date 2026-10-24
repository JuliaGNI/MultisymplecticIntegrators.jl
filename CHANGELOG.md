# Release Notes

All notable changes to MultisymplecticIntegrators.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually
changed, so that a compat-only bump can be told apart from a rename or a change in results.

This file was started on 2026-08-31 and deliberately holds no entries. Nothing has been
released yet — there are no tags, and `Project.toml` stands at `0.1.0` — so the development
history that predates this file is in `git log` alone. It is named as a gap rather than
reconstructed, because a changelog assembled after the fact loses exactly the reasoning that
makes it worth keeping.

## [Unreleased] — targeting 0.1.0

### Changed

- **CI coverage and cache.** CI uploads coverage from the `Julia 1 - ubuntu-latest` job instead of
  `Julia min`, and a test job saves the Julia cache only when it succeeds.
- **The 64- and 128-node Gauss–Legendre rules come from QuadratureRules.** `GaussQuadrature64`
  and `GaussQuadrature128`, two functions that returned hard-coded tables, are deleted.
  `composite_quadrature` and `construct_quadrature_grid_with_boundary` build every rule as
  `GaussLegendreQuadrature(k; fast = true)`; the exact default needs more than five minutes for 128
  nodes. The fast rules agree with the deleted tables to within `4 eps()`, which
  `test/quadrature/gauss_tables.jl` checks. Without the 1 624-character table lines,
  `src/utils/common.jl` is formatted and is no longer in the `ignore` list of
  `.JuliaFormatter.toml`, which supersedes the *Bug Fixes* entry that excluded it.
- **Repository hygiene.** `figures/` (13 files that nothing referenced), the tracked `.DS_Store`
  files and `rename_error.jl` are deleted.

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
  broken marks of the full-multiplier test and the final-time error assertion of the FEM
  test are `@test_broken`, and four Aqua checks are `broken` (issues #2–#7).
- **Reference values of the current integrators.** `scripts/reference_values.jl` runs every
  integrator family on short `Wave` and `SineGordon` cases and writes
  `test/helpers/reference_values.jl`: the solution after each step, the energy error, and the
  Newton iterations and residuals of each step, each entry with its constructor call. The solver
  options that every Newton case shares are written once, as `SOLVER_OPTIONS`. `converged` is
  SimpleSolvers' `isconverged`, which a small change of the iterate can meet at a large residual.
  A network case runs twice from the same seed, and the script stops if the two runs differ. The
  file also holds the error of the FEM stencil at `t = 5` and `t = 200` on three grids and at CFL
  1, and the observed order of the full-restriction spline integrator. Its header names the
  commit, the Julia version, the `--check-bounds` flag, the thread counts and the versions of the
  22 direct dependencies and OpenBLAS_jll. The values can depend on the `--check-bounds` flag, so
  a rewritten integrator must reproduce these values under the same flag. Two cases fail on the
  current code and record the exception and the step that raised it: `TrialNN_PDE_int` on `Wave`,
  and `Sindy_PDE_Integrator` on `Wave`, which fails at step 1. Both occur at the default
  `regularization_factor = 0`, and both stop with a NaN in the Newton direction. No test includes
  the file yet.

### Bug Fixes

- **`src/utils/common.jl` is excluded from JuliaFormatter.** The file holds four 64-element
  Gauss–Legendre literals on single lines, the longest 1 624 characters. JuliaFormatter 2.13.0
  kills the Julia process on it, and 2.14.0 does not finish in 300 s, so the pre-commit hook
  blocked any commit that staged the file. An `ignore` entry in `.JuliaFormatter.toml` skips it;
  the file itself is unchanged.
- **Two `show` methods print the x-breaks of the basis once.** `Dirichlet_BSpline2D` stores its
  knot interval and its knots in physical x, and its `show` and that of
  `Galerkin_Full_Restriction_Bspline_Integrator` multiplied both by the width of `xspan` again: on
  `xspan = (0.0, 2.0)` with a knot interval of 0.5 they showed `Break Length x: 1.0`. They print the
  stored values now.

### Breaking Changes

- **The package is renamed `MultiSymplectic` → `MultisymplecticIntegrators`.** The repository moved
  from `ZeyuanLee/MultiSymplectic.jl` to `JuliaGNI/MultisymplecticIntegrators.jl`, and the package
  name, the module and `src/MultisymplecticIntegrators.jl` follow it. The UUID is unchanged, as the
  package was never registered. Replace `using MultiSymplectic` and every `MultiSymplectic.` prefix.
  The documentation now deploys to `JuliaGNI.github.io/MultisymplecticIntegrators.jl`.
- **`tests/` renamed to `test/`.** Neither `Pkg.test()` nor `julia-actions/julia-runtest` looks at
  `tests/`, so the ten scripts there were unreachable by any standard tool. Anything referring to
  the old path by name needs updating.
- **`[compat] julia = "1.10"` added.** There was no Julia bound at all. 1.10 is the LTS and the
  floor across the tree, and the CI matrix resolves its lower entry from this field.
- **`test/Manifest.toml` untracked.** The root one was already removed; this one survived the
  `tests/` → `test/` rename while `.gitignore` claimed to ignore it. The test environment now
  resolves from `test/Project.toml`, so anyone relying on the pinned versions it recorded will get
  a fresh resolve instead.
- **`test/Project.toml` carries no `[compat]` bound for a dependency of the root `Project.toml`.**
  The six entries `BenchmarkTools`, `CairoMakie`, `GeometricIntegratorsBase`, `JLD2`, `Plots` and
  `Revise` are removed, and with them the whole `[compat]` table, which held nothing else. The test
  environment contains the package, so the resolver applies the root's bounds to every shared
  dependency; a test bound could only narrow them, and tests that run on narrower bounds than the
  package claims do not test what the package admits. This is the tree-wide rule of the test-suite
  unification.
- **Code that nothing reached is deleted; it stays reachable at `294f023`.** The four files that
  no `include` loaded go: `src/integrator/NN_PDE_LSGD_int.jl`, an older copy of
  `NN_PDE_int.jl`; `src/utils/PINN_for_init.jl`, a prototype script;
  `src/utils/Lagrangian_PDE_solution.jl` and `src/utils/solvers.jl`. So do the functions that
  nothing called: `box_init_plain`, `lagrangianPDE_derivatives`, the dense
  `spline2D_coeff_derivatives` with its `_time` and `_space` variants, `spacestep`,
  `generate_symbolic_u_trial`, the 4-argument `residual!` of `TrialNN_PDE_int`, and the
  `residual!` and `update!` of `FEM_Multisymplectic_Integrator`, which overrides
  `integrate_step!`. `TrialOGA2D` is gone, and so is the export of `GeometricPDESolution`, which no
  file defined (#3). Every integrator family stays. `src/` is 3 698 lines shorter than at
  `294f023`.
- **`show_status` is removed.** No integrator takes the keyword or stores the field any more, and
  a call that passes it throws a `MethodError`. The reports it switched on, errors against the
  exact solution printed during a step, are deleted, together with every unconditional `@show`,
  `println` and `@info`, every `@infiltrate` and the commented-out code in `src/`. With them goes
  the read of an undefined `tn` in `components!` of `NN_PDE_Integrator_Symbolic` and `ELM_PDE_int`
  under `show_status = true`. `Infiltrator` is no longer a dependency of the package or of its
  tests. The scripts in `scripts/` no longer pass the keyword.
