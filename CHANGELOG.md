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

### Breaking Changes

- **The package is rebuilt on the current GeometricBase stack, and every integrator family leaves
  `src/`** for a return of its own. `LPDEProblem`, `LPDESystem`, `PDEIntegrator` and every
  `*_Integrator` type are gone. The names that replace them are `LagrangianPDEProblem`,
  `LagrangianDensity` and `SpaceTimeQuadrature`, together with the `SpaceTimeAnsatz`,
  `SlabCoupling` and `SlabInitialGuess` abstract types. A caller of a family that has not returned
  yet works at `294f023`.
- **`julia = "1.11"` is the floor**, and every dependency has a `[compat]` entry.
  The floor goes down from 1.12 to 1.11, and CI's `min` job resolves this bound. The dependency
  list drops `AbstractNeuralNetworks`, `BSplineKit`, `CompactBasisFunctions`, `ForwardDiff`,
  `GeometricMachineLearning`, `IterTools`, `Parameters`, `Plots`, `SimpleSolvers`, `StaticArrays`,
  `SymbolicNeuralNetworks`, `Zygote`, `LinearAlgebra`, `Random` and `Statistics`. The package has
  no `[weakdeps]` and no `[extensions]`. Every `[compat]` entry is a patch-level floor
  (`GeometricBase = "0.15.0"`, `GeometricEquations = "0.21.5"`, `GeometricIntegratorsBase = "0.6.9"`,
  `GeometricSolutions = "0.6.6"`, `QuadratureRules = "0.2.2"`), except `Symbolics = "7"`, which
  names the major.
- **The package is renamed `MultiSymplectic` → `MultisymplecticIntegrators`.** The repository moved
  from `ZeyuanLee/MultiSymplectic.jl` to `JuliaGNI/MultisymplecticIntegrators.jl`, and the package
  name, the module and `src/MultisymplecticIntegrators.jl` follow it. The UUID is unchanged, as the
  package was never registered. Replace `using MultiSymplectic` and every `MultiSymplectic.` prefix.
  The documentation now deploys to `JuliaGNI.github.io/MultisymplecticIntegrators.jl`.
- **`tests/` renamed to `test/`.** Neither `Pkg.test()` nor `julia-actions/julia-runtest` looks at
  `tests/`, so the ten scripts there were unreachable by any standard tool. Anything referring to
  the old path by name needs updating.
- **`test/Manifest.toml` untracked.** The root one was already removed; this one survived the
  `tests/` → `test/` rename while `.gitignore` claimed to ignore it. The test environment now
  resolves from `test/Project.toml`, so anyone relying on the pinned versions it recorded will get
  a fresh resolve instead.
- **`test/Project.toml` carries no `[compat]` bound for a dependency of the root `Project.toml`.**
  The six entries `BenchmarkTools`, `CairoMakie`, `GeometricIntegratorsBase`, `JLD2`, `Plots` and
  `Revise` are removed, and its `[compat]` table keeps only `Aqua` and `ExplicitImports`, which the
  root does not depend on. The test environment contains the package, so the resolver applies the
  root's bounds to every shared dependency; a test bound could only narrow them, and tests that run
  on narrower bounds than the package claims do not test what the package admits. This is the
  tree-wide rule of the test-suite unification.
- **Code that nothing reached is deleted; it stays reachable at `294f023`.** The four files that
  no `include` loaded go: `src/integrator/NN_PDE_LSGD_int.jl`, an older copy of
  `NN_PDE_int.jl`; `src/utils/PINN_for_init.jl`, a prototype script;
  `src/utils/Lagrangian_PDE_solution.jl` and `src/utils/solvers.jl`. So do the functions that
  nothing called: `box_init_plain`, `lagrangianPDE_derivatives`, the dense
  `spline2D_coeff_derivatives` with its `_time` and `_space` variants, `spacestep`,
  `generate_symbolic_u_trial`, the 4-argument `residual!` of `TrialNN_PDE_int`, and the
  `residual!` and `update!` of `FEM_Multisymplectic_Integrator`, which overrides
  `integrate_step!`. `TrialOGA2D` is gone, and so is the export of `GeometricPDESolution`, which no
  file defined (#3); Aqua's undefined-exports check, `broken` until now, passes and is no longer
  marked.
- **`show_status` is removed.** No integrator in `src/` takes the keyword or stores the field any
  more. The reports it switched on, errors against the exact solution printed during a step, are
  deleted, together with every unconditional `@show`, `println` and `@info`, every `@infiltrate` and
  the commented-out code in `src/`. With them goes the read of an undefined `tn` in `components!` of
  `NN_PDE_Integrator_Symbolic` and `ELM_PDE_int` under `show_status = true`. The `atom repeated`
  warning of the OGA2D initial guess in `TrialNN_PDE_int` is the exception: it was printed only
  under `show_status = true` and now prints whenever OGA selects an atom it already holds, on any
  run. The only live use of the keyword left in `scripts/` is in `scripts/reference_values.jl`,
  which records the integrators at `294f023` and runs only against that commit.

### New Features

- **The problem layer.** `LagrangianDensity` builds `∂L/∂u`, `∂L/∂v` and `∂L/∂w` once with
  `Symbolics.build_function`, so an evaluation is a Julia call and not a re-derivation.
  `LagrangianPDEProblem{T}` holds its spans at the element type, passes its parameters to the
  initial, boundary and exact functions, and states its state as a `GeometricBase.State` of
  algebraic variables, so neither `_state` nor `_vectorfield` is pirated. Its grid must reach the
  right end of `xspan` to within `sqrt(eps(T))` times the largest of `xstep` and the magnitudes of
  the two ends of `xspan`, or the constructor throws an `ArgumentError`. `field_boundary_data`
  throws an `ArgumentError` unless `times` holds the two times `(t₀, t₁)` of the slab.
- **`Wave`, `VectorWave`, `SineGordon` and `LinearTransport`.** The scalar and vector wave
  equations, the sine-Gordon kink and the transport of a Gaussian, each with its closed-form exact
  solution and its parameters. `Wave.problem` is massless: its density `L = ½v² − ½c²w²` has the
  standing wave as exact solution, and the mass `m` and the coupling `β` are keywords of
  `Wave.density` alone. `Wave` takes the keyword `D`. `VectorWave.problem` takes one amplitude per
  component in each of `A1`, `A2`, `B1` and `B2`, so its parameters are `(c, l, A1, A2, B1, B2)`
  and `D` is the number of entries; `VectorWave.component_parameters(params, i)` gives component
  `i` its `Wave` parameters. The default parameters give the two components different phases, so
  the angular momentum `J` of the exact solution is not zero. The exact solution of the vector
  problem is that of the uncoupled equations. The vector density couples the components through
  the quartic term `−¼β|u|⁴`, which keeps it invariant under a rotation of `(u, v, w)` for any
  `β`, so `J` is conserved by the coupled equations too.
  `VectorWave` carries the angular momentum `J` as the conserved diagnostic of the vector case.
  `VectorWave.density` is the scalar density of one component and takes the mass `m` only;
  `vector_density(; params, D, m, β)` carries the coupling.
- **`SpaceTimeQuadrature{T}`.** Composite Gauss–Legendre nodes and weights on given breakpoints, in
  `t̂ ∈ [0, 1]` and in `x`, with the four boundary point sets of a slab and the element type kept.
- **`integrate` returns `(; solution, traces)`.** The solution on the output grid, and per step the
  top trace at the nodes of the space quadrature, where the energy diagnostics integrate it. A step
  whose state holds a NaN ends the run with a warning: `traces` ends at that step, and the solution
  holds the NaN there. `SpaceTimeAnsatz` defaults to `NoExtrapolation()`, which copies the initial
  data into the states of the solution.
- **The suite is the standard layout**: `core` testsets for the problem layer, the quadrature and
  the shell, `Aqua.test_all` with no `broken` mark, and `test_explicit_imports`. This closes the
  known issues K1 (the broken marks), K2 (the stale bounds), K3 (`Plots` and the GUI packages in
  the test environment), K5 (the two files the formatter could not write), KI-2 (the unused test
  dependencies), KI-3 (the mark that asserts nothing) and KI-4 (the README and the manual saying
  the package does not load).
- **Continuous integration.** `.github/workflows/` holds three workflows, `CI.yml`, `Documenter.yml`
  and `TagBot.yml`.
- **A documentation build.** `docs/` holds `make.jl`, `Project.toml` and a two-page manual, so
  `Documenter` has something to build and the doctest job has an environment.
- **`test/runtests.jl` runs each test file in its own `@safetestset`.** Empty `ARGS` runs the `core`
  and `slow` groups. The `core` group holds eight files: the Aqua and explicit-imports quality
  checks, the Lagrangian density and Lagrangian problem tests, the linear transport and vector wave
  tests, the space-time quadrature test and the integrator shell test. `runtests.jl` has no `slow`
  block, so the `slow` group runs no file yet.
- **Reference values of the integrators at `294f023`.** `scripts/reference_values.jl` runs every
  integrator family on short `Wave` and `SineGordon` cases and writes
  `test/helpers/reference_values.jl`. Two cases fail at `294f023`. The script does not run on this
  tree, the file it writes is absent here, and no test includes it.

### Changed

- **CI coverage and cache.** CI uploads coverage from the `Julia 1 - ubuntu-latest` job instead of
  `Julia min`, and a test job saves the Julia cache only when it succeeds.
- **Repository hygiene.** `figures/` (13 files that nothing referenced), the tracked `.DS_Store`
  files and `rename_error.jl` are deleted.
