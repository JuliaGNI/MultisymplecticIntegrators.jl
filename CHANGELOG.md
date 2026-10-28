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
  `GeometricMachineLearning`, `IterTools`, `Parameters`, `Plots`, `StaticArrays`,
  `SymbolicNeuralNetworks`, `Zygote`, `Random` and `Statistics`. The package has
  no `[weakdeps]` and no `[extensions]`. Every `[compat]` entry is a patch-level floor
  (`GeometricBase = "0.15.0"`, `GeometricEquations = "0.21.5"`, `GeometricIntegratorsBase = "0.6.9"`,
  `GeometricSolutions = "0.6.6"`, `QuadratureRules = "0.2.2"`, `SimpleSplines = "0.3.1"`,
  `SimpleSolvers = "0.14.1"`, `Symbolics = "7.18.1"`), except `LinearAlgebra = "1"` and `SparseArrays = "1"`,
  which name the major. `SimpleSplines` requires `DomainSets = "0.8"`, which `Symbolics` accepts only from `7.18.1`,
  so the `Symbolics` floor is that patch rather than the major.
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
  right end of `xspan` to within the tolerance
  `min(sqrt(eps(T)) * max(xstep, |xspan[1]|, |xspan[2]|), xstep / 4)`, which stays at or below a
  quarter of `xstep`, so a step that does not divide the span is rejected on a long span too;
  otherwise the constructor throws an `ArgumentError`. `field_boundary_data`
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
- **The first integrator family returns: the space-time Galerkin slab core.**
  `SpaceTimeGalerkin{A, C, G} <: SpaceTimeMethod` holds a tensor B-spline ansatz, a coupling of
  consecutive slabs and an initial guess, and one step of it solves one slab.
  `TensorBSplineAnsatz`, on `SimpleSplines`, is a clamped B-spline basis in `t̂ ∈ [0, 1]` and a
  B-spline basis in `x`, each of given degree and cell count — a number for both directions or a
  `(time, space)` tuple — and the space basis is recombined to a homogeneous `Dirichlet` value or
  left `Free()`, the natural condition. A Gauss rule per cell integrates the products of the basis
  functions and of their first derivatives exactly. It counts the degree, as `SimpleSplines` does,
  not the order: a basis of order `k` is one of degree `k - 1`, and both degrees are at least 1.
  `CoefficientRestriction` makes the previous slab's top coefficient row the bottom datum of the
  next slab, and reads the momentum flux off the row that slab already computed, so no flux is
  re-integrated from a trace and the top of a slab is measured rather than imposed. The unknown of
  a step is the coefficient array of the slab: the bottom row of the time direction is the
  coupling's datum, and the other `Nt - 1` rows of each space function and component are
  determined by the interior weak Euler–Lagrange form of `src/core/weak_form.jl`, tested against
  the basis functions that vanish at the top. That is a nonlinear system of `(Nt - 1) * Nx * D`
  equations, which the `SimpleSolvers` Newton solver solves; the cache keeps each step's iterate
  count, its final residual `‖F(x)‖₂` and its initial residual `r₀ = ‖F(x₀)‖₂`, beside the top
  trace at the nodes of the space quadrature. The initial-guess interface of `src/initial_guess/`
  fills the solver's vector before a step: `Continuation()`, the default, carries the previous
  slab's top row into every row of the new slab; `FitTo(target; method = nothing)` fits the
  ansatz to `InitialCondition()`, to `PreviousSlab()` — the polynomial piece of the top time cell
  expanded about its top, so a quadratic field continues as the same quadratic — to
  `ExactSolution()` or to a `TargetFunction(f)`, with `Projection()` where the ansatz names no fit
  of its own, and a user-defined target or fit method is one method of `evaluate!` or `fit!`. Four
  refusals are `ArgumentError`s at construction: `FitTo(ExactSolution())` on a problem with no
  exact solution, `Projection()` on an ansatz that is not linear in its coefficients, an ansatz of
  another element type than the problem's, and an ansatz on another space interval than the
  problem's. `output_grid` refuses a `xstep` that is not a positive finite number before it divides
  by it. `LagrangianPDEProblem` takes any density with the density protocol — a callable
  `L(u, v, w, params)` and a `derivatives` method — not a
  `LagrangianDensity` alone, so the density built by `VectorWave.vector_density`, which couples the
  components, reaches a problem; the problem's parameters default to the density's own `params`
  field where it has one. `Project.toml` gains `SimpleSplines` (compat `0.3.1`), `SimpleSolvers`
  (compat `0.14.1`), `LinearAlgebra` and `SparseArrays` (compat `1`). Three new test files — the tensor B-spline
  ansatz, the coefficient restriction and the initial guess — join the `core` group, with
  `test/helpers/fem_stencil.jl`, an independent FEM stencil oracle for the assembly. The §A
  oracle, `test/helpers/reference_values.jl`, which part C's test-tree commit had removed, is back
  in the tree byte for byte from `ed13663` (K18), and the three new tests read it.
- **`integrate` returns `(; solution, traces)`.** The solution on the output grid, and per step the
  top trace at the nodes of the space quadrature, where the energy diagnostics integrate it. Every
  call starts a new run: the method's cache is reset first, so the first slab takes the problem's
  initial data and the steps a method records belong to that run alone. A step whose state holds a
  NaN ends the run with a warning: `traces` ends at that step, and the solution holds the NaN
  there. `SpaceTimeMethod` defaults to `NoExtrapolation()`, so the history states of the solution
  step hold the initial data.
- **The method, the ansatz and the three roles of an initial guess have abstract types of their
  own.** `SpaceTimeMethod <: GeometricMethod` is the supertype of `SpaceTimeGalerkin`, and
  `SpaceTimeAnsatz` is the supertype of `TensorBSplineAnsatz` alone, which is no
  `GeometricMethod` any more. `SlabInitialGuess` covers `Continuation` and `FitTo`;
  `InitialCondition`, `PreviousSlab`, `ExactSolution` and `TargetFunction` are `SlabTarget`s, and
  `Projection` is a `SlabFitMethod`. A value in the wrong role fails where it is given:
  `SpaceTimeGalerkin(ansatz; initial_guess = InitialCondition())`, `FitTo(Continuation())` and
  `FitTo(target; method = Continuation())` raise a `TypeError` or a `MethodError` at
  construction, where each failed with a `MethodError` at the first step. `SpaceTimeGalerkin` of
  a method and `integrate(problem, ansatz)` raise a named `ArgumentError`. A user-defined target
  subtypes `SlabTarget`, and a user-defined fit method `SlabFitMethod`, whose `fit!` takes the
  cache as a fifth argument: the cache now holds the buffers of the guess, so `initial_guess!`
  allocates 0 B for `Continuation()` and for every `FitTo` guess, where a `FitTo` guess allocated
  about 17 kB per step.
- **A failed nonlinear solve ends the run with a warning, as a NaN does.** When a slab's Newton
  solve raises SimpleSolvers' `NonlinearSolverException`, `integrate` warns with the timestep and
  returns the steps already taken: `traces` ends at the step before, and the solution is valid up
  to that step, its later entries not computed. Any other exception still propagates. This closes
  the known issue K17.
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
  and `slow` groups. The `core` group holds twelve files: the Aqua and explicit-imports quality
  checks, the Lagrangian density and Lagrangian problem tests, the linear transport and vector wave
  tests, the space-time quadrature test, the integrator shell test, the tensor B-spline ansatz,
  coefficient restriction, weak-multiplier and initial-guess tests. `runtests.jl` has no `slow`
  block, so the `slow` group runs no file yet.
- **Reference values of the integrators at `294f023`.** `scripts/reference_values.jl` runs every
  integrator family on short `Wave` and `SineGordon` cases and writes
  `test/helpers/reference_values.jl`. Two cases fail at `294f023`. The script does not run on this
  tree, the file it writes is back, restored byte for byte from `ed13663`, and the slab core's
  three new tests include it.
- **The two weak-multiplier couplings.** `WeakInitialCondition()` imposes a slab's initial
  condition weakly, with the multiplier as an unknown of the step, and
  `WeakInitialAndBoundaryConditions(; stride = 1, multiplier_basis = :bspline)` imposes the space
  boundary conditions at `x = a` and `x = b` weakly as well, from the problem's `bcs_function` at
  every node of the time rule. The multiplier of a slab's top surface is the momentum it hands on:
  `copy_internal_variables!` carries it into the cache's datum of the next step and into the
  solution step's internal state, so the hook that issue #7 records as a no-op carries the
  coupling's own state. Both couplings take `multiplier_basis = :bspline`, the ansatz's own
  trace basis, or `:lagrange`, the nodal basis of the same space, and the second thins it by its
  `stride`: at `stride = 1` that coupling's bottom-surface multiplier set loses the two corner
  functions of the trace basis, where they would constrain the corners twice and leave the first
  slab rank deficient by two (§Audit 2.8). The first slab's bottom multiplier is the initial
  momentum read into the multiplier set — by the quadrature-weighted `L²` projection where the set
  is thinned, by the nodal least-squares fit where the whole trace basis is kept — factorised
  once per integrator (§Audit 5.4), and the multipliers of a step are guessed in `initial_guess!`
  rather than inside a residual evaluation (§Audit 2.2). The multiplier blocks of the `D > 1` case
  keep their components apart (§Audit 1.3). Each coupling accepts one space boundary condition and
  refuses the rest, naming the condition it needs and the one it got: `Dirichlet()` for
  `WeakInitialCondition` and `Free()` for `WeakInitialAndBoundaryConditions`. The two
  `Galerkin_Bspline_Integrator` entries of the reference values are unconverged solves of a
  singular layout, and the two couplings are compared with closed forms instead; K21 and K22 record
  why the entries cannot serve as references.

### Changed

- **CI coverage and cache.** CI uploads coverage from the `Julia 1 - ubuntu-latest` job instead of
  `Julia min`, and a test job saves the Julia cache only when it succeeds.
- **Repository hygiene.** `figures/` (13 files that nothing referenced), the tracked `.DS_Store`
  files and `rename_error.jl` are deleted.
