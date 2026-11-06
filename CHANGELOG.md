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
  `*_Integrator` type are gone. The names that replace them are `LagrangianPDEProblem` and
  `LagrangianDensity`, together with the `SpaceTimeAnsatz`, `SlabCoupling` and `SlabInitialGuess`
  abstract types. A caller of a family that has not returned yet works at `294f023`.
- **`SpaceTimeQuadrature` and `boundary_points` are removed, with their test.** Nothing in the
  package used them: the slab integrates on the tensor-product rule of `SimpleSplines` that the
  ansatz owns. The package no longer exports `nodes`, `weights` and `nnodes`, whose only methods of
  its own were those on `SpaceTimeQuadrature`.
- **`julia = "1.12"` is the floor**, and every dependency has a `[compat]` entry.
  The floor stays at 1.12, because every `SimpleSplines` release from 0.3.3 on requires Julia 1.12,
  and CI's `min` job resolves this bound. The dependency
  list drops `AbstractNeuralNetworks`, `BSplineKit`, `CompactBasisFunctions`, `ForwardDiff`,
  `GeometricMachineLearning`, `IterTools`, `Parameters`, `Plots`, `QuadratureRules`,
  `StaticArrays`, `SymbolicNeuralNetworks`, `Zygote`, `Random` and `Statistics`. The package
  carries no `Plots` dependency: the plotting layer of the entry below arrives as the weak
  dependency `[weakdeps] Makie` with `[extensions] MultisymplecticIntegratorsPlots = "Makie"`. Every
  `[compat]` entry is a patch-level floor
  (`GeometricBase = "0.15.0"`, `GeometricEquations = "0.21.5"`, `GeometricIntegratorsBase = "0.6.9"`,
  `GeometricSolutions = "0.6.6"`, `SimpleSplines = "0.3.4"`,
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
- **A density that couples the components of the field receives `NTuple`s, not `AbstractVector`
  views.** A custom density with its own `derivatives` method, such as `VectorWave.VectorDensity`,
  gets `u`, `v` and `w` as `NTuple{D}`s from the slab through `density_derivatives!`, and from its
  couplings through `momentum_load!` and `flux_load!`, which call `point_derivatives`; the slab
  previously passed it `AbstractVector` views. `energy_error` still passes `Vector`s for `D > 1` and
  scalars for `D = 1`. A `derivatives` method restricted to `AbstractVector` raises a `MethodError`
  in every coupling, so it must be widened to any indexable collection of length `D`. The call
  method is reached only by the energy diagnostics, with the same arguments. The docstrings of
  `LagrangianPDEProblem`, `derivatives` and `VectorWave.VectorDensity` state this contract. The
  derivative path that returns these tuples is described under the pointwise bullet in New
  Features.

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
  `vector_density(; params, D, m, β)` carries the coupling. `Wave.problem`, `SineGordon.problem`
  and `LinearTransport.problem` convert their parameters to `T`, as `VectorWave.problem` does, so a
  problem of `Float32` evaluates its density and energy in `Float32`.
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
  count, its final residual `‖F(x)‖₂`, its initial residual `r₀ = ‖F(x₀)‖₂` and whether it
  converged, beside the top trace at the nodes of the space quadrature. The initial-guess
  interface of `src/initial_guess/` fills the solver's vector before a step: `Continuation()`, the
  default, carries the previous slab's top row into every row of the new slab;
  `FitTo(target; method = nothing)` fits the ansatz to `InitialCondition()`, to `PreviousSlab()`
  — the polynomial piece of the top time cell expanded about its top, so a quadratic field
  continues as the same quadratic — to
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
  field where it has one. `Project.toml` gains `SimpleSplines` (compat `0.3.4`), `SimpleSolvers`
  (compat `0.14.1`), `LinearAlgebra` and `SparseArrays` (compat `1`). Three new test files — the tensor B-spline
  ansatz, the coefficient restriction and the initial guess — join the `core` group, with
  `test/helpers/fem_stencil.jl`, an independent FEM stencil oracle for the assembly. The §A
  oracle, `test/helpers/reference_values.jl`, which part C's test-tree commit had removed, is back
  in the tree byte for byte from `ed13663` (K18), and the three new tests read it.
- **The slab core reaches its ansatz and its coupling through one seam.** The cache holds an
  ansatz block, `TensorBSplineSlab` for the B-spline ansatz, with the coefficient array, the tables
  of the slab's quadrature and the evaluation of the top of the slab, and a coupling block that
  carries its coupling: `NoCouplingData`, `RestrictionData` with the restriction's bottom row and
  momentum load, or `MultiplierData`. Every function of a coupling that runs within a step
  dispatches on its block. Every coupling declares its own `free_rows`, and `coupling_term!` has no
  default, so a coupling that implements neither is refused with a `MethodError` that names the
  function. A coupling may transform the field through `transform_field!`, which the core applies
  in the residual, the trace and the solution. The trace and the solution come from one evaluation
  of the top of the slab. The manual's page *The slab seam* states the roles of the ansatz, the
  coupling and the guess, what each keeps in the cache, and the functions of the seam.
- **`integrate` returns `(; solution, traces, stats)`.** The solution on the output grid, per step
  the top trace at the nodes of the space quadrature, where the energy diagnostics integrate it,
  and `stats = (; iterations, residuals, initial_residuals, converged)`, the record of each step's
  Newton solve, with one entry per stored trace. A step has converged when its residual is at most
  `f_abstol + f_reltol · r₀`, with the tolerances of the integrator's own solver options; the term
  in `r₀` counts only for a finite `r₀`, so a residual that is not finite never converges. A step
  that has not converged warns with `on_failure = :warn`, the default, and throws with
  `on_failure = :error`; both name the step, and another value of `on_failure` throws an
  `ArgumentError`. `integrate(integrator; on_failure)` takes the keyword itself, so a keyword call
  on an integrator no longer falls through to GeometricIntegratorsBase's `integrate`, which threw a
  `MethodError` for `Solution`. Any other keyword is refused with Julia's `MethodError` for an
  unsupported keyword. This closes the known issue K10. A method extends `solve_stats(cache)` to
  give the record. Every call starts a new run: the method's cache is reset first, so the first
  slab takes the problem's initial data and the steps a method records belong to that run alone.
  A step whose state holds a NaN ends the run with a warning: `traces` ends at that step, and the
  solution holds the NaN there. `SpaceTimeMethod` defaults to `NoExtrapolation()`, so the history
  states of the solution step hold the initial data.
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
  cache as a fifth argument. A `FitTo` guess allocates its sample and its fitted coefficients once
  per step, and `Projection()` is `SimpleSplines`' `l2_projection!`; the cache holds no scratch of
  the guess.
- **A failed nonlinear solve ends the run with a warning, as a NaN does.** When a slab's Newton
  solve raises SimpleSolvers' `NonlinearSolverException`, `integrate` warns with the timestep and
  returns the steps already taken: `traces` ends at the step before, and the solution is valid up
  to that step, its later entries not computed. Any other exception still propagates. This closes
  the known issue K17.
- **The suite is the standard layout**: `core` testsets for the problem layer and the shell,
  `Aqua.test_all` with no `broken` mark, and `test_explicit_imports`. This closes the known issues
  K1 (the broken marks), K2 (the stale bounds), K3 (`Plots` and the GUI packages in
  the test environment), K5 (the two files the formatter could not write), KI-2 (the unused test
  dependencies), KI-3 (the mark that asserts nothing) and KI-4 (the README and the manual saying
  the package does not load).
- **Continuous integration.** `.github/workflows/` holds three workflows, `CI.yml`, `Documenter.yml`
  and `TagBot.yml`.
- **A documentation build.** `docs/` holds `make.jl`, `Project.toml` and a four-page manual, so
  `Documenter` has something to build and the doctest job has an environment. The page
  `docs/src/seam.md`, *The slab seam*, states the seam of the slab integrator and the action form
  that it admits. The library is split over two pages, the package module and its submodules, so
  that each page stays below Documenter's size threshold.
- **`test/runtests.jl` runs each test file in its own `@safetestset`.** Empty `ARGS` runs the `core`
  and `slow` groups. The `core` group holds fifteen files: the Aqua, explicit-imports and sources
  quality checks, the Lagrangian density and Lagrangian problem tests, the linear transport and
  vector wave tests, the tensor B-spline ansatz, coefficient restriction, coupling seam and
  initial-guess tests, the integrator shell and archive tests, and the diagnostics and
  plotting tests. The
  `slow` group holds the weak-multiplier test and the six property files of the entry below.
- **Reference values of the integrators at `294f023`.** `scripts/reference_values.jl` runs every
  integrator family on short `Wave` and `SineGordon` cases and writes
  `test/helpers/reference_values.jl`. Two cases fail at `294f023`. The script does not run on this
  tree, the file it writes is back, restored byte for byte from `ed13663`, and the slab core's
  three new tests include it.
- **The two weak-multiplier couplings.** `WeakInitialCondition()` imposes a slab's initial condition
  weakly, with the multiplier as an unknown of the step, and
  `WeakInitialAndBoundaryConditions(; multiplier_basis = :bspline)` imposes the space boundary
  conditions at `x = a` and `x = b` weakly as well, from the problem's `bcs_function` at every node
  of the time rule. The multiplier of a slab's top surface is the momentum it hands on:
  `slab_bottom!` carries it into the datum of the next step, and `copy_internal_variables!` into
  the solution step's internal state, so the hook that issue #7 records as a no-op carries the
  coupling's own state. Both couplings take `multiplier_basis = :bspline`, the ansatz's own trace
  basis, or `:lagrange`, the nodal basis of the same space, and the second's bottom-surface
  multiplier set leaves out the two corner functions of the trace basis, where they would constrain
  the corners twice and leave the first slab rank deficient by two (§Audit 2.8). The first slab's
  bottom multiplier is the initial momentum read into the multiplier set by the
  quadrature-weighted `L²` projection, factorised once per integrator (§Audit 5.4), its bottom load
  vector is cached once per step in `slab_bottom!`, and the multipliers of a step are guessed in
  `initial_guess!` rather than inside a residual evaluation (§Audit 2.2). The multiplier blocks of
  the `D > 1` case keep their components apart (§Audit 1.3). Each coupling accepts one space
  boundary condition and refuses the rest, naming the condition it needs and the one it got:
  `Dirichlet()` for `WeakInitialCondition` and `Free()` for `WeakInitialAndBoundaryConditions`. The
  two `Galerkin_Bspline_Integrator` entries of the reference values are unconverged solves of a
  singular layout, so the coupling that imposes the boundary conditions is compared with closed
  forms instead, and the one that imposes the initial condition alone is compared with a reference of
  its own: `WAVE_WEAK_INITIAL_CONDITION` in `test/helpers/weak_initial_condition_reference.jl`, which
  `scripts/weak_initial_condition_reference.jl` writes, with the commit, the Julia version, the
  `--check-bounds` flag and the dependency versions in its header. It is not the
  `WAVE_GALERKIN_DIRICHLET_BSPLINE` entry of `test/helpers/reference_values.jl`: that entry does not
  use the quadrature-weighted `L²` projection above, so its `u` differs from this coupling's solution
  by up to 1.18e-7 over the five steps and 21 grid nodes. The comparison keeps the bound
  `800 eps(T)`, 100 times the Newton tolerance `8 eps(T)`, in `Float64` and `Float32`, and
  `.JuliaFormatter.toml` ignores the generated file, as it does `reference_values.jl`;
  K21 and K22 record why the entries of the full-multiplier layout cannot serve as references.
- **The energy diagnostics.** `energy_density(density, u, v, w, params)` is the Hamiltonian density
  `v ∂L/∂v − L` of a Lagrangian density, built from the density's own derivative and value, so every
  problem the package integrates has a conserved quantity without a formula of its own. A density
  of one component at a time is applied to each component of the field, and a density that couples
  the components — `VectorWave.vector_density` for one — takes the vectors of a point and closes
  the product as their inner product. `energy_error(problem, method, traces)` integrates it over
  the space quadrature at the nodes the trace is stored on, taking the weights from the ansatz, and
  returns the relative deviation from the first stored step: one value per step, the first of which
  is zero. A first step whose energy vanishes falls back to the absolute deviation rather than a
  `NaN`, and a run that stored no trace raises an `ArgumentError`; so does a trace whose `u`, `v`
  and `w` hold different numbers of entries, or whose length is not a whole number of the field's
  components. `solution_data(problem, result)` is the plain `(t, x, u)` a figure is drawn from, so
  an archived run can be redrawn without a solution object tied to the version that built it.
- **The plotting extension.** The `Diagnostics` submodule holds the declarations of three figures —
  a space-time plot of a solution, the energy error of several runs over time on a logarithmic
  axis, and an animation of the field over the stored steps — together with the dispatcher that
  turns one archive into the figure its `"kind"` asks for, so a renderer needs no second registry of
  which experiment produces which picture. `using Makie`, or a backend such
  as `CairoMakie`, loads `MultisymplecticIntegratorsPlots`, which fills them in and applies one
  theme. The names live in a submodule rather than at the package's top level because
  `plot_solution` and `plot_energy_error` are names `GeometricProblems` already exports, and the
  extension therefore extends `Diagnostics.plot_solution` instead of shadowing it. Fonts, line
  widths and colours come from `Diagnostics.plot_theme()`, applied inside each figure method, so no
  figure sets a font, a line width or a colour of its own; the colour cycle is the colour-blind-safe
  palette Makie provides. The dependency is the weak dependency `Makie = "0.24"` with
  `[extensions] MultisymplecticIntegratorsPlots = "Makie"`, so a caller who loads no backend pays
  nothing for the layer.
- **`scripts/` is a project of its own.** It declares the packages its scripts use, with this
  package as a path source, so a script finds `JLD2` and `CairoMakie` where the package's own
  environment holds neither. `archives.jl` holds the two output directories, the option parser that
  refuses an unknown flag, and the writer and reader of the archive: one JLD2 file per run at
  `runs/<stem>.jld2`, with a `"kind"` and a stem that `store_run!` requires to be non-empty, not to
  start with a dot and to hold no path separator, and `load_runs()` reading the directory back, so
  a renderer needs no registry to know what an archive is, and the schema itself is stated in
  `scripts/README.md`. `experiments.jl` is the registry, which holds no family's entries and runs
  nothing on `include`; `run_family.jl` runs the smallest entry of the family it is named,
  `figures.jl` is the only place a figure is saved, and the energy scan, the phase-drift scan and
  the residual profile are carried against the merged family: the quadrature check that was a
  script is a test of the energy diagnostic, and the two scans and the profile are rewritten for
  the new constructors. `runs/` and `results/` are anchored in `.gitignore`. The archive test,
  `test/integration/archives.jl`, checks `atomic_write`, the stem refusal of `store_run!` and the
  dot-skip of `load_runs`; this closes the known issue K32.
- **The old `scripts/` are deleted (K6).** The drivers of the removed families stay reachable at
  `294f023`. The placeholder drivers of the network and symbolic families, which printed that the
  registry held no entry of their family and why, are deleted with them, so the registry holds
  `:spline` alone and a family without a model file is refused by name rather than driven.
- **The scripts are one append-only registry and one driver, and the plotting layer returns what it
  draws.** `scripts/experiments.jl` is the registry — `register_family!(name, entries; run)`,
  `family_entries` and `family_run` — and holds no family's entries: each family's entries and the
  `run` that turns one of them into an archive live in its own `scripts/<family>_model.jl`, and
  `spline_model.jl` registers `:spline`. A name already registered and a name the registry does not
  hold each raise an `ArgumentError` naming the family, so two model files that claim one name are
  reported rather than settled in favour of the later one. `scripts/run_family.jl` replaces
  `run_spline.jl`, `run_network.jl` and `run_symbolic.jl`, which are deleted: it includes every
  `*_model.jl` beside it in sorted order, takes the family name and then the stems of its entries,
  and returns 0 when every selected entry was archived and 1 when one failed. A run that throws is
  reported on stderr with its full backtrace, archives nothing, and makes the driver exit 1, so a
  failed family is distinguishable from one that ran. The file is includable — `exit(main(ARGS))`
  runs only under `PROGRAM_FILE` — so a caller can register a family of its own and run the real
  driver over it, which is what makes a new family a new file and no edit to the registry or the
  driver. Every file the scripts write goes through `atomic_write(write, path)`: a sibling temporary
  carrying the destination's extension — a rename within one file system, in the format a writer
  takes from the file name — renamed over the destination, and removed with the writer's exception
  rethrown when the writer fails, so `store_run!` and `figures.jl` leave an earlier file whole
  rather than truncated. The `"solution"` archive drops the four keys nothing read — `final_time`,
  `dimension`, `energy_t` and `energy_error` — and holds `kind`, `stem`, `label`, `problem_label`,
  `timestep`, `t`, `x` and `u`, whose row of the schema table in `scripts/README.md` is now the one
  statement of it; a run prints its own maximum energy error from its own series instead of
  archiving a copy that could only disagree with it. Two silent mistakes of the spline family now
  say what is wrong: `spline_method` raises an `ArgumentError` naming an unknown coupling symbol
  instead of silently running `:restriction`, and `phase_drift` measures at the last stored trace,
  `solution.t[length(result.traces)]`, raising an `ArgumentError` for a run that stored nothing, so
  a run that stopped early no longer reports a drift of 1.0.
  `Diagnostics.figures(data)` returns the `Figure` itself — one figure per archive, and a
  `D`-component solution as one figure of `D` panels — and names nothing, so `figures.jl` writes
  `results/<stem>.pdf` through `atomic_write`. `SolutionFields` and the extension's four
  "Implemented in the … extension" stub docstrings are gone, so each name has the one docstring of
  its declaration in `src/plots.jl`, and `draw_solution` no longer shadows Makie's `heatmap` with a
  local of that name. `title = nothing` takes the archive's heading and `title = ""` draws none, in
  the declarations of `plot_solution` and `plot_energy_error` and in their methods.
  `animate_solution` takes `title = nothing` as its default too, where `nothing` and `""` draw no
  heading before the time, and a given title is drawn before it. `plot_energy_error` adds a
  `Legend` only when it drew a series, so an all-zero error figure is
  drawn without one instead of throwing. In `src/diagnostics.jl` a node's `D` values are a static
  tuple rather than a heap `Vector`, so the energy of a vector trace builds no container per node,
  and `scripts/Project.toml` gains a `[compat]` entry for each of its remaining ten unbounded
  dependencies, so no package of the scripts' environment is unbounded.
- **A property suite over seeded generated problems** checks the slab core against an oracle that
  shares none of its code. `test/helpers/problem_generator.jl` draws 24 problems from a keyword
  RNG alone: `D = 1` and `D = 2`, potentials of degree 2, 3 and 4 with random coefficients, the
  rotation-invariant vector density, random wave speeds, timesteps, spans, start times, degrees,
  cell counts and Gauss counts, Noether problems and problems symmetric under `x ↦ a + b − x`.
  `test/helpers/action_oracle.jl` builds the discrete slab action from the density, the data
  functions and SimpleSplines' evaluation of the ansatz, with each coupling's terms and datums, and
  differentiates it with ForwardDiff. Six files under `test/verification/properties/`, in the group
  `slow`, assert in `Float64` and `Float32` the generator's controls and the oracle's independence,
  and under all three couplings: that `residual!` is the gradient of the action on the first and the
  second slab; that the solver's Jacobian equals central differences; that a linear density
  converges in at most two Newton iterations; that the rotation charge of a Noether problem is
  conserved; and that symmetric data give a symmetric top row and symmetric multipliers. The test
  environment adds `ForwardDiff`, `Random` and `SimpleSolvers`, and the `SimpleSplines` bound is
  `0.3.4`, the release whose mesh ends at `b` exactly and whose Greville nodes stay inside the span.
  This closes the known issue K18, whose fix had already landed.
- **`scripts/action_residual_spike.jl` measures the residual as the gradient of a discrete action.**
  It differentiates the property suite's slab action under `CoefficientRestriction` on `Wave` and
  `SineGordon`, checks the gradient against `residual!` and a block of the Hessian against the
  solver's Jacobian at each size, and then times both against the package's code, with ForwardDiff
  and with Enzyme (Mooncake where Enzyme fails), one fresh process per backend and problem.
  `--quick` and `--selfcheck` are its gate. The `scripts/` environment adds `BenchmarkTools`,
  `DifferentiationInterface`, `Enzyme`, `Mooncake`, `SimpleSolvers` and the packages of the two
  test helpers it includes.

- **Direct single-factorisation solve for linear densities.** `SpaceTimeGalerkin` takes the keyword
  `linear_solver = false`. When `linear_solver = true`, `check_method` verifies that the ansatz is
  linear in its coefficients and that the problem's density is linear (`is_linear_density`, checked
  symbolically when `LagrangianDensity` is built and true for `VectorWave.VectorDensity` when
  `β == 0`), refusing a nonlinear ansatz or density with an `ArgumentError`. On the first step,
  `integrate_step!` evaluates and LU-factorises the constant slab Jacobian once, and every step
  then solves from zero with that factorisation and takes one step of iterative refinement with
  it. The run factorises once (`factorisations(int) == 1`), and `stats.iterations` records 2 for
  each step. Over five slabs the deviation of the solution from the Newton solve is at most
  9.5 eps(T) in `Float32` and 13.9 eps(T) in `Float64`.
- **One pointwise density-derivative path and a function barrier for the step loop.**
  `density_derivatives!`, `momentum_load!` and `flux_load!` each call `point_derivatives`,
  `WeakInitialCondition` and `WeakInitialAndBoundaryConditions` compute `bottom_load` once per step
  in `slab_bottom!`, and `VectorWave.derivatives(::VectorDensity, …)` returns `NTuple{D, T}` tuples
  on the stack rather than heap `Vector` comprehensions, so `residual!` allocates 0 bytes on
  `VectorWave` as well as `Wave`, closing K19. `integrate` delegates its step loop to
  `integrate_steps!`, where the solution, solution step, trace vector, integrator and cache have
  concrete types, closing K30.

### Changed

- **CI coverage and cache.** CI uploads coverage from the `Julia 1 - ubuntu-latest` job instead of
  `Julia min`, and a test job saves the Julia cache only when it succeeds.
- **Test floors for the merged downgrade check.** CI's Downgrade job resolves `Project.toml` and
  `test/Project.toml` together at their lower bounds and requires each bound to be the version it
  resolves to. `test/Project.toml` raises `ForwardDiff` from `0.10` to `0.10.36, 1`, the lowest
  release that the resolve of the merged environment reaches. It adds `SafeTestsets = "0.1"`, so
  that every test dependency outside the standard library has a bound; 0.1.0 is again the lowest
  release that the merged resolve reaches.
- **`SimpleSplines` 0.3.5 is the floor**, from 0.3.4. With FFTA 0.3.2 or later loaded beside
  FFTW, as CairoMakie loads it, SimpleSplines 0.3.4 builds the periodic mass solver through
  FFTA's `plan_rfft`, which takes no `flags` keyword. The ansatz with a `Periodic()` boundary then
  raises a `MethodError`, and a weak-multiplier coupling no longer refuses it with an
  `ArgumentError`. SimpleSplines 0.3.5 calls FFTW's method by `invoke`.
- **Repository hygiene.** `figures/` (13 files that nothing referenced), the tracked `.DS_Store`
  files and `rename_error.jl` are deleted.
