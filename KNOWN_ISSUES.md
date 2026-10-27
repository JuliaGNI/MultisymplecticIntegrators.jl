# Known issues

Defects found in review and recorded, not fixed. Each entry names its kind and its evidence.

### K6 · The driver scripts still name the removed integrator families.

- location: `scripts/`
- evidence: The package no longer holds the old integrators, and `Project.toml` no longer holds
  their dependencies. The scripts under `scripts/` still call `Galerkin_Bspline_Integrator`,
  `Galerkin_Dirichlet_Bspline_Integrator`, `Galerkin_Full_Restriction_Bspline_Integrator`,
  `FEM_Multisymplectic_Integrator`, `SpaceTime_Spline_Wave_Integrator`, `NN_PDE_Integrator`,
  `NN_PDE_Integrator_Symbolic`, `ELM_PDE_int`, `TrialNN_PDE_int`, `Sindy_PDE_Integrator`,
  `PDEIntegrator` and `OGA2D`, and they still import `BSplineKit`, `Plots`, `Zygote`,
  `Infiltrator` and `GeometricMachineLearning`; `grep -rnw 'Galerkin_Bspline_Integrator' scripts/`
  names them. They cannot run until the family each one drives returns, and `scripts/` gets its
  own environment and a registry in the part that owns that environment.
- kind: defect
- found: 2026-10-26

### K7 · The four problem constructors repeat themselves.

- location: `src/problems/wave.jl`, `src/problems/sine_gordon.jl`,
  `src/problems/linear_transport.jl`
- evidence: each `problem` function of `Wave`, `VectorWave`, `SineGordon` and `LinearTransport`
  converts its spans, builds the two closures and calls `LagrangianPDEProblem` in the same way,
  about twenty lines four times. One shared constructor taking the three exact functions would
  remove about sixty lines. Recorded rather than done, because the shared form has to be chosen
  against the boundary interface that the slab core defines.
- kind: defect
- found: 2026-10-26

### K10 · `integrate(integrator; keyword)` falls through to the ecosystem's method.

- location: `src/core/integrate.jl`
- evidence: the package's `integrate(::GeometricIntegrator…)` takes no keyword, so Julia's keyword
  dispatch chooses `GeometricIntegratorsBase`'s, whose body throws
  `MethodError: no method matching Solution(::LagrangianPDEProblem…)`. The documented forms are
  `integrate(problem, method; kwargs...)` and `integrate(integrator)`. A guard that names the
  supported form would read better.
- kind: defect
- found: 2026-10-26

### K11 · `field_exact` infers a small union on a scalar call.

- location: `src/problems/lagrangian_problem.jl`
- evidence: `Base.return_types(problem.exact_u, Tuple{Float64, Float64, typeof(problem.params)})`
  gives `Union{Float64, Vector{Float64}}`, because `D` is a runtime value: a scalar `x` returns a
  scalar for `D = 1` and a vector of `D` values otherwise. `Wave.exact_u` itself infers `Float64`.
  Closing it needs `D` in the type of the wrapper, or one wrapper per case.
- kind: defect
- found: 2026-10-26

### K14 · `∂L/∂u` of the wave density is a literal zero of type `Int`.

- location: `src/problems/lagrangian_density.jl`
- evidence: the wave density has no `u` term, so `build_function` returns the literal `0`:
  `derivatives(Wave.density(), 0.3, 0.7, -0.2).u` is `0 :: Int64` where the other two derivatives
  have the element type of the arguments, and the kept test asserts the type of `.v` only. A
  density with a mass term returns `Float64`, so the clause holds wherever `∂L/∂u` is not
  identically zero.
- kind: defect
- found: 2026-10-26

### K16 · The transport density is a null Lagrangian.

- location: `src/problems/linear_transport.jl`
- evidence: `L = ½u(v + c w)` is a total divergence: its Euler–Lagrange expression is
  `(v + c w)/2 − ∂ₜu/2 − c ∂ₓu/2 ≡ 0`, so the density alone carries no dynamics. The transport
  equation enters through the constraint `v = −c w`, which the ansatz of the family that integrates
  this problem imposes; this package ships no integrator, so nothing here uses it. A density with a
  non-zero Euler–Lagrange equation, such as a potential formulation with an auxiliary field, is a
  design question for the part that integrates the symbolic and network families.
- kind: defect
- found: 2026-10-26

### K17 · `integrate` does not catch a failed nonlinear solve.

- location: `src/core/integrate.jl`
- evidence: the ecosystem's `integrate!` (GeometricIntegratorsBase 0.6.10, `src/integrate.jl`)
  catches `NonlinearSolverException`, warns, and returns the solution computed so far; the
  package's `integrate` stops on a NaN the same way but lets the exception propagate, so a caller
  loses the steps already taken. The exception is owned by SimpleSolvers, which
  GeometricIntegratorsBase does not export. The slab core of `SpaceTimeGalerkin` is the first method
  of this package that solves a nonlinear system, and it depends on SimpleSolvers, so the fix is the
  `try`/`catch` of the ecosystem's loop and no dependency change.
- kind: defect
- found: PR #31

### K18 · The §A oracle was absent from the base tree.

- location: `test/helpers/reference_values.jl`
- evidence: part C's test-tree commit removed `test/helpers/reference_values.jl`, although C's spec
  does not list it, and the file is absent at `9d706ba`;
  `git show ed13663:test/helpers/reference_values.jl` returns the 93-line file with its twenty-five
  `const` entries. The part that returns the slab core restored it byte for byte from `ed13663`. It
  is the oracle the tests read, not a testset: `test/runtests.jl` lists it under no
  `@safetestset`.
- kind: defect
- found: 2026-10-27

### K19 · `output_grid` raises `InexactError` for a zero or non-finite `xstep`.

- location: `src/ansatz/tensor_bspline.jl:138`
- evidence: `output_grid` scales `round(Int, Δx / xstep)` before its `n ≥ 2` guard, so
  `output_grid(ansatz, 0.0)` raises `InexactError: Int64(Inf)` and `output_grid(ansatz, NaN)`
  raises `InexactError: Int64(NaN)`; a step of `2.0` on `xspan = (0, 1)` does reach the guard and
  raises `ArgumentError`. A finiteness and sign check before the division would name the fault.
- kind: defect
- found: 2026-10-27

### K20 · The slab core does not compare the ansatz's `xspan` with the problem's.

- location: `src/core/slab_integrator.jl:143`
- evidence: `Cache{ST}(problem, method)` reads the space interval from the ansatz and the output
  grid from the problem, `SlabEvaluation(a, xgrid(problem), …)`, and checks neither against the
  other, so an ansatz on another interval than the problem's integrates without a word. §Audit 2.4
  records the same mixture of two `xspan` sources on the old stack.
- kind: defect
- found: 2026-10-27

### K21 · A `Float32` problem with a `Float64` ansatz is not refused.

- location: `src/core/slab_integrator.jl:184`
- evidence: `CacheType` dispatches on a problem and an ansatz of the same element type, but
  `Cache{ST}` itself takes any pair, so a `Float32` problem with a `Float64` ansatz builds a cache
  whose tabulation is `Float64` and whose coefficients are `Float32`. No clause names the mixed
  case and no test reaches it.
- kind: defect
- found: 2026-10-27

### K22 · The ansatz accepts a degree-0 basis.

- location: `src/ansatz/tensor_bspline.jl:46`
- evidence: `TensorBSplineAnsatz(degree = 0, ncells = (2, 20), xspan = (0, 1))` builds an ansatz
  with two time functions, eighteen space functions and eighteen free coefficients, so the
  piecewise-constant basis runs without a refusal. The slab core takes the gradient `∂ₓu` from the
  same space basis, which a degree-0 basis does not carry, and no check or test names the lowest
  degree the core is written for.
- kind: defect
- found: 2026-10-27

### K23 · Three exported names are called nowhere.

- location: `src/initial_guess/fit_methods.jl:26`, `src/ansatz/tensor_bspline.jl:120`,
  `src/core/slab_integrator.jl:394`
- evidence: `default_fit`, `space_quadrature_weights` and `total_free_coefficients` are exported
  and no test or source file calls them; `grep -rn 'default_fit\|space_quadrature_weights\|
  total_free_coefficients' src test` finds their definitions, their docstrings and the module's
  `export` and `import` lines, and no call.
  `default_fit(ansatz)` is the extension point §Guess names for the later families, and the other
  two are the accessors those families need, so they stay until their callers land.
- kind: dead code
- found: 2026-10-27

### K24 · The ansatz holds its boundary condition in an abstract field, and the cache boxes its step.

- location: `src/ansatz/tensor_bspline.jl:41`, `src/core/slab_integrator.jl:138`
- evidence: `TensorBSplineAnsatz.x_boundary::BoundaryCondition` is an abstract field, so the
  ansatz is not concrete over the condition it was built with and the reads of the field — at
  construction and in `show` — dispatch on its runtime type; its accessor `x_boundary(ansatz)` has
  no caller. The cache's `steps::Ref{Int}` is a box that `slab_bottom!` reads and `update!` writes
  on every step, through a pointer rather than a field.
  A type parameter for the condition and a plain `Int` for the counter would remove both; the
  counter reaches into the `CacheType` signature that the part adding the multiplier blocks
  extends.
- kind: defect
- found: 2026-10-27

### K25 · No test integrates a `SineGordon` problem with the slab core.

- location: `test/ansatz/tensor_bspline.jl`
- evidence: the requirement of the slab core names `Wave` and a `SineGordon` problem, and the
  three new test files integrate `Wave`, `VectorWave` and the coupled vector density only;
  `grep -rn SineGordon test/ansatz test/couplings test/initial_guess` finds nothing, and the §A
  entry `SINE_GORDON_GALERKIN_BSPLINE` is read by no test.
- kind: missing test
- found: 2026-10-27

### K26 · `assign!` and `trace!` repeat their loop, and so do the two `momentum_load!` methods.

- location: `src/core/slab_integrator.jl:286`, `src/core/slab_integrator.jl:315`,
  `src/couplings/coefficient_restriction.jl:69`
- evidence: `assign!` and `trace!` differ only in the evaluation table they multiply the top row
  with, and each repeats the same loop over the components, the space functions and the nodes; the
  two `momentum_load!` methods iterate the same quadrature grid for a scalar density and for a
  coupled one. The first pair is one function taking the table; the second is the two-density
  protocol of `density_derivatives!`, which is already written that way.
- kind: defect
- found: 2026-10-27

### K27 · `test/Project.toml` lists `LinearAlgebra` beyond the part's collision row.

- location: `test/Project.toml:8`
- evidence: the collision row of this part allows the line for `SimpleSplines` alone, and the
  branch adds `LinearAlgebra` as well, which the ansatz test's energy diagnostic needs for `dot`;
  the file holds no `[compat]` entry for `SimpleSplines`, where the package's own project bounds it
  at `0.3.1`.
- kind: defect
- found: 2026-10-27
