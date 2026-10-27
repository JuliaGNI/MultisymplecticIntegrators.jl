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
  loses the steps already taken. The exception is owned by SimpleSolvers, which is not a dependency
  and which GeometricIntegratorsBase does not export. Deferred until the first integrator family
  returns, because no method of this package solves a nonlinear system before then; the fix then
  adds SimpleSolvers to `[deps]` and the `try`/`catch` of the ecosystem's loop.
- kind: defect
- found: PR #31

### K18 · The §A oracle was absent from the base tree.

- location: `test/helpers/reference_values.jl`
- evidence: part C's test-tree commit removed `test/helpers/reference_values.jl`, although C's spec
  does not list it, and the file is absent at `9d706ba`;
  `git show ed13663:test/helpers/reference_values.jl` returns the 93-line file with its twenty-five
  `const` entries. The part that returns the slab core restored it byte for byte from `ed13663`, and
  no test file includes it: it is the oracle the later parts read, not a testset.
- kind: defect
- found: 2026-10-27

### K19 · The bilinear space-time element is not exact at CFL 1.

- location: `src/ansatz/tensor_bspline.jl`, `test/ansatz/tensor_bspline.jl`
- evidence: the §A entries `FEM_STENCIL_CFL1_T5` and `FEM_STENCIL_CFL1_T200` record an error of
  4.2e-14 and 7.6e-13 against the exact solution at `h = 0.05`, `xstep = 0.025` and `c = 0.5`, and
  the plan's clause 5 asks the degree-1 integrator to be exact there to `1e-11`. The element
  `FEM_Multisymplectic_Integrator` closes over,
  `M (u^{n+1} - 2u^n + u^{n-1}) = -(h^2 c^2/6) K (u^{n-1} + 4u^n + u^{n+1})`
  (`src/integrator/FEM_Multisymplectic_int.jl` at `294f023`,
  `lhs_full = mass_matrix / h^2 - c^2 * laplace_matrix / (6 * dx^2)`), is the degree-1 space-time
  Galerkin of this package: its five steps of `Wave` at `h = 0.05` and `xstep = 0.025` differ from
  the exact solution by 2.8e-5 at the first step and 1.4e-4 at the fifth, a phase error that grows
  linearly. The test `A1d the CFL-1 exactness` holds the clause as a `@test_broken`.
- kind: defect
- found: 2026-10-27

### K20 · A Newton solve of a linear problem takes one iteration from any guess.

- location: `test/initial_guess/guess.jl`
- evidence: the plan's clause 10 asks `Continuation()` to need fewer Newton iterations than
  `FitTo(InitialCondition())` from the second step on, and the spec's scenario A4a repeats it. The
  wave problem is linear in the coefficients, so one Newton step from any guess lands on the
  solution: with `min_iterations = 1` every guess records the same count, and the strict inequality
  cannot hold. The test records the counts of all five options, asserts that they agree, and holds
  the strict comparison as a `@test_broken`.
- kind: defect
- found: 2026-10-27

### K21 · `PreviousSlab()` is not distinguishable by an iteration count.

- location: `src/initial_guess/guess.jl`
- evidence: the plan's mutant MU6 expects a `PreviousSlab()` that copies the top row to converge
  more slowly than `FitTo(InitialCondition())`, so that the iteration count of the third step
  catches it. The wave problem is linear, so no guess changes the count (see K20) and the mutant
  survives that assertion. The extrapolation is checked by its value instead: the test of the
  initial-guess interface compares the coefficients the two targets fit, which differ.
- kind: defect
- found: 2026-10-27

### K22 · The §A CFL-1 entries name a scheme that does not have their property.

- location: `test/helpers/reference_values.jl`
- evidence: `FEM_STENCIL_CFL1_T5` and `FEM_STENCIL_CFL1_T200` hold `stencil_error` and
  `package_error` of the bilinear element, both at `1e-13`, whereas the bilinear element's own error
  at those settings is `1e-4` (see K19). A `@test` against those numbers fails on a correct
  implementation of the element that the same file's `FEM_STENCIL_H005_*` entries agree with. The
  file is the restored oracle and is not edited to fit a new integrator.
- kind: defect
- found: 2026-10-27

### K23 · The problem constructor takes a scalar density, so `β ≠ 0` cannot be carried.

- location: `src/problems/lagrangian_problem.jl`
- evidence: `LagrangianPDEProblem(density::LagrangianDensity; kwargs...)` is the only keyword
  constructor, and `VectorWave.problem` passes its scalar per-component density, so the vector
  density of `VectorWave.vector_density(; β = 0.5)` has no way in:
  `LagrangianPDEProblem(VectorWave.vector_density(; β = 0.5); D = 2, …)` raises
  `MethodError: no method matching LagrangianPDEProblem(::VectorDensity; …)`, with the
  `LagrangianDensity` keyword method as the closest candidate. The slab core evaluates a vector
  density already (`density_derivatives!` has a method for it), but the scenario A5a's `β ≠ 0` half
  cannot be constructed until the problem layer accepts one, which is the problem layer's part.
- kind: defect
- found: 2026-10-27

### K24 · The slab energy deviates by 8e-4 over 200 steps, not 1e-10.

- location: `src/core/slab_integrator.jl`, `test/ansatz/tensor_bspline.jl`
- evidence: the plan's clause 7 asks the relative error of the slab energy to stay below `1e-10`
  over 200 steps of `Wave` at CFL 0.5. The energy taken from the trace at the Gauss nodes with the
  Gauss weights deviates by 8.2e-4 relative over 200 steps at degree 2 with one time cell per slab
  (`A1f the energy error of 200 steps` records the number). The scheme carries the top flux of the
  weak form into the next slab, which preserves a discrete momentum in space, not the quadrature
  energy in time; §A's own `energy_error` column is a relative error against the exact solution at
  `2.7e-6`, so no run of this element reaches `1e-10` against the exact value either. The clause is
  held as a `@test_broken`.
- kind: defect
- found: 2026-10-27
