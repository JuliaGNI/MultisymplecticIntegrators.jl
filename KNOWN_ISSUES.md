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
