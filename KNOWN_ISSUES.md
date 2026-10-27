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

### K19 · The coupled density's derivatives allocate at every quadrature point.

- location: `src/problems/wave.jl`
- evidence: `derivatives(::VectorWave.VectorDensity, …)` builds three heap vectors and an array of
  named tuples per point, so the slab core's `residual!` on a `D = 2` problem allocates 184 320 B
  per call — 360 quadrature points at 512 B each — where the scalar path allocates 0. Measured by
  the slab core's pre-pull-request verification pass as the minimum of `@allocated` over 20 warm
  calls in a fresh process under `--check-bounds=auto`; `derivatives(vd, u, v, w, params)` alone
  allocates 1056 B and is unchanged when given `view` arguments. No clause of the part binds this
  path, and the fix belongs in `src/problems/wave.jl`.
- kind: defect
- found: 2026-10-27
