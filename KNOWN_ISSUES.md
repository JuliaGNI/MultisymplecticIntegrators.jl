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

### K7 · The three problem constructors repeat themselves.

- location: `src/problems/wave.jl`, `src/problems/sine_gordon.jl`,
  `src/problems/linear_transport.jl`
- evidence: each `problem` function converts its spans, builds the two closures and calls
  `LagrangianPDEProblem` in the same way, about twenty lines three times. One shared constructor
  taking the three exact functions would remove about forty lines. Recorded rather than done,
  because the shared form has to be chosen against the boundary interface that the slab core
  defines.
- kind: defect
- found: 2026-10-26

### K8 · The second component's boundary data has no closed-form test.

- location: `test/problems/lagrangian_problem.jl`
- evidence: the `D = 2` case of the boundary testset asserts the lengths of `bottom` and `left` and
  the first component's values, but nothing compares the second component with a closed form. A
  mutant that zeroes component 2 in `across` or in `at` of `field_boundary_data` survives every
  core test file. The code is right: the full vector matches the exact functions.
- kind: missing test
- found: 2026-10-26

### K9 · A boundary call with one time throws `BoundsError`.

- location: `src/problems/lagrangian_problem.jl`
- evidence: `problem.bcs_function(0.3, params)` destructures `times` and so throws
  `BoundsError: attempt to access Float64 at index [2]`, where an `ArgumentError` naming the
  `(t₀, t₁)` form would say what is wrong. No clause covers a single time.
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
  gives `Union{Float64, Vector{Float64}}`, because the runtime `D` stops the `x isa AbstractVector`
  branch from folding; `Wave.exact_u` itself infers `Float64`. A branch order, not a type
  parameter, closes it.
- kind: quality
- found: 2026-10-26

### K12 · The field data helpers concatenate twice.

- location: `src/problems/lagrangian_problem.jl`
- evidence: `field_initial_data` and `field_boundary_data` apply `reduce(vcat, …)` to a
  comprehension that already built the vector: 10 960 B against 1 568 B for the same three vectors
  as plain comprehensions, for a result of 960 B (fresh process, minimum of 20 warm calls).
- kind: quality
- found: 2026-10-26

### K13 · The scalar path of `field_exact` allocates 64 B.

- location: `src/problems/lagrangian_problem.jl`
- evidence: `problem.exact_u(0.1, 0.3, params)` allocates 64 B for the one-element `Vector` the
  wrapper builds before returning the scalar; `Wave.exact_u(0.1, 0.3, params)` allocates nothing.
- kind: quality
- found: 2026-10-26

### K14 · `∂L/∂u` of the wave density is a literal zero of type `Int`.

- location: `src/problems/lagrangian_density.jl`
- evidence: the wave density has no `u` term, so `build_function` returns the literal `0`:
  `derivatives(Wave.density(), 0.3, 0.7, -0.2).u` is `0 :: Int64` where A1a asks for the element
  type of the arguments, and the kept test asserts the type of `.v` only. A density with a mass
  term returns `Float64`, so the clause holds wherever `∂L/∂u` is not identically zero.
- kind: defect
- found: 2026-10-26
