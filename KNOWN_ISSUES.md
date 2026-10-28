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

### K20 · `output_grid` is exported and tested, and nothing in `src` calls it.

- location: `src/ansatz/tensor_bspline.jl`
- evidence: `grep -rn 'output_grid' src/` finds the definition and the export only. The slab
  integrator reports its solution on `xgrid(problem)`, and `SlabEvaluation` tabulates the space
  basis there, so `output_grid` is a public helper that the integrator does not call yet. It holds
  a deferred `InexactError`: `output_grid(ansatz, 1e-30)` passes the positive-finite check and then
  fails in `round(Int, Δx / step)` with `InexactError: Int64(9.999999999999999e29)`, where the
  documented refusal is an `ArgumentError`. Kept rather than deleted or wired into
  `SlabEvaluation`.
- kind: dead code
- found: 2026-10-27

### K21 · The `WAVE_GALERKIN_BSPLINE` record is an unconverged solve of a singular layout.

- location: `test/helpers/reference_values.jl:45`
- evidence: the entry's own Newton record is `iterations = [9, 11, 8, 5, 5]` with
  `residual = [9.92e-6, 5.43e-5, 3.28e-4, 1.95e-3, 1.28e-2]` and `converged = Bool[1, 1, 1, 1, 1]`:
  `isconverged` reports a solve whose residual grows by three orders over the run. Its value at
  `x = 0` reaches `-0.256` where the Dirichlet data of the problem is `0`, its `energy_error` reaches
  `4.6867`, and its `u` differs from the exact solution of `Wave` by `1.60e-4 … 2.56e-1`. The layout
  those values come from constrains the two corner functions of its multiplier basis twice, which
  leaves the first slab's Jacobian rank deficient by two (§Audit 2.8), so the record is an iterate
  and not a solution. It is kept as the record of that run and is not a reference: the two
  couplings that replace the family are compared with closed forms, and a later part that wants a
  full-multiplier comparison takes a converged solve of the corner-dropped layout.
- kind: defect
- found: 2026-10-28

### K22 · The `SINE_GORDON_GALERKIN_BSPLINE` record stops an order of magnitude above its tolerance.

- location: `test/helpers/reference_values.jl:67`
- evidence: the entry's `iterations = [6, 16, 12]` end at
  `residual = [4.88e-8, 2.27e-8, 6.55e-9]`, where `SOLVER_OPTIONS` asks for
  `f_abstol = 1.7763568394002505e-15` and the converged entries of the same file reach `1e-15`. Its
  `energy_error` is `-7.17e-7 … 7.40e-8`. It is the same full-multiplier layout as K21, so it is a
  record of that layout's run and not a reference: sine-Gordon's order at the head is measured
  against its exact solution instead.
- kind: defect
- found: 2026-10-28

### K23 · A `stride` above 1 thins the multiplier basis until the coupling stops converging.

- location: `src/couplings/weak_multipliers.jl:121`
- evidence: `WeakInitialAndBoundaryConditions(; stride)` accepts any `stride ≥ 1`, and only
  `stride = 1` converges, because the multipliers of the bottom surface and of the two ends of
  `xspan` are every `stride`-th function of the trace basis, and a thinned set of them no longer
  reproduces a constant. On `Wave` at its default parameters, at the meshes of the coupling's test
  file — five, ten and twenty space cells, two time cells, `timestep = 0.25 / (2 * cells / 5)`,
  `Float64` — the error against the exact solution at `t = 0.25` is `9.00e-4, 5.76e-5, 3.57e-6` at
  `stride = 1`, `5.31e-2, 1.67e-2, 1.27e-2` at `stride = 2` and `1.37e-1, 1.08e-1, 7.08e-2` at
  `stride = 3`, so the observed orders are `3.97, 4.01` against `1.67, 0.40` and `0.35, 0.61`. The
  coupling's docstring says the same, and the part's only tests of another stride are the
  multiplier counts of `stride = 2`. Whether the keyword is fixed at 1 or removed is the user's
  decision, and it is recorded in the pull request body.
- kind: defect
- found: 2026-10-28
