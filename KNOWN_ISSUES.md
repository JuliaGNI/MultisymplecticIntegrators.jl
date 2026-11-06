# Known issues

Defects found in review and recorded, not fixed. Each entry names its kind and its evidence.

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

### K14 · `∂L/∂u` of the wave density is a literal zero of type `Int`.

- location: `src/problems/lagrangian_density.jl`
- evidence: the wave density has no `u` term, so `build_function` returns the literal `0`:
  `derivatives(Wave.density(), 0.3, 0.7, -0.2).u` is `0 :: Int64` where the other two derivatives
  have the element type of the arguments, and the kept test asserts the type of `.v` only. A
  density with a mass term returns `Float64`, so the clause holds wherever `∂L/∂u` is not
  identically zero.
- kind: defect
- found: 2026-10-26

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

### K22 · The `SINE_GORDON_GALERKIN_BSPLINE` record stops seven orders of magnitude above its tolerance.

- location: `test/helpers/reference_values.jl:67`
- evidence: the entry's `iterations = [6, 16, 12]` end at
  `residual = [4.88e-8, 2.27e-8, 6.55e-9]`, where `SOLVER_OPTIONS` asks for
  `f_abstol = 1.7763568394002505e-15` and the converged entries of the same file reach `1e-15`. Its
  `energy_error` runs `-7.17e-7` to `2.27e-7`, and its last step's is `7.40e-8`. It is the same
  full-multiplier layout as K21, so it is a record of that layout's run and not a reference:
  sine-Gordon's order at the head is measured against its exact solution instead.
- kind: defect
- found: 2026-10-28; one issue with K21

### K31 · A7a's `integrate!` run does not assert convergence.

- location: `test/couplings/weak_multipliers.jl:94-113`
- evidence: `multiplier_momentum`, which A7a calls, drives a 50-step `VectorWave` run (β = 0.5)
  through `integrate!` and does not read `cache(int).converged`. The `integrate` run of the same
  problem and method in A7a (`:503-506`) asserts `all(result.stats.converged)`, so a step that does
  not converge still fails A7a, but the run whose momenta A7a checks has no assertion of its own.
- kind: missing test
- found: 2026-11-01

### K34 · CI never evaluates the allocation assertions that a bounds check would change.

- location: `test/ansatz/tensor_bspline.jl`, `test/diagnostics.jl`
- evidence: `.github/workflows/CI.yml` runs `julia-actions/julia-runtest@v1` without a
  `check_bounds` input, and the action's `action.yml` gives `check_bounds` the default `yes`. Every
  `@allocated` assertion of the testset "R2 single density_derivatives! and zero-allocation
  VectorWave residual!" sits under `if Base.JLOptions().check_bounds != 1`: `residual!` of the
  `D = 2` vector wave with the three couplings, `density_derivatives!`, `momentum_load!` and
  `flux_load!` at `D = 3`, and `residual!` of `Wave.problem(D = D)` at `D = 4` and `5`, all at
  0 B. The `D = 1` assertion of A1g in the same file and the `derivatives` assertion of S3b in
  `test/couplings/coefficient_restriction.jl` sit under the same guard, and the 512 B bound on
  `energy_error` of a coupled density in `test/diagnostics.jl` is skipped under
  `Base.JLOptions().check_bounds == 1`. A CI run passes these
  testsets without evaluating the bounds, so only a local run under `--check-bounds=auto` checks
  them.
- kind: missing test
- found: #40
