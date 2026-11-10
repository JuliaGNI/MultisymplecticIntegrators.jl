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

- location: `test/couplings/weak_multipliers.jl:107-126`
- evidence: `multiplier_momentum`, which A7a calls, drives a 50-step `VectorWave` run (β = 0.5)
  through `integrate!` and does not read `cache(int).converged`. The `integrate` run of the same
  problem and method in A7a (`:511-514`) asserts `all(result.stats.converged)`, so a step that does
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

### K35 · The side-data testset's symmetric span cannot see `left` and `right` swapped.

- location: `test/couplings/weak_multipliers.jl:561`
- evidence: the testset "the side data of every component reaches the slab" builds its problem on
  `xspan = (0.2, 0.8)`, and `Wave`'s closed form is `sin(πx/l)` in space, whose two ends carry the
  same value: `Wave.solution_u(0.0, 0.2, Wave.default_parameters())` is `0.7053423027509678` and
  the same call at `0.8` is `0.7053423027509679`. A target filled with `bcs.right.u[d]` at the left
  end and `bcs.left.u[d]` at the right passes all 25 assertions of each precision. The mutant that
  swaps them is caught by A6a of the same file, whose travelling wave differs at its two ends.
- kind: missing test
- found: 2026-11-07

### K36 · No test passes a zero or a negative `xstep` to a problem constructor.

- location: `src/problems/lagrangian_problem.jl:84`
- evidence: `grep -rn 'xstep = 0[,)]\|xstep = zero\|xstep = -' test/` finds nothing, so every test
  problem passes a positive step and the guard `xh > zero(T) || throw(ArgumentError("xstep must be
  positive, got $xh"))` is unasserted: the mutant that deletes the line survived the round-2 run of
  `test/problems/lagrangian_problem.jl` (`E7: a non-positive xstep is accepted`). Without the guard,
  `xstep = 0.0` reaches `round(Int, (xsp[2] - xsp[1]) / xh)` and throws `InexactError`.
- kind: missing test
- found: 2026-11-07

### K37 · A `xstep` at the smallest positive `Float64` throws `InexactError`, not `ArgumentError`.

- location: `src/problems/lagrangian_problem.jl:210`
- evidence: `Wave.problem(xstep = nextfloat(0.0))` passes the positive-step guard and throws
  `InexactError: Int64(Inf)` from `round(Int, (xsp[2] - xsp[1]) / xh)`, where the constructor's
  refusals are `ArgumentError`s. A stepped range of that step fails alike:
  `collect(0.0:nextfloat(0.0):1.0)` throws `InexactError: Int64(Inf)`.
- kind: defect
- found: 2026-11-07

### K38 · The `WAVE_NN_FULLY` run is an unconverged iterate.

- location: `test/helpers/reference_values.jl:55`, `src/ansatz/network.jl`
- evidence: the §A case of the entry — `Wave` at `h = 0.05`, `xspan = (0, 1)`, `S = 10`, `tanh`,
  `Nw = Nb = 50`, the exact solution as the guess with `OrthogonalGreedy()` — run with
  `NetworkAnsatz(; mode = :all)` and `WeakInitialAndBoundaryConditions()` takes
  `iterations = [54, 77, 51, 53, 51]` at `residual = [0.1088, 0.1346, 0.2991, 0.6403, 0.6917]`,
  each from an initial residual of `[15.31, 70.66, 79.15, 50.10, 68.46]`, and `converged = false`
  at every step, in `Float64`. The entry itself reports `converged = true` at
  `residual = [0.602, 4.789, 270.7, 39.61, 1575.0]`. Its sup-norm deviation from the entry is
  `0.5518` over the five steps in `Float64` and `14.74` in `Float32`, on
  `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia 1.13.3, two BLAS threads; the same entry's
  `:all` arm with the initial condition as the guess deviates by `0.6894` and `1.340` there. The
  record is that run and not a reference: the network's rows are the gradient of a discrete action
  the Newton solve cannot drive to round-off, which is the measured result and not a defect of the
  assembly.
- kind: defect
- found: 2026-11-07

### K39 · The `WAVE_NN_PARTIALLY` run does not converge at the solver's default tolerances.

- location: `test/helpers/reference_values.jl:57`, `src/ansatz/network.jl`
- evidence: the same §A case with `mode = :last_layer` takes `iterations = [3, 4, 4, 3, 3]` at
  `residual = [9.41e-4, 1.59e-3, 2.34e-3, 3.12e-3, 2.28e-3]`, each from an initial residual of
  `[2.175, 32.10, 61.90, 92.79, 119.7]`, and `converged = false` at every step, in `Float64`: the
  coupling's `16` constraint rows reach only the `10` output weights, so the step's system is rank
  deficient by at least `6`. The run uses the solver's default tolerances, `f_abstol = 1.78e-15` as
  in the options of the §A entry and `f_reltol = 1.49e-8`, not their `4.44e-16`, and does not meet
  them: its residuals stay near `1e-3`, at `4.3e-4`, `5.0e-5`, `3.8e-5`, `3.4e-5` and `1.9e-5` of
  the initial residual. The run does not use `network_f_reltol`; the last four ratios lie below its
  `1e-4`, which says nothing about convergence under the run's own settings. The run completes and
  records its stats, and its sup-norm deviation from the §A entry is `581.5` in `Float64` and
  `580.4` in `Float32`, on `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia 1.13.3, the entry being
  a diverged iterate whose own `u` reaches `581.76`; the deviation records that entry and not the
  accuracy of this run. The initial condition as the guess instead of the exact solution gives
  `580.5` and `580.5` in that order. The affine control at `S = 20`, where the freed output weights
  carry every row, converges in two iterations to `7.8e-6` of its initial residual in `Float64` and
  `6.4e-3` in `Float32`, which is what the family's `network_f_reltol` records.
- kind: defect
- found: 2026-11-07

### K40 · The `SINE_GORDON_NN_FULLY` run is an unconverged iterate.

- location: `test/helpers/reference_values.jl:69`, `src/ansatz/network.jl`
- evidence: the three-step sine-Gordon case of the entry takes `iterations = [134, 62, 118]` at
  `residual = [1.79e-3, 5.19e-3, 2.26e-3]`, each from an initial residual of `[0.6580, 0.5470,
  0.5158]`, and `converged = false` in `Float64`; the entry reports `converged = true` at
  `residual = [0.2339, 1.549, 139.9]`. Its sup-norm deviation from the entry is `0.0137` in
  `Float64` and `0.0222` in `Float32` over the three steps, on `arm64-apple-darwin27.0.0`
  (`apple-m4`) at Julia 1.13.3. As for K38 the record is that run and not a reference.
- kind: defect
- found: 2026-11-07

### K42 · The `WAVE_NN_SYMBOLIC_FULLY` run is an unconverged iterate of the same case.

- location: `test/helpers/reference_values.jl:59`, `src/ansatz/network.jl`
- evidence: the entry is the symbolic integrator of the old stack on the §A case of K38, and this
  tree's counterpart to it is the `:all` arm of K38, whose deviation from it is `0.2451` in
  `Float64` and `14.74` in `Float32` over the five steps, on `arm64-apple-darwin27.0.0`
  (`apple-m4`) at Julia 1.13.3. The entry's own `u` differs from
  `WAVE_NN_FULLY`'s by up to `0.7615`, so the two entries are separate records and neither is the
  other's reference; `maximum(abs, WAVE_NN_FULLY.u)` is `1.48`. The entry reports `converged = true`
  at `iterations = [5, 2, 27, 7, 1]` and `residual = [0.602, 8.766, 6.202, 65.82, 47.99]`, so as
  for K38 the record is that run and not a reference.
- kind: defect
- found: 2026-11-08

### K43 · The network family's surface code repeats the symbolic ansatz's.

- location: `src/ansatz/network.jl`, `src/couplings/weak_multipliers.jl`
- evidence: about 180 lines of the network family nearly repeat the code of `SymbolicAnsatz`:
  `edge!`, `slab_weak_form!`, `free_residual!`, `keep_top!`, `top_values!`, `initsolver` (an
  identical body), the `check_coupling` refusals, `coupling_term!` and `coupling_guess!`. They differ
  only in the surface rule and in the names of the scratch buffers. `network_target_values!`, through
  its helper `point_target!`, repeats `target_values!` of `src/initial_guess/guess.jl`, whose
  `ansatz` argument is unused, and the
  network's `evaluate!(::InitialCondition)` differs from the tensor one only in `trace_basis`. A
  shared supertype with a surface-rule accessor removes the copy; a reuse of `target_values!` widens
  its signature.
- kind: defect
- found: #44

### K44 · The `WAVE_ELM_LSGD` sounding is an unconverged iterate of another architecture.

- location: `test/helpers/reference_values.jl:61`, `src/initial_guess/trained_hidden_layer.jl`
- evidence: the entry is the old stack's `LSGD` branch on `Wave` at `h = 0.05` over five steps, a
  network of two hidden layers of twenty units whose field's derivative came from `Zygote`, and its
  own record is `iterations = [4, 6, 7, 4, 4]` at `residual = [3.46e-6, 2.25e-4, 2.13e-3, 2.69e-2,
  6.22e-1]`, with `energy_error` rising to `2.74e14`. This tree's counterpart is the `:last_layer`
  network at `S = 20` fitted by `TrainedHiddenLayer(epochs = 100, learning_rate = 1e-5)` from the
  seed 1234, `degree = 3` and `ncells = (10, 10)`, the old constructor's cells: it takes
  `iterations = [3, 3, 3, 3, 5]` at `residual = [9.61e-5, 8.91e-5, 1.55e-4, 2.21e-4, 3.87e-4]`,
  each from an initial residual of `[7.72, 6.92, 8.65, 10.43, 12.42]`, and `converged = false` at
  every step under the default `f_reltol = √eps(Float64)`. Its sup-norm deviation from the entry's
  `u` is `1.79e6`, measured in `Float64` in a cold process on `arm64-apple-darwin27.0.0`
  (`apple-m4`) at Julia 1.13.3, four BLAS threads and `--check-bounds=auto`. The record is that run
  and not a reference: the two are unconverged iterates of two architectures, one with two hidden
  layers and one with one.
- kind: defect
- found: 2026-11-08

### K45 · The `SINE_GORDON_ELM_LSGD` sounding is an unconverged iterate of another architecture.

- location: `test/helpers/reference_values.jl:71`, `src/initial_guess/trained_hidden_layer.jl`
- evidence: the entry is the same `LSGD` branch on `SineGordon` over three steps at `h = 0.05`, and
  its own record is `iterations = [13, 3, 27]` at `residual = [1.56e-7, 2.17e-2, 5.56e-3]`, with
  `energy_error` rising to `4.89e6`. This tree's counterpart, the same `:last_layer` method of K44
  on the same cells, takes `iterations = [4, 6, 4]` at `residual = [1.23e-5, 1.22e-5, 1.66e-5]`,
  each from an initial residual of `[4.31, 4.46, 4.37]`, and `converged = false` at every step. Its
  sup-norm deviation from the entry's `u` is `151.3`, on the machine and with the settings of K44.
  As for K44 the record is that run and not a reference.
- kind: defect
- found: 2026-11-08

### K46 · The rotation charge of a `:last_layer` run drifts, and no cutoff conserves it.

- location: `test/initial_guess/trained_hidden_layer.jl`, `src/ansatz/network.jl`
- evidence: `J_h = Σⱼ (u¹ⱼ λ²ⱼ − u²ⱼ λ¹ⱼ)` over the top trace and the top multiplier of a
  `VectorWave` run at `β = 0`, `S = 20`, `D = 2`, `degree = 3`, `ncells = (2, 5)`,
  `f_abstol = 0` and `f_reltol = trained_f_reltol(T)` drifts from `-0.4362` to `-0.5558` over ten
  steps in `Float64`, a largest deviation of `0.1196`, and from `-0.4101` to `-0.4783` in
  `Float32`, `0.0682`; both runs converge at every step in two iterations, at largest step ratios
  of `3.26e-5` and `4.26e-3`. `λ_top` lies in the near-null space of the `:last_layer` saddle
  point, whose constraint block reaches only the output weights — the `LinearTransport` Jacobian
  at `S = 10` and `D = 2` has rank 40 of 52 — so resolving it makes the run unstable and
  truncating it leaves the slab's equations unsatisfied there: at a cutoff of `1e-14` the
  residual reaches round-off and the charge grows to `7.3e16`, at `1e-10` and `1e-12` to `-1763`,
  and the drift is identical to five digits at `1e-2` uncapped, at a cap of two iterations, and at
  `1e-12` with fifty. The factorisation's cutoff is LAPACK's default, `sqrt(eps(T))`, because
  `initsolver` builds `LapackPivotedQR()` with no argument. Measured in `Float64` and `Float32` on
  `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia 1.13.3, four BLAS threads and
  `--check-bounds=auto`. The weights are the coupling's own: the trace rule's weights pair `λ_top`
  with the trace, and the unweighted sum drifts by `8.97`.
- kind: defect
- found: 2026-11-08

### K49 · No live-branch variant of the fixed-boundary network converges, and its `:last_layer` affine control has no solution under `FitTo(Predictor())`.

- location: `src/couplings/fixed_boundary.jl`, `src/core/levenberg_marquardt.jl`,
  `scripts/experiments.jl`
- evidence: the family's live branch is `FitTo(Predictor())` with Levenberg–Marquardt, in both
  bubble forms and with and without the projected solve, and the matrix's thirty rows for it are
  built by `scripts/experiments.jl` without being run. A two-step run at `max_iterations = 3` on
  `Wave` at `h = 0.05`, `Sx = 8`, `St = 10`, `ncells = 20` and `nq = (6, 4)` fails U12's rule at both
  steps in all four combinations, at final residuals of `9.26e-4` and `1.78e-4` (joint, `:abg`),
  `0.200` and `4.60e-3` (projected, `:abg`), `1.03e-4` and `1.03e-4` (joint, `:tsr`) and `0.234` and
  `4.14e-3` (projected, `:tsr`), from initial residuals of `6.82e-3` (`:abg`) and `6.83e-3`
  (`:tsr`) at the first step, and `0.0766`, `0.335`, `5.74e-3` and `0.335` at the second; while the
  dead branch — `Continuation()` with `Newton()` on a pivoted QR cut at `1e-12` — converges 20 of 20
  steps at a worst ratio of `2.96e-10` and an error of `6.6156e-4` at `t = 1`. The affine control
  under this guess has no solution at all: `FitTo(Predictor())` moves level 1's hidden layer away
  from level 0's by `35.7` on a quadratic entry, and the Jacobian keeps four null directions at
  `D = 1` and two at `D = 2` in the `c₁` columns and `c₀` rows, so that entry leaves `0.302` of
  `r₀` unreachable by any step — an LU solve leaves `1.8e-2`. The family's own affine control is
  therefore `Continuation()`, which holds the spatial hidden layer at `φ₀`'s fit. The
  non-convergence is a measured result of the method and not a defect of its solve: the matrix
  compares it, a row's `converged` is recorded and never asserted, and a fix would be a different
  solve rather than a different tolerance.
- kind: defect
- found: 2026-11-09

### K50 · `integrate!` applies a method's initial guess twice, so a guess that reads state it writes starts the step from its own output.

- location: `src/core/integrate.jl:90`, `src/core/slab_integrator.jl:651`
- evidence: `integrate_steps!` drives each step with the ecosystem's `integrate!(solstep, integrator)`,
  which calls `initial_guess!` (`GeometricIntegratorsBase`'s `src/integrate.jl:10`) and then
  `integrate_step!`, which calls it again (`src/core/slab_integrator.jl:651`). A guess whose fit and
  target read the block's parameter vector measures its own first output on the second application.
  Measured on `Wave` over two steps, step 2's datum `φ₀` differs from step 1's solved `φ₁` by `5.90`
  at `Sx = St = 4` and by `0.427` at `Sx = 8`, `St = 10`, and by `0` through `integrate_step!`
  alone. The double call is the base's; the family keeps its datum, its momentum and its top
  velocity in the copies `coupling_update!` writes (`phi_next`, `p_next`, `v_top`), and its fits read
  the level the fit builds, so that a second application of its guess cannot move the level, the
  momentum or the velocity the next slab reads.
- kind: upstream
- found: 2026-11-09

### K51 · The bubble's output block is rank deficient at `Continuation()`'s first slab, so a projected solve there eliminates an `a` that is not differentiable.

- location: `src/couplings/fixed_boundary.jl`, `src/ansatz/network.jl` (`bubble_start`)
- evidence: the bubble's start grid is antisymmetric under `k ↦ St + 1 − k`: the units at `k` and at
  `St + 1 − k` are exact negatives, measured to `0.0` at `St = 10`, so a first slab solved from
  `Continuation()` differentiates a bubble whose hidden atoms pair off. At `Wave`, `h = 0.05`,
  `Sx = 8`, `St = 10`, `ncells = 20` and `nq = (6, 4)`, the `∂S/∂a` block `J_aa` of that slab has
  rank 4 of 10 at the cut `√eps(T) · σmax` — 5 of 10 lie above `1e-9` — with singular values
  `0.746, 5.34e-3, 6.94e-5, 7.24e-7, 5.93e-9, 3.7e-17 … 3.6e-25` under `:abg`. A projected step
  from that guess therefore eliminates, by its inner solve, an `a*(z)` that a perturbation of a
  hidden entry breaks, and its reduced Jacobian is not the derivative of `z ↦ F_o(z, a*(z))`. The
  testset "FB5 variable projection" takes its measurements from `FitTo(Predictor())`, whose `J_aa`
  is full rank, and no test asserts a projected solve from `Continuation()`. A `bubble_start`
  without the pairing would be a different guess for every run of the family, and the guess is what
  the matrix compares.
- kind: defect
- found: 2026-11-10

### K52 · The plateau probe's extended-precision row cannot run, because `GeometricSolutions` cannot build a solution at `BigFloat`.

- location: `scripts/experiments.jl` (the `BigFloat` plateau row);
  `GeometricSolutions`, `src/timeseries.jl:8`
- evidence: `TimeSeries(ti::AbstractVector, Δt)` builds `new{T, n, typeof(t)}` with
  `n = length(ti) - 1`, and the `length` of a `BigFloat` range is a `BigInt` — it is a
  `StepRangeLen{BigFloat, BigFloat, BigFloat, BigInt}` — so `GeometricSolution(problem)` on a
  `BigFloat` `Wave` raises `TypeError: in Type, in parameter, expected Int64, got a value of type
  BigInt` before a step is taken. The row stays in the registry and in the build-only testset
  "FB18", which builds it and never runs it; the fix is a pull request of its own to
  `GeometricSolutions`, and the step runs as a row of M's matrix once it lands.
- kind: upstream
- found: 2026-11-10
