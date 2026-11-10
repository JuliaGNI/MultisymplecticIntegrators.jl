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

### K47 · The `WAVE_TRIALNN_OGA2D` sounding under the lifting is an unconverged iterate.

- location: `test/helpers/reference_values.jl:63`, `src/couplings/hard_constraint_lifting.jl`
- evidence: the §A case of the entry — `Wave` at `h = 0.05`, `xspan = (0, 1)`, `xstep = 0.05`,
  `S = 10`, `tanh`, `degree = 3`, `ncells = (2, 5)`, the exact solution as the guess with
  `OrthogonalGreedy(Nw = 50, Nb = 50)`, five steps — run under `HardConstraintLifting` takes
  `iterations = [56, 70, 75, 54, 80]` at `residual = [0.01289, 0.07862, 0.0009895, 0.006161,
  0.005156]`, each from an initial residual of `[2.090, 5.846, 9.451, 12.25, 11.16]`, and
  `converged = false` at every step, in `Float64`. The entry itself reports `status = :failed` after
  four steps, at a `NonlinearSolverException`, so the run's sup-norm deviation from its `u` is taken
  over those four steps and is `0.08319`, where the entry's `u` peaks at `1.2526`. Measured in
  `Float64` in a cold `run-tests.jl` run on `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia 1.13.3,
  four BLAS threads and `--check-bounds=auto`; the test asserts that the deviation is finite and
  writes it with `@info`, and no bound is asserted on it. The record is that run and not a
  reference: the network's rows are the gradient of a discrete action the Newton solve cannot drive
  to round-off, as for K38, and the two runs are iterates of two formulations.
- kind: defect
- found: 2026-11-09

### K48 · The `SINE_GORDON_TRIALNN_OGA2D` sounding under the lifting is an unconverged iterate.

- location: `test/helpers/reference_values.jl:73`, `src/couplings/hard_constraint_lifting.jl`
- evidence: the same method on `SineGordon` over three steps at `h = 0.05` takes
  `iterations = [63, 54, 63]` at `residual = [0.004098, 0.06072, 0.002168]`, each from an initial
  residual of `[0.5118, 1.0346, 1.5374]`, and `converged = false` at every step, in `Float64`. The
  entry reports `status = :ok` after its three steps, and the run's sup-norm deviation from its `u`
  is `0.02603` there, where the entry's `u` peaks at `5.045`. Measured in `Float64` in a cold
  `run-tests.jl` run on `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia 1.13.3, four BLAS threads
  and `--check-bounds=auto`; the deviation is recorded with `@info` and no bound is asserted on it.
  As for K47 the record is that run and not a reference.
- kind: defect
- found: 2026-11-09

### K49 · `coupling_data`'s docstring says the block is built once per integrator.

- location: `src/core/slab_integrator.jl:192`
- evidence: the docstring of `coupling_data` reads "The coupling's block of the slab cache, built
  once per integrator", and the core builds it inside `Cache{ST}` (`src/core/slab_integrator.jl:427`):
  an integrator builds the cache of the problem's element type, and a solver then builds a cache of
  the dual type on its first Jacobian, so the block is built once per cache. A coupling's own
  `coupling_data` docstring that counts on the rate — the lifting's says which cache speaks —
  therefore disagrees with the generic one. Recorded rather than fixed: the part's decision keeps
  the core's diff against `0e2af8d` to the transform's signature, its docstring and its two call
  sites, and this line is none of the three.
- kind: docs
- found: 2026-11-10

