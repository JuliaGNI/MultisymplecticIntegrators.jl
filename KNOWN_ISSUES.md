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

### K38 · `NetworkAnsatz` in `:all` mode is declined by the method: no solver variant converges it, and its three records are unconverged iterates.

- location: `test/helpers/reference_values.jl:55`, `:59` and `:69`, `src/ansatz/network.jl`
- evidence: the §A case of the `WAVE_NN_FULLY` entry — `Wave` at `h = 0.05`, `xspan = (0, 1)`,
  `S = 10`, `tanh`, `Nw = Nb = 50`, the exact solution as the guess with `OrthogonalGreedy()` — run
  with `NetworkAnsatz(; mode = :all)` and `WeakInitialAndBoundaryConditions()` takes
  `iterations = [54, 77, 51, 53, 51]` at `residual = [0.1088, 0.1346, 0.2991, 0.6403, 0.6917]`,
  each from an initial residual of `[15.31, 70.66, 79.15, 50.10, 68.46]`, and `converged = false`
  at every step, in `Float64`. The entry itself reports `converged = true` at
  `residual = [0.602, 4.789, 270.7, 39.61, 1575.0]`. Its sup-norm deviation from the entry is
  `0.5518` over the five steps in `Float64` and `14.74` in `Float32`, on
  `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia 1.13.3, two BLAS threads; the same entry's
  `:all` arm with the initial condition as the guess deviates by `0.6894` and `1.340` there. The
  record is that run and not a reference: the network's rows are the gradient of a discrete action
  the Newton solve cannot drive to round-off, which is the measured result and not a defect of the
  assembly. The three-step sine-Gordon case of the `SINE_GORDON_NN_FULLY` entry takes
  `iterations = [134, 62, 118]` at `residual = [1.79e-3, 5.19e-3, 2.26e-3]`, each from an initial
  residual of `[0.6580, 0.5470, 0.5158]`, and `converged = false` in `Float64`; the entry reports
  `converged = true` at `residual = [0.2339, 1.549, 139.9]`. Its sup-norm deviation from the entry
  is `0.0137` in `Float64` and `0.0222` in `Float32` over the three steps, on the same machine and
  Julia, and that record too is the run and not a reference. The `WAVE_NN_SYMBOLIC_FULLY` entry is
  the symbolic integrator of the old stack on the same §A case, and this tree's counterpart to it is
  the `:all` arm above, whose deviation from it is `0.2451` in `Float64` and `14.74` in `Float32`
  over the five steps, on the same machine and Julia. The entry's own `u` differs from
  `WAVE_NN_FULLY`'s by up to `0.7615`, so the two entries are separate records and neither is the
  other's reference; `maximum(abs, WAVE_NN_FULLY.u)` is `1.48`. The entry reports
  `converged = true` at `iterations = [5, 2, 27, 7, 1]` and
  `residual = [0.602, 8.766, 6.202, 65.82, 47.99]`, so that record too is the run and not a
  reference. The family is declined by the method and kept in the package for comparison. The
  measurement behind that is a run of this package at `4adae16` with a position-dependent density
  parameter added, at Julia 1.13 and SimpleSolvers 0.14, in `Float64`, over three slab steps, at
  `S = 10`, `tanh` and a degree-3 trace basis on `(2, n)` cells. The package's solver runs at its
  defaults, `f_abstol = 8 eps = 1.78e-15` and `f_reltol = 1.49e-8`, and the Levenberg–Marquardt
  and variable-projection variants stop on the relative test `r ≤ 1.49e-8 r₀` alone. The layer
  case is the linear-layer Klein–Gordon case of a FEEC–FDTD benchmark at 5 GHz (`h = 0.1499`, 71
  space intervals), whose density depends on the position; the package does not hold that problem.
  On the layer case with 71 trace cells the step has no root, with 82 data rows against 40
  parameters, and stops at `r/r₀ = 4.5e-6, 7.2e-6, 5.1e-5`. On `Wave` with 5 cells it is a
  nonlinear crawl, at `r/r₀ = 2.2e-4, 0.33, 2.9e-3`: the residual is in the range of `J`, and the
  true residual grows along the Newton step. Levenberg–Marquardt from the stall reaches `1.6e-5`
  on `Wave` after 3000 trials, and variable projection reaches `1.2e-5, 1.2e-6, 5.6e-6` on `Wave`
  and errors of 1.5 to 13 on the layer case: no variant converges on either problem.
- kind: defect
- found: 2026-11-07; declined 2026-11-22

### K39 · `NetworkAnsatz` in `:last_layer` mode is declined by the method: its slab has no root, its `WAVE_NN_PARTIALLY` run does not converge, and no cutoff conserves its rotation charge.

- location: `test/helpers/reference_values.jl:57`, `test/initial_guess/trained_hidden_layer.jl`,
  `src/ansatz/network.jl`
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
  `6.4e-3` in `Float32`, which is what the family's `network_f_reltol` records. The rotation charge
  of a `:last_layer` run drifts, and no cutoff conserves it: `J_h = Σⱼ (u¹ⱼ λ²ⱼ − u²ⱼ λ¹ⱼ)` over
  the top trace and the top multiplier of a `VectorWave` run at `β = 0`, `S = 20`, `D = 2`,
  `degree = 3`, `ncells = (2, 5)`, `f_abstol = 0` and `f_reltol = trained_f_reltol(T)` drifts from
  `-0.4362` to `-0.5558` over ten steps in `Float64`, a largest deviation of `0.1196`, and from
  `-0.4101` to `-0.4783` in `Float32`, `0.0682`; both runs converge at every step in two
  iterations, at largest step ratios of `3.26e-5` and `4.26e-3`. `λ_top` lies in the near-null
  space of the `:last_layer` saddle point, whose constraint block reaches only the output weights —
  the `LinearTransport` Jacobian at `S = 10` and `D = 2` has rank 40 of 52 — so resolving it makes
  the run unstable and truncating it leaves the slab's equations unsatisfied there: at a cutoff of
  `1e-14` the residual reaches round-off and the charge grows to `7.3e16`, at `1e-10` and `1e-12`
  to `-1763`, and the drift is identical to five digits at `1e-2` uncapped, at a cap of two
  iterations, and at `1e-12` with fifty. The factorisation's cutoff is LAPACK's default,
  `sqrt(eps(T))`, because `initsolver` builds `LapackPivotedQR()` with no argument. Measured in
  `Float64` and `Float32` on `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia 1.13.3, four BLAS
  threads and `--check-bounds=auto`. The weights are the coupling's own: the trace rule's weights
  pair `λ_top` with the trace, and the unweighted sum drifts by `8.97`. The family is declined by
  the method and kept in the package for comparison. The measurement behind that is a run of this
  package at `4adae16` with a position-dependent density parameter added, at Julia 1.13 and
  SimpleSolvers 0.14, in `Float64`, over three slab steps unless a count is named, with `tanh` and
  a degree-3 trace basis on `(2, n)` cells. The package's solver runs at its defaults,
  `f_abstol = 8 eps = 1.78e-15` and `f_reltol = 1.49e-8`, and the Levenberg–Marquardt variant stops
  on the relative test `r ≤ 1.49e-8 r₀` alone. The layer case is the linear-layer Klein–Gordon
  case of a FEEC–FDTD benchmark at 5 GHz (`h = 0.1499`, 71 space intervals), whose density depends
  on the position; the package does not hold that problem. The slab has no root: 97–100 % of the
  residual left at the stall is outside the range of the slab Jacobian. On `Wave` at `S = 10` with
  5 cells the steps stop at `r/r₀ = 1.5e-3, 6.2e-5, 7.0e-5`, with a constraint block of 16 × 10;
  on the layer case at `S = 40` with 20 cells the block is 31 × 40 of rank 28. The affine control
  at `S = 20` has a root behind the `√eps` rank cut: Levenberg–Marquardt over the full SVD
  converges at 20 of 20 steps on `Wave`, with an error of `5.7e-3 → 0.17` over the 20 steps,
  against `1.5e-3 → 3.5e-2` for the package's unconverged run. With the data rows reduced to `k`
  directions of the range of the constraint block, the dropped data grow: on the layer case at
  `S = 40` and `k = 29`, the block's rank, the error grows from `2.6` to `6.5e24` over 10 steps,
  and on `Wave` at `k = 6` of rank 10 the dropped data grow from `3.2e-3` to `0.22` in 5 steps.
- kind: defect
- found: 2026-11-07; declined 2026-11-22

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

### K44 · The ELM family, a `:last_layer` `NetworkAnsatz` fitted by `TrainedHiddenLayer`, is declined by the method: its slab has no root, and its two soundings are unconverged iterates of another architecture.

- location: `test/helpers/reference_values.jl:61` and `:71`,
  `src/initial_guess/trained_hidden_layer.jl`
- evidence: the `WAVE_ELM_LSGD` entry is the old stack's `LSGD` branch on `Wave` at `h = 0.05`
  over five steps, a network of two hidden layers of twenty units whose field's derivative came
  from `Zygote`, and its own record is `iterations = [4, 6, 7, 4, 4]` at
  `residual = [3.46e-6, 2.25e-4, 2.13e-3, 2.69e-2, 6.22e-1]`, with `energy_error` rising to
  `2.74e14`. This tree's counterpart is the `:last_layer`
  network at `S = 20` fitted by `TrainedHiddenLayer(epochs = 100, learning_rate = 1e-5)` from the
  seed 1234, `degree = 3` and `ncells = (10, 10)`, the old constructor's cells: it takes
  `iterations = [3, 3, 3, 3, 5]` at `residual = [9.61e-5, 8.91e-5, 1.55e-4, 2.21e-4, 3.87e-4]`,
  each from an initial residual of `[7.72, 6.92, 8.65, 10.43, 12.42]`, and `converged = false` at
  every step under the default `f_reltol = √eps(Float64)`. Its sup-norm deviation from the entry's
  `u` is `1.79e6`, measured in `Float64` in a cold process on `arm64-apple-darwin27.0.0`
  (`apple-m4`) at Julia 1.13.3, four BLAS threads and `--check-bounds=auto`. The record is that run
  and not a reference: the two are unconverged iterates of two architectures, one with two hidden
  layers and one with one. The `SINE_GORDON_ELM_LSGD` entry is the same `LSGD` branch on
  `SineGordon` over three steps at `h = 0.05`, and its own record is `iterations = [13, 3, 27]` at
  `residual = [1.56e-7, 2.17e-2, 5.56e-3]`, with `energy_error` rising to `4.89e6`. This tree's
  counterpart, the same `:last_layer` method on the same cells, takes `iterations = [4, 6, 4]` at
  `residual = [1.23e-5, 1.22e-5, 1.66e-5]`, each from an initial residual of `[4.31, 4.46, 4.37]`,
  and `converged = false` at every step. Its sup-norm deviation from the entry's `u` is `151.3`, on
  the same machine and with the same settings, and that record too is the run and not a reference.
  The family is declined by the method and kept in the package for comparison. The measurement
  behind that is a run of this package at `4adae16` with a position-dependent density parameter
  added, at Julia 1.13 and SimpleSolvers 0.14, in `Float64`, over three slab steps, at `S = 20`,
  `tanh` and a degree-3 trace basis on `(2, n)` cells, with the package's solver at its defaults,
  `f_abstol = 8 eps = 1.78e-15` and `f_reltol = 1.49e-8`. The layer case is the linear-layer
  Klein–Gordon case of a FEEC–FDTD benchmark at 5 GHz (`h = 0.1499`, 71 space intervals), whose
  density depends on the position; the package does not hold that problem. The slab has no root.
  On `Wave` with 5 cells the steps stop at `r/r₀ = 1.9e-5, 1.0e-5, 1.3e-5`, with a constraint block
  of 16 × 20 of rank 15; on the layer case with 71 cells the block is 82 × 20 of rank 18, 17 and 4,
  and `‖r‖` goes `0.033 → 0.15 → 30`.
- kind: defect
- found: 2026-11-08; declined 2026-11-22

### K47 · `HardConstraintLifting` on a `NetworkAnsatz`, the TrialNN family, is declined by the method: it converges on `Wave` only under variable projection, with the error of the unconverged run, and its two soundings are unconverged iterates.

- location: `test/helpers/reference_values.jl:63` and `:73`,
  `src/couplings/hard_constraint_lifting.jl`
- evidence: the §A case of the `WAVE_TRIALNN_OGA2D` entry — `Wave` at `h = 0.05`,
  `xspan = (0, 1)`, `xstep = 0.05`, `S = 10`, `tanh`, `degree = 3`, `ncells = (2, 5)`, the exact
  solution as the guess with `OrthogonalGreedy(Nw = 50, Nb = 50)`, five steps — run under
  `HardConstraintLifting` takes
  `iterations = [56, 70, 75, 54, 80]` at `residual = [0.01289, 0.07862, 0.0009895, 0.006161,
  0.005156]`, each from an initial residual of `[2.090, 5.846, 9.451, 12.25, 11.16]`, and
  `converged = false` at every step, in `Float64`. The entry itself reports `status = :failed` after
  four steps, at a `NonlinearSolverException`, so the run's sup-norm deviation from its `u` is taken
  over those four steps and is `0.08319`, where the entry's `u` peaks at `1.2526`. Measured in
  `Float64` in a cold `run-tests.jl` run on `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia 1.13.3,
  four BLAS threads and `--check-bounds=auto`; the test asserts that the deviation is finite and
  writes it with `@info`, and no bound is asserted on it. The record is that run and not a
  reference: the network's rows are the gradient of a discrete action the Newton solve cannot drive
  to round-off, as for K38, and the two runs are iterates of two formulations. The same method on
  `SineGordon`, the case of the `SINE_GORDON_TRIALNN_OGA2D` entry, over three steps at `h = 0.05`
  takes `iterations = [63, 54, 63]` at `residual = [0.004098, 0.06072, 0.002168]`, each from an
  initial residual of `[0.5118, 1.0346, 1.5374]`, and `converged = false` at every step, in
  `Float64`. The entry reports `status = :ok` after its three steps, and the run's sup-norm
  deviation from its `u` is `0.02603` there, where the entry's `u` peaks at `5.045`, measured on the
  same machine and with the same settings; the deviation is recorded with `@info` and no bound is
  asserted on it, and that record too is the run and not a reference. The family is declined by the
  method and kept in the package for comparison; the lifting on a `TensorBSplineAnsatz` is not part
  of that decision. The measurement behind it is a run of this package at `4adae16` with a
  position-dependent density parameter added, at Julia 1.13 and SimpleSolvers 0.14, in `Float64`,
  over three slab steps, at `S = 10`, `tanh` and a degree-3 trace basis on `(2, n)` cells. The
  package's solver runs at its defaults, `f_abstol = 8 eps = 1.78e-15` and `f_reltol = 1.49e-8`,
  and the Levenberg–Marquardt and variable-projection variants stop on the relative test
  `r ≤ 1.49e-8 r₀` alone. The layer case is the linear-layer Klein–Gordon case of a FEEC–FDTD
  benchmark at 5 GHz (`h = 0.1499`, 71 space intervals), whose density depends on the position; the
  package does not hold that problem. Variable projection converges on `Wave` at every step, in 17
  to 79 trials, with the error of the unconverged run: `8.8e-2` after step 3, against `9.1e-2` for
  plain Levenberg–Marquardt and `0.11` for the package's own solve. On the layer case no variant
  converges: variable projection ends 113 times above the tolerance at step 1, and 6 and 2 times
  above it at steps 2 and 3, at an error of `0.16`.
- kind: defect
- found: 2026-11-09; declined 2026-11-22

### K50 · No live-branch variant of the fixed-boundary network converges, and its `:last_layer` affine control has no solution under `FitTo(Predictor())`.

- location: `src/couplings/fixed_boundary.jl`, `src/core/levenberg_marquardt.jl`,
  `scripts/experiments.jl`
- evidence: the family's live branch is `FitTo(Predictor())` with Levenberg–Marquardt, in both
  bubble forms and with and without the projected solve, and the matrix's thirty rows for it are
  built by `scripts/experiments.jl` without being run. A two-step run at `max_iterations = 3` on
  `Wave` at `h = 0.05`, `Sx = 8`, `St = 10`, `ncells = 20` and `nq = (6, 4)` fails the convergence
  rule at both steps in all four combinations, at final residuals of `9.26e-4` and `1.78e-4` (joint,
  `:abg`), `0.200` and `4.60e-3` (projected, `:abg`), `1.03e-4` and `1.03e-4` (joint, `:tsr`) and
  `0.234` and `4.14e-3` (projected, `:tsr`), from initial residuals of `6.82e-3` (`:abg`) and
  `6.83e-3` (`:tsr`) at the first step, and `0.0766`, `0.335`, `5.74e-3` and `0.335` at the second;
  while the dead branch — `Continuation()` with `Newton()` on a pivoted QR cut at `1e-12` —
  converges 20 of 20 steps at a worst ratio of `2.96e-10` and an error of `6.6156e-4` at `t = 1`.
  The affine control under this guess has no solution at all: `FitTo(Predictor())` moves level 1's
  hidden layer away from level 0's by `35.7` on a quadratic entry, and the Jacobian keeps four null
  directions at `D = 1` and two at `D = 2` in the `c₁` columns and `c₀` rows, so that entry leaves
  `0.302` of `r₀` unreachable by any step — an LU solve leaves `1.8e-2`. The family's own affine
  control is therefore `Continuation()`, which holds the spatial hidden layer at `φ₀`'s fit. The
  non-convergence is a measured result of the method and not a defect of its solve: the matrix
  compares it, a row's `converged` is recorded and never asserted, and a fix would be a different
  solve rather than a different tolerance.
- kind: defect
- found: 2026-11-09

### K51 · `integrate!` applies a method's initial guess twice, so a guess that reads state it writes starts the step from its own output.

- location: `src/core/integrate.jl:92`, `src/core/slab_integrator.jl:681`
- evidence: `integrate_steps!` drives each step with the ecosystem's `integrate!(solstep, integrator)`,
  which calls `initial_guess!` (`GeometricIntegratorsBase`'s `src/integrate.jl:10`) and then
  `integrate_step!`, which calls it again (`src/core/slab_integrator.jl:681`). A guess whose fit and
  target read the block's parameter vector measures its own first output on the second application.
  Measured on `Wave` over two steps, step 2's datum `φ₀` differs from step 1's solved `φ₁` by `5.90`
  at `Sx = St = 4` and by `0.427` at `Sx = 8`, `St = 10`, and by `0` through `integrate_step!`
  alone. The double call is the base's; the family keeps its datum, its momentum and its top
  velocity in the copies `coupling_update!` writes (`phi_next`, `p_next`, `v_top`), and its fits read
  the level the fit builds, so that a second application of its guess cannot move the level, the
  momentum or the velocity the next slab reads.
- kind: upstream
- found: 2026-11-09

### K52 · The bubble's output block is rank deficient at `Continuation()`'s first slab, so a projected solve there eliminates an `a` that is not differentiable.

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

### K53 · The plateau probe's extended-precision entry cannot run, because the fit of the first datum takes an SVD that `LinearAlgebra` does not provide at `BigFloat`.

- location: `scripts/experiments.jl` (the `BigFloat` plateau entry);
  `src/core/levenberg_marquardt.jl:182` (`least_squares!`), reached from
  `src/couplings/fixed_boundary.jl:632` (`first_datum!`)
- evidence: `integrate` of a `BigFloat` `Wave` with
  `FixedBoundaryNetwork(Sx = 4, St = 4, ncells = 3, nq = (6, 4))`, `solver =
  LevenbergMarquardt()` and `f_reltol = big"1e-30"` raises `MethodError: no method matching
  svd!(::Matrix{BigFloat}; full::Bool, alg::LinearAlgebra.DivideAndConquer, atol::Int64,
  rtol::Int64)` in `least_squares!`, which `first_datum!` calls from `slab_bottom!` before the
  first slab is solved (Julia 1.13.4, GeometricSolutions 0.7.2, the package's own environment).
  `GeometricSolution(problem)` builds at `BigFloat`. The entry names its own `f_reltol =
  big"1e-8"`, because the family measures no tolerance at `BigFloat` and refuses a caller there
  that names none, and `fixed_boundary_solver_keywords` hands the entry's value to the solver, so
  an integrator that names `LevenbergMarquardt()` builds, as "FB18" builds it; under the default
  `Newton()` the build stops earlier, at SimpleSolvers' refusal of `LapackPivotedQR` at `BigFloat`
  (FB8). The entry stays in the registry and in the build-only testset "FB18", which builds it and
  never runs it. No environment of the package loads a generic `svd!` for `BigFloat`; whether
  `GenericLinearAlgebra` supplies one with these keywords is not measured.
- kind: defect
- found: 2026-11-10

### K54 · A reference run that itself failed part-way makes the driver throw after every row has archived.

- location: `scripts/benchmark.jl:265` (`fine_time_rows`), reached from `main` through
  `attach_references!`
- evidence: a problem without an exact solution whose finest run fails at a step raises
  `ArgumentError: the time 0.2 of the coarse run is not a time of the finer run` from
  `fine_time_rows`, so the driver ends with an exception after every selected row has written its
  archive. No registered row is measured against a reference run, so the input lies outside every
  clause's domain and nothing in the suite reaches it; comparing at the times both runs stored, and
  recording `NaN` past them, would make it a result.
- kind: defect
- found: 2026-11-11

### K56 · The box scheme copies the initial level into its cache twice on the first step.

- location: `src/baselines/box_scheme.jl:457` (`initial_guess!`) and `:508` (`read_initial_level!`)
- evidence: `initial_guess!` copies `sol.u`, `sol.v` and `sol.w` into the cache when `c.steps == 0`,
  and `integrate_step!` then calls `read_initial_level!` (`:492`), which copies the same three
  arrays again. Removing the second copy makes a bare `integrate_step!` start from a zero cache, so
  the choice of the one copy that stays is a design decision.
- kind: dead code
- found: #59

### K58 · The TrialNN family has no measured solver tolerance, and the ELM and TrialNN runs solve at the generic defaults.

- location: `scripts/trialnn_matrix_model.jl:29` (`TRIALNN_MATRIX`), `scripts/elm_model.jl:72` and
  `scripts/trialnn_model.jl:80`
- evidence: a network family solves at `f_abstol = 0` and a relative `f_reltol` that it measures
  on its affine control. The network family measured
  `network_f_reltol(Float64) = 1.0e-4` (`test/helpers/testsetup.jl:59`) and the ELM family
  `trained_f_reltol(Float64) = 1.0e-2` (`test/helpers/testsetup.jl:44`); no test of
  `HardConstraintLifting` names an `f_reltol`, so the TrialNN rows take the network family's value.
  The run families call `integrate(problem, method)` with no solver keywords, so their steps solve
  at `f_abstol = 8 eps = 1.78e-15` and `f_reltol = √eps = 1.49e-8`, and their archives' convergence
  is judged on another rule than the matrix rows of the same method.
- kind: defect
- found: 2026-11-12

### K59 · A network's generated kernels follow the GC history, so a trained run reproduces only within one process.

- location: `src/ansatz/network.jl:150` (`network_kernels`) and `:117` (`network_functions`);
  `SymbolicUtils` 4.56.1, `src/code.jl:1949` (`_cse_compute`) and `src/symbolic_ops/addsub.jl:129`
  (`_dict_copy_into!` in the call method of `AddWorkerBuffer`, `:96`, which `add_worker` reaches)
- evidence: `network_kernels` builds the `field` and `gradient` kernels of an architecture once
  per process, through `network_functions` and `SymbolicNeuralNetworks`' `build_flat_function`,
  and every `NetworkAnsatz` of that architecture reads them. The CSE pass of `SymbolicUtils` emits a
  sum's terms in the order of `arguments(expr)`, not `sorted_arguments`, and that order is the term
  dictionary of whichever hash-consed copy of the term the weak cache holds, so it follows the
  process's allocation and `GC.gc()` history. 4.56 seeds a sum's dictionary with a bulk copy of an
  existing one (`_dict_copy_into!`), which keeps that dictionary's slot order, where 4.55 inserts
  the terms one by one; that this change is the cause is likely and not shown.
  `julia --startup-file=no --project=. scripts/kernel_build_order.jl 20 5 30 3` at `6036a47`
  (Julia 1.13.3, `SymbolicUtils` 4.56.1, no `--check-bounds` flag) builds the kernels of
  `S = 20`, `D = 1` 20 times around the cache, with a `GC.gc()` after every fifth build: the builds
  fall into two classes, 1–10 and 11–20 in `Float64` and 1–5 and 6–20 in `Float32`, whose kernel
  values differ by up to `2.5e-15` and `2.2e-7` relative. One differing line is
  `ps.L1.W[21] * x1[2, k]` against `ps.L1.W[1] * x1[1, k]` in the first term of a sum. The same
  builds at `1e0301e`, which has no cache, are one class with 4.55.0. The 20 epochs of a
  `TrainedHiddenLayer` fit carry the rounding difference to the step: at `1e0301e`, where every
  `NetworkAnsatz` builds its own kernels, test E4a's pair of first steps, looped 30 times per
  element type with 3 builds and one `GC.gc()` between its two steps, differs in 2 of 30 `Float64`
  and 2 of 30 `Float32` trials, by up to `3.1e-12` and `1.3e-3` relative, with 4.56.1, and in none
  with 4.55.0. At `6036a47` the script's part 2 counts none of 30 per element type, because both
  steps read the one cached pair. Within one process the kernels are therefore one build; two
  processes whose history differs before that build can generate different kernels, so a trained
  run does not reproduce bit for bit from one process to the next. Sorted arguments in
  `SymbolicUtils`' CSE would remove the dependence at its source.
- kind: upstream
- found: 2026-11-13

### K61 · JET reports runtime dispatches in the box scheme's `residual!` and in the construction of its cache.

- location: `src/baselines/box_scheme.jl:353`
- evidence: `JET.report_opt` with `target_modules = (MultisymplecticIntegrators,)` reports on
  `residual!` and on `cache(int, T)` of a `BoxScheme()` integrator at all eight entries of
  `test/helpers/quality_cases.jl` (`Float64` and `Float32`, `D = 1` and `D = 2`, unforced on
  `Wave.problem` and `VectorWave.problem` and forced on the manufactured problem): six reports on
  `residual!` and five on `cache` at each entry, on Julia 1.13.4 with JET 0.13.1. Five are in
  `check_box_density` (`:134`), which runs once per cache, when `cache(int, T)` builds it, and
  which `residual!` reaches through the cache: the call of its closure `partials` (`:140`), whose
  `point_derivatives(…, Val(D))` takes the runtime `D = components(problem)`, and the `all` of its
  checks at `:152`, `:169`, `:173` and `:180`, which read the partials that call returns. The
  sixth, on `residual!` alone, is the call through the barrier
  `box_residual!(b, x, c, problem, Val(components(problem)), tc)` (`:353`), dispatched at run time
  one call below `residual!`, because `components(problem)` is a runtime integer. JET stops at this
  dispatch, so while it stays the forced entries of the box scheme repeat the reports of the
  unforced ones and do not see the forced row of `box_residual!`: with a forcing that returns its
  value through `Base.inferencebarrier`, the count on `residual!` stays at six at the four forced
  box-scheme entries, where it goes from none to two at the forced slab entries. The sixteen
  checks of the testset "JET: box scheme" in `test/quality/jet.jl` are `@test_broken` for these
  findings, so a fix fails them as unexpected passes. Issue #70 tracks it.
- kind: defect
- found: 2026-11-15

### K62 · The B-spline slab integrator on the wave equation is stable only up to about `c·h/Δx = 1`.

- location: `src/core/weak_form.jl`
- evidence: above the bound the high space modes grow by a factor `ρ` per step from round-off. The
  spectral radius of the one-step map of `Wave` (`c = 0.5`) under `CoefficientRestriction` on
  `2f916a2`, Julia 1.13.4, is `1` to round-off at every ratio measured up to `c·h/Δx = 1`, and
  `ρ ≈ 3` at `c·h/Δx = 2` and `3.7` at 16 for degree 1, `1.13` at 2 and `5.7` at 16 for degree 2;
  bisected at `nx = 40`, the bound is `1.0022` at degree 1 and `0.99988` at degree 2. At degree 1
  the step is the Newmark scheme with `β = 1/6`, `γ = 1/2`, whose spectral radius equals the
  measured one to about `1e-14`. Every step converges, so a short run looks right:
  `test/verification/long_time_energy.jl` runs the three couplings at `c·h/Δx = 0.9` for 80 steps
  under a largest `L²` error of `1e-2`, and records the largest error of 40 steps at `c·h/Δx = 4`
  without a bound. The first rung of the self-difference ladder of
  `test/verification/manufactured.jl`, `h = 0.1` at `nx = 20`, runs at `c·h/Δx = 1`, at the bound.
  A stable time rule is not part of the package; it is planned as separate work.
- kind: defect
- found: 2026-11-14

### K63 · Under the two weak couplings a non-finite residual ends `integrate` with a `SingularException`.

- location: `src/core/integrate.jl`
- evidence: `integrate` records a failed solve only for a `NonlinearSolverException` and rethrows
  every other exception. On the manufactured problem of `test/helpers/manufactured.jl`
  (`h = 0.1`, `nx = 10`, three steps, degree 2), a forcing that returns `NaN` or `Inf` makes
  `WeakInitialCondition` and `WeakInitialAndBoundaryConditions` throw
  `SingularException: matrix is singular; factorization failed. Zero pivot found at index 1` and
  `… index 3`, from SimpleSolvers' LU, where `CoefficientRestriction` and the box scheme return
  with 0 traces and `converged = Bool[]`. The unforced `Wave.problem(timespan = (0.0, 0.2),
  timestep = 0.1, xstep = 0.1)` with a `NaN` initial velocity gives the same four outcomes, so the
  forcing does not cause it. `test/verification/manufactured.jl` asserts the recorded run under
  `CoefficientRestriction` and the box scheme only.
- kind: defect
- found: 2026-11-15

### K64 · The hidden-row check of the per-component lifting holds to `16 * eps(T)`, a bound set from one measurement.

- location: `test/couplings/hard_constraint_lifting.jl:438`
- evidence: the testset "L4 the lifting acts per component on VectorWave, $T" asserts
  `maximum(abs, b₂[1:(3S)] .- 2 * b₁[1:(3S)]) ≤ 16 * eps(T) * maximum(abs, b₁)` for `Float64`
  and `Float32`. The largest measured difference is about 11.4 eps of `maximum(abs, b₁)`, in
  `Float64` on Julia 1.13.4, `windows-latest`
  (run 43575557665, job 132388889007), so the margin is about 1.4. The bound is the measured
  worst case rounded up; it is not derived from the error of the sum over the two components.
- kind: missing test
- found: #73

### K66 · No forced run has a space span other than `[0, 1]` or a time span that starts away from `t = 0`.

- location: `test/helpers/manufactured.jl:65`
- evidence: every forced problem of the tests is `manufactured_problem`, on the spans
  `(0, tend)` and `(0, 1)`. On `x ∈ [0, 1]` a physical position and a position on the unit
  interval agree, so no test tells the space nodes at which the slab and the box scheme read the
  forcing from their unit-interval counterparts. The slab reads it at the nodes `xnodes` of
  `quadrature_points(c.ansatz)` and at the time `slab_start(problem, sol) + h·t̂`
  (`add_forcing!` in `src/core/slab_integrator.jl`); the runs of several steps reach the time
  offset, but none starts at a time other than `0`.
- kind: missing test
- found: #72

### K72 · A cache lookup of a `NetworkAnsatz` integrator allocates.

- location: `src/ansatz/network.jl:608`
- evidence: `ansatz_data_type` of a `NetworkAnsatz` returns
  `NetworkSlab{ST, T, typeof(a), typeof(parameter_set(a, Vector{ST}(undef, a.np)))}`, and
  `parameter_set` builds a parameter set of views into that new vector. The type folds to a
  constant, but the compiler keeps the call, so `cache(int, T)` allocates the vector at every
  lookup, and `residual!` and the solve's other calls of `cache(int, ST)` pay for it.
  `@allocated` of a warm `cache(int, T)` with `T` known at the call site, over the unforced
  entries of `test/helpers/quality_cases.jl`: 192 and 224 bytes at `Float64` (`D = 1`, `2`) and
  128 and 144 bytes at `Float32` for the three network methods, and 0 bytes for the other six slab
  methods, on Julia 1.13.4. A type-level form of the parameter set's type would remove it.
- kind: defect
- found: 2026-11-15

### K76 · `test/ansatz/tensor_bspline.jl` reaches the 60 s that a `core` file may take.

- location: `test/ansatz/tensor_bspline.jl`
- evidence: two cold runs of `run-tests.jl <repository> core` on Julia 1.13.4, 13 files, all pass:
  196 s at load 6.25 and 188 s at load 7.39. The file took 61.1 s in the first run and less than
  60 s in the second. It is a `core` file at `3cef8d8` too. Whether it is above the limit at the
  machine's usual load is open.
- kind: not verified
- found: 2026-11-21

### K77 · The hard-constraint lifting on a Dirichlet B-spline ansatz does not converge on `Wave`: its slab Jacobian is singular to round-off.

- location: `src/couplings/hard_constraint_lifting.jl`
- evidence: the call that `docs/src/bspline.md` shows and does not run, `integrate(problem,
  SpaceTimeGalerkin(ansatz, HardConstraintLifting()))` with `Wave.problem(timespan = (0.0, 0.2),
  timestep = 0.1, xstep = 0.05)` and `TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
  xspan = (0.0, 1.0), x_boundary = Dirichlet())`, records on macOS `iterations = [9, 18]`,
  `residuals = [0.026945, 0.026945]` from `initial_residuals = [0.026946, 0.026947]` and
  `converged = [false, false]`, on Julia 1.13.4. On ubuntu-latest the same call throws
  `SingularException: matrix is singular; factorization failed. Zero pivot found at index 59` from
  SimpleSolvers' `ldiv!` (the Documentation job of CI run 44897936595 on pull request #82). The
  slab Jacobian at zero that `linear_solver = true` factorises, read as
  `jacobianmatrix(solver(int))` after `integrate(int)`, is 60 × 60 with numerical rank 40 on
  macOS, Julia 1.13.4: its 40th singular value is `0.0987`, the 41st `1.69e-16` and the smallest
  `4.7e-19`, against a largest of `1.057`. Under `CoefficientRestriction` the same ansatz gives a
  40 × 40 Jacobian of full rank, smallest singular value `0.0827`, and converges at both steps in
  one iteration, as under `WeakInitialCondition` on the same page. The kernel of 20 dimensions is
  measured in `Float64`; whether it is exact is open. The testset
  "L5 a B-spline ansatz runs under the lifting" of `test/couplings/hard_constraint_lifting.jl`
  runs this arm on another configuration, `lifting_problem` with `ncells = (2, 5)` and
  `FitTo(ExactSolution())`, logs its iterations and residuals with `@info`, and asserts nothing
  about their convergence. Whether the arm must converge is open.
- kind: not verified
- found: 2026-11-21

### K78 · The affine controls of the manual's network and ELM examples do not converge under the default solver options.

- location: `docs/src/network.md:29`
- evidence: on `Wave.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05)`, the second
  example of `docs/src/network.md`, `NetworkAnsatz(S = 4, mode = :last_layer, xspan = (0.0, 1.0),
  degree = 2, ncells = (2, 4))` under `WeakInitialAndBoundaryConditions()`, records
  `iterations = [3, 3]`, `residuals = [0.005585, 0.005472]` from
  `initial_residuals = [1.2278, 0.5532]` and `converged = [false, false]`. The second example of
  `docs/src/elm.md`, the same ansatz with `FitTo(PreviousSlab(); method =
  TrainedHiddenLayer(epochs = 5))` and `linear_solver = true`, records `iterations = [2, 2]`,
  `residuals = [0.07234, 0.02230]` from `initial_residuals = [0.5681, 11.260]`; its Newton run in
  the first example ends at the same residuals. The docstring of `NetworkAnsatz`
  (`src/ansatz/network.jl:22`) says that two Newton iterations reach the tolerance of a
  `:last_layer` step wherever the output weights carry every constraint row of the coupling.
  `network_f_reltol` of `test/helpers/testsetup.jl` is measured at `S = 20`, with step
  ratios up to `7.8e-6`; the examples use `S = 4` and `integrate`'s default options. Whether a
  step at `S = 4` has a root that the residual norm can reach is open.
- kind: not verified
- found: 2026-11-21

### K79 · Documenter renders the two Markdown assets of `docs/src/assets/` as pages of their own.

- location: `docs/src/assets/convergence_tables.md`
- evidence: a build of the manual writes `docs/build/assets/convergence_tables.html` and
  `docs/build/assets/long_time_energy.html` beside the page *Verification*, which reads both files
  with `Markdown.parse` (`docs/src/verification.md:22`, `:32`). Neither page is in the `pages` of
  `docs/make.jl`, so each is reachable by its URL and from no menu.
- kind: docs
- found: 2026-11-21

### K81 · Three library pages of the manual are above Documenter's `size_threshold_warn`.

- location: `docs/src/ansatz.md`
- evidence: a build of the manual warns "Generated HTML over size_threshold_warn limit" for
  `couplings.md` (129.05 KiB), `library.md` (141.24 KiB) and `ansatz.md` (153.52 KiB), against
  `size_threshold_warn` 100 KiB and `size_threshold` 200 KiB. Each of the three pages holds an
  `@autodocs` block.
- kind: docs
- found: 2026-11-21

### K83 · `src/couplings/weak_multipliers.jl` has 841 lines, against a target of about 350.

- location: `src/couplings/weak_multipliers.jl`
- evidence: `wc -l src/couplings/weak_multipliers.jl` prints `841`. The file holds
  `WeakInitialCondition` and `WeakInitialAndBoundaryConditions`, and the audit of 2026-10-31 set
  its target at about 350 lines. It has methods for `TensorBSplineAnsatz` (`:95`),
  `SymbolicAnsatz` (`:480`) and `NetworkAnsatz` (`:653`). The redraw of the kernels and couplings
  (`e7df04f`) left 395 lines; the symbolic ansatz added 179 (`3094f5f`) and the network ansatz 199
  (`72a9df4`), as `git diff --shortstat <c>^ <c> -- src/couplings/weak_multipliers.jl` prints.
  The file also contracts its trace tables in scalar loops that a `mul!` on the tables would
  replace: `row_dot` (`:223`), a row of a table against a vector, and the sum in `basis_load!`
  (`:194`) of the multipliers' coefficients against the basis at each node. They stay as they are
  until the rewrite to about 350 lines, of which replacing them is a part.
- kind: defect
- found: 2026-10-31

### K84 · The slab Jacobian is dense and built by ForwardDiff over the whole residual.

- location: `src/core/slab_integrator.jl:101`
- evidence: `default_solver(::SpaceTimeGalerkin) = Newton()`, and the package gives the solver no
  Jacobian of its own, so SimpleSolvers' `Newton()` builds it with `ForwardDiff.jacobian!` over the
  whole slab residual into a dense matrix, which `factorize!(lsolver, jacobianmatrix(s))` factorises
  (`src/core/slab_integrator.jl:690-691`) under `linear_solver = true`. The fixed-boundary
  network's `LevenbergMarquardt` takes `ForwardDiff`'s `jacobian` of the residual as well
  (`jacobian_of`, `src/core/levenberg_marquardt.jl:159`). Under a B-spline ansatz with a
  coupling that adds no lifting, a coefficient reaches only the residual rows of its own support,
  so the matrix is sparse there and is stored and factorised dense. Under `HardConstraintLifting`,
  `transform_field!` subtracts `(1 - t̂)` times the coefficient-dependent bottom correction from
  every time row, so a coefficient with a nonzero bottom trace also reaches the rows outside its
  temporal support; a sparse pattern for that coupling holds these entries too. A sparse Jacobian
  is deferred to a follow-up task.
- kind: defect
- found: 2026-10-31

### K89 · A problem constructor throws `OutOfMemoryError` for an `xstep` whose grid an `Int` counts but memory cannot hold.

- location: `src/problems/lagrangian_problem.jl:100`
- evidence: `_xgrid` (`:268-276`) refuses only a count that is not below `typemax(Int)`, and the
  constructor then calls `ics_function(collect(grid), params)`. `Wave.problem(xstep = 1e-17)`
  asks for about `10^17` points and throws `OutOfMemoryError` from `collect`, not an
  `ArgumentError` that names `xstep`. A bound on the memory of the grid is a design choice that
  the constructor does not make.
- kind: defect
- found: #85

### K90 · The bubble fit builds its whole dictionary on every fit, also when it keeps its workspace.

- location: `src/couplings/fixed_boundary.jl:922`
- evidence: `fit_bubble!` calls `bubble_atoms(key, ST)` before it decides whether to rebuild the
  workspace, and uses the result there only to compare sizes and, on a rebuild, to store it; a fit
  that keeps the workspace allocates an `nα nβ nγ × 3` matrix and discards it. The greedy fit of a
  network builds its dictionary only inside its rebuild branch (`fit_network!`,
  `src/initial_guess/orthogonal_greedy.jl:53-56`). The call is the same at `dd937ed`.
- kind: defect
- found: #85

### K91 · The relative convergence rule asks a fixed-boundary step with a small initial residual for a residual below the floor its slab reaches.

- location: `src/couplings/fixed_boundary.jl:219` (`fixed_boundary_f_reltol`), `:407`
  (`default_options`)
- evidence: a step converges at `r ≤ f_abstol + f_reltol · r₀`, and the family's defaults are
  `f_reltol = 1e-8` and `f_abstol = 0`. The case is a problem with a position-dependent density
  parameter, which the package does not hold, on 71 cells at `h = 0.1499` over 119 steps, `:tsr`,
  with `FixedBoundaryNetwork(Sx = 8, St = 10, ncells = 20, nq = (6, 4))`, `Continuation()` and the
  default `Newton()`, in `Float64` (Julia 1.13, SimpleSolvers 0.14). Step 80 stops after 37
  iterations at `r₀ = 174.7` and `r/r₀ = 1.71e-8`: SimpleSolvers reports stagnation at
  `‖r‖ = 2.98e-6`, a floor that no step lowers, because the residual lies outside the range of the
  Jacobian (share 1.000), and Levenberg–Marquardt from that iterate stops at its μ cap without
  moving. Seven converged steps of the same run end above that floor, at `3.0e-6 … 6.2e-6`. `r₀` is
  110–175 at the turning points of the field (steps 1, 20–21, 40–41, 60–61, 80–81) and up to `6.0e3`
  between them, so with `f_abstol = 0` the relative rule asks those steps for the smallest absolute
  residual of the run. Which steps fail depends on round-off in the history: a 21-step run of the
  same case differs from the 119-step run from step 7 on, as K59 records for the network's
  generated kernels. The tolerance stays as it is; an `f_abstol` would be a change of the rule.
- kind: defect
- found: 2026-11-23

### K92 · The family's pivoted QR keeps a direction that the SVD's rank cut drops, and the Newton step along it grows the residual.

- location: `src/couplings/fixed_boundary.jl:463` (`initsolver(::Newton, …)`), `:202`
  (`fixed_boundary_qr_cut`)
- evidence: on the case of K91, step 21 stops at the cap of 300 iterations at `r₀ = 129.3` and
  `r/r₀ = 9.97e-3`. At the cut `1e-12 σ₁` the SVD of the slab Jacobian keeps 30 directions, and
  its step leaves `1.3e-4` of `‖r‖`. `LapackPivotedQR(rtol = 1e-12)` keeps rank 31, because its
  31st `|R_ii|` lies above the cut while `σ₃₁/σ₁ = 9.0e-13` lies below it; it takes a step 150
  times longer (`5.0e-3` against `3.3e-5`) and multiplies `‖r‖` by 2.9 at the full step.
  Levenberg–Marquardt from the same iterate converges in 20 trials (`r/r₀ = 3.4e-9`). Which steps
  fail depends on round-off in the history: the 21-step run of the same case converges at step 21,
  as K59 records for the network's generated kernels. The route of the fix is the cut SVD of the
  port to `GeometricSolvers`; the package keeps the pivoted QR until then.
- kind: defect
- found: 2026-11-23

### K93 · SimpleSolvers' Newton loop makes no progress for 298 iterations at a step that two plain Newton steps solve.

- location: `src/couplings/fixed_boundary.jl:463` (`initsolver(::Newton, …)`); SimpleSolvers 0.14,
  its Newton loop
- evidence: `Wave` at `h = 0.05` over 160 steps to `t = 8`, `:tsr`, with
  `FixedBoundaryNetwork(Sx = 8, St = 10, ncells = 20, nq = (6, 4))`, `Continuation()` and the
  default `Newton()`, in `Float64` (Julia 1.13, SimpleSolvers 0.14), converges at 159 of 160
  steps. Step 25 (`t = 1.25`) stops at the cap of 300 iterations at `r₀ = 1.014` and
  `r/r₀ = 1.445e-4`, with no progress for 298 iterations and no line-search failure in the status.
  At the final iterate the QR step and the SVD step agree (rank 27, `‖δ‖ = 2.9e-3`), and two plain
  full Newton steps of the same factorization converge (`r/r₀ = 2.1e-6`, then `1.3e-11`), so the
  stall is in the loop of `SimpleSolvers` and not in the factorization. Which steps fail depends
  on round-off in the history, as K59 records for the network's generated kernels.
- kind: upstream
- found: 2026-11-23

### K94 · On the coarse quadrature of `ncells = 3`, the default `Newton()` stalls at the cap where Levenberg–Marquardt reduces the residual by six orders.

- location: `src/couplings/fixed_boundary.jl:391` (`default_solver`), `:463`
  (`initsolver(::Newton, …)`)
- evidence: `Wave` at `h = 0.05` over two steps, with
  `FixedBoundaryNetwork(Sx = 8, St = 10, ncells = 3, nq = (6, 4))`, `Continuation()` and the
  family's default options, in `Float64` (Julia 1.13.4, SimpleSolvers 0.14.4). Under the default
  `Newton()` both steps stop at the cap of 300 iterations, in both forms: from `r₀ = [0.4689,
  0.0544]` to `r = [0.4559, 0.0317]`. Under `LevenbergMarquardt()` both steps also stop at 300
  iterations, at `r = [1.18e-7, 1.37e-7]` (`:abg`) and `[6.96e-7, 6.07e-7]` (`:tsr`) from
  `r₀ = [0.4689, 0.398]`, so their ratios are above `f_reltol = 1e-8` as well. Every measurement behind the
  default is on `ncells = 20`, and the default's docstring is limited to it. `fixed_ansatz` in
  `test/helpers/fixed_boundary_setup.jl` builds this quadrature, and FB10, which solves one slab of
  it, names `LevenbergMarquardt()`.
- kind: defect
- found: #91
