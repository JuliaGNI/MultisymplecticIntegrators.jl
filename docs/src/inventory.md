```@meta
CurrentModule = MultisymplecticIntegrators
```

# From the old stack

Every feature of the old stack at `294f023`, the test file and the testset that check it on this
version, and the page of this manual that documents it. A row that held no functionality of its own
names the pull request that removed it.

| # | feature at `294f023` | test file | testset | page |
|--:|:--|:--|:--|:--|
| 1 | `Wave`, `SineGordon`, `LinearTransport` problems, with exact solutions and energies | `test/problems/lagrangian_problem.jl` | "the wave functions see the problem's parameters", "the sine-Gordon functions see the problem's parameters" and "the transport functions see the problem's parameters" for the exact solutions; "S2a the energy density is the density's Hamiltonian" of `test/diagnostics.jl` for the wave energy | [Example problems and diagnostics](problems.md) |
| 2 | Lagrangians as Symbolics expressions, derivatives generated | `test/problems/lagrangian_density.jl` | "the wave density against its closed form" | [Example problems and diagnostics](problems.md) |
| 3 | Full-multiplier spline Galerkin (`Galerkin_Bspline_Integrator`), with `multiplier_stride` | `test/couplings/weak_multipliers.jl` | "A2a the plane wave" | [The B-spline family](bspline.md) |
| 4 | Dirichlet spline Galerkin with an initial multiplier (`Galerkin_Dirichlet_Bspline_Integrator`) | `test/couplings/weak_multipliers.jl` | "A1a the Dirichlet reference" | [The B-spline family](bspline.md) |
| 5 | Full-restriction spline Galerkin (`Galerkin_Full_Restriction_Bspline_Integrator`) | `test/couplings/coefficient_restriction.jl` | "A1c the full-restriction reference" | [The B-spline family](bspline.md) |
| 6 | Space-time spline wave integrator (`SpaceTime_Spline_Wave_Integrator`), linear, factorised once | `test/ansatz/tensor_bspline.jl` | "A1b the space-time spline equivalence", and "R2 linear_solver = true direct LU path and linearity guards" | [The B-spline family](bspline.md#The-direct-linear-solve) |
| 7 | Bilinear FEM wave integrator (`FEM_Multisymplectic_Integrator`), with `:exact` and Taylor start-up and BDF2 velocity | `test/ansatz/tensor_bspline.jl` | "A1a the FEM equivalence at degree 1" | [The B-spline family](bspline.md) |
| 8 | Network integrator (`NN_PDE_Integrator`), `:Fully` and `:Partially` | `test/ansatz/network.jl` | "N1 the network integrates Wave under both modes" | [The multiplier network](network.md) |
| 9 | Multiplier bases `:BSplineDirichlet` and `:Lagrange` (nodal) | `test/couplings/weak_multipliers.jl` | "A3a the two bases agree as the mesh refines" | [The B-spline family](bspline.md) |
| 10 | Symbolic network (a hand-built Symbolics expression) | `test/ansatz/symbolic.jl` | "A4a the closed form over five slabs" | [The symbolic family](symbolic.md) |
| 11 | SINDy integrator (`Sindy_PDE_Integrator`, a closed-form symbolic ansatz) | `test/ansatz/symbolic.jl` | "A4b the vector field of two expressions" | [The symbolic family](symbolic.md) |
| 12 | ELM (`ELM_PDE_int`): random-feature hidden layers, output layer solved | `test/initial_guess/trained_hidden_layer.jl` | "E4a the family on Wave without an exact solution" | [The ELM family](elm.md) |
| 13 | ELM's LSGD training of the hidden layers | `test/initial_guess/trained_hidden_layer.jl` | "E1a the fit at epochs = 0 and at epochs > 0" | [The ELM family](elm.md) |
| 14 | ELM's direct least-squares branch (`IPMT <: ELM`, `problem_matrix_assemble`) | `test/initial_guess/trained_hidden_layer.jl` | "E3b one factorisation, two iterations and the recorded residual ratio" | [The ELM family](elm.md) |
| 15 | TrialNN (`TrialNN_PDE_int`): hard-constraint trial solution, no multipliers | `test/couplings/hard_constraint_lifting.jl` | "L1 the field is the data at the three boundary parts" | [The TrialNN family](trialnn.md) |
| 16 | TrialNN's PINN pre-training branch, and `PINN_for_init.jl` | `test/initial_guess/pinn_training.jl` | "P1 the fit improves the network on a sampled target" | [The TrialNN family](trialnn.md) |
| 17 | Orthogonal greedy seed `OGA2D` | `test/initial_guess/orthogonal_greedy.jl` | "G1 the fit reproduces the seed on the §A case" | [The initial guess](guess.md) |
| 18 | Exact-solution initial guesses | `test/initial_guess/guess.jl` | "A4a every option converges to the same solution" | [The initial guess](guess.md) |
| 19 | Momentum carry between slabs; TrialNN has none, on purpose | `test/couplings/weak_multipliers.jl` | "A1c the momentum carry" | [The B-spline family](bspline.md) |
| 26 | Vector-valued fields, `D > 1` | `test/problems/vector_wave.jl` | "each component carries its own standing wave, component-major" | [Example problems and diagnostics](problems.md) |
| 20 | `show_status` error reports against the exact solution | `test/diagnostics.jl` | "the error of a run against the exact solution" | [Library](library.md) |
| 21 | Quadrature values per step returned by `integrate!` | `test/integration/integrator_shell.jl` | "the trace of every step, at the quadrature nodes" | [Library](library.md) |
| 22 | The scripts' scans: energy, phase drift, regularisation, profiling, parallel sweeps | `test/integration/benchmark.jl` | "the drift and the charge a run archives" | [Verification](verification.md#The-scans-and-the-benchmark-matrix); the parallel sweeps `scripts/parallel_run.sh` and `scripts/parallel_run_sindy.sh` were a loop over registry entries and were removed by #34 |
| 23 | `NN_PDE_LSGD_int.jl` | — | — | **no own functionality**: an older copy of row 8, removed by #25 |
| 24 | `Lagrangian_PDE_solution.jl` (`LPDE_solution`) | — | — | **no own functionality**: never included, superseded by `GeometricSolution`, removed by #25 |
| 25 | `solvers.jl` (`initsolver` with backtracking) | — | — | **no own functionality**: never included, names undefined at load, removed by #25 |

Row 1's tests check the wave energy alone. The sine-Gordon and transport energies are checked only
for their element type, by "a Float32 SineGordon problem evaluates its energy in Float32" of
`test/problems/lagrangian_problem.jl` and "a Float32 problem evaluates its energy in Float32" of
`test/problems/linear_transport.jl`. Row 3's `multiplier_stride` is gone: every trace function
carries a multiplier, the corners excepted. Row 7 needs no start-up, because the step is a one-step
form, and its velocity is the ansatz's `∂u/∂t̂ / h`. Row 19's carry is each coupling's; the lifting
of the TrialNN family carries none, and the other couplings' carries are tested in
`test/couplings/coefficient_restriction.jl` and `test/couplings/weak_multipliers.jl`.
