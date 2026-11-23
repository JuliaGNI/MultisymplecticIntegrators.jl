using GeometricBase: parameters, timestep, timespan
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS, ColumnNorm, diag, norm, qr
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, current, integrate!, reset!,
                                 solutionstep, solver
import GeometricBase: ntime, timesteps

include("../helpers/problem_generator.jl")
include("../helpers/action_oracle.jl")
include("../helpers/testsetup.jl")
# The helpers this file shares with the other files of `test/couplings/fixed_boundary*.jl`, which
# hold the coupling's other testsets.
include("../helpers/fixed_boundary_setup.jl")

# ---- FB17 ----------------------------------------------------------------------------

@testset "FB17 the gauge's two periods are a record" begin
    # N21: the dead branch over two wave periods with `Continuation()` and the minimum-norm step,
    # at the clause's rule of 20 cells. The numbers are a record — the largest `‖(w, b)‖∞` over the
    # run, the steps that converge, the error at the end and the run's cost — and the testset
    # asserts none of them; the two statements the clause does make, that the action holds no gauge
    # term and that the solve re-fits no `φ` to a fixed field, are its source's and not a
    # measurement. The record carries the resolution, the machine and the run it was taken on, and
    # no number of it is compared with the spike's.
    problem = WF.problem(timespan = (0.0, 8.0), timestep = 0.05, xstep = 0.05,
        xspan = (0.0, 1.0))
    ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = :abg, xspan = (0.0, 1.0),
        ncells = 20, nq = (6, 4))
    method = SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = Continuation())
    int = GeometricIntegrator(problem, method; solver = Newton(), max_iterations = 60,
        f_abstol = 0.0, f_reltol = fixed_boundary_f_reltol(Float64), verbosity = 0)
    c = cache(int)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    steps = length(timesteps(solution)) - 1
    started = time()
    # `‖(w, b)‖∞` read after every step and kept as the largest: the parameters of a step's last
    # accepted iterate are the state the drift is measured on, and the value after the last step
    # alone misses a crossing that returns
    drift = zero(Float64)
    for n in 1:steps
        reset!(solstep, timesteps(solution)[n])
        integrate!(solstep, int)
        copy!(solution, current(solstep), n)
        drift = max(drift, maximum(abs, c.ansatz.params[1:(2 * ansatz.Sx)]))
    end
    elapsed = time() - started
    xs = collect(range(xspan(problem)[1], xspan(problem)[2]; length = 21))
    t_end = timesteps(solution)[end]
    final_error = maximum(abs(WF.solution_u(t_end, x, parameters(problem)) -
                              solution.u[end][i])
    for (i, x) in enumerate(xs))
    record = (steps = length(c.converged), converged = count(c.converged), drift = drift,
        error = final_error, T = Float64, nodes = length(ansatz.tabulation.xnode),
        h = timestep(problem), seconds = round(elapsed; digits = 1), version = VERSION,
        machine = Sys.MACHINE, cpu = Sys.CPU_NAME, threads = Threads.nthreads(),
        blas_threads = BLAS.get_num_threads(), check_bounds = Base.JLOptions().check_bounds)
    @info "FB17 the gauge's two periods" record
    @test steps == length(c.converged)
    @test all(isfinite, c.residuals)
    @test all(isfinite, c.ansatz.params)
end
