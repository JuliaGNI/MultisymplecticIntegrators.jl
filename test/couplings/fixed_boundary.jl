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
# The registry of the scripts and no part of the package: the family's rows are built here.
include(joinpath(@__DIR__, "..", "..", "scripts", "experiments.jl"))
# The helpers this file shares with the other files of `test/couplings/fixed_boundary*.jl`, which
# hold the coupling's other testsets.
include("../helpers/fixed_boundary_setup.jl")

# ---- FB0 -----------------------------------------------------------------------------

@testset "FB0 the two types and their pair" begin
    problem = PROBLEMS[:wave]()
    a = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0))
    @test FixedBoundary() isa SlabCoupling
    @test a isa SpaceTimeAnsatz
    @test (M.nspatial(a), M.nbubble(a), M.nfree(a)) == (12, 16, 28)
    last_layer = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0), mode = :last_layer)
    @test (M.nspatial(last_layer), length(M.spatial_free(last_layer)),
        M.nfree(last_layer)) == (12, 4, 8)
    # At `D = 2` a level frees `Sx·D` output weights, which are not the first `Sx` of them: the
    # level's parameters are `Sx` weights, `Sx` biases and then `Sx·D` output weights.
    wide = FixedBoundaryNetwork(
        Sx = 8, St = 10, xspan = (0.0, 1.0), mode = :last_layer, D = 2)
    @test M.nspatial(wide) == 2 * 8 + 8 * 2
    @test M.spatial_free(wide) == (2 * 8 + 1):(2 * 8 + 8 * 2)
    @test M.bubble_free(wide) == (3 * 10 + 1):(3 * 10 + 10 * 2)
    @test M.nfree(wide) == 8 * 2 + 10 * 2
    @test M.free_indices(wide) == vcat(collect(17:32), 32 .+ collect(31:50))
    network = NetworkAnsatz(S = 4, xspan = (0.0, 1.0), ncells = (2, 5), degree = 3)
    spline = TensorBSplineAnsatz(degree = 3, ncells = (2, 5), xspan = (0.0, 1.0))
    # `@test_throws` matches its `Regex` against `sprint(showerror, e)`, which prefixes the
    # exception's type, so the anchors hold only when the type is part of the pattern.
    @test_throws r"^ArgumentError: FixedBoundary takes a FixedBoundaryNetwork, got NetworkAnsatz$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(network, FixedBoundary()))
    @test_throws r"^ArgumentError: FixedBoundary takes a FixedBoundaryNetwork, got TensorBSplineAnsatz$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(spline, FixedBoundary()))
    for coupling in (WeakInitialAndBoundaryConditions(), WeakInitialCondition(),
        CoefficientRestriction())
        @test_throws Regex("^ArgumentError: a FixedBoundaryNetwork takes the FixedBoundary " *
                           "coupling, got $(nameof(typeof(coupling)))\$") GeometricIntegrator(
            problem, SpaceTimeGalerkin(a, coupling))
    end
    # `HardConstraintLifting` takes a `NetworkAnsatz` or a `TensorBSplineAnsatz` and names those two
    # in its own refusal; a `FixedBoundaryNetwork` is neither, so the refusal that answers is this
    # intersection's.
    @test_throws r"^ArgumentError: a FixedBoundaryNetwork takes the FixedBoundary coupling, got HardConstraintLifting$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(a, HardConstraintLifting()))
    method = SpaceTimeGalerkin(a, FixedBoundary())
    int = GeometricIntegrator(problem, method)
    @test default_solver(method) isa LevenbergMarquardt
    @test solver(int) isa M.LevenbergMarquardtSolver
    @test length(cache(int).x) == 28
    # A `PINNTraining` fit on a fixed-boundary network matches the network's refusal for any fit
    # method and `PINNTraining`'s refusal for any ansatz, so it reaches this intersection's.
    @test_throws r"^ArgumentError: a FixedBoundaryNetwork is fitted by BubbleGreedy, got PINNTraining$" M.fit!(
        zeros(1), PINNTraining(epochs = 0), a, nothing, cache(int))
    @test M.coupling_unknowns(FixedBoundary(), a, problem) == 0
    @test M.free_rows(FixedBoundary(), a) == 28
    @test total_free_coefficients(method, problem) == 28
    # A `PreviousSlab` target has no field map for this ansatz, so the guess is refused and the
    # refusal names the target it was given.
    @test_throws r"^ArgumentError: a FixedBoundaryNetwork takes InitialCondition, ExactSolution, TargetFunction and Predictor as targets, got PreviousSlab$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = FitTo(PreviousSlab())))
end

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

# ---- FB18 ----------------------------------------------------------------------------

@testset "FB18 the family's rows build and the plateau is a record" begin
    # N23: the registry holds the family's 32 entries — 30 rows of the matrix and 2 runs of the
    # plateau probe — each with a distinct stem, one of the plateau's at `Sx = 12`, `St = 16` and
    # `h = 0.025` and one at `BigFloat`, and the plateau's numbers are a record and no assertion.
    plateau = family_entries(:fixed_boundary)
    matrix = family_entries(:fixed_boundary_matrix)
    @test length(plateau) == 2
    @test length(matrix) == 30
    @test allunique(entry.stem for entry in Iterators.flatten((plateau, matrix)))
    @test all(entry -> occursin("plateau", entry.stem), plateau)
    @test !any(entry -> occursin("plateau", entry.stem), matrix)
    @test all(entry -> entry.steps ≥ 1, plateau)
    @test all(entry -> entry.parameters.steps ≥ 1, matrix)
    @test length(Set((entry.problem, entry.parameters.form, entry.parameters.solve,
                         entry.resolution.timestep) for entry in matrix)) == 30
    # a matrix row integrates its ladder up to `t = 0.2`, the final time of the other families'
    # rows of its problem, so that the rows of one problem compare errors at one time
    @test all(entry -> entry.parameters.steps * entry.resolution.timestep ≈ 0.2, matrix)
    @test sort(unique(entry.parameters.steps for entry in matrix)) == [2, 4, 8]
    # a driver given no stem runs the plateau's `Float64` entry, and not the `BigFloat` one, by its
    # stem and whatever the order of the family's tuple: reversed, the cheapest entry of the tie is
    # the `BigFloat` one, and the default stays
    default = default_entry(:fixed_boundary)
    @test default.stem == "wave-fixed-boundary-plateau-h0.025"
    @test default.T === Float64
    held = FAMILIES[:fixed_boundary]
    try
        FAMILIES[:fixed_boundary] = merge(held, (entries = reverse(held.entries),))
        @test smallest_entry(family_entries(:fixed_boundary)).T === BigFloat
        @test default_entry(:fixed_boundary) === default
    finally
        FAMILIES[:fixed_boundary] = held
    end
    # a plateau entry names its solve as a row's `parameters` do; `solver` is a row's solver method
    @test all(entry -> entry.solve in (:newton, :joint, :projected), plateau)
    @test !any(entry -> haskey(entry, :solver), plateau)
    # the cheapest row, the largest time step, runs at `Float64`
    largest = maximum(entry.resolution.timestep for entry in matrix)
    @test all(entry -> entry.parameters.T === :Float64,
        filter(entry -> entry.resolution.timestep == largest, matrix))
    entries = vcat(collect(plateau), collect(matrix))
    @test all(entry -> entry.problem in (:wave, :sine_gordon), entries)
    @test any(entry -> entry.Sx == 12 && entry.St == 16 && entry.timestep == 0.025, plateau)
    @test any(entry -> entry.T === BigFloat, plateau)
    @test all(entry -> entry.steps == 1, filter(entry -> entry.T === BigFloat, plateau))
    # N23: the extended-precision entry names its own tolerance, because the family measures none at
    # `BigFloat`; the entry's driver and this testset build the solver's keywords through one
    # function, so the entry's value is the one the solver is built with
    extended = only(filter(entry -> entry.T === BigFloat, plateau))
    @test extended.f_reltol == big"1e-8"
    extended_int = GeometricIntegrator(fixed_boundary_problem(extended),
        fixed_boundary_method(extended); solver = M.LevenbergMarquardt(),
        fixed_boundary_solver_keywords(extended)...)
    @test config(solver(extended_int)).f_reltol == extended.f_reltol
    for entry in entries
        @test fixed_boundary_problem(entry) isa M.LagrangianPDEProblem
        @test fixed_boundary_method(entry) isa SpaceTimeGalerkin
    end
end
