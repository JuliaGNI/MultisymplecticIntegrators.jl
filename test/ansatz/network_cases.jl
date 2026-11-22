# The network ansatz on its integrated cases: the boundary points against the discrete action, a
# network with two outputs on VectorWave, and the §A case in both modes. One of the files of
# `test/ansatz/network*.jl`, which share `test/helpers/network_setup.jl`.
using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS, norm
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using SimpleSplines
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, current, nlsolution, reset!,
                                 solutionstep, solver, integrate!
import SimpleSolvers: jacobian!, jacobianmatrix

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")
include("../helpers/action_oracle.jl")
include("../helpers/network_setup.jl")

@testset "Network ansatz: cases" begin
    # The edges against the discrete action: a bottom or a top point read in the normalised
    # coordinate moves the residual away from the oracle by a quarter of its scale, where the
    # unmutated residual agrees with it to round-off.
    @testset "N3 the boundary points are physical" for T in (Float64, Float32)
        xspan = (T(-0.5), zero(T))
        problem = LinearTransport.problem(T = T, timespan = (zero(T), T(0.15)),
            timestep = T(0.05), xstep = T(0.05))
        ansatz = network_ansatz(T; degree = 3, ncells = (2, 5), D = 2, xspan = xspan)
        method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(ExactSolution(); method = OrthogonalGreedy()))
        int = GeometricIntegrator(problem, method)
        result = integrate(int)
        c = cache(int)
        @test length(result.traces) == 3
        @test length(result.stats.iterations) == 3
        @test length(result.stats.residuals) == 3
        # a fresh integrator's first slab, its datum fixed, at a vector away from its solution
        slab = network_slab(T; D = 2, xspan = xspan, problem = problem)
        oracle = ActionOracle(slab.problem, slab.method)
        params = parameters(slab.problem)
        N = length(cache(slab.int).x)
        @test N == unknown_count(oracle)
        θ = T.(2 .* rand(Xoshiro(1234), N) .- 1)
        b = zeros(T, N)
        M.residual!(b, θ, nothing, params, slab.int)
        ∇S = oracle_residual(oracle, first_datum(oracle), θ)
        @test all(isfinite, b)
        @test maximum(abs, ∇S) ≥ T(1 // 100)
        @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * maximum(abs, ∇S)
        # the block's own nodes are the physical ones, and the initial condition is read there
        tnodes, xnodes = M.quadrature_points(c.ansatz)
        xt, _ = gauss_rule(T, xspan[1], xspan[2], 5,
            M.quadrature_order_of(M.trace_basis(ansatz))[2])
        @test collect(extrema(xnodes)) ≈ collect(extrema(xt))
        @test length(xnodes) == length(xt)
        sample = zeros(T, length(tnodes), length(xnodes), 2)
        M.evaluate!(sample, InitialCondition(), c, ansatz, problem,
            solutionstep(int,
                GeometricSolution(problem)[0]))
        values = problem.ics_function(collect(xt), parameters(problem))
        scale = maximum(abs, values.u)
        for r in eachindex(xt), d in 1:2

            @test abs(sample[1, r, d] - T(values.u[r, d])) ≤
                  16 * eps(T) * max(scale, one(T))
        end
        # a node set read in the normalized coordinate is not the physical one
        @test extrema(xnodes) != extrema(range(zero(T), one(T); length = length(xnodes)))
    end

    @testset "N4 a network with two outputs integrates VectorWave" for T in (Float64, Float32)
        params = VectorWave.default_parameters()
        problem = LagrangianPDEProblem(
            VectorWave.vector_density(; params = params, β = T(0.5));
            D = 2,
            ics_function = VectorWave.problem(T = T; params = params, timestep = T(0.05)).ics_function,
            bcs_function = VectorWave.problem(T = T; params = params, timestep = T(0.05)).bcs_function,
            solutions = VectorWave.problem(T = T; params = params, timestep = T(0.05)).solutions,
            timespan = (zero(T), T(0.5)), timestep = T(0.05), xspan = (zero(T), one(T)),
            xstep = T(0.05), params = params)
        ansatz = network_ansatz(T; S = 10, D = 2)
        method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        int = GeometricIntegrator(problem, method)
        result = integrate(int)
        c = cache(int)
        @test length(result.traces) == 10
        @test finite_states(result.solution.u)
        @test all(isfinite, c.trace_u)
        @test all(isfinite, c.trace_v)
        # each component holds its own output weights, one column of the fit's coefficients
        @test M.output_weights(c.ansatz.params, ansatz, 1) !=
              M.output_weights(c.ansatz.params, ansatz, 2)
        # the step's residual is recorded: a network step is a record, and it may fail (U13, U24)
        @test result.stats.residuals[1] ≤ result.stats.initial_residuals[1]
        # the output dimension is the problem's, and the input layer's is two in either case: a
        # network of two outputs on a scalar problem is refused when its cache is built
        @test M.noutputs(ansatz) == 2
        scalar = SpaceTimeGalerkin(network_ansatz(T; D = 2),
            WeakInitialAndBoundaryConditions())
        @test_throws ArgumentError GeometricIntegrator(wave_problem(T), scalar)
        @test_throws ArgumentError NetworkAnsatz(
            S = 4, xspan = (zero(T),), ncells = (2, 5),
            degree = 3)
        @test_throws ArgumentError NetworkAnsatz(S = 4, D = 0, xspan = (zero(T), one(T)),
            ncells = (2, 5), degree = 3)
        @test_throws ArgumentError NetworkAnsatz(S = 0, xspan = (zero(T), one(T)),
            ncells = (2, 5), degree = 3)
        @test_throws ArgumentError NetworkAnsatz(S = 4, xspan = (zero(T), one(T)),
            ncells = (2, 5), degree = 3, mode = :full)
    end

    @testset "A1 the §A case in both modes" for T in (Float64, Float32)
        for (mode, name, entry) in ((:all, :WAVE_NN_FULLY, WAVE_NN_FULLY),
            (:last_layer, :WAVE_NN_PARTIALLY, WAVE_NN_PARTIALLY))
            _, _, _, result = network_run(T; mode = mode,
                guess = FitTo(ExactSolution(); method = OrthogonalGreedy()))
            @test length(result.traces) == NETWORK_STEPS
            @test length(result.stats.iterations) == NETWORK_STEPS
            @test finite_states(result.solution.u)
            deviation = sup_deviation(result, entry)
            @test isfinite(deviation)
            record_deviation(:A1, name, T, deviation)
            if mode === :all
                symbolic = sup_deviation(result, WAVE_NN_SYMBOLIC_FULLY)
                @test isfinite(symbolic)
                record_deviation(:A1, :WAVE_NN_SYMBOLIC_FULLY, T, symbolic)
            end
        end
        # the three-step SineGordon case of the entry
        problem = SineGordon.problem(
            T = T, timespan = (zero(T), T(0.15)), timestep = T(0.05),
            xstep = T(0.05))
        method = SpaceTimeGalerkin(network_ansatz(T), WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(ExactSolution(); method = OrthogonalGreedy()))
        result = integrate(GeometricIntegrator(problem, method))
        @test length(result.traces) == 3
        deviation = sup_deviation(result, SINE_GORDON_NN_FULLY)
        @test isfinite(deviation)
        record_deviation(:A1, :SINE_GORDON_NN_FULLY, T, deviation)
    end
end
