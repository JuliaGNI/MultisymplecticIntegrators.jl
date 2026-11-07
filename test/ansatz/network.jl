using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: norm
using MultisymplecticIntegrators
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, nlsolution, reset!,
                                 solutionstep, integrate!

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")
include("../helpers/action_oracle.jl")

const M = MultisymplecticIntegrators

"""
The largest deviation of a network run from its §A entry, in the sup norm over the output grid, with
the value this tree measures beside it.

`WAVE_NN_FULLY`: `0.7553` over the five steps, in `Float64` — the entry is the run of the old
integrator, and this bound holds the run of this one. `WAVE_NN_PARTIALLY`: `580.4`, because the
entry is a diverged iterate, its own `u` reaching `1.3e2` where this run's stays under `0.7`; the
bound records that run and not the accuracy of this one. `SINE_GORDON_NN_FULLY`: `0.046`.

Neither entry can be reproduced bit for bit: they were made before the two defects the network
family fixes — the bottom points read in the normalized coordinate (`xspan` is `(0, 1)`, so there
they coincide, and this bound is the fit's alone) and the duplicate atom of the old dictionary,
which cost the greedy a whole selection.
"""
const NETWORK_SUP_BOUND = (WAVE_NN_FULLY = 1.0, WAVE_NN_PARTIALLY = 600.0,
    SINE_GORDON_NN_FULLY = 0.1)

"""The steps of the §A case."""
const NETWORK_STEPS = 5

"""The most Newton iterations the affine control of a `:last_layer` step may take."""
const LAST_LAYER_ITERATIONS = 2

function network_ansatz(::Type{T}; S = 10, D = 1, mode = :all, degree = 3, ncells = (2, 5),
        xspan = (zero(T), one(T)), quadrature_order = nothing) where {T}
    return NetworkAnsatz(S = S, D = D, mode = mode, xspan = xspan, degree = degree,
        ncells = ncells, quadrature_order = quadrature_order)
end

function wave_problem(::Type{T}; timespan = (zero(T), T(0.25)), timestep = T(0.05),
        xstep = T(0.05), D = 1) where {T}
    return Wave.problem(
        T = T, D = D, timespan = timespan, timestep = timestep, xstep = xstep)
end

"""A whole run of one network mode on `Wave`: five steps of `h = 0.05`."""
function network_run(::Type{T}; mode = :all, D = 1, S = 10,
        guess = FitTo(InitialCondition(); method = OrthogonalGreedy()),
        problem = wave_problem(T; D = D), steps = NETWORK_STEPS) where {T}
    method = SpaceTimeGalerkin(network_ansatz(T; mode = mode, D = D, S = S),
        WeakInitialAndBoundaryConditions(); initial_guess = guess)
    int = GeometricIntegrator(problem, method)
    result = integrate(int)
    return problem, method, int, result
end

"""The sup-norm deviation of a run from the `u` of a §A entry, over the steps it took."""
function sup_deviation(result, entry)
    return maximum(maximum(abs.(result.solution.u[n] .- entry.u[n, :]))
    for n in 1:length(result.traces))
end

# The physical Gauss nodes and weights of one direction of a slab, computed here so that a node set
# read in another coordinate shows up as a difference rather than as an error.
function gauss_rule(::Type{T}, a, b, ncells::Integer, n::Integer = 4) where {T}
    ξ = T[-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
    ω = T[0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
    nodes = T[]
    weights = T[]
    width = (b - a) / ncells
    for c in 0:(ncells - 1), k in 1:n

        push!(nodes, a + width * (ξ[k] + one(T)) / 2)
        push!(weights, width / 2 * ω[k])
    end
    return nodes, weights
end

@testset "Network ansatz" begin
    @testset "S1b a projection on a network is refused" for T in (Float64, Float32)
        problem = wave_problem(T)
        ansatz = network_ansatz(T)
        make(fit) = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = fit)
        err = try
            GeometricIntegrator(problem, make(FitTo(InitialCondition(); method = Projection())))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Projection()", err.msg)
        @test occursin("NetworkAnsatz", err.msg)
        @test GeometricIntegrator(problem,
            make(FitTo(InitialCondition()))) isa GeometricIntegrator
        @test GeometricIntegrator(problem, make(Continuation())) isa GeometricIntegrator
    end

    @testset "S2b the size of one slab's system" for T in (Float64, Float32)
        problem = wave_problem(T)
        for mode in (:all, :last_layer), S in (4, 10)

            ansatz = network_ansatz(T; S = S, mode = mode)
            method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
            int = GeometricIntegrator(problem, method)
            @test length(nlsolution(int)) == total_free_coefficients(method, problem)
            @test length(nlsolution(int)) ==
                  M.nfree(ansatz) +
                  M.coupling_unknowns(coupling(method), ansatz,
                problem)
            free = mode === :last_layer ? S : 3S + S
            @test M.nfree(ansatz) == free
        end
        # the tensor path keeps the base's count: free rows × space functions × components
        spline = TensorBSplineAnsatz(degree = 2, ncells = (2, 5), xspan = (zero(T), one(T)))
        method = SpaceTimeGalerkin(spline, WeakInitialAndBoundaryConditions())
        int = GeometricIntegrator(problem, method)
        @test length(nlsolution(int)) == total_free_coefficients(method, problem)
        @test length(nlsolution(int)) ==
              M.free_rows(coupling(method), spline) *
              M.nspace_functions(spline) * 1 +
              M.coupling_unknowns(coupling(method), spline, problem)
    end

    @testset "N2 the parameters live in the cache, not in the method" for T in (Float64, Float32)
        problem = wave_problem(T)
        ansatz = network_ansatz(T; S = 4)
        before = (ansatz.mode, ansatz.S, ansatz.D, ansatz.np, ansatz.nfree,
            ansatz.first_free, ansatz.xspan, ansatz.chain)
        first_method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        second_method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        first_int = GeometricIntegrator(problem, first_method)
        second_int = GeometricIntegrator(problem, second_method)
        first = integrate(first_int)
        second = integrate(second_int)
        again = integrate(first_int)
        for run in (second, again)
            @test run.solution.u ≈ first.solution.u
            @test run.solution.v ≈ first.solution.v
            @test run.stats.iterations == first.stats.iterations
            @test run.stats.residuals == first.stats.residuals
        end
        after = (ansatz.mode, ansatz.S, ansatz.D, ansatz.np, ansatz.nfree,
            ansatz.first_free, ansatz.xspan, ansatz.chain)
        @test all(before[k] === after[k] for k in eachindex(before))
    end

    @testset "N5 the element type is the problem's" for T in (Float64, Float32)
        _, _, int, result = network_run(T; steps = 2)
        c = cache(int)
        assert_no_upcast(result.solution.u, T)
        assert_no_upcast(result.solution.v, T)
        assert_no_upcast(c.trace_u, T)
        assert_no_upcast(c.x, T)
        assert_no_upcast(c.ansatz.params, T)
        assert_no_upcast(c.ansatz.u, T)
        assert_no_upcast(c.ansatz.Ju, T)
        assert_no_upcast(c.ansatz.gradient, T)
        assert_no_upcast(result.traces, T)
    end

    @testset "N1 the network integrates Wave under both modes" for T in (Float64, Float32)
        for (mode, entry, bound) in ((:all, WAVE_NN_FULLY, NETWORK_SUP_BOUND.WAVE_NN_FULLY),
            (:last_layer, WAVE_NN_PARTIALLY, NETWORK_SUP_BOUND.WAVE_NN_PARTIALLY))
            _, _, int, result = network_run(T; mode = mode)
            c = cache(int)
            @test length(result.traces) == NETWORK_STEPS
            @test length(result.stats.iterations) == NETWORK_STEPS
            @test length(result.stats.residuals) == NETWORK_STEPS
            @test length(result.stats.converged) == NETWORK_STEPS
            @test minimum(result.stats.iterations) ≥ 1
            @test all(isfinite, result.solution.u)
            @test all(isfinite, result.solution.v)
            @test all(isfinite, c.trace_u)
            if mode === :last_layer
                @test maximum(result.stats.iterations) ≤ LAST_LAYER_ITERATIONS
            end
            deviation = sup_deviation(result, entry)
            @test deviation ≤ bound
            assert_no_upcast(result.solution.u, T)
        end
    end

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
        # the block's own nodes are the physical ones, and the initial condition is read there
        tnodes, xnodes = M.quadrature_points(c.ansatz)
        xt, _ = gauss_rule(T, xspan[1], xspan[2], 3,
            M.quadrature_order_of(M.trace_basis(ansatz))[2])
        @test extrema(xnodes) ≈ extrema(xt)
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
        @test all(isfinite, result.solution.u)
        @test all(isfinite, c.trace_u)
        @test all(isfinite, c.trace_v)
        # the second component's rows carry its own target, not the first one's
        @test M.output_weights(c.ansatz.params, ansatz, 1) !=
              M.output_weights(c.ansatz.params, ansatz, 2)
        # the step's residual is recorded: a network step is a record, and it may fail (U13, U24)
        @test result.stats.residuals[1] ≤ result.stats.initial_residuals[1]
    end

    @testset "A1 the §A case in both modes" for T in (Float64, Float32)
        for (mode, entry, bound) in ((:all, WAVE_NN_FULLY, NETWORK_SUP_BOUND.WAVE_NN_FULLY),
            (:last_layer, WAVE_NN_PARTIALLY, NETWORK_SUP_BOUND.WAVE_NN_PARTIALLY))
            _, _, _, result = network_run(T; mode = mode,
                guess = FitTo(ExactSolution(); method = OrthogonalGreedy()))
            @test length(result.traces) == NETWORK_STEPS
            @test length(result.stats.iterations) == NETWORK_STEPS
            @test all(isfinite, result.solution.u)
            deviation = sup_deviation(result, entry)
            @test deviation ≤ bound
        end
        # the three-step SineGordon case of the entry
        problem = SineGordon.problem(
            T = T, timespan = (zero(T), T(0.15)), timestep = T(0.05),
            xstep = T(0.05))
        method = SpaceTimeGalerkin(network_ansatz(T), WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(ExactSolution(); method = OrthogonalGreedy()))
        result = integrate(GeometricIntegrator(problem, method))
        @test length(result.traces) == 3
        @test sup_deviation(result, SINE_GORDON_NN_FULLY) ≤
              NETWORK_SUP_BOUND.SINE_GORDON_NN_FULLY
    end
end
