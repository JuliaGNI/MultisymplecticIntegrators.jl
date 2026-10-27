using GeometricBase
using GeometricIntegratorsBase
using MultisymplecticIntegrators
using SimpleSplines
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")

const M = MultisymplecticIntegrators

"""
An ansatz that is not linear in its coefficients: it stands in for the network ansatz of a later
part, whose `Projection()` must be refused.
"""
struct NonlinearFixtureAnsatz <: SpaceTimeAnsatz end

function guess_run(::Type{T}, guess; timestep = T(0.05), xstep = T(0.05),
        problem = Wave.problem(T = T, timespan = (zero(T), T(0.25)), timestep = timestep,
            xstep = xstep)) where {T}
    ansatz = TensorBSplineAnsatz(degree = 2, ncells = (2, 20),
        xspan = (zero(T), one(T)), n = 3)
    method = SpaceTimeGalerkin(ansatz, CoefficientRestriction(); initial_guess = guess)
    int = GeometricIntegrator(problem, method)
    return int, integrate(int)
end

@testset "Initial guess" begin
    @testset "A4a every option converges to the same solution" for T in (Float64, Float32)
        newton_tolerance = T(SOLVER_OPTIONS.f_abstol / eps(Float64)) * eps(T)
        options = [Continuation(), FitTo(InitialCondition()), FitTo(PreviousSlab()),
            FitTo(ExactSolution()),
            FitTo(TargetFunction((t, x, p) -> Wave.exact_u(t, x, p)))]
        solutions = Vector{Vector{T}}()
        for option in options
            int, result = guess_run(T, option)
            c = cache(int)
            @test length(c.iterations) == 5
            @test length(c.initial_residuals) == 5
            @test all(≥(1), c.iterations)
            @test all(r -> isfinite(r) && r ≥ zero(T), c.initial_residuals)
            push!(solutions, result.solution.u[end])
        end
        for solution in solutions[2:end]
            @test maximum(abs.(solution .- solutions[1])) ≤ T(100) * newton_tolerance
        end
    end

    @testset "A4c the continuation's guess and the extrapolation's value" for T in (Float64, Float32)
        continuation, _ = guess_run(T, Continuation())
        initial, _ = guess_run(T, FitTo(InitialCondition()))
        rc = cache(continuation).initial_residuals
        ri = cache(initial).initial_residuals
        @test length(rc) == 5
        @test length(ri) == 5
        # the carried top row is the better guess from the second step on
        for n in 2:5
            @test rc[n] < ri[n]
        end
        # the extrapolation is the polynomial piece of the top time cell, about its top
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (3, 20),
            xspan = (zero(T), one(T)), n = 3)
        problem = Wave.problem(T = T, timespan = (zero(T), T(0.25)), timestep = T(0.05),
            xstep = T(0.05))
        c = cache(GeometricIntegrator(problem,
            SpaceTimeGalerkin(ansatz, CoefficientRestriction();
                initial_guess = FitTo(PreviousSlab()))))
        field(t̂, x) = (one(T) + t̂ + t̂^2) * x * (one(T) - x)
        c.coeffs[:, :, 1] .= SimpleSplines.l2_projection(quadrature(ansatz),
            p -> field(p[1], p[2]))
        c.steps[] = ncells(ansatz)[1]
        qt = M.nquadrature(c.tabulation.time)
        qx = M.nquadrature(c.tabulation.space)
        sample = Array{T, 3}(undef, qt, qx, 1)
        M.evaluate!(sample, PreviousSlab(), c, ansatz, problem, nothing)
        xq = space_quadrature_nodes(ansatz)
        scale = T(100) * eps(T) * maximum(abs, sample)
        for q in 1:qt, r in 1:qx

            u = one(T) + c.tabulation.time.node[q]
            @test abs(sample[q, r, 1] - field(u, xq[r])) ≤ scale
        end
    end

    @testset "A4b the two refusals and the element type of a target" begin
        T = Float64
        plain = Wave.problem(T = T, timespan = (zero(T), T(0.1)), timestep = T(0.05),
            xstep = T(0.05))
        without_exact = LagrangianPDEProblem(plain.density; D = 1,
            ics_function = plain.ics_function, bcs_function = plain.bcs_function,
            timespan = plain.timespan, timestep = plain.timestep, xspan = plain.xspan,
            xstep = plain.xstep, params = plain.params)
        @test without_exact.exact_u === nothing
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
            xspan = (zero(T), one(T)), n = 3)
        @test_throws ArgumentError GeometricIntegrator(without_exact,
            SpaceTimeGalerkin(ansatz, CoefficientRestriction();
                initial_guess = FitTo(ExactSolution())))
        # the projection's refusal names the reason: an ansatz that is not linear in its
        # coefficients. The fixture also makes the cache refuse its ansatz, so an assertion on the
        # type alone would pass on the cache's own `ArgumentError`.
        nonlinear = SpaceTimeGalerkin(NonlinearFixtureAnsatz(), CoefficientRestriction();
            initial_guess = FitTo(InitialCondition()))
        @test_throws "Projection() needs an ansatz that is linear in its coefficients" GeometricIntegrator(
            plain, nonlinear)
        # the control: a guess that is no projection passes the same check
        @test M.check_method(plain,
            SpaceTimeGalerkin(NonlinearFixtureAnsatz(), CoefficientRestriction())) ===
              nothing
        # a target that returns another element type gives the problem's element type
        small = Float32
        int, result = guess_run(small,
            FitTo(TargetFunction((t, x, p) -> Float64(Wave.exact_u(Float64(t), Float64(x), p))));
            timestep = small(0.05), xstep = small(0.05))
        @test eltype(cache(int).x) === small
        assert_no_upcast(cache(int).coeffs, small)
        assert_no_upcast(result.solution.u, small)
        # a target is evaluated in physical time: the step that ends at `t = 0.1` starts at
        # `t = 0.05`, so the sample of `f(t, x) = t` is `0.05 + 0.05 t̂`
        c = cache(GeometricIntegrator(plain,
            SpaceTimeGalerkin(ansatz, CoefficientRestriction())))
        qt, qx = M.nquadrature(c.tabulation.time), M.nquadrature(c.tabulation.space)
        sample = Array{T, 3}(undef, qt, qx, 1)
        M.target_values!(sample, (t, x, p) -> t, c.tabulation, ansatz, plain, (t = T(0.1),))
        for q in 1:qt, r in 1:qx

            @test abs(sample[q, r, 1] - (T(0.05) + T(0.05) * c.tabulation.time.node[q])) ≤
                  tolerance(T)
        end
    end
end
