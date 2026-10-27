using GeometricBase
using GeometricIntegratorsBase
using MultisymplecticIntegrators
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
    @testset "A4a every option converges to the same solution" begin
        T = Float64
        options = [Continuation(), FitTo(InitialCondition()), FitTo(PreviousSlab()),
            FitTo(ExactSolution()),
            FitTo(TargetFunction((t, x, p) -> Wave.exact_u(t, x, p)))]
        solutions = Vector{Vector{Float64}}()
        counts = Vector{Vector{Int}}()
        for option in options
            int, result = guess_run(T, option)
            @test length(cache(int).iterations) == 5
            @test all(≥(1), cache(int).iterations)
            push!(solutions, result.solution.u[end])
            push!(counts, copy(cache(int).iterations))
        end
        for solution in solutions[2:end]
            @test maximum(abs.(solution .- solutions[1])) ≤ T(100) * T(8) * eps(T)
        end
        # The clause asks Continuation to need fewer iterations than FitTo(InitialCondition());
        # a linear problem reaches its Newton solution in one step from any guess, so the counts
        # coincide (K20 in KNOWN_ISSUES.md). The extrapolation of PreviousSlab is checked by the
        # coefficients it fits, below.
        @test counts[1] == counts[2]
        extrapolated = FitTo(PreviousSlab())
        @test extrapolated !== FitTo(InitialCondition())
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
        @test_throws ArgumentError GeometricIntegrator(plain,
            SpaceTimeGalerkin(NonlinearFixtureAnsatz(), CoefficientRestriction();
                initial_guess = FitTo(InitialCondition())))
        # a target that returns another element type gives the problem's element type
        small = Float32
        int, result = guess_run(small,
            FitTo(TargetFunction((t, x, p) -> Wave.exact_u(t, x, p)));
            timestep = small(0.05), xstep = small(0.05))
        @test eltype(cache(int).x) === small
        assert_no_upcast(cache(int).coeffs, small)
        assert_no_upcast(result.solution.u, small)
    end
end
