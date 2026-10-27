using GeometricBase
using GeometricIntegratorsBase
using MultisymplecticIntegrators
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")

const M = MultisymplecticIntegrators

function slab_run(::Type{T}; degree = 2, ncells = (2, 20), n = 3, guess = Continuation(),
        problem = Wave.problem(T = T, timespan = (zero(T), T(0.25)), timestep = T(0.05),
            xstep = T(0.05))) where {T}
    ansatz = TensorBSplineAnsatz(degree = degree, ncells = ncells,
        xspan = (zero(T), one(T)), n = n)
    method = SpaceTimeGalerkin(ansatz, CoefficientRestriction(); initial_guess = guess)
    int = GeometricIntegrator(problem, method)
    return problem, method, int, integrate(int)
end

@testset "Coefficient restriction" begin
    @testset "A1c the full-restriction reference" begin
        T = Float64
        problem, method, int, result = slab_run(T)
        reference = WAVE_GALERKIN_FULL_RESTRICTION_BSPLINE.u
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- reference[n, :])) ≤
                  T(100) * T(8) * eps(T)
        end
    end

    @testset "A3a the coupling carries the top trace across a step" begin
        T = Float64
        problem, method, int, result = slab_run(T)
        c = cache(int)
        @test length(c.iterations) == 5
        @test all(≥(1), c.iterations)
        @test length(c.residuals) == 5
        reference = WAVE_GALERKIN_FULL_RESTRICTION_BSPLINE.u
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- reference[n, :])) ≤
                  T(100) * T(8) * eps(T)
        end
        # the top trace of a slab is the bottom datum of the next one
        @test c.flux == c.residual[size(c.residual, 1), :, :]
        @test result.traces[end].u == vec(c.trace_u)
        @test length(result.traces) == 5
    end

    @testset "A5a the two components and the discrete momentum" for T in (Float64, Float32)
        params = VectorWave.default_parameters()
        problem = VectorWave.problem(T = T, params = params, timespan = (zero(T), T(0.25)),
            timestep = T(0.05), xstep = T(0.05))
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
            xspan = (zero(T), one(T)), n = 3)
        method = SpaceTimeGalerkin(ansatz, CoefficientRestriction())
        result = integrate(problem, method)
        @test components(problem) == 2
        assert_no_upcast(result.solution.u, T)
        # each component equals its own scalar run with the component's parameters
        for d in 1:2
            scalar = Wave.problem(
                T = T, params = VectorWave.component_parameters(params, d),
                timespan = (zero(T), T(0.25)), timestep = T(0.05), xstep = T(0.05))
            one_d = integrate(scalar, SpaceTimeGalerkin(ansatz, CoefficientRestriction()))
            component = result.solution.u[end][(d - 1) * length(xgrid(problem)) .+ (1:length(xgrid(problem)))]
            @test maximum(abs.(component .- one_d.solution.u[end])) ≤
                  T(10) * eps(T) * maximum(abs, one_d.solution.u[end])
        end
    end
end
