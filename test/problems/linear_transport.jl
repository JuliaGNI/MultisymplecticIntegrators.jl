using GeometricBase
using GeometricEquations
using MultisymplecticIntegrators
using Test

include("../helpers/testsetup.jl")

@testset "Linear transport" begin
    problem = LinearTransport.problem(timespan = (0.0, 2.0), timestep = 0.5, xstep = 0.125)

    @testset "the problem builds on the decided interval" begin
        @test xspan(problem) == (-0.5, 0.0)
        @test xgrid(problem) ≈ collect(-0.5:0.125:0.0)
        @test xspan(problem)[1] == -0.5 && xspan(problem)[2] == 0.0
        @test ntime(problem) == 4
        @test problem.D == 1
        @test length(problem.ics.u) == length(xgrid(problem))
    end

    @testset "the exact solution is the travelling profile" begin
        params = (c = 0.2,)
        for t in (0.0, 0.5, 1.0), x in (-0.5, -0.25, 0.0)

            ξ = x + 2 - params.c * t
            @test LinearTransport.exact_u(t, x, params) ≈ 0.5 * exp(-ξ^2) / sqrt(π)
            @test LinearTransport.exact_v(t, x, params) ≈
                  params.c * ξ * exp(-ξ^2) / sqrt(π)
            @test LinearTransport.exact_w(t, x, params) ≈ -ξ * exp(-ξ^2) / sqrt(π)
        end
    end

    @testset "the exact functions see the problem's parameters" begin
        fast = LinearTransport.problem(params = (c = 0.5,), timespan = (0.0, 1.0),
            timestep = 0.5, xstep = 0.25)
        slow = LinearTransport.problem(params = (c = 0.2,), timespan = (0.0, 1.0),
            timestep = 0.5, xstep = 0.25)
        @test fast.bcs_function(0.4, fast.params).bottom.u !=
              slow.bcs_function(0.4, slow.params).bottom.u
        @test fast.exact_u(0.4, -0.25, fast.params) != slow.exact_u(0.4, -0.25, slow.params)
        @test fast.exact_u(0.4, -0.25, fast.params) ≈
              LinearTransport.exact_u(0.4, -0.25, (c = 0.5,))
    end

    @testset "the solution shifts by the speed" begin
        params = (c = 0.2,)
        x = collect(xgrid(LinearTransport.problem()))
        u₀ = [LinearTransport.exact_u(0.0, xi, params) for xi in x]
        u₁ = [LinearTransport.exact_u(1.0, xi + params.c, params) for xi in x]
        @test u₀ ≈ u₁ rtol = 16 * eps(Float64)
    end
end
