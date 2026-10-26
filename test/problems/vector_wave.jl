using MultisymplecticIntegrators
using Test

include("../helpers/testsetup.jl")

@testset "Vector wave" begin
    @testset "the two components are carried component-major" begin
        problem = VectorWave.problem(timespan = (0.0, 0.4), timestep = 0.1, xstep = 0.25)
        x = xgrid(problem)
        @test problem.D == 2
        @test length(problem.ics.u) == 2 * length(x)
        @test problem.ics.u[1:length(x)] ≈ problem.ics.u[(length(x) + 1):end]
        @test problem.ics.u[1:length(x)] ≈
              Wave.exact_u(0.0, x, VectorWave.default_parameters())
    end

    @testset "the vector density is the scalar one per component" begin
        params = (c = 0.5, m = 0.0, β = 0.0)
        scalar = Wave.density(; params = params)
        vector = VectorWave.vector_density(; params = params, D = 2)
        u = [0.7, -1.3]
        v = [0.2, 0.9]
        w = [-0.4, 0.1]
        dscalar = derivatives(scalar, u[1], v[1], w[1], params)
        @test VectorWave.derivatives(vector, u, v, w, params).u[1] ≈ dscalar.u
        @test VectorWave.derivatives(vector, u, v, w, params).v[2] ≈
              derivatives(scalar, u[2], v[2], w[2], params).v
        @test vector(u, v, w, params) ≈
              sum(scalar(u[i], v[i], w[i], params) for i in eachindex(u))
        @test_throws ArgumentError VectorWave.derivatives(
            vector, [0.7], [0.2], [-0.4], params)
    end

    @testset "the angular momentum of the exact solution is conserved" begin
        x = collect(0.0:0.05:1.0)
        first = (c = 0.5, A1 = 0.8, A2 = 0.0, B1 = 0.4, B2 = 0.0, l = 1.0)
        second = (c = 0.5, A1 = 0.8, A2 = 0.0, B1 = 0.0, B2 = 0.0, l = 1.0)
        momentum(t) = begin
            a = Wave.initial_data(x, first, 1, t)
            b = Wave.initial_data(x, second, 1, t)
            VectorWave.angular_momentum(a.u, b.u, a.v, b.v, x)
        end
        j₀ = momentum(0.0)
        @test j₀ != 0.0
        @test momentum(0.35) ≈ j₀ rtol = 16 * eps(Float64)
        @test momentum(1.25) ≈ j₀ rtol = 16 * eps(Float64)
    end
end
