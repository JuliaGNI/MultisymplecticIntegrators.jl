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
        params = VectorWave.default_parameters()
        phase = params.A1 + params.B1 * sin(π / 6)
        @test problem.ics.u[1:length(x)] ≈ [phase * sin(π * xi / params.l) for xi in x]
    end

    @testset "the vector density is the scalar one per component" begin
        params = (c = 0.5,)
        scalar = Wave.density(; params = params, m = 0.7)
        vector = VectorWave.vector_density(; params = params, m = 0.7, D = 2)
        u = [0.7, -1.3]
        v = [0.2, 0.9]
        w = [-0.4, 0.1]
        got = VectorWave.derivatives(vector, u, v, w, params)
        for i in eachindex(u)
            one = derivatives(scalar, u[i], v[i], w[i], params)
            @test got.u[i] ≈ one.u
            @test got.v[i] ≈ one.v
            @test got.w[i] ≈ one.w
        end
        @test got.u[1] != 0.0
        @test vector(u, v, w, params) ≈
              sum(scalar(u[i], v[i], w[i], params) for i in eachindex(u))
        @test_throws ArgumentError VectorWave.derivatives(
            vector, [0.7], [0.2], [-0.4], params)
        @test_throws ArgumentError vector([0.7], [0.2], [-0.4], params)
    end

    @testset "the angular momentum of the exact solution is conserved" begin
        x = collect(0.0:0.05:1.0)
        p₁ = (c = 0.5, A1 = 0.8, A2 = 0.0, B1 = 0.4, B2 = 0.0, l = 1.0)
        p₂ = (c = 0.5, A1 = 0.8, A2 = 0.0, B1 = 0.0, B2 = 0.0, l = 1.0)
        momentum(t) = begin
            a = Wave.initial_data(x, p₁, 1, t)
            b = Wave.initial_data(x, p₂, 1, t)
            VectorWave.angular_momentum(a.u, b.u, a.v, b.v, x)
        end
        j₀ = momentum(0.0)
        # The closed form of `J = ∫ (u₁v₂ − u₂v₁) dx` for this data, whose second component has
        # `B₁ = 0` and therefore a vanishing velocity at t = 0.
        ω = π * p₁.c / p₁.l
        @test j₀ ≈ -p₁.A1 * p₁.B1 * ω * cos(π / 6) / 2 rtol = 16 * eps(Float64)
        @test momentum(0.35) ≈ j₀ rtol = 16 * eps(Float64)
        @test momentum(1.25) ≈ j₀ rtol = 16 * eps(Float64)
    end
end
