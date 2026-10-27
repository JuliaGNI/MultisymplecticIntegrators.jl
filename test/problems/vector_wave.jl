using MultisymplecticIntegrators
using Test

include("../helpers/testsetup.jl")

@testset "Vector wave" begin
    @testset "each component carries its own standing wave, component-major" begin
        problem = VectorWave.problem(timespan = (0.0, 0.4), timestep = 0.1, xstep = 0.25)
        params = VectorWave.default_parameters()
        x = xgrid(problem)
        n = length(x)
        ω = π * params.c / params.l
        phase(i, t) = params.A1[i] * cos(ω * t) + params.B1[i] * sin(ω * t + π / 6)
        @test problem.D == 2
        @test length(problem.ics.u) == 2 * n
        for i in 1:2
            block = ((i - 1) * n + 1):(i * n)
            @test problem.ics.u[block] ≈
                  [phase(i, 0.0) * sin(π * xi / params.l) for xi in x]
            @test problem.ics.v[block] ≈
                  [params.B1[i] * ω * cos(π / 6) * sin(π * xi / params.l) for xi in x]
        end
        @test !(problem.ics.u[1:n] ≈ problem.ics.u[(n + 1):end])

        times = (0.0, 0.3)
        data = problem.bcs_function(times, params)
        @test data.bottom.u ≈ problem.ics.u
        @test data.top.u ≈
              [phase(i, times[2]) * sin(π * xi / params.l) for i in 1:2 for xi in x]
        @test data.left.w ≈ [phase(i, t) * π / params.l for i in 1:2 for t in times]
        @test data.right.w ≈ [-phase(i, t) * π / params.l for i in 1:2 for t in times]

        @test problem.exact_u(0.0, x, params) ≈ problem.ics.u
        @test problem.exact_u(0.2, 0.3, params) ≈
              [phase(i, 0.2) * sin(π * 0.3 / params.l) for i in 1:2]
        @test_throws ArgumentError VectorWave.problem(params = merge(params, (B2 = (0.0,),)))
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
            each = derivatives(scalar, u[i], v[i], w[i], params)
            @test got.u[i] ≈ each.u
            @test got.v[i] ≈ each.v
            @test got.w[i] ≈ each.w
        end
        @test got.u[1] != 0.0
        @test vector(u, v, w, params) ≈
              sum(scalar(u[i], v[i], w[i], params) for i in eachindex(u))
        @test_throws ArgumentError VectorWave.derivatives(
            vector, [0.7], [0.2], [-0.4], params)
        @test_throws ArgumentError vector([0.7], [0.2], [-0.4], params)
    end

    @testset "the quartic coupling is invariant under a rotation" begin
        params = (c = 0.5,)
        m, β = 0.7, 1.3
        vector = VectorWave.vector_density(; params = params, m = m, β = β, D = 2)
        u = [0.7, -1.3]
        v = [0.2, 0.9]
        w = [-0.4, 0.1]
        r² = sum(abs2, u)
        @test vector(u, v, w, params) ≈
              sum(abs2, v) / 2 - params.c^2 * sum(abs2, w) / 2 - m^2 * r² / 2 - β * r²^2 / 4
        for θ in (π / 7, π / 4, 2.0)
            R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
            @test vector(R * u, R * v, R * w, params) ≈ vector(u, v, w, params) rtol = 16 *
                                                                                       eps(Float64)
        end
        got = VectorWave.derivatives(vector, u, v, w, params)
        @test got.u ≈ @. -m^2 * u - β * r² * u
        @test got.v ≈ v
        @test got.w ≈ -params.c^2 .* w
    end

    @testset "the angular momentum of the problem's exact solution" begin
        problem = VectorWave.problem(timespan = (0.0, 1.0), timestep = 0.1, xstep = 0.05)
        params = problem.params
        x = xgrid(problem)
        n = length(x)
        momentum(t) = begin
            u = problem.exact_u(t, x, params)
            v = problem.exact_v(t, x, params)
            VectorWave.angular_momentum(u[1:n], u[(n + 1):end], v[1:n], v[(n + 1):end], x)
        end
        # Each component is `aᵢ(t) sin(πx/l)`, so `J = (a₁a₂′ − a₂a₁′) l/2`; at `t = 0`,
        # `aᵢ = A1ᵢ + B1ᵢ/2` and `aᵢ′ = B1ᵢ ω cos(π/6)`.
        ω = π * params.c / params.l
        j₀ = ω * cos(π / 6) * params.l / 2 *
             (params.A1[1] * params.B1[2] - params.A1[2] * params.B1[1])
        @test abs(j₀) > 0.1
        @test momentum(0.0) ≈ j₀ rtol = 16 * eps(Float64)
        @test momentum(0.35) ≈ j₀ rtol = 16 * eps(Float64)
        @test momentum(1.25) ≈ j₀ rtol = 16 * eps(Float64)
    end
end
