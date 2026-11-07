using MultisymplecticIntegrators
using Symbolics
using Test

include("../helpers/testsetup.jl")

@testset "Lagrangian density" begin
    u = Symbolics.variable(:u)
    v = Symbolics.variable(:v)
    w = Symbolics.variable(:w)
    c = Symbolics.variable(:c)

    @testset "the wave density against its closed form" begin
        density = LagrangianDensity(v^2 / 2 - c^2 * w^2 / 2, u, v, w; params = (c = 0.5,))
        points = ((0.3, 0.7, -0.2), (-1.0, 2.0, 0.5), (0.0, 0.0, 1.0), (2.5, -0.25, 0.75),
            (-0.5, 1.5, -1.5))
        for T in (Float32, Float64), (u₀, v₀, w₀) in points

            uₜ, vₜ, wₜ = T(u₀), T(v₀), T(w₀)
            @test derivative_u(density, uₜ, vₜ, wₜ) ≈ zero(T) atol = tolerance(T)
            @test derivative_v(density, uₜ, vₜ, wₜ) ≈ vₜ rtol = tolerance(T)
            @test derivative_w(density, uₜ, vₜ, wₜ) ≈ -(T(0.5)^2) * wₜ rtol = tolerance(T)
            @test eltype(derivatives(density, uₜ, vₜ, wₜ).v) === T
        end
    end

    @testset "the parameters reach every built function" begin
        density = LagrangianDensity(v^2 / 2 - c^2 * w^2 / 2, u, v, w; params = (c = 0.5,))
        @test derivative_w(density, 0.0, 0.0, 1.0, (c = 2.0,)) ≈ -4.0
        @test derivative_v(density, 0.0, 3.0, 0.0, (c = 2.0,)) ≈ 3.0
        @test density(0.0, 2.0, 1.0, (c = 2.0,)) ≈ 2.0 - 2.0
    end

    @testset "the sine-Gordon density" begin
        density = SineGordon.density()
        for T in (Float32, Float64), u₀ in (T(0.5), T(-1.25), T(0.0))

            @test derivative_u(density, u₀, zero(T), zero(T)) ≈ -sin(u₀) rtol = tolerance(T)
            @test derivative_v(density, u₀, T(2), zero(T)) ≈ T(4) * T(2) rtol = tolerance(T)
            @test derivative_w(density, u₀, zero(T), T(3)) ≈ -T(3) rtol = tolerance(T)
        end
    end

    @testset "the transport density" begin
        # The doubled field's density, `L = ψ (v₁ + c w₁)`: component 1 is the transported field,
        # whose velocity and slope enter it, and component 2 the adjoint field, whose value does.
        points = ((0.3, 1.1, 0.7, -0.2), (-1.0, 0.4, 2.0, 0.5), (0.0, 2.5, 0.0, 1.0))
        for T in (Float32, Float64), (u₁, u₂, v₁, w₁) in points

            c = T(0.2)
            density = LinearTransport.density(; params = (c = c,))
            u, v, w = T[u₁, u₂], T[v₁, 0.6], T[w₁, -0.25]
            @test density(u, v, w) ≈ T(u₂) * (T(v₁) + c * T(w₁)) rtol = tolerance(T)
            partials = derivatives(density, u, v, w)
            @test collect(partials.u) ≈ [zero(T), T(v₁) + c * T(w₁)] atol = tolerance(T)
            @test collect(partials.v) ≈ [T(u₂), zero(T)] atol = tolerance(T)
            @test collect(partials.w) ≈ [c * T(u₂), zero(T)] atol = tolerance(T)
            @test typeof(density(u, v, w)) === T
        end
    end
end
