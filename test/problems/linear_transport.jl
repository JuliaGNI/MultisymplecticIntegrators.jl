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
        @test problem.D == 2
        @test size(problem.ics.u) == (length(xgrid(problem)), 2)
        @test all(iszero, problem.ics.u[:, 2])
    end

    @testset "the doubled density and its derivatives" begin
        params = (c = 0.2,)
        density = LinearTransport.density(; params = params)
        # The field `(u, ψ)`, and its derivatives, at one point.
        u = [0.3, 1.1]
        v = [0.7, 0.4]
        w = [-0.2, 0.6]
        # The closed form `L = ψ (v₁ + c w₁)` and its three partials, written here.
        @test density(u, v, w, params) ≈ 1.1 * (0.7 + params.c * -0.2)
        @test collect(derivatives(density, u, v, w, params).u) ≈
              [0.0, 0.7 + params.c * -0.2]
        @test collect(derivatives(density, u, v, w, params).v) ≈ [1.1, 0.0]
        @test collect(derivatives(density, u, v, w, params).w) ≈ [params.c * 1.1, 0.0]
        @test density.params == params
    end

    @testset "the exact solution is the travelling profile" begin
        params = (c = 0.2,)
        for t in (0.0, 0.5, 1.0), x in (-0.5, -0.25, 0.0)

            ξ = x + 2 - params.c * t
            u = LinearTransport.solution_u(t, x, params)
            v = LinearTransport.solution_v(t, x, params)
            w = LinearTransport.solution_w(t, x, params)
            # component 1 is the transported field, component 2 the adjoint field's zero data
            @test u[1] ≈ 0.5 * exp(-ξ^2) / sqrt(π)
            @test v[1] ≈ params.c * ξ * exp(-ξ^2) / sqrt(π)
            @test w[1] ≈ -ξ * exp(-ξ^2) / sqrt(π)
            @test iszero(u[2]) && iszero(v[2]) && iszero(w[2])
            # `v` and `w` are the derivatives of the profile, so `v = −c w` on it.
            @test v[1] ≈ -params.c * w[1]
        end
    end

    @testset "the exact functions see the problem's parameters" begin
        fast = LinearTransport.problem(params = (c = 0.5,), timespan = (0.0, 1.0),
            timestep = 0.5, xstep = 0.25)
        slow = LinearTransport.problem(params = (c = 0.2,), timespan = (0.0, 1.0),
            timestep = 0.5, xstep = 0.25)
        @test fast.bcs_function(0.4, fast.params).left.u !=
              slow.bcs_function(0.4, slow.params).left.u
        @test fast.solutions.u(0.4, -0.25, fast.params) !=
              slow.solutions.u(0.4, -0.25, slow.params)
        @test fast.solutions.u(0.4, -0.25, fast.params)[1] ≈
              LinearTransport.solution_u(0.4, -0.25, (c = 0.5,))[1]
    end

    @testset "the solution shifts by the speed" begin
        params = (c = 0.2,)
        x = collect(xgrid(LinearTransport.problem()))
        u₀ = [LinearTransport.solution_u(0.0, xi, params)[1] for xi in x]
        u₁ = [LinearTransport.solution_u(1.0, xi + params.c, params)[1] for xi in x]
        @test u₀ ≈ u₁ rtol = 16 * eps(Float64)
    end

    @testset "a Float32 problem evaluates its energy in Float32" begin
        T = Float32
        problem = LinearTransport.problem(
            T = T, timespan = (zero(T), T(0.1)), timestep = T(0.05), xstep = T(0.125))
        @test all(value -> typeof(value) === T, problem.params)
        ansatz = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 4), xspan = xspan(problem), n = 3)
        weights = MultisymplecticIntegrators.space_quadrature_weights(ansatz)
        entries = problem.D * length(weights)
        trace = (u = fill(T(0.3), entries), v = fill(T(0.7), entries),
            w = fill(T(-0.2), entries))
        energy = @inferred MultisymplecticIntegrators.step_energy(
            problem.density, problem.params, weights, trace, 1, Val(problem.D))
        @test energy isa T
        @test energy != zero(T)
    end
end
