using ForwardDiff
using GeometricBase: timesteps
using GeometricEquations
using GeometricIntegratorsBase: GeometricIntegrator, cache, reset!, solutionstep
using GeometricSolutions: GeometricSolution
using LinearAlgebra: rank, svdvals
using MultisymplecticIntegrators
using SimpleSplines: Free
using Test

include("../helpers/testsetup.jl")

# The ansatz of the transport measurement: two time cells and `cells` space cells of degree 2, three
# Gauss–Legendre nodes per cell in each direction, and a free boundary on `xspan = (-0.5, 0.0)`,
# which leaves the inflow end to the problem.
function transport_ansatz(::Type{T}, cells) where {T}
    TensorBSplineAnsatz(degree = (2, 2),
        ncells = (2, cells), xspan = (T(-0.5), zero(T)), n = 3, x_boundary = Free())
end

# The transport problem on the mesh of `cells` cells, integrated under `coupling` to `t = 2`.
function transport_run(::Type{T}, cells, coupling) where {T}
    problem = LinearTransport.problem(T = T, timespan = (zero(T), T(2)),
        timestep = T(2) / cells, xstep = T(0.5) / cells)
    result = integrate(problem, SpaceTimeGalerkin(transport_ansatz(T, cells), coupling))
    assert_no_upcast(result.solution.u[end], T)
    return problem, result
end

# The error of component 1 at the final time over the three meshes of the measurement, on the whole
# grid and on the part of it that the initial data alone determines.
function transport_errors(::Type{T}, coupling) where {T}
    whole, masked, iterations, converged = T[], T[], Int[], Bool[]
    for cells in (5, 10, 20)
        problem, result = transport_run(T, cells, coupling)
        grid = collect(xgrid(problem))
        exact = problem.solutions.u(problem.timespan[2], grid, problem.params)
        nx = length(grid)
        error = result.solution.u[end][1:nx] .- exact[1:nx]
        keep = grid .>= grid[1] + problem.params.c * problem.timespan[2]
        push!(whole, maximum(abs, error))
        push!(masked, maximum(abs, error[keep]))
        push!(iterations, maximum(result.stats.iterations))
        push!(converged, all(result.stats.converged))
    end
    return whole, masked, iterations, converged
end

# The Jacobian of the first 20-cell slab of `CoefficientRestriction` at a fixed point of the
# coefficient space, with its size, its rank and its extreme singular values.
function transport_jacobian(::Type{T}) where {T}
    problem, _ = transport_run(T, 20, CoefficientRestriction())
    method = SpaceTimeGalerkin(transport_ansatz(T, 20), CoefficientRestriction())
    int = GeometricIntegrator(problem, method)
    sol = GeometricSolution(problem)
    step = solutionstep(int, sol[0])
    reset!(step, timesteps(sol)[1])
    function residual(y)
        b = zeros(eltype(y), length(y))
        MultisymplecticIntegrators.residual!(b, y, step, problem, int)
        return b
    end
    unknowns = length(cache(int).x)
    J = ForwardDiff.jacobian(residual, fill(T(0.01), unknowns))
    σ = svdvals(J)
    return (; unknowns, rank = rank(J), smallest = σ[end], largest = σ[1])
end

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

    @testset "the doubled field integrates under both couplings" begin
        # The three meshes, under both couplings. `CoefficientRestriction` imposes the
        # bottom row and no side datum, so the initial data alone determine the field on
        # `x ≥ xspan[1] + c t` and the order is read there; `WeakInitialAndBoundaryConditions`
        # imposes the inflow value as well, and its order is read on the whole grid.
        for T in (Float64, Float32)
            _, masked, restriction_iterations, restriction_converged = transport_errors(
                T, CoefficientRestriction())
            whole, _, multiplier_iterations, multiplier_converged = transport_errors(
                T, WeakInitialAndBoundaryConditions())
            @test all(restriction_converged) && all(multiplier_converged)
            @test maximum(restriction_iterations) <= 2
            @test maximum(multiplier_iterations) <= 2
            # An observed order needs an error above the precision's rounding floor; at 20 cells
            # the masked error is `1.1e-8` in `Float64` and `3.9e-8` against `eps(Float32) = 1.2e-7`
            # in `Float32`, so the two orders are asserted in `Float64` alone.
            if T === Float64
                @test log2(masked[end - 1] / masked[end]) >= 1.9
                @test log2(whole[end - 1] / whole[end]) >= 1.9
            end
        end

        # The Legendre map of the doubled density is degenerate (`p_u = ψ`, `p_ψ = 0`), so the slab
        # step's solvability is measured: the first 20-cell slab of `CoefficientRestriction` is
        # non-singular and not ill-conditioned.
        for T in (Float64, Float32)
            jacobian = transport_jacobian(T)
            @test jacobian.rank == jacobian.unknowns
            @test jacobian.smallest / jacobian.largest >= 1e-3
        end
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
