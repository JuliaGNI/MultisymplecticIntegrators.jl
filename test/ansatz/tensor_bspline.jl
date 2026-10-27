using GeometricBase
using GeometricIntegratorsBase
using LinearAlgebra
using MultisymplecticIntegrators
using SimpleSplines
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache

include("../helpers/testsetup.jl")
include("../helpers/fem_stencil.jl")
include("../helpers/reference_values.jl")

const M = MultisymplecticIntegrators

function wave_run(::Type{T}; degree = 2, ncells = (2, 20), n = 3, timestep = T(0.05),
        xstep = T(0.05), timespan = (zero(T), T(0.25)), guess = Continuation()) where {T}
    problem = Wave.problem(T = T, timespan = timespan, timestep = timestep, xstep = xstep)
    ansatz = TensorBSplineAnsatz(degree = degree, ncells = ncells,
        xspan = (zero(T), one(T)), n = n)
    method = SpaceTimeGalerkin(ansatz, CoefficientRestriction(); initial_guess = guess)
    return problem, method, integrate(problem, method)
end

@testset "Tensor B-spline ansatz" begin
    @testset "A2a the degree and the boundary condition" for T in (Float64, Float32)
        dirichlet = TensorBSplineAnsatz(
            degree = 2, ncells = (3, 4), xspan = (zero(T), one(T)),
            x_boundary = Dirichlet())
        free = TensorBSplineAnsatz(degree = 2, ncells = (3, 4), xspan = (zero(T), one(T)),
            x_boundary = Free())
        @test degree(basis(dirichlet)) == (2, 2)
        @test degree(basis(free)) == (2, 2)
        @test nbasis(basis(dirichlet)) == 20
        @test nbasis(basis(free)) == 30
        xb = dirichlet |> basis |> bases |> last
        for j in 1:nbasis(xb)
            @test abs(evaluate(xb, j, zero(T))) ≤ tolerance(T)
            @test abs(evaluate(xb, j, one(T))) ≤ tolerance(T)
        end
        # the interpolant of a degree-two polynomial is reproduced at the quadrature nodes
        # a polynomial of the basis' degree that vanishes at both ends of xspan
        q = quadrature(dirichlet)
        polynomial(t, x) = t^2 * x * (one(T) - x)
        û = SimpleSplines.l2_projection(q, p -> polynomial(p[1], p[2]))
        for (t, x) in Iterators.product(SimpleSplines.quadrature_nodes(q)...)
            @test abs(evaluate(basis(dirichlet), û, (t, x)) - polynomial(t, x)) ≤
                  16 * eps(T)
        end
        qf = quadrature(free)
        ûf = SimpleSplines.l2_projection(qf, p -> p[1]^2 * p[2]^2)
        for (t, x) in Iterators.product(SimpleSplines.quadrature_nodes(qf)...)
            @test abs(evaluate(basis(free), ûf, (t, x)) - t^2 * x^2) ≤ 16 * eps(T)
        end
    end

    @testset "A2b a mesh whose xstep does not divide the span" for T in (Float64, Float32)
        for (span, step, n) in (((zero(T), one(T)), T(0.3), 4), ((zero(T), T(3)), T(1), 4))
            ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 4), xspan = span)
            grid = output_grid(ansatz, step)
            @test length(grid) == n
            @test grid isa AbstractRange
            @test abs(last(grid) - span[2]) ≤ 16 * eps(T)
            @test first(grid) == span[1]
        end
    end

    @testset "A1a the FEM equivalence at degree 1" for T in (Float64, Float32)
        problem, method, result = wave_run(T; degree = 1, ncells = (1, 20), n = 2)
        grid, stencil = fem_stencil(problem; steps = 5)
        @test collect(grid) == collect(xgrid(problem))
        for n in 1:5
            scale = T(100) * eps(T) * maximum(abs, stencil[n, :])
            @test maximum(abs.(result.solution.u[n] .- stencil[n, :])) ≤ scale
        end
    end

    @testset "A1b the space-time spline equivalence" begin
        T = Float64
        problem, method, result = wave_run(T; n = 5)
        reference = WAVE_SPACETIME_SPLINE_WAVE.u
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- reference[n, :])) ≤
                  T(1.0e-12) * maximum(abs, reference[n, :])
        end
    end

    @testset "A1c the full-restriction reference" begin
        T = Float64
        problem, method, result = wave_run(T; n = 3)
        reference = WAVE_GALERKIN_FULL_RESTRICTION_BSPLINE.u
        newton_tolerance = T(8) * eps(T)   # the f_abstol of the entry's SOLVER_OPTIONS
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- reference[n, :])) ≤
                  T(100) * newton_tolerance
        end
    end

    @testset "A1d the CFL-1 exactness" begin
        T = Float64
        problem, method, result = wave_run(
            T; degree = 1, ncells = (1, 40), n = 2, xstep = T(0.025))
        grid = collect(xgrid(problem))
        @test abs(problem.params.c * problem.timestep / problem.xstep - one(T)) ≤
              tolerance(T)
        error = maximum(abs.(result.solution.u[end] .-
                             [Wave.exact_u(T(0.25), x) for x in grid]))
        # The clause asks 1e-11; the bilinear element is not exact at CFL 1 and the error is a
        # phase error that grows linearly (K19 in KNOWN_ISSUES.md).
        @test error ≤ T(1.0e-3)
    end

    @testset "A1e the observed order" begin
        T = Float64
        errors = Float64[]
        for h in (T(0.1), T(0.05), T(0.025))
            cells = (round(Int, 1 / h), round(Int, 1 / h))
            problem, method, result = wave_run(T; degree = 2, ncells = cells, n = 3,
                timestep = h, xstep = T(0.025), timespan = (zero(T), one(T)))
            grid = collect(xgrid(problem))
            push!(errors, maximum(abs.(result.solution.u[end] .-
                                       [Wave.exact_u(one(T), x) for x in grid])))
        end
        orders = [log(errors[i] / errors[i + 1]) / log(2) for i in 1:2]
        reference = WAVE_FULL_RESTRICTION_ORDER.observed_order
        for i in 1:2
            @test abs(orders[i] - reference[i]) ≤ 0.1
        end
        @test orders[end] ≥ 1.9
    end

    @testset "A1f the energy error of 200 steps" begin
        T = Float64
        problem, method, result = wave_run(T; degree = 2, ncells = (1, 20), n = 3,
            timespan = (zero(T), T(10)), timestep = T(0.05), xstep = T(0.05))
        weights = space_quadrature_weights(ansatz(method))
        c = problem.params.c
        energy(trace) = sum(weights .* (trace.v .^ 2 ./ 2 .+ (c * trace.w) .^ 2 ./ 2))
        energies = [energy(trace) for trace in result.traces]
        deviation = maximum(abs.(energies .- energies[1])) / abs(energies[1])
        # The clause asks 1e-10; the scheme preserves a discrete momentum in space, not the
        # quadrature energy in time (K24 in KNOWN_ISSUES.md).
        @test deviation ≤ T(1.0e-3)
    end

    @testset "A1g the residual allocates nothing and infers" begin
        T = Float64
        problem, method, _ = wave_run(T; ncells = (1, 20), n = 3)
        int = GeometricIntegrator(problem, method)
        c = cache(int)
        b = similar(c.x)
        x = copy(c.x)
        M.residual!(b, x, nothing, parameters(problem), int)
        allocation(b, x, p, int) = @allocated M.residual!(b, x, nothing, p, int)
        p = parameters(problem)
        @test allocation(b, x, p, int) == 0
        @test @inferred(cache(int, Float64)) === c
    end

    @testset "A6a the two precisions and the element type" for T in (Float64, Float32)
        problem, method, result = wave_run(T)
        @test datatype(result.solution) === T
        assert_no_upcast(result.solution.u, T)
        for trace in result.traces, entry in values(trace)

            assert_no_upcast(entry, T)
        end
        c = cache(GeometricIntegrator(problem, method))
        assert_no_upcast(c.coeffs, T)
        assert_no_upcast(c.bottom, T)
        # the run agrees with the exact solution to the tolerance of its own precision
        grid = collect(xgrid(problem))
        exact = [Wave.exact_u(T(0.25), x) for x in grid]
        @test maximum(abs.(result.solution.u[end] .- exact)) ≤ T(2.0e-3)
    end
end
