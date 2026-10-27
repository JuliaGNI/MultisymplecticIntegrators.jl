using GeometricBase
using GeometricIntegratorsBase
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
        # the condition is part of the ansatz's concrete type, not a field of an abstract type
        @test isconcretetype(fieldtype(typeof(dirichlet), :x_boundary))
        @test fieldtype(typeof(free), :x_boundary) === typeof(Free())
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

    @testset "A2c the refusals of the grid and the degree" for T in (Float64, Float32)
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 4), xspan = (zero(T), one(T)))
        for step in (zero(T), T(NaN), -T(0.05))
            @test_throws ArgumentError("output_grid needs a positive finite xstep, got $step") output_grid(
                ansatz, step)
        end
        @test_throws ArgumentError("degree must be ≥ 1 in the time direction, got (0, 2)") TensorBSplineAnsatz(
            degree = (0, 2), ncells = (2, 4), xspan = (zero(T), one(T)))
        @test_throws ArgumentError("degree must be ≥ 1 in the space direction, got (2, 0)") TensorBSplineAnsatz(
            degree = (2, 0), ncells = (2, 4), xspan = (zero(T), one(T)))
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

    @testset "A1b the space-time spline equivalence" for T in (Float64, Float32)
        problem, method, result = wave_run(T; n = 5)
        reference = WAVE_SPACETIME_SPLINE_WAVE.u
        @test datatype(result.solution) === T
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- reference[n, :])) ≤
                  T(4500) * eps(T) * maximum(abs, reference[n, :])
        end
    end

    @testset "A1c the full-restriction reference" for T in (Float64, Float32)
        problem, method, result = wave_run(T; n = 3)
        reference = WAVE_GALERKIN_FULL_RESTRICTION_BSPLINE.u
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- reference[n, :])) ≤
                  T(100) * newton_tolerance(T, SOLVER_OPTIONS)
        end
    end

    @testset "A1d the CFL-1 exactness" for T in (Float64, Float32)
        # A zero initial velocity removes the load startup, whose O(h²) velocity defect is the
        # whole error at CFL 1: the bilinear element is d'Alembert's shift there, so this run is
        # exact to roundoff.
        rest = merge(Wave.default_parameters(), (B1 = zero(T),))
        ansatz = TensorBSplineAnsatz(degree = 1, ncells = (1, 40),
            xspan = (zero(T), one(T)), n = 2)
        for t in (T(0.25), T(5))
            problem = Wave.problem(T = T, params = rest, timespan = (zero(T), t),
                timestep = T(0.05), xstep = T(0.025))
            @test abs(problem.params.c * problem.timestep / problem.xstep - one(T)) ≤
                  tolerance(T)
            result = integrate(problem, SpaceTimeGalerkin(ansatz, CoefficientRestriction()))
            grid = collect(xgrid(problem))
            exact = [Wave.exact_u(t, x, rest) for x in grid]
            @test maximum(abs.(result.solution.u[end] .- exact)) ≤ T(45_000) * eps(T)
        end
        # at the default parameters the same setting is the FEM stencil of the same settings
        problem, _, result = wave_run(T; degree = 1, ncells = (1, 40), n = 2,
            xstep = T(0.025))
        grid, stencil = fem_stencil(problem; steps = 5)
        @test collect(grid) == collect(xgrid(problem))
        for n in 1:5
            @test maximum(abs.(result.solution.u[n] .- stencil[n, :])) ≤
                  T(100) * eps(T) * maximum(abs, stencil[n, :])
        end
    end

    @testset "A1e the observed order" begin
        T = Float64
        # The entry's own cells: `t_knot_interval = 0.5` on `timespan = (0, 1)` is two time cells,
        # and its `x_knot_interval = h` refines the space direction alone.
        function errors(; degree)
            e = T[]
            for h in (T(0.1), T(0.05), T(0.025))
                problem, method, result = wave_run(T; degree = degree,
                    ncells = (2, round(Int, 1 / h)), n = degree + 1, timestep = h,
                    xstep = T(0.025), timespan = (zero(T), one(T)))
                grid = collect(xgrid(problem))
                push!(e, maximum(abs.(result.solution.u[end] .-
                                      [Wave.exact_u(one(T), x) for x in grid])))
            end
            return e
        end
        orders(e) = [log(e[i] / e[i + 1]) / log(2) for i in 1:2]
        reference = WAVE_FULL_RESTRICTION_ORDER.observed_order
        observed = orders(errors(degree = 2))
        for i in 1:2
            @test abs(observed[i] - reference[i]) ≤ 0.1
        end
        # degree 1 on the same three space resolutions, where the clause asks for at least 1.9
        for order in orders(errors(degree = 1))
            @test order ≥ T(1.9)
        end
    end

    @testset "A1f the discrete energy of 200 steps" for T in (Float64, Float32)
        problem, _, result = wave_run(T; degree = 1, ncells = (1, 20), n = 2,
            timestep = T(0.05), xstep = T(0.05), timespan = (zero(T), T(10)))
        M, K = fem_matrices(problem)
        h, c = convert(T, problem.timestep), convert(T, problem.params.c)
        function energy(a, b)
            δ = b .- a
            return sum(δ .* (M * δ)) / (2 * h^2) +
                   c^2 / 12 * (sum(b .* (K * b)) + sum(a .* (K * a)) +
                    4 * sum(b .* (K * a)))
        end
        u = [collect(step) for step in result.solution.u]
        @test length(u) == 201
        energies = [energy(u[n], u[n + 1]) for n in 0:199]
        @test eltype(energies) === T
        @test maximum(abs.(energies .- energies[1])) / abs(energies[1]) <
              T(450_000) * eps(T)
    end

    @testset "A1g the residual allocates nothing and infers" for T in (Float64, Float32)
        problem, method, _ = wave_run(T; ncells = (1, 20), n = 3)
        int = GeometricIntegrator(problem, method)
        c = cache(int)
        b = similar(c.x)
        x = copy(c.x)
        p = parameters(problem)
        residual(y) = M.residual!(b, y, nothing, p, int)
        allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
        # the helper sees an allocation, so a zero below is the residual's and not the helper's
        @test allocations(collect, x) > 0
        # the clause's domain is `--check-bounds=auto`; `Pkg.test()` runs the floor under
        # `--check-bounds=yes`, so the count is asserted under the clause's flag only
        if Base.JLOptions().check_bounds != 1
            @test allocations(residual, x) == 0
        end
        @test @inferred(cache(int, T)) === c
        # the step counter is a field of the cache, not a box that its readers dereference
        @test fieldtype(typeof(c), :steps) === Int
    end

    @testset "A1h SineGordon on the slab core" for T in (Float64, Float32)
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (2, 20),
            xspan = (zero(T), one(T)), n = 3, x_boundary = Free())
        problem = SineGordon.problem(T = T, timespan = (zero(T), T(0.15)),
            timestep = T(0.05), xstep = T(0.05))
        int = GeometricIntegrator(problem, SpaceTimeGalerkin(ansatz, CoefficientRestriction()))
        result = integrate(int)
        c = cache(int)
        @test length(c.iterations) == 3
        @test all(≥(1), c.iterations)
        @test all(isfinite, c.residuals)
        @test datatype(result.solution) === T
        assert_no_upcast(result.solution.u, T)
        grid = collect(xgrid(problem))
        for n in 1:3
            exact = [SineGordon.exact_u(n * T(0.05), x, problem.params) for x in grid]
            @test maximum(abs.(result.solution.u[n][3:(end - 2)] .-
                               exact[3:(end - 2)])) ≤ T(0.01)
            # the free boundary's error travels inward from both ends at 1/√c, so by the last
            # step it has not reached x ∈ [0.3, 0.7]: over the indices 7:end-6 the solution
            # must meet the tighter bound
            @test maximum(abs.(result.solution.u[n][7:(end - 6)] .-
                               exact[7:(end - 6)])) ≤ T(1e-3)
        end
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
