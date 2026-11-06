using GeometricBase
using GeometricEquations
using GeometricSolutions: GeometricSolution
using MultisymplecticIntegrators
using Test

include("../helpers/testsetup.jl")

@testset "Lagrangian problem" begin
    @testset "the step count of a problem" begin
        # The step count is the one a `GeometricSolution` of the problem holds, and not a rounded
        # division: at `(0.0, 0.38)` with `0.076` the division rounds up to 6 where the solution
        # takes 5.
        for (T, span, h) in ((Float64, (0.0, 0.38), 0.076), (Float64, (0.0, 1.0), 0.1),
            (Float64, (0.0, 3.0), 0.3), (Float64, (0.0, 0.15), 0.05),
            (Float32, (0.0f0, 0.38f0), 0.076f0))
            problem = Wave.problem(T = T, timespan = span, timestep = h, xstep = T(0.25))
            @test ntime(problem) == ntime(GeometricSolution(problem))
        end
        @test ntime(Wave.problem(timespan = (0.0, 0.38), timestep = 0.076, xstep = 0.25)) ==
              5
    end

    @testset "the exact solution is a vector at both call shapes" for T in (Float64, Float32)
        # The wrapper returns a `Vector` at a scalar point and at a grid for every problem, so its
        # return type is no union of a scalar and a vector (`Base.return_types`, because `D` is a
        # runtime value).
        for problem in (Wave.problem(T = T), SineGordon.problem(T = T),
            VectorWave.problem(T = T), LinearTransport.problem(T = T))
            @test problem.solutions !== nothing
            @test Base.return_types(problem.solutions.u,
                Tuple{T, T, typeof(problem.params)}) == Any[Vector{T}]
            scalar = problem.solutions.u(T(0.1), T(0.3), problem.params)
            @test scalar isa Vector{T}
            @test length(scalar) == components(problem)
            grid = problem.solutions.u(T(0.1), collect(xgrid(problem)), problem.params)
            @test length(grid) == components(problem) * length(xgrid(problem))
        end
    end

    @testset "the spans have the element type" begin
        for T in (Float32, Float64)
            problem = Wave.problem(T = T, timespan = (zero(T), T(1)), timestep = T(0.1),
                xstep = T(0.25))
            @test typeof(problem.timespan) === NTuple{2, T}
            @test typeof(problem.timestep) === T
            @test typeof(problem.xspan) === NTuple{2, T}
            @test typeof(problem.xstep) === T
            @test eltype(problem.ics.u) === T
            @test datatype(problem) === T
            @test timetype(problem) === T
        end
    end

    @testset "the wave functions see the problem's parameters" begin
        # The closed form is written here, and not read from the module: a test that calls the
        # routine under test twice compares nothing.
        params = (c = 0.9, A1 = 1.0, A2 = 0.0, B1 = 0.5, B2 = 0.0, l = 2.0)
        problem = Wave.problem(params = params, timespan = (0.0, 1.0), timestep = 0.1,
            xspan = (0.0, 2.0), xstep = 0.5)
        x = xgrid(problem)
        ω = π * params.c / params.l
        phase = params.A1 * cos(0.0) + params.B1 * sin(π / 6)
        @test vec(problem.ics.u) ≈ [phase * sin(π * xi / params.l) for xi in x]
        @test vec(problem.ics.v) ≈
              [params.B1 * ω * cos(π / 6) * sin(π * xi / params.l) for xi in x]
        @test vec(problem.ics.w) ≈
              [phase * π / params.l * cos(π * xi / params.l) for xi in x]
        @test only(problem.solutions.u(0.3, 0.7, params)) ≈
              (params.A1 * cos(ω * 0.3) + params.B1 * sin(ω * 0.3 + π / 6)) *
              sin(π * 0.7 / params.l)
        @test parameters(problem) === params
    end

    @testset "the sine-Gordon functions see the problem's parameters" begin
        params = (c = 4.0, velocity = 0.1)
        problem = SineGordon.problem(
            params = params, timespan = (0.0, 2.0), timestep = 1.0,
            xstep = 0.25)
        γ = 1 / sqrt(1 - params.c * params.velocity^2)
        x = xgrid(problem)
        @test vec(problem.ics.u) ≈ [4 * atan(exp(γ * xi)) for xi in x]
        @test only(problem.solutions.u(0.5, 0.3, params)) ≈
              4 * atan(exp(γ * (0.3 - params.velocity * 0.5)))
        @test parameters(problem) === params
        @test_throws ArgumentError SineGordon.problem(params = (c = 4.0, velocity = 0.75))
    end

    @testset "the transport functions see the problem's parameters" begin
        params = (c = 0.5,)
        problem = LinearTransport.problem(params = params, timespan = (0.0, 2.0),
            timestep = 0.5, xstep = 0.25)
        x = xgrid(problem)
        @test vec(problem.ics.u[:, 1]) ≈ [0.5 * exp(-(xi + 2)^2) / sqrt(π) for xi in x]
        @test all(iszero, problem.ics.u[:, 2])
        @test problem.solutions.u(1.0, -0.25, params)[1] ≈
              0.5 * exp(-(−0.250 + 2 - params.c * 1.0)^2) / sqrt(π)
        @test iszero(problem.solutions.u(1.0, -0.25, params)[2])
        @test parameters(problem) === params
    end

    @testset "the boundary data of a slab" begin
        problem = Wave.problem(
            params = (c = 0.5, A1 = 0.8, A2 = 0.0, B1 = 0.4, B2 = 0.0,
                l = 1.0),
            timespan = (0.0, 1.0), timestep = 0.1, xstep = 0.25)
        data = problem.bcs_function(0.0, problem.params)
        @test keys(data) == (:left, :right)
        @test length(data.left.u) == 1
        @test data.left.u[1] == 0.0
        @test data.left.w[1] != data.right.w[1]
        later = problem.bcs_function(0.4, problem.params)
        @test later.left.w != data.left.w
    end

    @testset "the boundary data sees the problem's parameters" begin
        params = (c = 0.9, A1 = 1.0, A2 = 0.0, B1 = 0.5, B2 = 0.0, l = 2.0)
        problem = Wave.problem(params = params, timespan = (0.0, 1.0), timestep = 0.1,
            xspan = (0.0, 2.0), xstep = 0.5)
        ω = π * params.c / params.l
        phase(t) = params.A1 * cos(ω * t) + params.B1 * sin(ω * t + π / 6)
        a, b = xspan(problem)
        for t in (0.3, 0.7)
            data = problem.bcs_function(t, params)
            @test data.left.u ≈ [phase(t) * sin(π * a / params.l)]
            @test data.right.u ≈ [phase(t) * sin(π * b / params.l)]
            @test data.left.w ≈ [phase(t) * π / params.l * cos(π * a / params.l)]
            @test data.right.w ≈ [phase(t) * π / params.l * cos(π * b / params.l)]
        end

        kink = (c = 4.0, velocity = 0.1)
        nonlinear = SineGordon.problem(
            params = kink, timespan = (0.0, 2.0), timestep = 1.0,
            xstep = 0.25)
        γ = 1 / sqrt(1 - kink.c * kink.velocity^2)
        ka, kb = xspan(nonlinear)
        for t in (0.3, 0.7)
            kink_data = nonlinear.bcs_function(t, kink)
            @test kink_data.left.u ≈ [4 * atan(exp(γ * (ka - kink.velocity * t)))]
            @test kink_data.right.u ≈ [4 * atan(exp(γ * (kb - kink.velocity * t)))]
        end

        vector = Wave.problem(
            D = 2, params = params, timespan = (0.0, 1.0), timestep = 0.1,
            xspan = (0.0, 2.0), xstep = 0.5)
        vector_data = vector.bcs_function(0.3, params)
        @test length(vector_data.left.u) == 2
        @test length(vector_data.left.w) == 2
        for component in 1:2
            @test vector_data.left.u[component] ≈ phase(0.3) * sin(π * a / params.l)
            @test vector_data.right.u[component] ≈ phase(0.3) * sin(π * b / params.l)
            @test vector_data.left.w[component] ≈
                  phase(0.3) * π / params.l * cos(π * a / params.l)
        end
    end

    @testset "initial conditions merge the initial time" begin
        problem = Wave.problem(timespan = (1.0, 2.0), timestep = 0.5, xstep = 0.5)
        ics = initial_conditions(problem)
        @test keys(ics) == (:t, :u, :v, :w)
        @test ics.t == 1.0
        @test ics.u == problem.ics.u
    end

    @testset "the decided edges" begin
        density = Wave.density()
        ics = (x, p) -> (u = zeros(length(x)), v = zeros(length(x)), w = zeros(length(x)))
        # `ics` returns a length-`length(x)` vector, which no problem accepts: it refuses a `D = 2`
        # problem at the initial data's size. `ics1` returns the `(length(x), 1)` matrices a `D = 1`
        # problem accepts, so that the refusals below are the constructor's own and not the initial
        # data's size.
        ics1 = (x, p) -> (u = zeros(length(x), 1), v = zeros(length(x), 1),
            w = zeros(length(x), 1))
        bcs = (times, p) -> nothing
        @test_throws ArgumentError LagrangianPDEProblem(density; D = 2, ics_function = ics,
            bcs_function = bcs, timespan = (0.0, 1.0), timestep = 0.1, xspan = (0.0, 1.0),
            xstep = 0.25)
        # A grid of one point is refused by the grid's own guard, and not by the initial data's
        # size.
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics1,
            bcs_function = bcs, timespan = (0.0, 1.0), timestep = 0.1, xspan = (0.0, 1.0),
            xstep = 2.0)
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics1,
            bcs_function = bcs, timespan = (Float16(0), Float16(1)), timestep = Float16(0.1),
            xspan = (Float16(0), Float16(1)), xstep = Float16(0.25))
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics1,
            bcs_function = bcs, timespan = (1.0, 0.0), timestep = 0.1, xspan = (0.0, 1.0),
            xstep = 0.25)
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics1,
            bcs_function = bcs, timespan = (0.0, 1.0), timestep = 0.1, xspan = (1.0, 0.0),
            xstep = 0.25)
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics1,
            bcs_function = bcs, timespan = (0.0, 1.0), timestep = 0.0, xspan = (0.0, 1.0),
            xstep = 0.25)
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics1,
            bcs_function = bcs, timespan = (0.0, Inf), timestep = 0.1, xspan = (0.0, 1.0),
            xstep = 0.25)

        # A Float32 range that ends at zero stops a rounding error past it, which a relative
        # comparison with zero rejects.
        ending_at_zero = LinearTransport.problem(T = Float32, xspan = (-0.3f0, 0.0f0),
            xstep = 0.3f0 / 15)
        @test length(xgrid(ending_at_zero)) == 16
        @test abs(last(xgrid(ending_at_zero))) ≤ eps(Float32)

        # The grid holds `n = round(Int, Δx / xstep) + 1` points and ends on the span's end, so a
        # step that divides the span builds on a long span too.
        spanning = LinearTransport.problem(T = Float32, xspan = (-1000.0f0, 0.0f0),
            xstep = 0.3f0)
        @test length(xgrid(spanning)) == 3334
        @test last(xgrid(spanning)) == 0.0f0
        @test length(xgrid(LinearTransport.problem(T = Float32, xspan = (-1000.0f0, 0.0f0),
            xstep = 0.25f0))) == 4001
    end

    @testset "the grid of a mesh" for T in (Float64, Float32)
        # `n = round(Int, Δx / xstep) + 1`: the last mesh divides the span 27 times but for a
        # rounding error, so it holds 28 points, where a floor holds 27.
        for (span, step, n) in (((zero(T), one(T)), T(0.3), 4), ((zero(T), T(3)), T(1), 4),
            ((T(0.1), T(0.7)), T(0.6) / 27, 28))
            problem = Wave.problem(T = T, timespan = (zero(T), T(1)), timestep = T(0.5),
                xspan = span, xstep = step)
            grid = xgrid(problem)
            @test length(grid) == n
            @test grid isa AbstractRange
            @test abs(last(grid) - span[2]) ≤ 16 * eps(T)
            @test first(grid) == span[1]
        end
    end

    @testset "a run that starts at a non-zero time" begin
        problem = Wave.problem(timespan = (1.0, 1.4), timestep = 0.1, xstep = 0.5)
        params = Wave.default_parameters()
        ω = π * params.c / params.l
        phase = params.A1 * cos(ω) + params.B1 * sin(ω + π / 6)
        @test ntime(problem) == 4
        @test vec(problem.ics.u) ≈ [phase * sin(π * xi / params.l) for xi in xgrid(problem)]
    end

    @testset "a Float32 $(nameof(family)) problem evaluates its energy in Float32" for family in (
        Wave, SineGordon)
        T = Float32
        problem = family.problem(
            T = T, timespan = (zero(T), T(0.1)), timestep = T(0.05), xstep = T(0.25))
        @test all(value -> typeof(value) === T, problem.params)
        ansatz = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 4), xspan = (zero(T), one(T)), n = 3)
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
