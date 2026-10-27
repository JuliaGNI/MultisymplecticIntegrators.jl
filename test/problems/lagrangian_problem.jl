using GeometricBase
using GeometricEquations
using MultisymplecticIntegrators
using Test

include("../helpers/testsetup.jl")

@testset "Lagrangian problem" begin
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
        @test problem.ics.u ≈ [phase * sin(π * xi / params.l) for xi in x]
        @test problem.ics.v ≈
              [params.B1 * ω * cos(π / 6) * sin(π * xi / params.l) for xi in x]
        @test problem.ics.w ≈ [phase * π / params.l * cos(π * xi / params.l) for xi in x]
        @test problem.exact_u(0.3, 0.7, params) ≈
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
        @test problem.ics.u ≈ [4 * atan(exp(γ * xi)) for xi in x]
        @test problem.exact_u(0.5, 0.3, params) ≈
              4 * atan(exp(γ * (0.3 - params.velocity * 0.5)))
        @test parameters(problem) === params
        @test_throws ArgumentError SineGordon.problem(params = (c = 4.0, velocity = 0.75))
    end

    @testset "the transport functions see the problem's parameters" begin
        params = (c = 0.5,)
        problem = LinearTransport.problem(params = params, timespan = (0.0, 2.0),
            timestep = 0.5, xstep = 0.25)
        x = xgrid(problem)
        @test problem.ics.u ≈ [0.5 * exp(-(xi + 2)^2) / sqrt(π) for xi in x]
        @test problem.exact_u(1.0, -0.25, params) ≈
              0.5 * exp(-(−0.250 + 2 - params.c * 1.0)^2) / sqrt(π)
        @test parameters(problem) === params
    end

    @testset "the boundary data of a slab" begin
        problem = Wave.problem(
            params = (c = 0.5, A1 = 0.8, A2 = 0.0, B1 = 0.4, B2 = 0.0,
                l = 1.0),
            timespan = (0.0, 1.0), timestep = 0.1, xstep = 0.25)
        x = xgrid(problem)
        times = (0.0, 1.0)
        data = problem.bcs_function(times, problem.params)
        @test keys(data) == (:bottom, :top, :left, :right)
        @test length(data.bottom.u) == length(x)
        @test length(data.left.u) == length(times)
        @test data.bottom.u != data.top.u
        @test data.left.u[1] == 0.0
        @test data.right.w[1] != data.left.w[1]
    end

    @testset "the boundary data sees the problem's parameters" begin
        params = (c = 0.9, A1 = 1.0, A2 = 0.0, B1 = 0.5, B2 = 0.0, l = 2.0)
        problem = Wave.problem(params = params, timespan = (0.0, 1.0), timestep = 0.1,
            xspan = (0.0, 2.0), xstep = 0.5)
        times = (0.3, 0.7)
        ω = π * params.c / params.l
        phase(t) = params.A1 * cos(ω * t) + params.B1 * sin(ω * t + π / 6)
        data = problem.bcs_function(times, params)
        x = xgrid(problem)
        @test data.bottom.u ≈ [phase(times[1]) * sin(π * xi / params.l) for xi in x]
        @test data.top.u ≈ [phase(times[2]) * sin(π * xi / params.l) for xi in x]
        @test data.left.w ≈ [phase(t) * π / params.l for t in times]
        @test data.right.w ≈ [-phase(t) * π / params.l for t in times]

        kink = (c = 4.0, velocity = 0.1)
        nonlinear = SineGordon.problem(
            params = kink, timespan = (0.0, 2.0), timestep = 1.0,
            xstep = 0.25)
        γ = 1 / sqrt(1 - kink.c * kink.velocity^2)
        kink_data = nonlinear.bcs_function(times, kink)
        kx = xgrid(nonlinear)
        @test kink_data.bottom.u ≈ [4 * atan(exp(γ * (xi - kink.velocity * times[1])))
               for xi in kx]
        @test kink_data.top.u ≈ [4 * atan(exp(γ * (xi - kink.velocity * times[2])))
               for xi in kx]
        @test kink_data.left.u ≈ [4 * atan(exp(γ * (-kink.velocity * t))) for t in times]

        vector = Wave.problem(
            D = 2, params = params, timespan = (0.0, 1.0), timestep = 0.1,
            xspan = (0.0, 2.0), xstep = 0.5)
        vector_data = vector.bcs_function(times, params)
        @test length(vector_data.bottom.u) == 2 * length(x)
        @test length(vector_data.left.u) == 2 * length(times)
        for component in 1:2
            span = ((component - 1) * length(x) + 1):(component * length(x))
            ends = ((component - 1) * length(times) + 1):(component * length(times))
            @test vector_data.bottom.u[span] ≈
                  [phase(times[1]) * sin(π * xi / params.l) for xi in x]
            @test vector_data.top.u[span] ≈
                  [phase(times[2]) * sin(π * xi / params.l) for xi in x]
            @test vector_data.left.w[ends] ≈ [phase(t) * π / params.l for t in times]
            @test vector_data.right.w[ends] ≈ [-phase(t) * π / params.l for t in times]
        end

        @test_throws ArgumentError problem.bcs_function(0.3, params)
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
        bcs = (times, p) -> nothing
        @test_throws ArgumentError LagrangianPDEProblem(density; D = 2, ics_function = ics,
            bcs_function = bcs, timespan = (0.0, 1.0), timestep = 0.1, xspan = (0.0, 1.0),
            xstep = 0.25)
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics,
            bcs_function = bcs, timespan = (Float16(0), Float16(1)), timestep = Float16(0.1),
            xspan = (Float16(0), Float16(1)), xstep = Float16(0.25))
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics,
            bcs_function = bcs, timespan = (0.0, 1.0), timestep = 0.1, xspan = (1.0, 0.0),
            xstep = 0.25)
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics,
            bcs_function = bcs, timespan = (0.0, 1.0), timestep = 0.0, xspan = (0.0, 1.0),
            xstep = 0.25)
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics,
            bcs_function = bcs, timespan = (0.0, Inf), timestep = 0.1, xspan = (0.0, 1.0),
            xstep = 0.25)
        @test_throws ArgumentError LagrangianPDEProblem(density; ics_function = ics,
            bcs_function = bcs, timespan = (0.0, 1.0), timestep = 0.1, xspan = (0.0, 1.0),
            xstep = 0.3)

        # A Float32 range that ends at zero stops a rounding error past it, which a relative
        # comparison with zero rejects.
        ending_at_zero = LinearTransport.problem(T = Float32, xspan = (-0.3f0, 0.0f0),
            xstep = 0.3f0 / 15)
        @test length(xgrid(ending_at_zero)) == 16
        @test abs(last(xgrid(ending_at_zero))) ≤ eps(Float32)

        # On a long span the rounding tolerance would exceed a cell; a step that stops a third
        # of a cell short of the end is still rejected.
        @test_throws ArgumentError LinearTransport.problem(T = Float32,
            xspan = (-1000.0f0, 0.0f0), xstep = 0.3f0)
        @test length(xgrid(LinearTransport.problem(T = Float32, xspan = (-1000.0f0, 0.0f0),
            xstep = 0.25f0))) == 4001
    end

    @testset "a run that starts at a non-zero time" begin
        problem = Wave.problem(timespan = (1.0, 1.4), timestep = 0.1, xstep = 0.5)
        params = Wave.default_parameters()
        ω = π * params.c / params.l
        phase = params.A1 * cos(ω) + params.B1 * sin(ω + π / 6)
        @test ntime(problem) == 4
        @test problem.ics.u ≈ [phase * sin(π * xi / params.l) for xi in xgrid(problem)]
    end
end
