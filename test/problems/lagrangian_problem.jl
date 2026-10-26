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

    @testset "the functions see the problem's parameters" begin
        params = (c = 0.9, A1 = 1.0, A2 = 0.0, B1 = 0.5, B2 = 0.0, l = 2.0)
        problem = Wave.problem(params = params, timespan = (0.0, 1.0), timestep = 0.1,
            xspan = (0.0, 2.0), xstep = 0.5)
        x = xgrid(problem)
        @test problem.ics.u ≈ Wave.exact_u(0.0, x, params)
        @test problem.ics.v ≈ Wave.exact_v(0.0, x, params)
        @test problem.ics.w ≈ Wave.exact_w(0.0, x, params)
        @test problem.bcs_function(0.3, params).bottom.u ≈ Wave.exact_u(0.3, x, params)
        @test problem.exact_u(0.3, 0.7, params) ≈ Wave.exact_u(0.3, 0.7, params)
        @test parameters(problem) === params
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
        bcs = (t, p) -> nothing
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
    end

    @testset "a run that starts at a non-zero time" begin
        problem = Wave.problem(timespan = (1.0, 1.4), timestep = 0.1, xstep = 0.5)
        @test ntime(problem) == 4
        @test problem.ics.u ≈ Wave.exact_u(1.0, xgrid(problem), Wave.default_parameters())
    end
end
