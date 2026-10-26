using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using Test

import GeometricIntegratorsBase: Cache, CacheType, IntegratorCache, integrate_step!

include("../helpers/testsetup.jl")

# A fixture method: it advances nothing, because this package ships no integrator. It exists to
# exercise the shell — the solution the integrator builds, and the trace it stores beside it. Its
# cache holds the trace at the nodes of a space quadrature, as a method's cache does.
struct FixtureMethod{T} <: SpaceTimeAnsatz
    quadrature::SpaceTimeQuadrature{T}
end

function FixtureMethod(xspan; n::Integer = 1)
    FixtureMethod(SpaceTimeQuadrature{Float64}(n; xbreaks = collect(xspan)))
end

struct FixtureCache{DT} <: IntegratorCache{DT}
    u::Vector{DT}
    v::Vector{DT}
    w::Vector{DT}
end

function Cache{ST}(problem::LagrangianPDEProblem, method::FixtureMethod; kwargs...) where {ST}
    nodes = length(method.quadrature.x)
    return FixtureCache{ST}([problem.ics.u[1] for _ in 1:nodes],
        [problem.ics.v[1] for _ in 1:nodes],
        [problem.ics.w[1] for _ in 1:nodes])
end

CacheType(ST, ::LagrangianPDEProblem, ::FixtureMethod) = FixtureCache{ST}

function MultisymplecticIntegrators.top_trace(cache::FixtureCache)
    (u = cache.u, v = cache.v,
        w = cache.w)
end

function integrate_step!(sol, history, params,
        int::GeometricIntegrator{<:FixtureMethod, <:LagrangianPDEProblem})
    # The fixture counts its steps in the cache, so that a stored trace that aliases the cache
    # repeats the last value instead of holding each step's own.
    c = cache(int)
    c.u .+= 1
    return nothing
end

@testset "Integrator shell" begin
    problem = Wave.problem(timespan = (0.0, 0.4), timestep = 0.1, xstep = 0.25)
    method = FixtureMethod((-0.5, 0.0); n = 2)
    result = integrate(problem, method)

    @testset "the named tuple of the shell" begin
        @test keys(result) == (:solution, :traces)
        @test result.solution isa GeometricSolution
        @test datatype(result.solution) === Float64
    end

    @testset "the trace of every step, at the quadrature nodes" begin
        @test length(result.traces) == ntime(result.solution)
        @test !isempty(result.traces)
        @test isconcretetype(eltype(result.traces))
        quadrature_nodes = length(method.quadrature.x)
        interior_nodes = length(xgrid(problem))
        @test quadrature_nodes != interior_nodes
        for trace in result.traces, entry in values(trace)

            @test length(entry) == quadrature_nodes
        end
        @test result.traces[1].u != result.traces[2].u
        @test result.traces[2].u != result.traces[end].u
    end

    @testset "the solution carries the problem's state" begin
        @test keys(result.solution.dataser) == (:t, :u, :v, :w)
        @test result.solution.u[0] ≈ problem.ics.u
        @test result.solution.u[ntime(result.solution)] ≈ problem.ics.u
    end

    @testset "a run that starts at a time other than zero" begin
        late = Wave.problem(timespan = (1.0, 1.4), timestep = 0.1, xstep = 0.25)
        steps = integrate(late, method)
        @test ntime(steps.solution) == 4
        @test [steps.solution.t[n] for n in 1:4] ≈ [1.1, 1.2, 1.3, 1.4]
        @test length(steps.traces) == 4
        @test steps.traces[1] !== nothing
        @test steps.solution.u[1] ≈ late.ics.u
        @test steps.solution.u[0] ≈ late.ics.u
    end

    @testset "no silent upcast" begin
        small = Wave.problem(T = Float32, timespan = (Float32(0), Float32(0.4)),
            timestep = Float32(0.1), xstep = Float32(0.25))
        small_method = FixtureMethod((Float32(-0.5), Float32(0.0)); n = 2)
        small_result = integrate(small, small_method)
        assert_no_upcast(small.ics.u, Float32)
        assert_no_upcast(small_result.solution.u, Float32)
        for trace in small_result.traces, entry in values(trace)

            assert_no_upcast(entry, Float32)
        end
    end

    @testset "a method without a trace stores nothing" begin
        @test top_trace(nothing) === nothing
    end
end
