using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using Test

import GeometricIntegratorsBase: Cache, CacheType, IntegratorCache, integrate_step!

# A fixture method: it advances nothing, because part C ships no integrator. It exists to exercise
# the shell — the solution the integrator builds, and the trace it stores beside it.
struct FixtureMethod <: SpaceTimeAnsatz end

struct FixtureCache{DT} <: IntegratorCache{DT}
    u::Vector{DT}
    v::Vector{DT}
    w::Vector{DT}
end

function Cache{ST}(problem::LagrangianPDEProblem, ::FixtureMethod; kwargs...) where {ST}
    FixtureCache{ST}(copy(problem.ics.u), copy(problem.ics.v), copy(problem.ics.w))
end

CacheType(ST, ::LagrangianPDEProblem, ::FixtureMethod) = FixtureCache{ST}

function MultisymplecticIntegrators.top_trace(cache::FixtureCache)
    (u = cache.u, v = cache.v,
        w = cache.w)
end

function integrate_step!(sol, history, params,
        int::GeometricIntegrator{<:FixtureMethod, <:LagrangianPDEProblem})
    return nothing
end

@testset "Integrator shell" begin
    problem = Wave.problem(timespan = (0.0, 0.4), timestep = 0.1, xstep = 0.25)
    result = integrate(problem, FixtureMethod())

    @testset "the named tuple of the shell" begin
        @test keys(result) == (:solution, :traces)
        @test result.solution isa GeometricSolution
        @test datatype(result.solution) === Float64
    end

    @testset "the trace of every step" begin
        @test length(result.traces) == ntime(result.solution)
        @test !isempty(result.traces)
        nodes = length(xgrid(problem))
        for trace in result.traces, entry in values(trace)

            @test length(entry) == nodes
        end
    end

    @testset "the solution carries the problem's state" begin
        @test keys(result.solution.dataser) == (:t, :u, :v, :w)
        @test result.solution.u[0] ≈ problem.ics.u
        @test result.solution.u[ntime(result.solution)] ≈ problem.ics.u
    end

    @testset "a method without a trace stores nothing" begin
        @test top_trace(nothing) === nothing
    end
end
