# JET's optimisation analysis of the two entry points of the Newton solve, `residual!` and
# `cache(int, T)`, for every method at both element types and both component counts. The analysis
# of `cache` follows the cache dictionary's `getindex` into the construction of the cache, so a
# runtime dispatch in a check that runs when the cache is built counts as a report.

using JET
using MultisymplecticIntegrators
using GeometricIntegratorsBase: cache, residual!
using Test

include("../helpers/quality_cases.jl")

const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

function residual_reports(args)
    JET.get_reports(JET.report_opt(residual!, typeof.(args);
        target_modules = (MultisymplecticIntegrators,)))
end

function cache_reports(int, T)
    JET.get_reports(JET.report_opt(cache, Tuple{typeof(int), Type{T}};
        target_modules = (MultisymplecticIntegrators,)))
end

if !JET_WORKS
    @test_skip "JET does not work on this Julia version"  # aviatesk/JET.jl#681
else
    @testset "JET: $name" for (name, make) in QUALITY_METHODS
        @testset "T = $T, D = $D" for T in QUALITY_TYPES, D in QUALITY_COMPONENTS

            int, args = quality_entry(make, T, D)
            if name == "box scheme"
                @test_broken isempty(residual_reports(args))  # issue #70, K61
                @test_broken isempty(cache_reports(int, T))  # issue #70, K61
            else
                @test isempty(residual_reports(args))
                @test isempty(cache_reports(int, T))
            end
        end
    end
end
