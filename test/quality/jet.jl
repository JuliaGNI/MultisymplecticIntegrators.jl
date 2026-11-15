# JET's optimisation analysis of the two entry points of the Newton solve, `residual!` and
# `cache(int, T)`, for every method of `QUALITY_METHODS` at both element types and both component
# counts. The analysis of `cache` follows the cache dictionary's `getindex` into the construction
# of the cache, so a runtime dispatch in a check that runs when the cache is built counts as a
# report.

using JET
using MultisymplecticIntegrators
using GeometricIntegratorsBase: cache, residual!
using Test

include("../helpers/quality_cases.jl")

const JET_WORKS = JET.JET_AVAILABLE

function residual_reports(args)
    JET.get_reports(JET.report_opt(residual!, typeof.(args);
        target_modules = (MultisymplecticIntegrators,)))
end

# The call `cache(int, T)` with `T` known at the call site, as the integrator makes it. A reflection
# on `cache` at `Type{T}` analyses, from Julia 1.14, a specialisation that no real call compiles
# (JuliaLang/julia#63965), and reports runtime dispatches that the real call does not make (issue
# #74). The static parameter of `Val{T}` gives the call the signature that a real call compiles.
cache_at(int, ::Val{T}) where {T} = cache(int, T)

function cache_reports(int, T)
    JET.get_reports(JET.report_opt(cache_at, Tuple{typeof(int), Val{T}};
        target_modules = (MultisymplecticIntegrators, @__MODULE__)))
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
