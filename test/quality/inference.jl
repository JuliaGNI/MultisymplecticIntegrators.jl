# Inference of the two entry points of the Newton solve, `residual!` and `cache(int, T)`, for
# every method of `QUALITY_METHODS` at both element types and both component counts.

using MultisymplecticIntegrators
using GeometricIntegratorsBase: cache, residual!
using Test

include("../helpers/quality_cases.jl")

@testset "Inference: $name" for (name, make) in QUALITY_METHODS
    @testset "T = $T, D = $D" for T in QUALITY_TYPES, D in QUALITY_COMPONENTS

        int, args = quality_entry(make, T, D)
        @test (@inferred residual!(args...)) === nothing
        @test (@inferred cache(int, T)) === cache(int, T)
    end
end
