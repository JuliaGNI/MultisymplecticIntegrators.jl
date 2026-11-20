# The contract of `GeometricBase.AbstractMethod` on one instance of each concrete method type of
# the package: `interface_violations` lists no entry for any of them. Every method of
# `QUALITY_METHODS` at `T = Float64` and `D = 1`, the direct linear solve, both variants of
# `LevenbergMarquardt`, and `T()` of each owned type that has a constructor of no argument. The
# methods are built and never run.

using MultisymplecticIntegrators
using GeometricBase: AbstractMethod, interface_violations
using InteractiveUtils: subtypes
using Test

include("../helpers/quality_cases.jl")

function owned_method_types(T::Type = AbstractMethod)
    types = Type[]
    for S in subtypes(T)
        if isabstracttype(S)
            append!(types, owned_method_types(S))
        elseif Base.moduleroot(parentmodule(S)) === MultisymplecticIntegrators
            push!(types, S)
        end
    end
    return types
end

const OWNED_METHOD_TYPES = owned_method_types()

const INTERFACE_METHODS = [[name => make(Float64, 1) for (name, make) in QUALITY_METHODS];
                           "B-spline ansatz, coefficient restriction, linear solve" =>
                               SpaceTimeGalerkin(
                                   quality_spline(Float64, Dirichlet()), CoefficientRestriction();
                                   linear_solver = true);
                           "Levenberg–Marquardt, variable projection" =>
                               LevenbergMarquardt(variable_projection = true);
                           ["$(nameof(T))()" => T()
                            for T in OWNED_METHOD_TYPES if hasmethod(T, Tuple{})]]

@testset "Interface: $name" for (name, method) in INTERFACE_METHODS
    @test isempty(interface_violations(method))
end

@testset "Every owned method type has an instance" begin
    @test !isempty(OWNED_METHOD_TYPES)
    @testset "$(nameof(T))" for T in OWNED_METHOD_TYPES
        @test any(m -> m isa T, last.(INTERFACE_METHODS))
    end
end
