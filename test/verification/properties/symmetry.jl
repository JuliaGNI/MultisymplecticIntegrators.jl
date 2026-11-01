using GeometricBase: datatype, ntime, parameters, timespan, timestep
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, solver

include("../../helpers/problem_generator.jl")
include("../../helpers/action_oracle.jl")

# the largest difference between the rows of `a` and those of its mirror, `j ↦ n + 1 − j`
function asymmetry(a::AbstractMatrix)
    n = size(a, 1)
    return maximum(abs(a[j, d] - a[n + 1 - j, d]) for j in 1:n, d in axes(a, 2))
end

@testset "A7a symmetric data under x ↦ a + b − x, $T" for T in (Float64, Float32)
    entries = [e
               for e in generate_problems(T; rng = Xoshiro(SEED)) if e.kind === :symmetric]
    @testset "entry $i, $(nameof(typeof(coupling)))" for (i, entry) in enumerate(entries),
        coupling in COUPLINGS

        method = entry_method(entry, coupling)
        int = GeometricIntegrator(entry.problem, method)
        result = integrate(int)
        @test length(result.traces) == ntime(result.solution)
        f_abstol = SimpleSolvers.config(solver(int)).f_abstol
        # the last step's solved vector, decoded by the oracle's layout
        oracle = ActionOracle(entry.problem, method)
        ĉ, λ, μa, μb = decode(oracle, first_datum(oracle), copy(cache(int).x))
        top = ĉ[oracle.nt, :, :]
        @test asymmetry(top) ≤ 1000 * f_abstol * maximum(abs, top)
        coupling isa CoefficientRestriction && continue
        # the multipliers carry the conditioning of the saddle-point system, so their relations
        # hold to a multiple of cbrt(eps)
        bound = 10 * cbrt(eps(T))
        @test asymmetry(λ) ≤ bound * maximum(abs, λ)
        coupling isa WeakInitialAndBoundaryConditions || continue
        scale = maximum(abs, μa)
        @test scale ≥ T(1 // 100)
        @test maximum(abs, μa + μb) ≤ bound * scale
    end
end
