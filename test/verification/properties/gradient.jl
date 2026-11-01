using GeometricBase: parameters
using GeometricIntegratorsBase
using MultisymplecticIntegrators
using Random
using Test

import GeometricIntegratorsBase: cache, integrate!

include("../../helpers/problem_generator.jl")
include("../../helpers/action_oracle.jl")

const M = MultisymplecticIntegrators

"""
    compare_gradient(b, ∇S, T)

The package's residual `b` against the oracle's `∇S`: both finite, the gradient at least `1/100` in
its largest entry, and the difference at most `100 eps(T)` relative to that entry. No scale and no
sign is fitted between the two.
"""
function compare_gradient(b, ∇S, ::Type{T}) where {T}
    @test all(isfinite, b)
    @test all(isfinite, ∇S)
    scale = maximum(abs, ∇S)
    @test scale ≥ T(1 // 100)
    @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * scale
end

@testset "A3a the residual is the gradient of the action, $T" for T in (Float64, Float32)
    entries = generate_problems(T; rng = Xoshiro(SEED))
    rng = Xoshiro(SEED + 1)
    @testset "entry $i, $(nameof(typeof(coupling)))" for (i, entry) in enumerate(entries),
        coupling in COUPLINGS

        slab = first_slab(entry, coupling)
        oracle = ActionOracle(entry.problem, slab.method)
        c = cache(slab.int)
        params = parameters(entry.problem)
        N = length(c.x)
        @test N == unknown_count(oracle)
        b = zeros(T, N)
        # the first slab, at a random vector away from the solution
        θ₁ = 2 .* rand(rng, T, N) .- 1
        M.residual!(b, θ₁, nothing, params, slab.int)
        datum = first_datum(oracle)
        compare_gradient(b, oracle_residual(oracle, datum, θ₁), T)
        # the second slab, whose datum the oracle carries from the first slab's solved vector
        integrate!(slab.solstep, slab.int)
        solved = copy(c.x)
        next_slab!(slab, 2)
        θ₂ = 2 .* rand(rng, T, N) .- 1
        M.residual!(b, θ₂, nothing, params, slab.int)
        compare_gradient(b, oracle_residual(oracle, next_datum(oracle, datum, solved), θ₂), T)
    end
end
