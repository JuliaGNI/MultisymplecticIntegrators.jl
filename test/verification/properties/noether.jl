using GeometricBase: datatype, ntime
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using Test

import GeometricIntegratorsBase: cache, integrate!, solver

include("../../helpers/problem_generator.jl")
include("../../helpers/action_oracle.jl")

"""
    rotation_charges(entry, coupling)

The rotation charge `J_n = Σ_j (ĉ¹_Nt,j p²_j − ĉ²_Nt,j p¹_j)` of every step of a run, with `ĉ_Nt`
the top row of the step's solved vector and `p = ∂S/∂ĉ` of that row from the oracle's interior
action, and the `f_abstol` of the integrator's own solver.

The steps are solved one at a time, and the oracle carries each step's datum from the solved vector
of the step before it.
"""
function rotation_charges(entry::GeneratedProblem, coupling)
    T = datatype(entry.problem)
    slab = first_slab(entry, coupling)
    oracle = ActionOracle(entry.problem, slab.method)
    c = cache(slab.int)
    datum = first_datum(oracle)
    charges = T[]
    for n in 1:ntime(slab.solution)
        n > 1 && next_slab!(slab, n)
        integrate!(slab.solstep, slab.int)
        θ = copy(c.x)
        ĉ, _, _, _ = decode(oracle, datum, θ)
        p = top_momentum(oracle, datum, θ)
        top = ĉ[oracle.nt, :, :]
        push!(charges, sum(top[:, 1] .* p[:, 2] .- top[:, 2] .* p[:, 1]))
        datum = next_datum(oracle, datum, θ)
    end
    return charges, SimpleSolvers.config(solver(slab.int)).f_abstol
end

@testset "A6a the rotation charge over a run, $T" for T in (Float64, Float32)
    entries = [e for e in generate_problems(T; rng = Xoshiro(SEED)) if e.kind === :noether]
    @testset "entry $i, $(nameof(typeof(coupling)))" for (i, entry) in enumerate(entries),
        coupling in COUPLINGS

        charges, f_abstol = rotation_charges(entry, coupling)
        @test length(charges) ≥ 2
        @test all(isfinite, charges)
        @test abs(charges[1]) ≥ T(1 // 100)
        @test maximum(J -> abs(J - charges[1]), charges) ≤ 100 * f_abstol
    end
end
