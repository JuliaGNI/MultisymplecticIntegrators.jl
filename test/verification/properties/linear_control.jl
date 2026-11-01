using GeometricBase: ntime
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, solver

include("../../helpers/problem_generator.jl")

# A density of degree 2 gives a slab that is linear in its unknowns, so one Newton step with the
# exact Jacobian solves it up to round-off.
# The run bounds the iterations at two and asserts the convergence rule after them, with the
# tolerances of the integrator's own solver; it compares no iteration count exactly, because a
# count on a round-off edge differs between machines.
@testset "A5a a linear density in two iterations, $T" for T in (Float64, Float32)
    entries = [e for e in generate_problems(T; rng = Xoshiro(SEED)) if e.potential == 2]
    @testset "$(entry.kind), $(nameof(typeof(coupling)))" for entry in entries,
        coupling in COUPLINGS

        int = GeometricIntegrator(entry.problem, entry_method(entry, coupling);
            max_iterations = 2)
        result = integrate(int)
        c = cache(int)
        options = SimpleSolvers.config(solver(int))
        steps = ntime(result.solution)
        @test length(result.traces) == steps
        @test length(c.iterations) == steps
        @test length(c.residuals) == steps
        @test length(c.initial_residuals) == steps
        @test all(≥(T(1 // 100)), c.initial_residuals)
        @test all(≤(2), c.iterations)
        @test all(c.residuals .≤
                  options.f_abstol .+ options.f_reltol .* c.initial_residuals)
    end
end
