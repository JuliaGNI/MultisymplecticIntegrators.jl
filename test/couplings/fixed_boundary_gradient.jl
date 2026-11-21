# The fixed-boundary coupling: the residual against the action's gradient. One of the files of
# `test/couplings/fixed_boundary*.jl`, which share `test/helpers/fixed_boundary_setup.jl`.
using GeometricBase: parameters, timestep, timespan
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS, ColumnNorm, diag, norm, qr
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, current, integrate!, reset!,
                                 solutionstep, solver
import GeometricBase: ntime, timesteps

include("../helpers/problem_generator.jl")
include("../helpers/action_oracle.jl")
include("../helpers/testsetup.jl")
# The registry of the scripts and no part of the package: the family's rows are built here.
include(joinpath(@__DIR__, "..", "..", "scripts", "experiments.jl"))
include("../helpers/fixed_boundary_setup.jl")

# ---- FB1 -----------------------------------------------------------------------------

@testset "FB1 the residual is the action's gradient" begin
    rng = Xoshiro(7)
    # N4 and N18's first check: all 24 seeded entries at both element types, in both forms and both
    # modes, on the first and the second slab, at the clause's rule of 20 cells.
    @testset "generated entry, $T, $form, $mode" for T in (Float64, Float32),
        (i, entry) in enumerate(generate_problems(T; rng = Xoshiro(SEED))), form in (:abg, :tsr),
        mode in (:all, :last_layer)

        ansatz = clause_ansatz(entry.problem; form = form, mode = mode)
        problem = entry.problem
        # the step is capped: the second slab's datum is a step's final iterate, converged or not,
        # so the cap changes which datum the second slab carries and not the identity asserted on it
        slab = fixed_slab(problem, ansatz; max_iterations = 3, verbosity = 0)
        c = cache(slab.int)
        oracle = ActionOracle(problem, slab.method; block = c.ansatz)
        @test length(c.x) == unknown_count(oracle)
        θ = 2 .* rand(rng, T, length(c.x)) .- 1
        datum = fixed_datum(c)
        label = (; entry = i, form = form, mode = mode)
        compare_gradient(slab_residual(slab.int, θ, parameters(problem)),
            oracle_residual(oracle, datum, θ), T;
            oracle = oracle, datum = datum, θ = θ, label = (; label..., slab = 1))
        integrate!(slab.solstep, slab.int)
        next_fixed_slab!(slab, 2)
        θ₂ = 2 .* rand(rng, T, length(c.x)) .- 1
        datum₂ = fixed_datum(c)
        compare_gradient(slab_residual(slab.int, θ₂, parameters(problem)),
            oracle_residual(oracle, datum₂, θ₂), T;
            oracle = oracle, datum = datum₂, θ = θ₂, label = (; label..., slab = 2))
    end
    @testset "the two problems, first and second slab, $name, $form, $mode" for name in (:wave, :sinegordon),
        form in (:abg, :tsr), mode in (:all, :last_layer)

        problem = clause_problem(name; steps = 2)
        slab = fixed_slab(problem, clause_ansatz(problem; form = form, mode = mode);
            max_iterations = 3, verbosity = 0)
        c = cache(slab.int)
        oracle = ActionOracle(problem, slab.method; block = c.ansatz)
        θ = 2 .* rand(rng, Float64, length(c.x)) .- 1
        datum = fixed_datum(c)
        label = (; entry = name, form = form, mode = mode)
        compare_gradient(slab_residual(slab.int, θ, parameters(problem)),
            oracle_residual(oracle, datum, θ), Float64;
            oracle = oracle, datum = datum, θ = θ, label = (; label..., slab = 1))
        integrate!(slab.solstep, slab.int)
        next_fixed_slab!(slab, 2)
        θ₂ = 2 .* rand(rng, Float64, length(c.x)) .- 1
        datum₂ = fixed_datum(c)
        compare_gradient(slab_residual(slab.int, θ₂, parameters(problem)),
            oracle_residual(oracle, datum₂, θ₂), Float64;
            oracle = oracle, datum = datum₂, θ = θ₂, label = (; label..., slab = 2))
    end
end
