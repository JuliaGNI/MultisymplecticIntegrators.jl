# The fixed-boundary coupling: the Jacobian against differences of the residual. One of the files of
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

# ---- FB2 -----------------------------------------------------------------------------

@testset "FB2 the Jacobian is the derivative of the residual" begin
    rng = Xoshiro(11)
    drng = Xoshiro(12)
    # `jacobian!` is SimpleSolvers' and writes the solver's own matrix, so it exists for a
    # `NonlinearSolver`. The family's Levenberg–Marquardt keeps no such matrix — it takes one SVD of
    # the Jacobian at each accepted iterate — so the Jacobian of the residual is read through the
    # `Newton()` solver the family also offers, on the same residual and the same block.
    #
    # N5's and N18's second check: the legs are `Wave`, `SineGordon` and all 24 seeded entries, in
    # both forms and both modes, at 16 vectors each, and at both element types. Each vector's `J` is
    # read along three directions drawn from `drng`, so that `rng` draws the vectors alone; each `d`
    # is drawn in `Float64` and rounded to `T`. The difference along `d` checks `J · d` with two
    # residuals, and a `d` with no zero entry weights every column of `J`. The bound is
    # `sqrt(eps(T)) · max|J| · ‖d‖₂`. Its margin is measured, not derived: on this grid the worst
    # ratio of error to bound is 0.124 in `Float64` and 0.026 in `Float32` (issue #63). The
    # differences are taken on the `Float64` slab in both precisions, as the symbolic ansatz's are:
    # the residual's round-off at `Float32` sits in differences `cbrt(eps(Float32))` apart, where
    # its own truncation error is larger than the bound. The `Float64` slab is the same slab —
    # `transplant_datum!` copies the datum across, so that both element types differentiate one
    # function of the unknowns rather than two fits of it.
    for T in (Float64, Float32)
        entries = generate_problems(T; rng = Xoshiro(SEED))
        reference_entries = T === Float64 ? entries :
                            generate_problems(Float64; rng = Xoshiro(SEED))
        legs = vcat(
            [(label = name === :wave ? "wave" : "sinegordon",
                 problem = clause_problem(name; T = T),
                 reference = clause_problem(name)) for name in (:wave, :sinegordon)],
            [(label = "entry $i", problem = entry.problem,
                 reference = reference_entries[i].problem)
             for (i, entry) in enumerate(entries)])
        for leg in legs, form in (:abg, :tsr), mode in (:all, :last_layer)
            problem = leg.problem
            ansatz = clause_ansatz(problem; form = form, mode = mode)
            slab = fixed_slab(problem, ansatz; solver = Newton(), verbosity = 0)
            reference = fixed_slab(leg.reference,
                clause_ansatz(leg.reference; form = form, mode = mode);
                solver = Newton(), verbosity = 0)
            c = cache(slab.int)
            s = solver(slab.int)
            p = parameters(problem)
            # at the slab's guess, as a run reaches the Jacobian: a `:last_layer` slab whose held
            # hidden layer is still zero has a zero Jacobian, whatever its residual is
            M.initial_guess!(current(slab.solstep), nothing, p, slab.int)
            T === Float64 ||
                transplant_datum!(reference, slab)
            M.initial_guess!(current(reference.solstep), nothing,
                parameters(leg.reference), reference.int)
            for _ in 1:16
                θ = 2 .* rand(rng, T, length(c.x)) .- 1
                SimpleSolvers.jacobian!(s, θ, (current(slab.solstep), p, slab.int))
                J = copy(SimpleSolvers.jacobianmatrix(s))
                @test maximum(abs, J) ≥ T(1 // 100)
                for _ in 1:3
                    d = T.(2 .* rand(drng, length(c.x)) .- 1)
                    @test maximum(abs,
                        J * d - directional_differences(reference.int, Float64.(θ),
                            Float64.(d), parameters(leg.reference))) ≤
                          sqrt(eps(T)) * maximum(abs, J) * norm(d)
                end
            end
        end
    end
end
