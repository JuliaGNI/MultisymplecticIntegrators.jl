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

const M = MultisymplecticIntegrators
const WF = M.Wave
const SG = M.SineGordon

# ---- helpers -------------------------------------------------------------------------

"""
    oracle_floor(oracle, datum, θ; perturbations = 8, seed = 0x5eed, rows = :)

The round-off floor of the oracle's own gradient at `oracle_vector(oracle, datum, θ)`, which is the
datum level's free parameters followed by `θ`: the largest change in that gradient, in the
residual's layout — `p₀`'s rows and then `∂S/∂Θ`'s, as [`oracle_residual`](@ref) builds them —
over `perturbations` seeded perturbations, each of `eps(T)` relative to an entry and an absolute
`eps(T)` below one, read at the residual's rows `rows` alone (`:` is every row). It reads the
oracle's action and nothing of the package.
"""
function oracle_floor(oracle::FixedBoundaryOracle, datum, θ::AbstractVector{T};
        perturbations = 8, seed = 0x5eed, rows = :) where {T}
    a = oracle.ansatz
    nfp, nf = length(M.spatial_free(a)), M.nfree(a)
    z = oracle_vector(oracle, datum, θ)
    function gradient(v)
        g = action_gradient(w -> fixed_boundary_action(oracle, datum, w), v)
        return vcat(view(g, 1:nfp), view(g, (2nfp + 1):(nfp + nf)))
    end
    g₀ = gradient(z)
    rng = Xoshiro(seed)
    floor = zero(T)
    for _ in 1:perturbations
        direction = 2 .* rand(rng, T, length(z)) .- 1
        z₊ = z .+ eps(T) .* direction .* max.(one(T), abs.(z))
        floor = max(floor, maximum(abs, view(gradient(z₊) - g₀, rows)))
    end
    return floor
end

"""
    compare_gradient(b, ∇S, T; oracle, datum, θ, label)

The package's residual `b` against the oracle's gradient `∇S`: both finite, the gradient at least
`1/100` in its largest entry, and the difference at most `max(100 eps(T) · scale, F)`, `scale` that
largest entry and `F` the oracle's own gradient floor at this point, from
[`oracle_floor`](@ref). `F` is asserted at most `1e6 eps(T) · scale` and recorded, with the miss,
as a multiple of `eps(T) · scale` under `label`.
"""
function compare_gradient(b, ∇S, ::Type{T}; oracle, datum, θ, label = (;)) where {T}
    @test all(isfinite, b)
    @test all(isfinite, ∇S)
    scale = maximum(abs, ∇S)
    @test scale ≥ T(1 // 100)
    floor = oracle_floor(oracle, datum, θ)
    @test floor ≤ 1e6 * eps(T) * scale
    @test maximum(abs, b - ∇S) ≤ max(100 * eps(T) * scale, floor)
    @info("FB1 the gradient's floor", VERSION, T=string(T), entry=label.entry,
        form=label.form, mode=label.mode, slab=label.slab, scale=scale, floor=floor,
        ratio=Float64(floor/(eps(T)*scale)),
        miss=Float64(maximum(abs, b-∇S)/(eps(T)*scale)),
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(),
        check_bounds=Base.JLOptions().check_bounds)
    return nothing
end

"""A fixed-boundary network on a problem's span and component count."""
function fixed_ansatz(problem; Sx = 8, St = 10, form = :abg, mode = :all, ncells = 3,
        nq = (6, 4))
    return FixedBoundaryNetwork(Sx = Sx, St = St, bubble_form = form, mode = mode,
        xspan = xspan(problem), nq = nq, ncells = ncells, D = components(problem))
end

"""
The clause's own slab of a problem: the network and the bubble of `Sx = 8` and `St = 10` units, on
`nq = (6, 4)` nodes and 20 cells of its span. Every testset that asserts one of the section's
numbers runs on it; a testset whose claim holds at any rule and whose cost would otherwise dominate
— the continuation's layout, the predictor, the two fit workspaces, the refused options — runs
`fixed_ansatz`'s coarser rule with the same network and bubble.
"""
function clause_ansatz(problem; Sx = 8, St = 10, form = :abg, mode = :all)
    return fixed_ansatz(problem; Sx = Sx, St = St, form = form, mode = mode, ncells = 20)
end

"""
`Wave` or `SineGordon` at the clause's time step, on `[0, 1]` with `xstep = 0.05`, over `steps`
slabs of `timestep` each. A testset that runs more than one step names the count its clause gives.
"""
function clause_problem(name::Symbol; timestep = 0.05, steps = 1, T = Float64)
    problem = name === :wave ? WF.problem : SG.problem
    return problem(
        T = T, timespan = (zero(T), T(steps * timestep)), timestep = T(timestep),
        xstep = T(0.05), xspan = (zero(T), one(T)))
end

"""The integrator of one slab, and the solution step a run drives."""
function fixed_slab(problem, ansatz; guess = Continuation(), kwargs...)
    method = SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = guess)
    int = GeometricIntegrator(problem, method; kwargs...)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    slab = (; problem, method, int, solution, solstep, ansatz)
    next_fixed_slab!(slab, 1)
    return slab
end

"""Move a slab to step `n` and fix its datum, as a run does before it solves the slab."""
function next_fixed_slab!(slab, n::Integer)
    reset!(slab.solstep, timesteps(slab.solution)[n])
    c = cache(slab.int)
    M.slab_bottom!(c, c.coupling, slab.problem, slab.ansatz, current(slab.solstep))
    return nothing
end

"""The datum of the coupling's block, as the oracle reads it."""
function fixed_datum(c)
    (; phi = copy(c.coupling.phi), momentum = copy(c.coupling.momentum),
        start = c.coupling.start)
end

"""
    transplant_datum!(slab, source)

Give `slab` the datum of `source`'s slab, converted to `slab`'s element type: the level, the
momentum it received, the slab's side data at its nodes and its start time, and the block's
parameter vector, which holds the parameters a `:last_layer` mode keeps. Two slabs of one problem at
two element types then differentiate one function of the unknowns, and the `Float64` slab's central
differences are the `Float32` one's derivative.
"""
function transplant_datum!(slab, source)
    c, s = cache(slab.int), cache(source.int)
    copyto!(c.coupling.phi, eltype(c.coupling.phi).(s.coupling.phi))
    copyto!(c.coupling.momentum, eltype(c.coupling.momentum).(s.coupling.momentum))
    copyto!(c.coupling.ga, eltype(c.coupling.ga).(s.coupling.ga))
    copyto!(c.coupling.gb, eltype(c.coupling.gb).(s.coupling.gb))
    copyto!(c.coupling.dga, eltype(c.coupling.dga).(s.coupling.dga))
    copyto!(c.coupling.dgb, eltype(c.coupling.dgb).(s.coupling.dgb))
    c.coupling.start = eltype(c.coupling.start)(s.coupling.start)
    copyto!(c.ansatz.params, eltype(c.ansatz.params).(s.ansatz.params))
    return nothing
end

"""One component of a level's network, written here from its definition alone."""
function level_network(problem, a, φ, x, t, d)
    xa, xb = xspan(problem)
    L = xb - xa
    ŝ = (x - xa) / L
    p = parameters(problem)
    ga = problem.bcs_function(t, p).left.u[d]
    gb = problem.bcs_function(t, p).right.u[d]
    m = zero(eltype(φ))
    for i in 1:a.Sx
        m += φ[2a.Sx + (i - 1) * a.D + d] * tanh(φ[i] * ŝ + φ[a.Sx + i])
    end
    return (1 - ŝ) * ga + ŝ * gb + (x - xa) * (xb - x) * m
end

"""
    top_velocity(oracle, problem, ansatz, φ0, φ1, Θ, t₀, x, d)

The velocity of a solved slab's field at its top time, `∂_{t̂}u / h`, from the oracle's own field:
the value a predictor of the slab after it starts its substeps from. `φ0` is that slab's datum, `φ1`
the level its solve produced and `Θ` its bubble.
"""
function top_velocity(oracle, problem, ansatz, φ0, φ1, Θ, t₀, x, d)
    h = timestep(problem)
    p = parameters(problem)
    t₁ = t₀ + h
    at0 = problem.bcs_function(t₀, p)
    at1 = problem.bcs_function(t₁, p)
    dga = ForwardDiff.derivative(τ -> problem.bcs_function(τ, p).left.u[d], t₁)
    dgb = ForwardDiff.derivative(τ -> problem.bcs_function(τ, p).right.u[d], t₁)
    sides = (at0.left.u[d], at0.right.u[d], at1.left.u[d], at1.right.u[d],
        at1.left.u[d], at1.right.u[d], dga, dgb)
    _, ut, _ = oracle_component(oracle, ansatz, d, φ0, φ1, Θ, one(h), x, h, sides,
        ansatz.Sx, ansatz.St, ansatz.bubble_form)
    return ut / h
end

"""Central differences of a residual at `θ`, one column per unknown."""
function residual_differences(int, θ::AbstractVector{T}, params) where {T}
    n = length(θ)
    J = zeros(T, n, n)
    forward = zeros(T, n)
    backward = zeros(T, n)
    x = copy(θ)
    for k in 1:n
        δ = cbrt(eps(T)) * max(one(T), abs(θ[k]))
        x[k] = θ[k] + δ
        M.residual!(forward, x, nothing, params, int)
        up = x[k]
        x[k] = θ[k] - δ
        M.residual!(backward, x, nothing, params, int)
        J[:, k] .= (forward .- backward) ./ (up - x[k])
        x[k] = θ[k]
    end
    return J
end

"""The residual of a slab at `θ`, in a vector of `θ`'s element type."""
function slab_residual(int, θ, params)
    b = zeros(eltype(θ), length(θ))
    M.residual!(b, θ, nothing, params, int)
    return b
end

"""
    residual_floor(int, θ, params, cut)

The part of an affine slab's residual at `θ` that lies outside the range the solver keeps, relative to
that residual: the least ratio any step from `θ` can reach. The Jacobian is the central difference of
the residual at unit steps, which is exact for an affine residual. Its range is taken as the
solver's `LapackPivotedQR` takes it: the leading columns of `Q` of a column-pivoted QR, as long a run
as the moduli of `R`'s diagonal exceed `cut` times the first. An SVD's rank rule is a different
one, and at `Float32` it keeps up to four directions fewer on these slabs (`scripts/fb8_floor.jl`).
A residual that does not depend on `θ` has rank `0` and floor `1`.
"""
function residual_floor(int, θ::AbstractVector{T}, params, cut) where {T}
    r₀ = slab_residual(int, θ, params)
    J = similar(θ, length(r₀), length(θ))
    for j in eachindex(θ)
        e = zero(θ)
        e[j] = one(T)
        J[:, j] = (slab_residual(int, θ .+ e, params) .-
                   slab_residual(int, θ .- e, params)) ./ 2
    end
    F = qr(J, ColumnNorm())
    d = abs.(diag(F.R))
    rank = something(findfirst(≤(cut * d[1]), d), length(d) + 1) - 1
    Q = Matrix(F.Q)[:, 1:rank]
    return norm(r₀ - Q * (Q' * r₀)) / norm(r₀)
end

const PROBLEMS = Dict(
    :wave =>
        () -> WF.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05,
            xspan = (0.0, 1.0)),
    :sinegordon =>
        () -> SG.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05,
            xspan = (0.0, 1.0)))

# ---- FB0 -----------------------------------------------------------------------------

@testset "FB0 the two types and their pair" begin
    problem = PROBLEMS[:wave]()
    a = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0))
    @test FixedBoundary() isa SlabCoupling
    @test a isa SpaceTimeAnsatz
    @test (M.nspatial(a), M.nbubble(a), M.nfree(a)) == (12, 16, 28)
    last_layer = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0), mode = :last_layer)
    @test (M.nspatial(last_layer), length(M.spatial_free(last_layer)),
        M.nfree(last_layer)) == (12, 4, 8)
    # At `D = 2` a level frees `Sx·D` output weights, which are not the first `Sx` of them: the
    # level's parameters are `Sx` weights, `Sx` biases and then `Sx·D` output weights.
    wide = FixedBoundaryNetwork(
        Sx = 8, St = 10, xspan = (0.0, 1.0), mode = :last_layer, D = 2)
    @test M.nspatial(wide) == 2 * 8 + 8 * 2
    @test M.spatial_free(wide) == (2 * 8 + 1):(2 * 8 + 8 * 2)
    @test M.bubble_free(wide) == (3 * 10 + 1):(3 * 10 + 10 * 2)
    @test M.nfree(wide) == 8 * 2 + 10 * 2
    @test M.free_indices(wide) == vcat(collect(17:32), 32 .+ collect(31:50))
    network = NetworkAnsatz(S = 4, xspan = (0.0, 1.0), ncells = (2, 5), degree = 3)
    spline = TensorBSplineAnsatz(degree = 3, ncells = (2, 5), xspan = (0.0, 1.0))
    # `@test_throws` matches its `Regex` against `sprint(showerror, e)`, which prefixes the
    # exception's type, so the anchors hold only when the type is part of the pattern.
    @test_throws r"^ArgumentError: FixedBoundary takes a FixedBoundaryNetwork, got NetworkAnsatz$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(network, FixedBoundary()))
    @test_throws r"^ArgumentError: FixedBoundary takes a FixedBoundaryNetwork, got TensorBSplineAnsatz$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(spline, FixedBoundary()))
    for coupling in (WeakInitialAndBoundaryConditions(), WeakInitialCondition(),
        CoefficientRestriction())
        @test_throws Regex("^ArgumentError: a FixedBoundaryNetwork takes the FixedBoundary " *
                           "coupling, got $(nameof(typeof(coupling)))\$") GeometricIntegrator(
            problem, SpaceTimeGalerkin(a, coupling))
    end
    # `HardConstraintLifting` takes a `NetworkAnsatz` or a `TensorBSplineAnsatz` and names those two
    # in its own refusal; a `FixedBoundaryNetwork` is neither, so the refusal that answers is this
    # intersection's.
    @test_throws r"^ArgumentError: a FixedBoundaryNetwork takes the FixedBoundary coupling, got HardConstraintLifting$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(a, HardConstraintLifting()))
    method = SpaceTimeGalerkin(a, FixedBoundary())
    int = GeometricIntegrator(problem, method)
    @test default_solver(method) isa LevenbergMarquardt
    @test solver(int) isa M.LevenbergMarquardtSolver
    @test length(cache(int).x) == 28
    # A `PINNTraining` fit on a fixed-boundary network matches the network's refusal for any fit
    # method and `PINNTraining`'s refusal for any ansatz, so it reaches this intersection's.
    @test_throws r"^ArgumentError: a FixedBoundaryNetwork is fitted by BubbleGreedy, got PINNTraining$" M.fit!(
        zeros(1), PINNTraining(epochs = 0), a, nothing, cache(int))
    @test M.coupling_unknowns(FixedBoundary(), a, problem) == 0
    @test M.free_rows(FixedBoundary(), a) == 28
    @test total_free_coefficients(method, problem) == 28
    # A `PreviousSlab` target has no field map for this ansatz, so the guess is refused and the
    # refusal names the target it was given.
    @test_throws r"^ArgumentError: a FixedBoundaryNetwork takes InitialCondition, ExactSolution, TargetFunction and Predictor as targets, got PreviousSlab$" GeometricIntegrator(
        problem, SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = FitTo(PreviousSlab())))
end

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

# ---- FB2 -----------------------------------------------------------------------------

@testset "FB2 the Jacobian is the derivative of the residual" begin
    rng = Xoshiro(11)
    # `jacobian!` is SimpleSolvers' and writes the solver's own matrix, so it exists for a
    # `NonlinearSolver`. The family's Levenberg–Marquardt keeps no such matrix — it takes one SVD of
    # the Jacobian at each accepted iterate — so the Jacobian of the residual is read through the
    # `Newton()` solver the family also offers, on the same residual and the same block.
    #
    # N5's and N18's second check: the legs are `Wave`, `SineGordon` and all 24 seeded entries, in
    # both forms and both modes, at 16 vectors each, and at both element types. The differences are
    # taken on the `Float64` slab in both precisions, as the symbolic ansatz's are: the residual's
    # round-off at `Float32` sits in differences `cbrt(eps(Float32))` apart, where its own truncation
    # error is larger than the bound. The `Float64` slab is the same slab — `transplant_datum!`
    # copies the datum across, so that both element types differentiate one function of the unknowns
    # rather than two fits of it.
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
                @test maximum(abs,
                    J - residual_differences(reference.int, Float64.(θ),
                        parameters(leg.reference))) ≤ sqrt(eps(T)) * maximum(abs, J)
            end
        end
    end
end

# ---- FB3 -----------------------------------------------------------------------------

@testset "FB3 the first momentum and the momentum handed on" begin
    # N6: the datum of a run's first slab — the fit of the level to `u₀`, and `p₀` against the
    # oracle's own integral of the density's momentum, on the free parameters of the mode
    for name in (:wave, :sinegordon), mode in (:all, :last_layer)

        problem = clause_problem(name)
        ansatz = clause_ansatz(problem; mode = mode)
        slab = fixed_slab(problem, ansatz)
        c = cache(slab.int)
        p = parameters(problem)
        # the fit's sup error on the output grid, whose 21 points are `x = 0:0.05:1`
        xs = collect(range(xspan(problem)[1], xspan(problem)[2]; length = 21))
        initial = problem.ics_function(xs, p)
        fitted = [level_network(problem, ansatz, c.coupling.phi, x, 0.0, d)
                  for x in xs, d in 1:ansatz.D]
        @test maximum(abs, fitted .- initial.u) ≤ 1e-5
        # the covector of the mode's own parameters: `Sx·D` under `:last_layer`
        @test length(c.coupling.momentum) == length(M.spatial_free(ansatz))
        nodes, weights = ansatz.tabulation.xnode, ansatz.tabulation.xweight
        ics = problem.ics_function(collect(nodes), p)
        expect = zeros(length(c.coupling.momentum))
        for r in eachindex(nodes)
            g = ForwardDiff.gradient(
                φ -> sum(level_network(problem, ansatz, φ, nodes[r],
                             0.0, d) for d in 1:ansatz.D),
                c.coupling.phi)
            for (k, i) in enumerate(M.spatial_free(ansatz)), d in 1:ansatz.D

                expect[k] += weights[r] *
                             derivative_v(problem.density, ics.u[r, d],
                                 ics.v[r, d], ics.w[r, d], p) * g[i]
            end
        end
        @test all(isfinite, expect)
        @test maximum(abs, c.coupling.momentum - expect) ≤
              100 * eps(Float64) * maximum(abs, expect)
    end
    # N7: the momentum handed on, at the final iterate of a step that is capped at three LM
    # iterations and of one `Newton()` takes, unchanged by a further residual evaluation and by a
    # second step
    for solvermethod in (LevenbergMarquardt(), Newton())
        problem = clause_problem(:wave; steps = 2)
        ansatz = clause_ansatz(problem)
        method = SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = Continuation())
        int = GeometricIntegrator(problem, method; solver = solvermethod,
            max_iterations = 3, verbosity = 0)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        c = cache(int)
        p = parameters(problem)
        # A step's own time is the end of the slab it solves — `slab_start` reads one timestep back
        # from it — so a run's first step is placed at `timesteps(solution)[1]`, the first slab's
        # end, as the helper `next_fixed_slab!` places it.
        reset!(solstep, timesteps(solution)[1])
        M.slab_bottom!(c, c.coupling, problem, ansatz, current(solstep))
        datum = fixed_datum(c)
        oracle = ActionOracle(problem, method; block = c.ansatz)
        integrate!(solstep, int)
        solved = copy(c.x)
        handed = copy(c.coupling.p_next)
        solved_level = copy(c.ansatz.params[1:M.nspatial(ansatz)])
        scale = maximum(abs, handed)
        @test all(isfinite, handed)
        @test maximum(abs,
            oracle_momentum(oracle, datum, solved) - handed) ≤ 100 * eps(Float64) * scale
        # a further residual evaluation at another vector rewrites no part of the record — the
        # block's own parameters it does rewrite, which is why the level is read before it
        other = 2 .* rand(MersenneTwister(11), Float64, length(solved)) .- 1
        b = zeros(length(solved))
        M.residual!(b, other, nothing, p, int)
        @test c.coupling.p_next == handed
        # the second slab's datum is that block and that level, exactly
        next_fixed_slab!((; problem, method, int, solution, solstep, ansatz), 2)
        @test c.coupling.momentum == handed
        @test c.coupling.phi == solved_level
    end
end

# ---- FB4 -----------------------------------------------------------------------------

@testset "FB4 the solution carries the side data at every step" begin
    # The state is component-major: component `d`'s `nx` output-grid values start at
    # `(d - 1) * nx + 1`, and the grid's ends are the span's.
    function ends(u, D)
        nx = length(u) ÷ D
        return [(u[(d - 1) * nx + 1], u[d * nx]) for d in 1:D]
    end
    for name in (:wave, :sinegordon), form in (:abg, :tsr)

        problem = clause_problem(name; steps = 3)
        method = SpaceTimeGalerkin(clause_ansatz(problem; form = form), FixedBoundary())
        result = integrate(problem, method; max_iterations = 3, verbosity = 0)
        @test length(result.traces) == 3
        for n in eachindex(result.stats.converged)
            t = result.solution.t[n]
            bcs = problem.bcs_function(t, parameters(problem))
            scale = max(1, maximum(abs, result.solution.u[n]))
            for (d, (left, right)) in enumerate(ends(result.solution.u[n],
                components(problem)))
                @test abs(left - bcs.left.u[d]) ≤ 8 * eps(Float64) * scale
                @test abs(right - bcs.right.u[d]) ≤ 8 * eps(Float64) * scale
            end
        end
    end
    # The suite's three `:general` entries and one `:symmetric` entry, at the clause's 20 cells: the
    # identity holds at every node of any rule, and the two kinds carry different side data — a
    # `:symmetric` entry's two ends agree and vary in time, a `:general` entry's differ.
    entries = generate_problems(Float64; rng = Xoshiro(SEED))
    legs = vcat([(entry = entries[i], kind = :general) for i in (13, 14, 15)],
        [(entry = entries[9], kind = :symmetric)])
    for leg in legs, form in (:abg, :tsr)

        @test leg.entry.kind === leg.kind
        problem = leg.entry.problem
        method = SpaceTimeGalerkin(clause_ansatz(problem; form = form), FixedBoundary())
        result = integrate(problem, method; max_iterations = 3, verbosity = 0)
        for n in eachindex(result.stats.converged)
            bcs = problem.bcs_function(result.solution.t[n], parameters(problem))
            scale = max(1, maximum(abs, result.solution.u[n]))
            for (d, (left, right)) in enumerate(ends(result.solution.u[n],
                components(problem)))
                @test abs(left - bcs.left.u[d]) ≤ 8 * eps(Float64) * scale
                @test abs(right - bcs.right.u[d]) ≤ 8 * eps(Float64) * scale
            end
        end
    end
end

# ---- FB5 -----------------------------------------------------------------------------

@testset "FB5 variable projection" begin
    for name in (:wave, :sinegordon), form in (:abg, :tsr)

        problem = name === :wave ?
                  WF.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05,
            xspan = (0.0, 1.0)) :
                  SG.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05,
            xspan = (0.0, 1.0))
        ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = form,
            xspan = (0.0, 1.0), ncells = 20, nq = (6, 4), D = components(problem))
        slab = fixed_slab(problem, ansatz; guess = FitTo(Predictor()),
            solver = LevenbergMarquardt(variable_projection = true), max_iterations = 5,
            verbosity = 0)
        int, c = slab.int, cache(slab.int)
        p = parameters(problem)
        slot = length(M.variable_projection_indices(slab.method))
        @test slot == ansatz.St * ansatz.D
        @test M.variable_projection_indices(slab.method) ==
              collect((M.nfree(ansatz) - slot + 1):M.nfree(ansatz))
        projected = M.variable_projection_indices(slab.method)
        n = length(c.x)
        oracle = ActionOracle(problem, slab.method; block = c.ansatz)
        datum = fixed_datum(c)
        M.initial_guess!(current(slab.solstep), nothing, p, int)
        θ₀ = copy(c.x)
        # A fixed-boundary guess is a function of the slab's datum alone, so a second application of
        # it writes the same vector; that is what lets the two applications a step makes agree.
        M.initial_guess!(current(slab.solstep), nothing, p, int)
        @test c.x == θ₀
        f! = (b, z) -> (M.residual!(b, z, nothing, p, int); b)
        # 1. the inner solve: after it, the eliminated block is at a root of its own equations, and
        # the block was not already there — both at the oracle's gradient
        g₀ = oracle_residual(oracle, datum, θ₀)
        scale = maximum(abs, view(g₀, projected))
        @test scale ≥ sqrt(eps(Float64))
        x = copy(θ₀)
        r = slab_residual(int, x, p)
        reduced, _ = M.project!(x, f!, r, M.jacobian_of(f!, x, n), projected)
        floor = maximum(abs, view(oracle_residual(oracle, datum, x), projected))
        # the same rule as FB1's, at the rows FB5 measures: the inner solve's residual is compared
        # with the larger of the old bound and the round-off floor of the oracle's own gradient at
        # this point, read at the projected rows
        noise = oracle_floor(oracle, datum, x; rows = projected)
        @test noise ≤ 1e6 * eps(Float64) * scale
        @test floor ≤ max(100 * eps(Float64) * scale, noise)
        @info("FB5 the inner solve",
            VERSION, T="Float64", name=name, form=form, scale=scale,
            floor=floor, noise=noise, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
        @info("FB5 the inner solve's floor", VERSION, T="Float64", name=name, form=form,
            ratio=Float64(noise/(eps(Float64)*scale)),
            machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(),
            check_bounds=Base.JLOptions().check_bounds)
        # 2. the Schur complement against central differences of z ↦ F_o(z, a*(z)), each inner solve
        # warm-started at the solved `a`
        solved = M.lm_solved_indices(n, projected)
        function reduced_residual(z)
            rp = slab_residual(int, z, p)
            M.project!(copy(z), f!, rp, M.jacobian_of(f!, z, n), projected)
            return view(rp, solved)
        end
        z = copy(x)
        worst = zero(Float64)
        for (k, j) in enumerate(solved)
            δ = cbrt(eps(Float64)) / 10 * max(1, abs(x[j]))
            up = copy(z)
            up[j] += δ
            down = copy(z)
            down[j] -= δ
            difference = (reduced_residual(up) .- reduced_residual(down)) ./ (2δ)
            ratio = maximum(abs, difference - view(reduced, :, k)) /
                    (sqrt(eps(Float64)) * maximum(abs, reduced))
            worst = max(worst, ratio)
            @test ratio ≤ 1
        end
        @info("FB5 the reduced Jacobian against differences",
            VERSION, T="Float64", name=name, form=form,
            worst=worst, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
        # 3. one projected step: its recorded residual is the test's own full-system `‖F‖`
        guess = c.x
        r₀ = norm(slab_residual(int, guess, p))
        integrate!(slab.solstep, int)
        x₁ = copy(c.x)
        @test c.iterations[end] ≤ 5
        @test abs(c.residuals[end] - norm(slab_residual(int, x₁, p))) ≤
              8 * eps(Float64) * max(1, c.residuals[end])
        @test abs(c.initial_residuals[end] - r₀) ≤ 8 * eps(Float64) * max(1, r₀)
    end
end

# ---- FB6 -----------------------------------------------------------------------------

@testset "FB6 the record of a step under each form and solve" begin
    @test default_solver(SpaceTimeGalerkin(fixed_ansatz(PROBLEMS[:wave]()),
        FixedBoundary())) isa LevenbergMarquardt
    for name in (:wave, :sinegordon), form in (:abg, :tsr),
        solve in (:lm, :projected, :newton)
        problem = PROBLEMS[name]()
        a = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = form,
            xspan = xspan(problem), ncells = 20, nq = (6, 4), D = components(problem))
        m = SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = FitTo(Predictor()))
        solvermethod = if solve === :lm
            LevenbergMarquardt()
        elseif solve === :projected
            LevenbergMarquardt(variable_projection = true)
        else
            Newton()
        end
        int = GeometricIntegrator(problem, m; solver = solvermethod, max_iterations = 3,
            verbosity = 0)
        c = cache(int)
        p = parameters(problem)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        options = config(solver(int))
        ns = M.nspatial(a)
        solved = Float64[]
        for n in 1:2
            reset!(solstep, timesteps(solution)[n])
            M.slab_bottom!(c, c.coupling, problem, a, current(solstep))
            # The datum of a later slab is the level the step before it solved, and not whatever
            # the guess of that step left in the block's parameter vector: a run under `integrate!`
            # applies the guess twice, and the second application overwrote that vector.
            n == 2 && @test c.coupling.phi == solved
            # A fixed-boundary guess is a function of the slab's datum alone — the fit and the
            # predictor read the datum, not the block's parameter vector — so the vector measured
            # here is the one the step's own guess writes; FB5 asserts that at the first slab.
            M.initial_guess!(current(solstep), nothing, p, int)
            r₀ = norm(slab_residual(int, c.x, p))
            integrate!(solstep, int)
            r₁ = norm(slab_residual(int, c.x, p))
            @test length(c.iterations) == n
            @test c.iterations[end] ≤ 3
            # the record against the test's own full-system norms, at the guess and at the final one
            @test abs(c.residuals[end] - r₁) ≤ 8 * eps(Float64) * max(1, r₁)
            @test abs(c.initial_residuals[end] - r₀) ≤ 8 * eps(Float64) * max(1, r₀)
            @test isfinite(c.residuals[end]) && isfinite(c.initial_residuals[end])
            @test c.converged[end] ==
                  M.step_converged(c.residuals[end], c.initial_residuals[end], options)
            solved = copy(c.ansatz.params[1:ns])
        end
    end
end

# ---- FB7 -----------------------------------------------------------------------------

@testset "FB7 the LM routine against a closed form" begin
    f!(b, x) = (b .= atan.(x))
    x = [2.0, -3.0]
    result = M.lm_solve!(x, f!, 2; max_iterations = 300, f_abstol = 0.0, f_reltol = 1e-12)
    @test result.converged
    @test norm(x) ≤ 1e-8
    @test result.residual == norm(atan.(x))
    @test result.initial_residual > 1
    y = [NaN, 0.0]
    nan = M.lm_solve!(y, f!, 2; max_iterations = 300, f_abstol = 0.0, f_reltol = 1e-8)
    @test nan.iterations == 0
    @test !nan.converged
end

# ---- FB8 -----------------------------------------------------------------------------

@testset "FB8 the affine control in two iterations" begin
    # N13 and N18's third check: the plan's control is `:last_layer` on a linear density, with the
    # spatial hidden layer held at `φ₀`'s fit for the whole run — which is the `Continuation()`
    # guess — and the unknowns `c` and `a`, in both forms and in both element types.
    # `FitTo(Predictor())` is not the control: it moves level 1's hidden layer away from level 0's,
    # and the affine system then has no solution (K50).
    #
    # The slab's Jacobian is rank-deficient, and the part of the initial residual outside its range is
    # a floor that no step removes: on entry 10 of the generated family it is `1.51e-10` of the
    # initial residual at the first slab, and one, two, three and five iterations all end there. The
    # floor is a property of the slab's data, and it moves with the machine's round-off. So each step
    # is held to the larger of the tolerance and ten times its own floor, which a dense pivoted QR of
    # the step's Jacobian measures outside the solver, and the step converges wherever that floor is
    # a tenth of the tolerance or less. The floor itself is bounded by `√f_reltol`: the largest on
    # these slabs is `1.29e-9` at `Float64` and `1.63e-2` at `Float32`, and a residual that has lost
    # its dependence on the unknowns has floor `1`, which would otherwise pass every check here.
    for T in (Float64, Float32), form in (:abg, :tsr)

        linear = [entry.problem
                  for entry in generate_problems(T; rng = Xoshiro(SEED))
                  if entry.potential == 2]
        for (i, problem) in enumerate(vcat([clause_problem(:wave; T = T)], linear))
            ansatz = clause_ansatz(problem; mode = :last_layer, form = form)
            method = SpaceTimeGalerkin(ansatz, FixedBoundary();
                initial_guess = Continuation())
            int = GeometricIntegrator(
                problem, method; solver = Newton(), max_iterations = 2,
                f_reltol = fixed_boundary_f_reltol(T), verbosity = 0)
            c = cache(int)
            p = parameters(problem)
            solution = GeometricSolution(problem)
            solstep = solutionstep(int, solution[0])
            floors = T[]
            for n in 1:ntime(problem)
                reset!(solstep, timesteps(solution)[n])
                M.slab_bottom!(c, c.coupling, problem, ansatz, current(solstep))
                M.initial_guess!(current(solstep), nothing, p, int)
                push!(floors, residual_floor(int, c.x, p, M.fixed_boundary_qr_cut(T)))
                integrate!(solstep, int)
            end
            ratios = c.residuals ./ c.initial_residuals
            reltol = fixed_boundary_f_reltol(T)
            label = i == 1 ? "wave" : "entry $(i - 1)"
            @info("FB8 the affine control's ratio and floor per step", label, T, form,
                ratios, floors, cpu = Sys.CPU_NAME)
            @testset "$label, $T, $form" begin
                @test length(ratios) == ntime(problem)
                @test all(c.iterations .≤ 2)
                @test all(floors .≤ sqrt(reltol))
                @test all(ratios .≤ max.(reltol, 10 .* floors))
                @test all(c.converged .| (floors .> reltol / 10))
                @test all(c.initial_residuals .≥ T(1 // 100))
            end
        end
    end
    @test fixed_boundary_f_reltol(Float64) == 1e-8
    @test fixed_boundary_f_reltol(Float32) ≥ 1e-8
    @test fixed_boundary_f_reltol(Float32) != Float32(1e-8)
    # N14: a third element type is refused rather than given a tolerance the family never measured,
    # and the refusal names it — matched whole and anchored, as every refusal of this file is.
    @test_throws r"^ArgumentError: the fixed-boundary network has no measured f_reltol at BigFloat; name the tolerance as an option of its own$" fixed_boundary_f_reltol(
        BigFloat)
    # A caller who names the tolerance at such a type runs: the option reaches the solver's config
    # unmoved, and the same build without it raises the same refusal.
    big = WF.problem(T = BigFloat, timespan = (big"0.0", big"0.1"), timestep = big"0.05",
        xstep = big"0.05", xspan = (big"0.0", big"1.0"))
    big_method = SpaceTimeGalerkin(fixed_ansatz(big; Sx = 4, St = 4), FixedBoundary())
    big_int = GeometricIntegrator(big, big_method; f_reltol = big"1e-30")
    @test eltype(nlsolution(cache(big_int))) === BigFloat
    @test config(solver(big_int)).f_reltol == big"1e-30"
    @test_throws r"^ArgumentError: the fixed-boundary network has no measured f_reltol at BigFloat; name the tolerance as an option of its own$" GeometricIntegrator(
        big, big_method)
    # the family's rank cut is one value per element type, `1e-12` at `Float64` and `10 eps(T)`
    # for another type
    @test M.fixed_boundary_qr_cut(Float64) == 1e-12
    @test M.fixed_boundary_qr_cut(Float32) == 10 * eps(Float32)
    problem = PROBLEMS[:wave]()
    int = GeometricIntegrator(problem,
        SpaceTimeGalerkin(fixed_ansatz(problem), FixedBoundary()))
    options = config(solver(int))
    @test options.f_abstol == 0
    @test options.f_reltol == fixed_boundary_f_reltol(Float64)
    @test options.max_iterations == 300
end

# ---- N14 -----------------------------------------------------------------------------

@testset "N14 the family's tolerance reaches every solver" begin
    # N14: `default_options` hands the tolerance on unresolved, and the family's `initsolver` fills
    # it in — the one method every nonlinear solver of the family reaches, because
    # `GeometricIntegrator` merges `default_options` before the caller's keywords. A solver that
    # names no tolerance of its own is solved at the element type's measured value, whichever
    # nonlinear solver it is.
    for T in (Float64, Float32), s in (SimpleSolvers.DogLeg(), SimpleSolvers.Picard())

        problem = clause_problem(:wave; T = T)
        int = GeometricIntegrator(problem,
            SpaceTimeGalerkin(fixed_ansatz(problem), FixedBoundary());
            solver = s, max_iterations = 5)
        @test config(solver(int)).f_reltol == fixed_boundary_f_reltol(T)
    end
    # at an element type the family has not measured, a caller that names the tolerance keeps it ...
    big = WF.problem(T = BigFloat, timespan = (big"0.0", big"0.1"), timestep = big"0.05",
        xstep = big"0.05", xspan = (big"0.0", big"1.0"))
    big_method = SpaceTimeGalerkin(fixed_ansatz(big; Sx = 4, St = 4), FixedBoundary())
    named = GeometricIntegrator(big, big_method; solver = SimpleSolvers.DogLeg(),
        f_reltol = big"1e-30")
    @test config(solver(named)).f_reltol == big"1e-30"
    # ... and a caller that names none is refused, whole and anchored, before a solver is built
    @test_throws r"^ArgumentError: the fixed-boundary network has no measured f_reltol at BigFloat; name the tolerance as an option of its own$" GeometricIntegrator(
        big, big_method; solver = SimpleSolvers.DogLeg())
end

# ---- FB9 -----------------------------------------------------------------------------

@testset "FB9 the dead branch on Wave over twenty steps" begin
    problem = WF.problem(timespan = (0.0, 1.0), timestep = 0.05, xstep = 0.05,
        xspan = (0.0, 1.0))
    ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = :abg, xspan = (0.0, 1.0),
        ncells = 20, nq = (6, 4))
    method = SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = Continuation())
    result = integrate(problem, method; solver = Newton(), max_iterations = 60,
        f_abstol = 0.0, f_reltol = fixed_boundary_f_reltol(Float64), verbosity = 0)
    stats = result.stats
    @test length(stats.converged) == 20
    @test all(stats.converged)
    t = result.solution.t
    u = result.solution.u
    grid = xspan(problem)
    xs = collect(range(grid[1], grid[2]; length = 21))
    error = maximum(abs(WF.solution_u(t[end], x, parameters(problem)) - u[end][i])
    for (i, x) in enumerate(xs))
    @info("FB9 the dead branch's twenty steps", VERSION, T="Float64", error=error,
        worst_ratio=maximum(stats.residuals ./ stats.initial_residuals),
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(),
        check_bounds=Base.JLOptions().check_bounds)
    @test 5.96e-4 ≤ error ≤ 7.28e-4
    # N20's `Float32` arm: two steps, recorded, and only the element types asserted
    problem32 = WF.problem(T = Float32, timespan = (0.0f0, 0.1f0), timestep = 0.05f0,
        xstep = 0.05f0, xspan = (0.0f0, 1.0f0))
    ansatz32 = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = :abg,
        xspan = (0.0f0, 1.0f0), ncells = 20, nq = (6, 4))
    method32 = SpaceTimeGalerkin(ansatz32, FixedBoundary(); initial_guess = Continuation())
    result32 = integrate(problem32, method32; solver = Newton(), max_iterations = 60,
        f_abstol = 0.0f0, f_reltol = fixed_boundary_f_reltol(Float32), verbosity = 0)
    @info("FB9 the dead branch's Float32 arm", VERSION, T="Float32",
        ratios=result32.stats.residuals ./ result32.stats.initial_residuals,
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(),
        check_bounds=Base.JLOptions().check_bounds)
    @test length(result32.stats.converged) == 2
    assert_no_upcast(result32.solution.u, Float32)
    @test all(isreal, result32.stats.residuals)
end

# ---- FB10 ----------------------------------------------------------------------------

@testset "FB10 the continuation starts at φ₀ and the previous Θ" begin
    for form in (:abg, :tsr)
        problem = PROBLEMS[:wave]()
        ansatz = fixed_ansatz(problem; form = form)
        slab = fixed_slab(problem, ansatz)
        c = cache(slab.int)
        nfp = length(M.spatial_free(ansatz))
        M.initial_guess!(current(slab.solstep), nothing, parameters(problem), slab.int)
        start = M.bubble_start(ansatz, Float64)
        @test copy(c.x[1:nfp]) ≈ c.coupling.phi[M.spatial_free(ansatz)]
        @test copy(c.x[(nfp + 1):end]) ≈ start[M.bubble_free(ansatz)]
        @test all(iszero, view(start, (3ansatz.St + 1):length(start)))
        integrate!(slab.solstep, slab.int)
        solved = copy(c.x)
        next_fixed_slab!(slab, 2)
        M.initial_guess!(current(slab.solstep), nothing, parameters(problem), slab.int)
        @test copy(c.x[1:nfp]) ≈ c.coupling.phi[M.spatial_free(ansatz)]
        @test copy(c.x[(nfp + 1):end]) ≈ solved[(nfp + 1):end]
    end
end

# ---- FB11 ----------------------------------------------------------------------------

@testset "FB11 a unit at r = 0" begin
    problem = PROBLEMS[:wave]()
    a = FixedBoundaryNetwork(Sx = 8, St = 9, bubble_form = :tsr, xspan = (0.0, 1.0),
        ncells = 3, nq = (4, 3))
    @test_throws r"^ArgumentError: the bubble's unit 5 has α = β = 0, that is r = 0, and no \(θ, s, ρ\) coordinates: exp\(ρ\) is never zero$" M.bubble_start(
        a, Float64)
    abg = FixedBoundaryNetwork(Sx = 8, St = 9, bubble_form = :abg, xspan = (0.0, 1.0),
        ncells = 3, nq = (4, 3))
    @test length(M.bubble_start(abg, Float64)) == M.nbubble(abg)
    method = SpaceTimeGalerkin(abg, FixedBoundary())
    result = integrate(problem, method; max_iterations = 2, verbosity = 0)
    @test isfinite(norm(result.solution.u[end]))
    atoms = bubble_atoms(BubbleGreedy(), Float64)
    @test size(atoms, 1) == 2184
    @test !any(iszero(atoms[i, 1]) && iszero(atoms[i, 2]) for i in axes(atoms, 1))
end

# ---- FB12 ----------------------------------------------------------------------------

@testset "FB12 the predictor against the test's own copy" begin
    for name in (:wave, :sinegordon)
        problem = PROBLEMS[name]()
        ansatz = fixed_ansatz(problem)
        slab = fixed_slab(problem, ansatz; max_iterations = 3, verbosity = 0)
        c = cache(slab.int)
        tb = ansatz.tabulation
        p = parameters(problem)
        h = timestep(problem)
        grid = collect(range(xspan(problem)[1], xspan(problem)[2]; length = 101))
        dx = grid[2] - grid[1]
        dt = h / 10
        # The density's one constant `m`, computed here rather than read from the code under test:
        # `∂L/∂v = m v`, so `m` is that derivative at `v = 1` with `u` and `w` at zero.
        mass = derivative_v(problem.density, 0.0, 1.0, 0.0, p)
        # The test's own Störmer–Verlet: ten substeps, central differences, the side data imposed at
        # every substep of the slab that starts at `t₀`, and nothing of the package's predictor but
        # the density's own derivatives.
        function stormer_verlet(u, v, t₀)
            accel(u, v) = begin
                a = zeros(length(u))
                w = similar(u)
                for j in eachindex(u)
                    j1, j2 = max(j - 1, 1), min(j + 1, length(u))
                    w[j] = (u[j2] - u[j1]) / ((j2 - j1) * dx)
                end
                du = zeros(length(u))
                dw = zeros(length(u))
                for j in eachindex(u)
                    du[j] = derivative_u(problem.density, u[j], v[j], w[j], p)
                    dw[j] = derivative_w(problem.density, u[j], v[j], w[j], p)
                end
                for j in 2:(length(u) - 1)
                    a[j] = (du[j] - (dw[j + 1] - dw[j - 1]) / (2dx)) / mass
                end
                a
            end
            U = zeros(11, length(u))
            U[1, :] .= u
            a = accel(u, v)
            for k in 1:10
                v .+= dt / 2 .* a
                u .+= dt .* v
                u[1] = problem.bcs_function(t₀ + k * dt, p).left.u[1]
                u[end] = problem.bcs_function(t₀ + k * dt, p).right.u[1]
                a = accel(u, v)
                v .+= dt / 2 .* a
                U[k + 1, :] .= u
            end
            return U
        end
        at(values, x) = begin
            j = clamp(floor(Int, (x - grid[1]) / dx) + 1, 1, length(grid) - 1)
            λ = (x - grid[j]) / dx
            (1 - λ) * values[j] + λ * values[j + 1]
        end
        sample = zeros(Float64, length(tb.tnode), length(tb.xnode), ansatz.D)
        function compare(U)
            for q in eachindex(tb.tnode), r in eachindex(tb.xnode)

                position = tb.tnode[q] * 10
                k = clamp(floor(Int, position) + 1, 1, 10)
                λ = position - (k - 1)
                expect = (1 - λ) * at(U[k, :], tb.xnode[r]) +
                         λ * at(U[k + 1, :], tb.xnode[r])
                @test abs(sample[q, r, 1] - expect) ≤
                      1e3 * eps(Float64) * max(1, abs(expect))
            end
            return nothing
        end
        # the first slab: from the datum's level and the initial data's velocity
        M.evaluate!(sample, Predictor(), c, ansatz, problem, current(slab.solstep))
        initial = problem.ics_function(grid, p)
        u = [level_network(problem, ansatz, c.coupling.phi, x, 0.0, 1) for x in grid]
        compare(stormer_verlet(u, collect(initial.v), 0.0))
        # the second slab: from the level the step produced and the velocity at that slab's top
        datum = copy(c.coupling.phi)
        integrate!(slab.solstep, slab.int)
        ns = M.nspatial(ansatz)
        φ1 = copy(c.ansatz.params[1:ns])
        Θ = copy(c.ansatz.params[(ns + 1):end])
        oracle = ActionOracle(problem, slab.method; block = c.ansatz)
        velocity = [top_velocity(oracle, problem, ansatz, datum, φ1, Θ, 0.0, x, 1)
                    for x in grid]
        next_fixed_slab!(slab, 2)
        @test c.coupling.phi == φ1
        M.evaluate!(sample, Predictor(), c, ansatz, problem, current(slab.solstep))
        compare(stormer_verlet([level_network(problem, ansatz, φ1, x, h, 1) for x in grid],
            velocity, h))
        @test isfinite(maximum(abs, c.ansatz.top_sample))
    end
    transport = M.LinearTransport.problem(timespan = (0.0, 0.1), timestep = 0.05,
        xstep = 0.05, xspan = (0.0, 1.0))
    ansatz = fixed_ansatz(transport; Sx = 2, St = 2)
    @test_throws r"^ArgumentError: Predictor\(\) needs a density whose ∂L/∂v" GeometricIntegrator(
        transport,
        SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = FitTo(Predictor())))
    # A density whose `∂L/∂v` is nonlinear vanishes at zero velocity and gives one value at
    # `v = 1` too: only a second velocity shows that its ratio is not the constant `m`.
    u_, v_, w_ = Symbolics.variable.((:u, :v, :w))
    cubic = LagrangianDensity(v_^3 / 3, u_, v_, w_)
    @test_throws r"^ArgumentError: Predictor\(\) needs one constant ∂L/∂v / v" M.predictor_mass(
        cubic, NamedTuple(), Float64, 1)
    wave = PROBLEMS[:wave]()
    spline = TensorBSplineAnsatz(degree = 3, ncells = (1, 5), xspan = (0.0, 1.0))
    @test_throws r"^ArgumentError: Predictor\(\) needs a FixedBoundaryNetwork; TensorBSplineAnsatz has no slab predictor$" GeometricIntegrator(
        wave, SpaceTimeGalerkin(spline, WeakInitialCondition();
            initial_guess = FitTo(Predictor())))
end

# ---- FB13 ----------------------------------------------------------------------------

@testset "FB13 a second bubble fit on one block reads its own dictionary" begin
    problem = PROBLEMS[:wave]()
    ansatz = fixed_ansatz(problem)
    slab = fixed_slab(problem, ansatz)
    c = cache(slab.int)
    tb = ansatz.tabulation
    sample = zeros(Float64, length(tb.tnode), length(tb.xnode), ansatz.D)
    M.evaluate!(sample, InitialCondition(), c, ansatz, problem, current(slab.solstep))
    w = collect(c.ansatz.params)
    M.fit!(w, BubbleGreedy(γ_range = (-6.0, 6.0)), ansatz, sample, c)
    M.fit!(w, BubbleGreedy(γ_range = (-3.0, 3.0)), ansatz, sample, c)
    second = copy(w)
    chosen = c.ansatz.fit.selected[1:ansatz.St]
    @test all(c.ansatz.fit.atoms[i, 3] ≤ 3.0 for i in chosen)
    @test c.ansatz.fit.method.γ_range == (-3.0, 3.0)
    c.ansatz.fit.method = nothing
    w2 = collect(c.ansatz.params)
    M.fit!(w2, BubbleGreedy(γ_range = (-3.0, 3.0)), ansatz, sample, c)
    @test second ≈ w2
end

# ---- FB14 ----------------------------------------------------------------------------

@testset "FB14 no silent upcast" begin
    problem = WF.problem(T = Float32, timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05,
        xspan = (0.0, 1.0))
    for form in (:abg, :tsr), solve in (:newton, :lm)

        a = fixed_ansatz(problem; form = form)
        m = SpaceTimeGalerkin(a, FixedBoundary())
        solvermethod = solve === :lm ? LevenbergMarquardt() : Newton()
        # The cache is read after two steps, not as `Cache` built it: the datum, the momentum the
        # steps handed on and the fit's dictionary are written by the run.
        int = GeometricIntegrator(problem, m; solver = solvermethod, max_iterations = 2,
            verbosity = 0)
        c = cache(int)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        for n in 1:2
            reset!(solstep, timesteps(solution)[n])
            M.slab_bottom!(c, c.coupling, problem, a, current(solstep))
            M.initial_guess!(current(solstep), nothing, parameters(problem), int)
            integrate!(solstep, int)
        end
        result = integrate(problem, m; solver = solvermethod, max_iterations = 2,
            verbosity = 0)
        assert_no_upcast(result.solution.u, Float32)
        assert_no_upcast(c.x, Float32)
        assert_no_upcast(c.coupling.momentum, Float32)
        assert_no_upcast(c.coupling.phi, Float32)
        assert_no_upcast(c.coupling.p_next, Float32)
        assert_no_upcast(c.coupling.phi_next, Float32)
        assert_no_upcast(c.coupling.v_top, Float32)
        assert_no_upcast(c.ansatz.params, Float32)
        assert_no_upcast(c.ansatz.momentum, Float32)
        assert_no_upcast(c.ansatz.top_sample, Float32)
        assert_no_upcast(c.ansatz.fit.atoms, Float32)
        assert_no_upcast(c.ansatz.fit.table, Float32)
        assert_no_upcast(c.u, Float32)
        assert_no_upcast(c.v, Float32)
        assert_no_upcast(c.w, Float32)
    end
end

# ---- FB15 ----------------------------------------------------------------------------

@testset "FB15 the two forms are one scheme" begin
    # N8: at the same greedy atoms the bubble is one bubble in two charts, so the field and its two
    # derivatives agree at every node, the `:tsr` residual is the `:abg` residual after the chart's
    # transpose, and no parameter of either form is wrapped into a range.
    problem = clause_problem(:wave)
    abg = clause_ansatz(problem; form = :abg)
    tsr = clause_ansatz(problem; form = :tsr)
    slab_abg = fixed_slab(problem, abg)
    slab_tsr = fixed_slab(problem, tsr)
    ca, ct = cache(slab_abg.int), cache(slab_tsr.int)
    p = parameters(problem)
    M.initial_guess!(current(slab_abg.solstep), nothing, p, slab_abg.int)
    M.initial_guess!(current(slab_tsr.solstep), nothing, p, slab_tsr.int)
    # the same greedy atoms in both blocks, each in its own coordinates: the dictionary is one
    # dictionary, so atom `k` is the same bubble in either form, and the output weights are one
    # nonzero number in both
    atoms = bubble_atoms(BubbleGreedy(), Float64)
    St, nsp = tsr.St, M.nspatial(tsr)
    for k in 1:St
        i = 1 + (k - 1) * 218
        α, β, γ = atoms[i, 1], atoms[i, 2], atoms[i, 3]
        ca.x[nsp + k], ca.x[nsp + St + k], ca.x[nsp + 2St + k] = α, β, γ
        θ, s, ρ = M.bubble_chart(α, β, γ, k)
        ct.x[nsp + k], ct.x[nsp + St + k], ct.x[nsp + 2St + k] = θ, s, ρ
    end
    first_weight = nsp + 3St + 1
    ca.x[first_weight:end] .= 0.3
    ct.x[first_weight:end] .= 0.3
    hidden = (nsp + 1):(nsp + 3St)
    b_abg = zeros(Float64, length(ca.x))
    b_tsr = zeros(Float64, length(ct.x))
    M.residual!(b_abg, ca.x, nothing, p, slab_abg.int)
    M.residual!(b_tsr, ct.x, nothing, p, slab_tsr.int)
    scale = max(1, maximum(abs, ct.u))
    @test maximum(abs, ca.u - ct.u) ≤ 1e3 * eps(Float64) * scale
    @test maximum(abs, ca.v - ct.v) ≤ 1e3 * eps(Float64) * maximum(abs, ct.v)
    @test maximum(abs, ca.w - ct.w) ≤ 1e3 * eps(Float64) * maximum(abs, ct.w)
    # the chart, written here from its definition: `(α, β, γ) = (e^ρ cos θ, e^ρ sin θ, −e^ρ s)`
    chart(θ, s, ρ) = (exp(ρ) * cos(θ), exp(ρ) * sin(θ), -exp(ρ) * s)
    nf = length(ct.x)
    G = zeros(Float64, nf, nf)
    for k in 1:nf
        G[k, k] = 1
    end
    for k in 1:St
        θ, s, ρ = ct.x[nsp + k], ct.x[nsp + St + k], ct.x[nsp + 2St + k]
        J = ForwardDiff.jacobian(z -> collect(chart(z[1], z[2], z[3])), [θ, s, ρ])
        for i in 1:3, j in 1:3

            G[nsp + (i - 1) * St + k, nsp + (j - 1) * St + k] = J[i, j]
        end
    end
    r_abg = slab_residual(slab_abg.int, ca.x, p)
    r_tsr = slab_residual(slab_tsr.int, ct.x, p)
    # the comparison below is evidence only while the field depends on the atoms and the residual's
    # hidden rows are not zero: the first step's guess writes the output weights as zero, and at zero
    # the chart's transpose blocks multiply nothing
    @test maximum(abs, ca.x[first_weight:end]) > 0
    @test maximum(abs, r_abg[hidden]) > 0
    @test maximum(abs, r_tsr - transpose(G) * r_abg) ≤
          1e3 * eps(Float64) * maximum(abs, transpose(G) * r_abg)
    # nothing is wrapped or clamped: the `:tsr` start's angles are the chart's own, unwrapped
    start = M.bubble_start(tsr, Float64)
    grid = M.bubble_start(abg, Float64)
    for k in 1:St
        α, β, γ = grid[k], grid[St + k], grid[2St + k]
        @test start[k] === atan(β, α)
        @test start[St + k] === -γ / sqrt(α^2 + β^2)
    end
end

# ---- FB16 ----------------------------------------------------------------------------

@testset "FB16 a failed step is carried and its record kept" begin
    # N22: a step that does not reach the tolerance leaves its last accepted iterate, the momentum
    # of that iterate and its three numbers; the next slab starts from them, and `on_failure` is
    # the one place that acts on the record.
    problem = clause_problem(:wave; timestep = 0.05, steps = 2)
    method = SpaceTimeGalerkin(clause_ansatz(problem), FixedBoundary();
        initial_guess = FitTo(Predictor()))
    int = GeometricIntegrator(problem, method; solver = LevenbergMarquardt(),
        max_iterations = 1, verbosity = 0)
    c = cache(int)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    reset!(solstep, timesteps(solution)[1])
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), current(solstep))
    M.initial_guess!(current(solstep), nothing, parameters(problem), int)
    integrate!(solstep, int)
    level = copy(c.ansatz.params[1:M.nspatial(ansatz(method))])
    momentum = copy(c.coupling.p_next)
    @test !c.converged[end]
    @test c.iterations[end] == 1
    @test isfinite(c.residuals[end]) && isfinite(c.initial_residuals[end])
    # the next slab's datum is what the failed step left
    reset!(solstep, timesteps(solution)[2])
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), current(solstep))
    @test c.coupling.phi == level
    @test c.coupling.momentum == momentum
    # the run completes and records both steps, and `on_failure` acts on the record. A failing step
    # warns once, so the run's log is exactly the two failures and no other record.
    result = integrate(problem, method; solver = LevenbergMarquardt(), max_iterations = 1,
        verbosity = 0)
    @test length(result.stats.converged) == 2
    @test !result.stats.converged[1]
    @test length(result.traces) == 2
    @test_logs (:warn, r"did not converge at timestep n=1") (:warn,
        r"did not converge at timestep n=2") integrate(
        problem, method; solver = LevenbergMarquardt(), max_iterations = 1, verbosity = 0,
        on_failure = :warn)
    @test_throws r"did not converge at timestep n=1" integrate(problem, method;
        solver = LevenbergMarquardt(), max_iterations = 1, verbosity = 0,
        on_failure = :error)
end

# ---- FB17 ----------------------------------------------------------------------------

@testset "FB17 the gauge's two periods are a record" begin
    # N21: the dead branch over two wave periods with `Continuation()` and the minimum-norm step,
    # at the clause's rule of 20 cells. The numbers are a record — the largest `‖(w, b)‖∞` over the
    # run, the steps that converge, the error at the end and the run's cost — and the testset
    # asserts none of them; the two statements the clause does make, that the action holds no gauge
    # term and that the solve re-fits no `φ` to a fixed field, are its source's and not a
    # measurement. The record carries the resolution, the machine and the run it was taken on, and
    # no number of it is compared with the spike's.
    problem = WF.problem(timespan = (0.0, 8.0), timestep = 0.05, xstep = 0.05,
        xspan = (0.0, 1.0))
    ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = :abg, xspan = (0.0, 1.0),
        ncells = 20, nq = (6, 4))
    method = SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = Continuation())
    int = GeometricIntegrator(problem, method; solver = Newton(), max_iterations = 60,
        f_abstol = 0.0, f_reltol = fixed_boundary_f_reltol(Float64), verbosity = 0)
    c = cache(int)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    steps = length(timesteps(solution)) - 1
    started = time()
    # `‖(w, b)‖∞` read after every step and kept as the largest: the parameters of a step's last
    # accepted iterate are the state the drift is measured on, and the value after the last step
    # alone misses a crossing that returns
    drift = zero(Float64)
    for n in 1:steps
        reset!(solstep, timesteps(solution)[n])
        integrate!(solstep, int)
        copy!(solution, current(solstep), n)
        drift = max(drift, maximum(abs, c.ansatz.params[1:(2 * ansatz.Sx)]))
    end
    elapsed = time() - started
    xs = collect(range(xspan(problem)[1], xspan(problem)[2]; length = 21))
    t_end = timesteps(solution)[end]
    final_error = maximum(abs(WF.solution_u(t_end, x, parameters(problem)) -
                              solution.u[end][i])
    for (i, x) in enumerate(xs))
    record = (steps = length(c.converged), converged = count(c.converged), drift = drift,
        error = final_error, T = Float64, nodes = length(ansatz.tabulation.xnode),
        h = timestep(problem), seconds = round(elapsed; digits = 1), version = VERSION,
        machine = Sys.MACHINE, cpu = Sys.CPU_NAME, threads = Threads.nthreads(),
        blas_threads = BLAS.get_num_threads(), check_bounds = Base.JLOptions().check_bounds)
    @info "FB17 the gauge's two periods" record
    @test steps == length(c.converged)
    @test all(isfinite, c.residuals)
    @test all(isfinite, c.ansatz.params)
end

# ---- FB18 ----------------------------------------------------------------------------

@testset "FB18 the family's rows build and the plateau is a record" begin
    # N23: the registry holds the family's 32 entries — 30 of the matrix and 2 of the plateau probe
    # — each with a distinct stem and a resolution, one of them at `Sx = 12`, `St = 16` and
    # `h = 0.025` and one at `BigFloat`, and the plateau's numbers are a record and no assertion.
    entries = family_entries(:fixed_boundary)
    @test length(entries) == 32
    @test length(unique(entry.stem for entry in entries)) == 32
    @test all(entry -> entry.steps ≥ 1, entries)
    matrix = filter(
        entry -> occursin("fixed-boundary", entry.stem) &&
                 !occursin("plateau", entry.stem),
        entries)
    @test length(matrix) == 30
    @test length(Set((entry.problem, entry.form, entry.solver, entry.timestep)
    for entry in matrix)) == 30
    # a matrix row integrates its ladder up to `t = 1`, so that the three resolutions of one
    # problem compare one span
    @test all(entry -> entry.steps * entry.timestep ≥ 1 - 1e-12, matrix)
    # a driver given no stem runs the cheapest row: the largest time step, at `Float64`
    cheapest = smallest_entry(entries)
    @test cheapest.timestep == maximum(entry.timestep for entry in entries)
    @test cheapest.T === Float64
    plateau = filter(entry -> occursin("plateau", entry.stem), entries)
    @test length(plateau) == 2
    @test any(entry -> entry.Sx == 12 && entry.St == 16 && entry.timestep == 0.025, plateau)
    @test any(entry -> entry.T === BigFloat, plateau)
    @test all(entry -> entry.steps == 1, filter(entry -> entry.T === BigFloat, plateau))
    # N23: the extended-precision row names its own tolerance, because the family measures none at
    # `BigFloat`; the row's driver and this testset build the solver's keywords through one
    # function, so the entry's value is the one the solver is built with
    extended = only(filter(entry -> entry.T === BigFloat, plateau))
    @test extended.f_reltol == big"1e-8"
    extended_int = GeometricIntegrator(fixed_boundary_problem(extended),
        fixed_boundary_method(extended); solver = M.LevenbergMarquardt(),
        fixed_boundary_solver_keywords(extended)...)
    @test config(solver(extended_int)).f_reltol == extended.f_reltol
    for entry in entries
        @test fixed_boundary_problem(entry) isa M.LagrangianPDEProblem
        @test fixed_boundary_method(entry) isa SpaceTimeGalerkin
    end
end
