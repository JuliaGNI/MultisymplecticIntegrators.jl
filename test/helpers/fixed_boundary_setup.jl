# The shared setup of the fixed-boundary test files `test/couplings/fixed_boundary*.jl`: the oracle's
# gradient floor, the clause's slab and problem, and the differences the files compare against. Each
# file includes it after its own `using` lines and the helpers it names.

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

"""
    directional_differences(int, θ, d, params)

The central difference of a residual at `θ` along the direction `d`, `(r(θ + δd) − r(θ − δd)) / 2δ`
with `δ = cbrt(eps(T)) · max(1, max|θ|) / max|d|`: two residuals, which approximate `J · d`.
"""
function directional_differences(int, θ::AbstractVector{T}, d::AbstractVector{T}, params) where {T}
    δ = cbrt(eps(T)) * max(one(T), maximum(abs, θ)) / maximum(abs, d)
    forward = zeros(T, length(θ))
    backward = zeros(T, length(θ))
    M.residual!(forward, θ .+ δ .* d, nothing, params, int)
    M.residual!(backward, θ .- δ .* d, nothing, params, int)
    return (forward .- backward) ./ (2δ)
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
