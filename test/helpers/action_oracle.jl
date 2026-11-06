using ForwardDiff: ForwardDiff
using GeometricBase: parameters, timespan, timestep
using LinearAlgebra: Diagonal
using MultisymplecticIntegrators
import SimpleSplines

# The independent discrete action of one slab, the oracle of every residual of the property suite.
#
# The oracle reads the problem through its density, called as `L(u, v, w, params)`, its two data
# functions, its parameters, timestep, timespan, components and space interval; the ansatz through
# its basis and its quadrature, evaluated with SimpleSplines; and the coupling through its type
# alone. It differentiates the action with ForwardDiff. It reads nothing of the package's assembly,
# and lays out the solver's vector by its own formula.

"""
    ActionOracle(problem, method; restricted = coupling(method) isa CoefficientRestriction,
                 lift = nothing)

The tables of the discrete action of one slab of `method` on `problem`.

`restricted` says whether the coupling restricts the bottom row, as `CoefficientRestriction` does;
a test coupling that restricts it under another type says so here. `lift`, where given, is a
function `(t̂, x, d) -> (ℓ, ∂_t̂ℓ, ∂ₓℓ)` that the coupling adds to component `d` of the field: the
action is then that of the ansatz's field plus `ℓ`.

The action of the full coefficient array `ĉ`, of size `(Nt, Nx, D)`, is

    S(ĉ) = Σ_q Σ_r ωₜ_q ωₓ_r h L(u, ∂_t̂u / h, ∂ₓu)

at the nodes `(t̂_q, x_r)` of the ansatz's rule, and the coupling adds its terms to it
([`slab_action`](@ref)). `φ`, `φ′` are the time functions and their derivatives at the time nodes,
`ψ`, `ψ′` the space functions and their derivatives at the space nodes, `φ₀`, `φ₁` the time
functions at `t̂ = 0` and `t̂ = 1`, and `ψa`, `ψb` the space functions at the two ends of the span.
`Λ` holds the functions of the bottom and top multipliers at the space nodes — the space basis, or
the space basis without its two corner functions where the coupling imposes the space boundary
conditions — and `χ` those of the multipliers of the two ends at the time nodes, the whole time
basis. `nodes` and `interpolation` are the spline nodes of the space basis and the basis there.
"""
struct ActionOracle{T, P, C, F}
    problem::P
    coupling::C
    restricted::Bool
    lift::F
    nt::Int
    nx::Int
    D::Int
    nλ::Int
    nμ::Int
    h::T
    ωt::Vector{T}
    t̂::Vector{T}
    ωx::Vector{T}
    x::Vector{T}
    φ::Matrix{T}
    φ′::Matrix{T}
    ψ::Matrix{T}
    ψ′::Matrix{T}
    φ₀::Vector{T}
    φ₁::Vector{T}
    ψa::Vector{T}
    ψb::Vector{T}
    Λ::Matrix{T}
    χ::Matrix{T}
    nodes::Vector{T}
    interpolation::Matrix{T}
end

function ActionOracle(problem, method;
        restricted::Bool = coupling(method) isa CoefficientRestriction, lift = nothing)
    T = eltype(timespan(problem))
    a = ansatz(method)
    c = coupling(method)
    tb, xb = SimpleSplines.bases(SimpleSplines.basis(a))
    qt, qx = SimpleSplines.quadratures(MultisymplecticIntegrators.quadrature(a))
    t̂ = collect(T, SimpleSplines.quadrature_nodes(qt))
    x = collect(T, SimpleSplines.quadrature_nodes(qx))
    nt = SimpleSplines.nbasis(tb)
    nx = SimpleSplines.nbasis(xb)
    xa, xb_end = xspan(problem)
    table(basis, points, n, order) = T[SimpleSplines.evaluate(basis, j, p, order)
                                       for p in points, j in 1:n]
    ψ = table(xb, x, nx, 0)
    corners = c isa WeakInitialAndBoundaryConditions
    λindices = corners ? (2:(nx - 1)) : (1:nx)
    nodes = collect(T, SimpleSplines.nodes(xb))
    return ActionOracle{T, typeof(problem), typeof(c), typeof(lift)}(
        problem, c, restricted, lift, nt, nx, components(problem),
        restricted ? 0 : length(λindices), corners ? nt : 0,
        T(timestep(problem)),
        collect(T, SimpleSplines.quadrature_weights(qt)), t̂,
        collect(T, SimpleSplines.quadrature_weights(qx)), x,
        table(tb, t̂, nt, 0), table(tb, t̂, nt, 1), ψ, table(xb, x, nx, 1),
        vec(table(tb, [zero(T)], nt, 0)), vec(table(tb, [one(T)], nt, 0)),
        vec(table(xb, [xa], nx, 0)), vec(table(xb, [xb_end], nx, 0)),
        ψ[:, λindices], table(tb, t̂, nt, 0), nodes, table(xb, nodes, nx, 0))
end

restricted(o::ActionOracle) = o.restricted

"""The number of free time rows: `Nt - 1` under the restriction, whose bottom row is a datum."""
free_time_rows(o::ActionOracle) = restricted(o) ? o.nt - 1 : o.nt

"""The length of the solver's vector of a slab."""
unknown_count(o::ActionOracle) = free_time_rows(o) * o.nx * o.D + (o.nλ + 2 * o.nμ) * o.D

# The solver's index of the free row `a`, space function `j` and component `d`: the time index
# runs fastest, then the space index, then the component.
function coefficient_entry(o::ActionOracle, a, j, d)
    a + (j - 1) * free_time_rows(o) +
    (d - 1) * free_time_rows(o) * o.nx
end

# The first index of the multiplier blocks: the top multiplier, then the multipliers of `x = a`,
# then those of `x = b`, each component-major.
multiplier_base(o::ActionOracle) = free_time_rows(o) * o.nx * o.D

"""
    SlabDatum{T}

The datum of one slab, which the oracle computes itself: the slab's start time, the bottom row and
the momentum load `F` of the restriction, the trace `u₀` the initial condition is compared with, at
the space nodes, the bottom multiplier `λ₀`, and the side data `g_a`, `g_b` at the time nodes.
"""
struct SlabDatum{T}
    start::T
    bottom::Matrix{T}
    load::Matrix{T}
    target::Matrix{T}
    λ₀::Matrix{T}
    ga::Matrix{T}
    gb::Matrix{T}
end

"""
    decode(o, datum, θ)

The full coefficient array `ĉ` and the multipliers `λ`, `μ_a`, `μ_b` of the solver's vector `θ`.
"""
function decode(o::ActionOracle, datum::SlabDatum, θ::AbstractVector)
    S = eltype(θ)
    ĉ = zeros(S, o.nt, o.nx, o.D)
    shift = restricted(o) ? 1 : 0
    for d in 1:o.D, j in 1:o.nx

        restricted(o) && (ĉ[1, j, d] = datum.bottom[j, d])
        for a in 1:free_time_rows(o)
            ĉ[a + shift, j, d] = θ[coefficient_entry(o, a, j, d)]
        end
    end
    base = multiplier_base(o)
    λ = S[θ[base + (d - 1) * o.nλ + p] for p in 1:o.nλ, d in 1:o.D]
    base += o.nλ * o.D
    μa = S[θ[base + (d - 1) * o.nμ + p] for p in 1:o.nμ, d in 1:o.D]
    base += o.nμ * o.D
    μb = S[θ[base + (d - 1) * o.nμ + p] for p in 1:o.nμ, d in 1:o.D]
    return ĉ, λ, μa, μb
end

# The Lagrangian at one node: summed over the components for a density of one component, and
# called on the `D`-vectors of the node otherwise.
function node_lagrangian(L::LagrangianDensity, u, v, w, params)
    sum(
        L(u[d], v[d], w[d], params) for d in eachindex(u))
end
node_lagrangian(L, u, v, w, params) = L(u, v, w, params)

# The gradient of the action, with one chunk size for every slab, so that a slab of another size
# compiles no new method. Every slab of the suite has at least 8 coefficients.
function action_gradient(f, z)
    ForwardDiff.gradient(f, z,
        ForwardDiff.GradientConfig(f, z, ForwardDiff.Chunk{8}()))
end

# The momentum `∂L/∂v` at one node, by ForwardDiff of the Lagrangian of that node.
function node_momentum(L, u, v, w, params)
    ForwardDiff.gradient(
        s -> node_lagrangian(L, u, s, w, params), v)
end

"""
    interior_action(o, ĉ)

`S(ĉ) = Σ_q Σ_r ωₜ_q ωₓ_r h L(u, ∂_t̂u / h, ∂ₓu)`, with the field and its derivatives at the nodes
of the rule from the coefficients `ĉ`.
"""
function interior_action(o::ActionOracle, ĉ::AbstractArray{S, 3}) where {S}
    U = [o.φ * ĉ[:, :, d] * transpose(o.ψ) for d in 1:o.D]
    V = [o.φ′ * ĉ[:, :, d] * transpose(o.ψ) / o.h for d in 1:o.D]
    W = [o.φ * ĉ[:, :, d] * transpose(o.ψ′) for d in 1:o.D]
    if o.lift !== nothing
        for d in 1:o.D, q in eachindex(o.t̂), r in eachindex(o.x)
            ℓ, ℓt, ℓx = o.lift(o.t̂[q], o.x[r], d)
            U[d][q, r] += ℓ
            V[d][q, r] += ℓt / o.h
            W[d][q, r] += ℓx
        end
    end
    params = parameters(o.problem)
    action = zero(S)
    for q in eachindex(o.t̂), r in eachindex(o.x)

        L = node_lagrangian(o.problem.density, [U[d][q, r] for d in 1:o.D],
            [V[d][q, r] for d in 1:o.D], [W[d][q, r] for d in 1:o.D], params)
        action += o.ωt[q] * o.ωx[r] * o.h * L
    end
    return action
end

"""
    slab_action(o, datum, ĉ, λ₀, λ, μa, μb)

The action of the slab with the coupling's terms:

- the restriction: `+ Σ_{j,d} F_jd ĉ_1jd`;
- both multiplier couplings: `+ Σ_r ωₓ_r [λ₀(x_r)·(u(0, x_r) − u₀(x_r)) − λ(x_r)·u(1, x_r)]`;
- the coupling of the space boundary conditions in addition:
  `+ Σ_q h ωₜ_q [μ_a(t̂_q)·(u(t̂_q, a) − g_a(t_q)) − μ_b(t̂_q)·(u(t̂_q, b) − g_b(t_q))]`.
"""
function slab_action(o::ActionOracle, datum::SlabDatum, ĉ, λ₀, λ, μa, μb)
    action = interior_action(o, ĉ)
    if restricted(o)
        for d in 1:o.D, j in 1:o.nx

            action += datum.load[j, d] * ĉ[1, j, d]
        end
        return action
    end
    for d in 1:o.D
        bottom = o.ψ * (transpose(ĉ[:, :, d]) * o.φ₀)
        top = o.ψ * (transpose(ĉ[:, :, d]) * o.φ₁)
        incoming = o.Λ * λ₀[:, d]
        outgoing = o.Λ * λ[:, d]
        for r in eachindex(o.x)
            action += o.ωx[r] *
                      (incoming[r] * (bottom[r] - datum.target[r, d]) -
                       outgoing[r] * top[r])
        end
        o.nμ == 0 && continue
        left = o.φ * (ĉ[:, :, d] * o.ψa)
        right = o.φ * (ĉ[:, :, d] * o.ψb)
        ma = o.χ * μa[:, d]
        mb = o.χ * μb[:, d]
        for q in eachindex(o.t̂)
            action += o.h * o.ωt[q] *
                      (ma[q] * (left[q] - datum.ga[q, d]) -
                       mb[q] * (right[q] - datum.gb[q, d]))
        end
    end
    return action
end

"""
    oracle_residual(o, datum, θ)

The residual of the slab at the solver's vector `θ`, in the solver's layout, from the gradient of
[`slab_action`](@ref): `∂S/∂ĉ` on the equation rows of the coupling — the rows `1:Nt-1` under the
restriction, all `Nt` rows otherwise — then `−∂S/∂λ₀` on the rows of the top multiplier, and
`−∂S/∂μ_a` and `+∂S/∂μ_b` on the rows of the two ends.
"""
function oracle_residual(o::ActionOracle, datum::SlabDatum, θ::AbstractVector{T}) where {T}
    ĉ, λ, μa, μb = decode(o, datum, θ)
    parts = cumsum((0, length(ĉ), length(datum.λ₀), length(μa), length(μb)))
    z = vcat(vec(ĉ), vec(datum.λ₀), vec(μa), vec(μb))
    block(v, k) = v[(parts[k] + 1):parts[k + 1]]
    function action(z)
        return slab_action(o, datum, reshape(block(z, 1), o.nt, o.nx, o.D),
            reshape(block(z, 2), o.nλ, o.D), λ, reshape(block(z, 3), o.nμ, o.D),
            reshape(block(z, 4), o.nμ, o.D))
    end
    g = action_gradient(action, z)
    ∂c = reshape(block(g, 1), o.nt, o.nx, o.D)
    ∂λ₀, ∂μa, ∂μb = block(g, 2), block(g, 3), block(g, 4)
    R = zeros(T, unknown_count(o))
    for d in 1:o.D, j in 1:o.nx, k in 1:free_time_rows(o)
        R[coefficient_entry(o, k, j, d)] = ∂c[k, j, d]
    end
    base = multiplier_base(o)
    R[(base + 1):(base + length(∂λ₀))] .= -∂λ₀
    base += length(∂λ₀)
    R[(base + 1):(base + length(∂μa))] .= -∂μa
    base += length(∂μa)
    R[(base + 1):(base + length(∂μb))] .= ∂μb
    return R
end

"""
    top_momentum(o, datum, θ)

`p = ∂S/∂ĉ` of the top row of the interior action at the solver's vector `θ`, of size `(Nx, D)`.
"""
function top_momentum(o::ActionOracle, datum::SlabDatum, θ::AbstractVector)
    ĉ, _, _, _ = decode(o, datum, θ)
    g = action_gradient(c -> interior_action(o, c), ĉ)
    return g[o.nt, :, :]
end

# The initial data at the points `x`, as three `(length(x), D)` matrices.
function initial_values(o::ActionOracle{T}, x::AbstractVector) where {T}
    values = o.problem.ics_function(collect(x), parameters(o.problem))
    n = length(x)
    sample(f) = T[f[(d - 1) * n + r] for r in 1:n, d in 1:o.D]
    return sample(values.u), sample(values.v), sample(values.w)
end

# The side data at the nodes of the time rule of the slab that starts at `start`: component `d` of
# a side trace at the slab's first time is its entry `2(d − 1) + 1`.
function side_values(o::ActionOracle{T}, start::T) where {T}
    ga = zeros(T, length(o.t̂), o.D)
    gb = zeros(T, length(o.t̂), o.D)
    o.nμ == 0 && return ga, gb
    for q in eachindex(o.t̂)
        t = start + o.h * o.t̂[q]
        values = o.problem.bcs_function((t, t), parameters(o.problem))
        for d in 1:o.D
            ga[q, d] = values.left.u[2 * (d - 1) + 1]
            gb[q, d] = values.right.u[2 * (d - 1) + 1]
        end
    end
    return ga, gb
end

"""
    first_datum(o)

The datum of the first slab, from the problem alone: the bottom row interpolates `u₀` at the spline
nodes; the load is `F_jd = Σ_r ωₓ_r p₀(x_r) ψ_j(x_r)` with `p₀ = ∂L/∂v(u₀, v₀, w₀)`; the target is
`u₀` at the space nodes; `λ₀` is `p₀` read into the multiplier basis by the weighted `L²`
projection; and the side data are read at `t = timespan[1] + h t̂_q`.
"""
function first_datum(o::ActionOracle{T}) where {T}
    params = parameters(o.problem)
    u_nodes, _, _ = initial_values(o, o.nodes)
    bottom = o.interpolation \ u_nodes
    u₀, v₀, w₀ = initial_values(o, o.x)
    p₀ = zeros(T, length(o.x), o.D)
    for r in eachindex(o.x)
        p₀[r, :] = node_momentum(o.problem.density, u₀[r, :], v₀[r, :], w₀[r, :], params)
    end
    load = transpose(o.ψ) * Diagonal(o.ωx) * p₀
    λ₀ = if restricted(o)
        zeros(T, 0, o.D)
    else
        weighted = transpose(o.Λ) * Diagonal(o.ωx)
        (weighted * o.Λ) \ (weighted * p₀)
    end
    start = timespan(o.problem)[1]
    ga, gb = side_values(o, start)
    return SlabDatum{T}(start, bottom, load, u₀, λ₀, ga, gb)
end

"""
    next_datum(o, datum, θ)

The datum of the slab after the one that `datum` and its solved vector `θ` describe: the previous
top row and, as its load, `∂S/∂ĉ` of that top row; the previous top trace at the space nodes and the
previous top multiplier; and the side data at the new slab's times.
"""
function next_datum(o::ActionOracle{T}, datum::SlabDatum{T}, θ::AbstractVector{T}) where {T}
    ĉ, λ, _, _ = decode(o, datum, θ)
    bottom = ĉ[o.nt, :, :]
    load = top_momentum(o, datum, θ)
    target = reduce(hcat, [o.ψ * (transpose(ĉ[:, :, d]) * o.φ₁) for d in 1:o.D])
    start = datum.start + o.h
    ga, gb = side_values(o, start)
    return SlabDatum{T}(start, bottom, load, target, restricted(o) ? datum.λ₀ : λ, ga, gb)
end
