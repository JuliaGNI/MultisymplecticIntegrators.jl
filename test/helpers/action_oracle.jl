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
    return (u = T.(values.u), v = T.(values.v), w = T.(values.w))
end

# The side data at the nodes of the time rule of the slab that starts at `start`: entry `d` of a
# side datum is component `d`'s value at that end.
function side_values(o::ActionOracle{T}, start::T) where {T}
    ga = zeros(T, length(o.t̂), o.D)
    gb = zeros(T, length(o.t̂), o.D)
    o.nμ == 0 && return ga, gb
    for q in eachindex(o.t̂)
        values = o.problem.bcs_function(start + o.h * o.t̂[q], parameters(o.problem))
        for d in 1:o.D
            ga[q, d] = values.left.u[d]
            gb[q, d] = values.right.u[d]
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

# ---- the symbolic ansatz ---------------------------------------------------
#
# The oracle of a symbolic slab reads the field from a hand-written map of the test's own, and
# differentiates its action with ForwardDiff. The map is
#
#     field(t̂, x, p) -> (u = …, ut = …, ux = …)
#
# a named tuple whose three entries are the `D` components of the field, of its derivative in the
# normalised time and of its derivative in space. The tables are the trace basis' rule and the
# multiplier basis of the coupling, which the oracle builds for itself.

"""
    SymbolicActionOracle(problem, ansatz, field)

The tables of the discrete action of one symbolic slab.

`field` is a hand-written field map `field(t̂, x, p) -> (u = …, ut = …, ux = …)`. The action is

    S = Σ_q Σ_r ωₜ_q ωₓ_r h L(u, uₜ / h, u_x)
      + Σ_r ωₓ_r [λ₀(x_r)·(u(0, x_r) − target(x_r)) − λ(x_r)·u(1, x_r)]
      + Σ_q h ωₜ_q [μ_a(t̂_q)·(u(a, t̂_q) − g_a(t̂_q)) − μ_b(t̂_q)·(u(b, t̂_q) − g_b(t̂_q))],

where `λ`, `μ_a` and `μ_b` are the slab's multipliers and `λ₀` the incoming one of the datum. The
datum is fixed by [`symbolic_first_datum!`](@ref) and carried to the next slab by
[`symbolic_next_datum!`](@ref).
"""
mutable struct SymbolicActionOracle{T, P, A, F}
    problem::P
    ansatz::A
    field::F
    D::Int
    np::Int
    nλ::Int
    nμ::Int
    h::T
    ωt::Vector{T}
    t̂::Vector{T}
    ωx::Vector{T}
    x::Vector{T}
    Λ::Matrix{T}
    χ::Matrix{T}
    start::T
    target::Matrix{T}
    λ₀::Matrix{T}
    ga::Matrix{T}
    gb::Matrix{T}
end

function SymbolicActionOracle(problem, ansatz, field)
    T = eltype(timespan(problem))
    trace = ansatz.trace
    tb, xb = SimpleSplines.bases(SimpleSplines.basis(trace))
    qt, qx = SimpleSplines.quadratures(MultisymplecticIntegrators.quadrature(trace))
    t̂ = collect(T, SimpleSplines.quadrature_nodes(qt))
    x = collect(T, SimpleSplines.quadrature_nodes(qx))
    nx, nt = SimpleSplines.nbasis(xb), SimpleSplines.nbasis(tb)
    nλ, nμ = nx - 2, nt
    Λ = T[SimpleSplines.evaluate(xb, j, p) for p in x, j in 2:(nx - 1)]
    χ = T[SimpleSplines.evaluate(tb, j, p) for p in t̂, j in 1:nt]
    D = length(ansatz.expr)
    qxs, qts = length(x), length(t̂)
    zeros2 = zeros(T, qxs, D)
    return SymbolicActionOracle{T, typeof(problem), typeof(ansatz), typeof(field)}(
        problem, ansatz, field, D, ansatz.num_params, nλ, nμ, T(timestep(problem)),
        collect(T, SimpleSplines.quadrature_weights(qt)), t̂,
        collect(T, SimpleSplines.quadrature_weights(qx)), x,
        Λ, χ,
        timespan(problem)[1], zeros2, zeros(T, nλ, D), zeros(T, qts, D), zeros(T, qts, D))
end

# The side data at the nodes of the time rule of the slab that starts at `start`: entry `d` of a
# side datum is component `d`'s value at that end.
function symbolic_side_values!(o::SymbolicActionOracle{T}) where {T}
    for q in eachindex(o.t̂)
        values = o.problem.bcs_function(o.start + o.h * o.t̂[q], parameters(o.problem))
        for d in 1:o.D
            o.ga[q, d] = values.left.u[d]
            o.gb[q, d] = values.right.u[d]
        end
    end
    return nothing
end

"""
    symbolic_first_datum!(o)

The datum of the first slab, from the problem alone: the target is `u₀` at the space nodes, `λ₀` is
`∂L/∂v(u₀, v₀, w₀)` at those nodes read into the multiplier basis by the weighted `L²` projection,
and the side data are read at `t = timespan[1] + h t̂_q`.
"""
function symbolic_first_datum!(o::SymbolicActionOracle{T}) where {T}
    params = parameters(o.problem)
    values = o.problem.ics_function(collect(o.x), params)
    qx = length(o.x)
    u, v, w = T.(values.u), T.(values.v), T.(values.w)
    o.target .= u
    p₀ = zeros(T, qx, o.D)
    for r in 1:qx
        p₀[r, :] = node_momentum(o.problem.density, u[r, :], v[r, :], w[r, :], params)
    end
    weighted = transpose(o.Λ) * Diagonal(o.ωx)
    o.λ₀ .= (weighted * o.Λ) \ (weighted * p₀)
    o.start = timespan(o.problem)[1]
    symbolic_side_values!(o)
    return nothing
end

"""
    symbolic_next_datum!(o, θ)

The datum of the slab after the one whose solved vector is `θ`: the top trace of that slab at the
space nodes, its top multiplier, and the side data at the new slab's times.
"""
function symbolic_next_datum!(o::SymbolicActionOracle{T}, θ::AbstractVector) where {T}
    np, D, nλ = o.np, o.D, o.nλ
    p = θ[1:np]
    for r in eachindex(o.x), d in 1:D

        o.target[r, d] = o.field(one(T), o.x[r], p).u[d]
    end
    o.λ₀ .= reshape(θ[(np + 1):(np + nλ * D)], nλ, D)
    o.start += o.h
    symbolic_side_values!(o)
    return nothing
end

"""
    symbolic_action(o, z)

The total discrete action of the oracle's slab at `z = (p, λ₀, λ, μ_a, μ_b)`, each block in the
solver's layout.
"""
function symbolic_action(o::SymbolicActionOracle{T}, z::AbstractVector) where {T}
    np, D, nλ, nμ = o.np, o.D, o.nλ, o.nμ
    p = z[1:np]
    i = np
    λ₀ = reshape(z[(i + 1):(i + nλ * D)], nλ, D)
    i += nλ * D
    λ = reshape(z[(i + 1):(i + nλ * D)], nλ, D)
    i += nλ * D
    μa = reshape(z[(i + 1):(i + nμ * D)], nμ, D)
    i += nμ * D
    μb = reshape(z[(i + 1):(i + nμ * D)], nμ, D)
    params = parameters(o.problem)
    action = zero(T)
    for q in eachindex(o.t̂), r in eachindex(o.x)

        values = o.field(o.t̂[q], o.x[r], p)
        v = map(s -> s / o.h, values.ut)
        action += o.ωt[q] * o.ωx[r] * o.h *
                  node_lagrangian(o.problem.density, values.u, v, values.ux, params)
    end
    xa, xb = xspan(o.problem)
    for d in 1:D
        for r in eachindex(o.x)
            bottom = o.field(zero(T), o.x[r], p).u[d]
            top = o.field(one(T), o.x[r], p).u[d]
            incoming = sum(λ₀[p, d] * o.Λ[r, p] for p in 1:nλ; init = zero(T))
            outgoing = sum(λ[p, d] * o.Λ[r, p] for p in 1:nλ; init = zero(T))
            action += o.ωx[r] *
                      (incoming * (bottom - o.target[r, d]) - outgoing * top)
        end
        for q in eachindex(o.t̂)
            left = o.field(o.t̂[q], xa, p).u[d]
            right = o.field(o.t̂[q], xb, p).u[d]
            ma = sum(μa[p, d] * o.χ[q, p] for p in 1:nμ; init = zero(T))
            mb = sum(μb[p, d] * o.χ[q, p] for p in 1:nμ; init = zero(T))
            action += o.h * o.ωt[q] *
                      (ma * (left - o.ga[q, d]) - mb * (right - o.gb[q, d]))
        end
    end
    return action
end

"""
    symbolic_gradient(o, θ)

The residual of the oracle's slab at the solver's vector `θ = (p, λ, μ_a, μ_b)`, from the gradient
of [`symbolic_action`](@ref): `∂S/∂p` on the parameter rows, `−∂S/∂λ₀` on the rows of the top
multiplier, and `−∂S/∂μ_a` and `+∂S/∂μ_b` on the rows of the two ends.
"""
function symbolic_gradient(o::SymbolicActionOracle{T}, θ::AbstractVector{T}) where {T}
    np, D, nλ, nμ = o.np, o.D, o.nλ, o.nμ
    z = vcat(θ[1:np], vec(o.λ₀), θ[(np + 1):end])
    g = action_gradient(w -> symbolic_action(o, w), z)
    R = zeros(T, length(θ))
    R[1:np] .= g[1:np]
    # `z` is `(p, λ₀, λ, μ_a, μ_b)`, and `θ` is `(p, λ, μ_a, μ_b)`: the rows of the slab's
    # multiplier are the derivative of the action in the datum's incoming one.
    base = np + nλ * D
    R[(np + 1):base] .= -g[(np + 1):base]
    offset = np + 2 * nλ * D
    R[(base + 1):(base + nμ * D)] .= -g[(offset + 1):(offset + nμ * D)]
    offset += nμ * D
    R[(base + nμ * D + 1):(base + 2 * nμ * D)] .= g[(offset + 1):(offset + nμ * D)]
    return R
end

# ---- the network ansatz ----------------------------------------------------
#
# The oracle of a network slab writes the field out from the network's own definition — one point
# `[t̂, x]` and the flat parameter vector — rather than reading the ansatz's generated functions,
# and differentiates its action with ForwardDiff.
# The interior action is read on the ansatz's own rule and the four surface terms on the trace
# basis' rules, whose multiplier functions and weights the coupling's tables are built on.

"""
    NetworkActionOracle(problem, method)

The tables of the discrete action of one network slab.

The action is

    S = Σ_q Σ_r ωₜ_q ωₓ_r h L(u, uₜ / h, u_x)
      + Σ_r ωₓ_r [λ₀(x_r)·(u(0, x_r) − target(x_r)) − λ(x_r)·u(1, x_r)]
      + Σ_q h ωₜ_q [μ_a(t̂_q)·(u(a, t̂_q) − g_a(t̂_q)) − μ_b(t̂_q)·(u(b, t̂_q) − g_b(t̂_q))],

where the interior sum is taken at the nodes of the ansatz's own quadrature, `uₜ` and `u_x` are its
two derivatives there, and the multiplier functions `Λ` and `χ`, their weights and the four surfaces
are read from `trace_basis(ansatz)`. The datum is fixed by [`first_datum`](@ref) and carried to the
next slab by [`next_datum`](@ref).

The unknowns of the slab are the network's parameters and then the multipliers, so the oracle reads
every parameter: a `:last_layer` method frees a part of them, which the datum does not carry, and is
refused.
"""
struct NetworkActionOracle{T, P, A}
    problem::P
    ansatz::A
    np::Int
    D::Int
    nλ::Int
    nμ::Int
    h::T
    ωt::Vector{T}
    t̂::Vector{T}
    ωx::Vector{T}
    x::Vector{T}
    ωxt::Vector{T}
    xt::Vector{T}
    ωtt::Vector{T}
    tt::Vector{T}
    Λ::Matrix{T}
    χ::Matrix{T}
end

function ActionOracle(problem, method::SpaceTimeGalerkin{<:NetworkAnsatz}; kwargs...)
    isempty(kwargs) ||
        throw(ArgumentError("a network's action oracle takes no keyword argument"))
    return NetworkActionOracle(problem, method)
end

function NetworkActionOracle(problem, method)
    T = eltype(timespan(problem))
    a = ansatz(method)
    c = coupling(method)
    M = MultisymplecticIntegrators
    M.nfree(a) == M.nparameters(a) || throw(ArgumentError(
        "the action oracle of a network reads every parameter; a :last_layer method frees " *
        "$(M.nfree(a)) of $(M.nparameters(a))"))
    own = SimpleSplines.quadratures(M.quadrature(a))
    tr = M.trace_basis(a)
    trace = SimpleSplines.quadratures(M.quadrature(tr))
    nodes(rule) = collect(T, SimpleSplines.quadrature_nodes(rule))
    weights(rule) = collect(T, SimpleSplines.quadrature_weights(rule))
    xt, ωxt = nodes(trace[2]), weights(trace[2])
    tt, ωtt = nodes(trace[1]), weights(trace[1])
    tb, xb = SimpleSplines.bases(SimpleSplines.basis(tr))
    nt, nx = SimpleSplines.nbasis(tb), SimpleSplines.nbasis(xb)
    corners = c isa WeakInitialAndBoundaryConditions
    λindices = corners ? (2:(nx - 1)) : (1:nx)
    table(basis, points, indices) = T[SimpleSplines.evaluate(basis, j, p)
                                      for p in points, j in indices]
    return NetworkActionOracle{T, typeof(problem), typeof(a)}(
        problem, a, M.nparameters(a), a.D, length(λindices), corners ? nt : 0,
        T(timestep(problem)),
        weights(own[1]), nodes(own[1]), weights(own[2]), nodes(own[2]),
        ωxt, xt, ωtt, tt,
        table(xb, xt, λindices), table(tb, tt, 1:nt))
end

restricted(::NetworkActionOracle) = false

"""The length of the solver's vector of a network slab: its parameters, then its multipliers."""
unknown_count(o::NetworkActionOracle) = o.np + (o.nλ + 2 * o.nμ) * o.D

mutable struct NetworkDatum{T}
    start::T
    target::Matrix{T}
    λ₀::Matrix{T}
    ga::Matrix{T}
    gb::Matrix{T}
end

# The side data at the nodes of the trace basis' time rule of the slab that starts at `start`.
function network_side_values(o::NetworkActionOracle{T}, start) where {T}
    ga = zeros(T, length(o.tt), o.D)
    gb = zeros(T, length(o.tt), o.D)
    o.nμ == 0 && return ga, gb
    for q in eachindex(o.tt)
        values = o.problem.bcs_function(start + o.h * o.tt[q], parameters(o.problem))
        for d in 1:o.D
            ga[q, d] = values.left.u[d]
            gb[q, d] = values.right.u[d]
        end
    end
    return ga, gb
end

"""
    first_datum(o::NetworkActionOracle)

The datum of the first network slab, from the problem alone: the target is `u₀` at the trace basis'
space nodes, `λ₀` is `∂L/∂v(u₀, v₀, w₀)` there read into the multiplier basis by the weighted `L²`
projection, and the side data are read at `t = timespan[1] + h t̂_q`.
"""
function first_datum(o::NetworkActionOracle{T}) where {T}
    params = parameters(o.problem)
    values = o.problem.ics_function(collect(o.xt), params)
    qx = length(o.xt)
    u, v, w = T.(values.u), T.(values.v), T.(values.w)
    p₀ = zeros(T, qx, o.D)
    for r in 1:qx
        p₀[r, :] = node_momentum(o.problem.density, u[r, :], v[r, :], w[r, :], params)
    end
    weighted = transpose(o.Λ) * Diagonal(o.ωxt)
    λ₀ = (weighted * o.Λ) \ (weighted * p₀)
    start = timespan(o.problem)[1]
    ga, gb = network_side_values(o, start)
    return NetworkDatum{T}(start, u, λ₀, ga, gb)
end

"""
    next_datum(o, datum, θ)

The datum of the slab after the one whose solved vector is `θ`: the top trace of that slab at the
trace basis' space nodes, its top multiplier, and the side data at the new slab's times.
"""
function next_datum(o::NetworkActionOracle{T}, datum::NetworkDatum{T},
        θ::AbstractVector{T}) where {T}
    p = θ[1:o.np]
    for r in eachindex(o.xt), d in 1:o.D

        datum.target[r, d] = network_field(o, p, one(T), o.xt[r])[d]
    end
    datum.λ₀ .= reshape(θ[(o.np + 1):(o.np + o.nλ * o.D)], o.nλ, o.D)
    datum.start += o.h
    datum.ga, datum.gb = network_side_values(o, datum.start)
    return datum
end

# The values `[u; ∂_t̂u; ∂ₓu]` of the network at one point, of the given parameter vector, written
# here from the network's own definition rather than read from the ansatz's generated functions:
# with `S` hidden units of `tanh` and `D` outputs,
# `u_d = Σ_s W2[s, d] tanh(W1[s, 1] t̂ + W1[s, 2] x + b[s])`, and each derivative carries the slope
# `1 - tanh²` of its hidden unit times its input weight. The parameters are read in the order
# `L1.W`, `L1.b`, `L2.W`, with `L1.W` laid out as `[W1[:, 1]; W1[:, 2]]`.
function network_field(o::NetworkActionOracle, p, t̂, x)
    S, D = o.ansatz.S, o.D
    values = zeros(eltype(p), 3D)
    for s in 1:S
        σ = tanh(p[s] * t̂ + p[S + s] * x + p[2S + s])
        slope = one(σ) - abs2(σ)
        for d in 1:D
            weight = p[3S + (s - 1) * D + d]
            values[d] += weight * σ
            values[D + d] += weight * slope * p[s]
            values[2D + d] += weight * slope * p[S + s]
        end
    end
    return values
end

"""
    network_action(o, datum, z)

The total discrete action of the oracle's slab at `z = (p, λ₀, λ, μ_a, μ_b)`, each block in the
solver's layout.
"""
function network_action(o::NetworkActionOracle{T}, datum::NetworkDatum{T},
        z::AbstractVector) where {T}
    np, D, nλ, nμ = o.np, o.D, o.nλ, o.nμ
    p = z[1:np]
    i = np
    λ₀ = reshape(z[(i + 1):(i + nλ * D)], nλ, D)
    i += nλ * D
    λ = reshape(z[(i + 1):(i + nλ * D)], nλ, D)
    i += nλ * D
    μa = reshape(z[(i + 1):(i + nμ * D)], nμ, D)
    i += nμ * D
    μb = reshape(z[(i + 1):(i + nμ * D)], nμ, D)
    params = parameters(o.problem)
    action = zero(T)
    for q in eachindex(o.t̂), r in eachindex(o.x)

        values = network_field(o, p, o.t̂[q], o.x[r])
        u = [values[d] for d in 1:D]
        v = [values[D + d] / o.h for d in 1:D]
        w = [values[2D + d] for d in 1:D]
        action += o.ωt[q] * o.ωx[r] * o.h *
                  node_lagrangian(o.problem.density, u, v, w, params)
    end
    xa, xb = xspan(o.problem)
    for d in 1:D
        for r in eachindex(o.xt)
            bottom = network_field(o, p, zero(T), o.xt[r])[d]
            top = network_field(o, p, one(T), o.xt[r])[d]
            incoming = sum(λ₀[j, d] * o.Λ[r, j] for j in 1:nλ; init = zero(T))
            outgoing = sum(λ[j, d] * o.Λ[r, j] for j in 1:nλ; init = zero(T))
            action += o.ωxt[r] * (incoming * (bottom - datum.target[r, d]) - outgoing * top)
        end
        for q in eachindex(o.tt)
            left = network_field(o, p, o.tt[q], xa)[d]
            right = network_field(o, p, o.tt[q], xb)[d]
            ma = sum(μa[j, d] * o.χ[q, j] for j in 1:nμ; init = zero(T))
            mb = sum(μb[j, d] * o.χ[q, j] for j in 1:nμ; init = zero(T))
            action += o.h * o.ωtt[q] *
                      (ma * (left - datum.ga[q, d]) - mb * (right - datum.gb[q, d]))
        end
    end
    return action
end

"""
    oracle_residual(o::NetworkActionOracle, datum, θ)

The residual of the oracle's network slab at the solver's vector `θ = (p, λ, μ_a, μ_b)`, from the
gradient of [`network_action`](@ref): `∂S/∂p` on the parameter rows, `−∂S/∂λ₀` on the rows of the
top multiplier, and `−∂S/∂μ_a` and `+∂S/∂μ_b` on the rows of the two ends.
"""
function oracle_residual(o::NetworkActionOracle{T}, datum::NetworkDatum{T},
        θ::AbstractVector{T}) where {T}
    np, D, nλ, nμ = o.np, o.D, o.nλ, o.nμ
    z = vcat(θ[1:np], vec(datum.λ₀), θ[(np + 1):end])
    g = action_gradient(w -> network_action(o, datum, w), z)
    R = zeros(T, length(θ))
    R[1:np] .= g[1:np]
    base = np + nλ * D
    R[(np + 1):base] .= -g[(np + 1):base]
    offset = np + 2 * nλ * D
    R[(base + 1):(base + nμ * D)] .= -g[(offset + 1):(offset + nμ * D)]
    offset += nμ * D
    R[(base + nμ * D + 1):(base + 2 * nμ * D)] .= g[(offset + 1):(offset + nμ * D)]
    return R
end
