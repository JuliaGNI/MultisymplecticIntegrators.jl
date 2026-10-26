"""
    SpaceTimeQuadrature{T}

Composite Gauss–Legendre rule on one space-time slab.

The rule takes `n` Gauss–Legendre nodes per cell in each direction, placed between the breakpoints
`tbreaks` of the normalised time coordinate `t̂ ∈ [0, 1]` and `xbreaks` of the physical space
coordinate `x`. Because a cell's rule is exact for polynomials of degree `2n − 1`, the composite
rule integrates `t̂ᵃ xᵇ` exactly for `a ≤ 2n − 1` and `b ≤ 2n − 1` on any mesh, uniform or not.

The four boundary point sets of a slab are `bottom` and `top`, at `t̂ = 0` and `t̂ = 1` across the
`x` nodes, and `left` and `right`, at the two ends of `xspan` across the `t̂` nodes.

```julia
q = SpaceTimeQuadrature{Float64}(3; xbreaks = [0.0, 0.5, 1.0])
q((t, x) -> t^2 * x)     # the rule applied to a function of (t, x)
```
"""
struct SpaceTimeQuadrature{T}
    n::Int
    tbreaks::Vector{T}
    xbreaks::Vector{T}
    t̂::Vector{T}
    ωt::Vector{T}
    x::Vector{T}
    ωx::Vector{T}
end

function SpaceTimeQuadrature{T}(n::Integer;
        tbreaks::AbstractVector = T[zero(T), one(T)],
        xbreaks::AbstractVector = T[zero(T), one(T)]) where {T <: AbstractFloat}
    n ≥ 1 || throw(ArgumentError("n must be ≥ 1, got $n"))
    tb = _breakpoints(T, tbreaks)
    xb = _breakpoints(T, xbreaks)
    tb[1] == zero(T) && tb[end] == one(T) ||
        throw(ArgumentError("tbreaks must run from 0 to 1, got $(tb[1]) to $(tb[end])"))
    xb[1] < xb[end] ||
        throw(ArgumentError("xbreaks must increase, got $(xb[1]) to $(xb[end])"))
    rule = GaussLegendreQuadrature(T, Int(n); fast = true)
    ξ, η = nodes(rule), weights(rule)
    t̂, ωt = _composite(T, ξ, η, tb)
    x, ωx = _composite(T, ξ, η, xb)
    return SpaceTimeQuadrature{T}(Int(n), tb, xb, t̂, ωt, x, ωx)
end

SpaceTimeQuadrature(n::Integer; kwargs...) = SpaceTimeQuadrature{Float64}(n; kwargs...)

"""The normalised time nodes and weights, and the physical space nodes and weights."""
nodes(q::SpaceTimeQuadrature) = (q.t̂, q.x)
weights(q::SpaceTimeQuadrature) = (q.ωt, q.ωx)

"""The number of nodes in each direction."""
nnodes(q::SpaceTimeQuadrature) = (length(q.t̂), length(q.x))

"""
    boundary_points(q::SpaceTimeQuadrature)

The four boundary point sets of the slab: `bottom` and `top` at `t̂ = 0` and `t̂ = 1` across the
`x` nodes, `left` and `right` at the two ends of `xspan` across the `t̂` nodes.
"""
function boundary_points(q::SpaceTimeQuadrature{T}) where {T}
    return (bottom = (t̂ = T[zero(T)], x = copy(q.x)),
        top = (t̂ = T[one(T)], x = copy(q.x)),
        left = (t̂ = copy(q.t̂), x = T[q.xbreaks[1]]),
        right = (t̂ = copy(q.t̂), x = T[q.xbreaks[end]]))
end

"""
    (q::SpaceTimeQuadrature)(f)

Apply the rule to `f(t̂, x)`: the tensor product of the two one-dimensional composite rules.
"""
function (q::SpaceTimeQuadrature{T})(f) where {T}
    s = zero(T)
    for (i, t) in enumerate(q.t̂), (j, x) in enumerate(q.x)

        s += f(t, x) * q.ωt[i] * q.ωx[j]
    end
    return s
end

function _breakpoints(::Type{T}, breaks::AbstractVector) where {T}
    length(breaks) ≥ 2 ||
        throw(ArgumentError("at least two breakpoints are needed, got $(length(breaks))"))
    b = T[convert(T, x) for x in breaks]
    issorted(b) && all(diff(b) .> zero(T)) ||
        throw(ArgumentError("the breakpoints must increase, got $b"))
    return b
end

function _composite(::Type{T}, ξ::AbstractVector, η::AbstractVector,
        breaks::AbstractVector) where {T}
    m = length(breaks) - 1
    v = Vector{T}(undef, m * length(ξ))
    w = Vector{T}(undef, m * length(ξ))
    for (c, (a, b)) in enumerate(zip(breaks[1:(end - 1)], breaks[2:end]))
        h = b - a
        for (k, (ξk, ηk)) in enumerate(zip(ξ, η))
            i = (c - 1) * length(ξ) + k
            v[i] = a + h * convert(T, ξk)
            w[i] = h * convert(T, ηk)
        end
    end
    return v, w
end
