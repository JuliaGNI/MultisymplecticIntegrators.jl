"""
    LagrangianDensity(L, u, v, w; params = NamedTuple())

The Lagrangian density `L(u, v, w; params)` of a field theory in one space dimension, where
`v = ∂ₜu` and `w = ∂ₓu`, together with its three partial derivatives `∂u`, `∂v` and `∂w`.

`L`, `u`, `v` and `w` are symbolic. Each of the four functions is built once by
`Symbolics.build_function(…; expression = Val(false))`, so an evaluation is an ordinary Julia call
and not a re-derivation of the expression. The names of `params` are the parameters of the
density: every built function takes their values as further positional arguments.

# Fields

- `L(u, v, w, p...)`: the density.
- `∂u(u, v, w, p...)`, `∂v(u, v, w, p...)`, `∂w(u, v, w, p...)`: its partial derivatives.
- `names`: the names of the parameters, in the order of the built functions' arguments.
- `params`: the default parameter values.
- `linear`: whether the partial derivatives are affine in `(u, v, w)`.

# Example

```julia
using Symbolics
@variables u v w
d = LagrangianDensity(v^2 / 2 - w^2 / 2, u, v, w; params = (c = 0.5,))
d(0.0, 1.0, 0.0, (c = 0.5,))   # the density, with the parameters named
d.∂v(0.0, 1.0, 0.0, 0.5)       # ∂L/∂v, with the values in argument order
```
"""
struct LagrangianDensity{LT, UT, VT, WT, NT, PT}
    L::LT
    ∂u::UT
    ∂v::VT
    ∂w::WT
    names::NT
    params::PT
    linear::Bool
end

function LagrangianDensity(L, ∂u, ∂v, ∂w, names, params)
    return LagrangianDensity(L, ∂u, ∂v, ∂w, names, params, false)
end

function LagrangianDensity(L, u, v, w; params = NamedTuple())
    names = Tuple(keys(params))
    psyms = map(name -> Symbolics.variable(name), names)
    args = (u, v, w, psyms...)
    build(expression) = Symbolics.build_function(expression, args...; expression = Val(false))
    du = Symbolics.derivative(L, u)
    dv = Symbolics.derivative(L, v)
    dw = Symbolics.derivative(L, w)
    hessian = (
        Symbolics.derivative(du, u),
        Symbolics.derivative(du, v),
        Symbolics.derivative(du, w),
        Symbolics.derivative(dv, v),
        Symbolics.derivative(dv, w),
        Symbolics.derivative(dw, w))
    linear = all(
        iszero(h) || all(
            let d = Symbolics.derivative(h, x)
                iszero(d) || iszero(Symbolics.simplify(d))
            end for x in (u, v, w))
    for h in hessian)
    return LagrangianDensity(
        build(L), build(du), build(dv), build(dw), names, params, linear)
end

"""Whether the density's partial derivatives are affine in `(u, v, w)`."""
is_linear_density(density::LagrangianDensity) = density.linear
is_linear_density(::Any) = false

"""
The values of `params` in the order of the density's built functions.

When the names of the default parameters are the density's `names`, in their order, the type of
the defaults fixes that order for the compiler; otherwise each name is looked up in `params`.
"""
@inline function parameter_values(
        density::LagrangianDensity{LT, UT, VT, WT, NT, PT}, params::NamedTuple) where {
        LT, UT, VT, WT, NT, PT <: NamedTuple}
    density.names === fieldnames(PT) ||
        return map(name -> getfield(params, name), density.names)
    return Tuple(NamedTuple{fieldnames(PT)}(params))
end

@inline function parameter_values(density::LagrangianDensity, params)
    return map(name -> getfield(params, name), density.names)
end

function (density::LagrangianDensity)(u, v, w, params = density.params)
    density.L(u, v, w, parameter_values(density, params)...)
end

"""`∂L/∂u` at `(u, v, w)`."""
function derivative_u(density::LagrangianDensity, u, v, w, params = density.params)
    density.∂u(u, v, w, parameter_values(density, params)...)
end

"""`∂L/∂v` at `(u, v, w)`."""
function derivative_v(density::LagrangianDensity, u, v, w, params = density.params)
    density.∂v(u, v, w, parameter_values(density, params)...)
end

"""`∂L/∂w` at `(u, v, w)`."""
function derivative_w(density::LagrangianDensity, u, v, w, params = density.params)
    density.∂w(u, v, w, parameter_values(density, params)...)
end

"""
The three derivatives of a density at `(u, v, w)`, as a named tuple.

A density that couples the `D` components of the field defines its own method. The couplings of
the slab call it with `NTuple{D}`s for `u`, `v` and `w`, so a method restricted to
`AbstractVector` does not apply; see [`LagrangianPDEProblem`](@ref).
"""
function derivatives(density::LagrangianDensity, u, v, w, params = density.params)
    return (u = derivative_u(density, u, v, w, params),
        v = derivative_v(density, u, v, w, params),
        w = derivative_w(density, u, v, w, params))
end

"""The three partial derivatives of `density` at point `(q, r)` of a slab grid, as `D`-tuples."""
@inline function point_derivatives(density::LagrangianDensity,
        u::AbstractArray{ST, 3}, v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3},
        q::Integer, r::Integer, params, ::Val{D} = Val(1)) where {ST, D}
    return (
        u = ntuple(
            d -> ST(derivative_u(density, u[q, r, d], v[q, r, d], w[q, r, d], params)),
            Val(D)),
        v = ntuple(
            d -> ST(derivative_v(density, u[q, r, d], v[q, r, d], w[q, r, d], params)),
            Val(D)),
        w = ntuple(
            d -> ST(derivative_w(density, u[q, r, d], v[q, r, d], w[q, r, d], params)),
            Val(D)))
end

@inline function point_derivatives(density,
        u::AbstractArray{ST, 3}, v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3},
        q::Integer, r::Integer, params, ::Val{D} = Val(2)) where {ST, D}
    u_pt = ntuple(d -> u[q, r, d], Val(D))
    v_pt = ntuple(d -> v[q, r, d], Val(D))
    w_pt = ntuple(d -> w[q, r, d], Val(D))
    return derivatives(density, u_pt, v_pt, w_pt, params)
end

"""The three partial derivatives of `density` at node `r` of a boundary trace, as `D`-tuples."""
@inline function point_derivatives(density::LagrangianDensity,
        u::AbstractMatrix, v::AbstractMatrix, w::AbstractMatrix,
        r::Integer, params, ::Val{D} = Val(1)) where {D}
    T = promote_type(eltype(u), eltype(v), eltype(w))
    return (
        u = ntuple(
            d -> T(derivative_u(density, u[r, d], v[r, d], w[r, d], params)), Val(D)),
        v = ntuple(
            d -> T(derivative_v(density, u[r, d], v[r, d], w[r, d], params)), Val(D)),
        w = ntuple(
            d -> T(derivative_w(density, u[r, d], v[r, d], w[r, d], params)), Val(D)))
end

@inline function point_derivatives(density,
        u::AbstractMatrix, v::AbstractMatrix, w::AbstractMatrix,
        r::Integer, params, ::Val{D} = Val(2)) where {D}
    u_pt = ntuple(d -> u[r, d], Val(D))
    v_pt = ntuple(d -> v[r, d], Val(D))
    w_pt = ntuple(d -> w[r, d], Val(D))
    return derivatives(density, u_pt, v_pt, w_pt, params)
end

function Base.show(io::IO, density::LagrangianDensity)
    print(io, "LagrangianDensity with parameters ", density.names)
end
