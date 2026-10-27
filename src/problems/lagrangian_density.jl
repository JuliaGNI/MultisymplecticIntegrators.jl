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
end

function LagrangianDensity(L, u, v, w; params = NamedTuple())
    names = Tuple(keys(params))
    psyms = map(name -> Symbolics.variable(name), names)
    args = (u, v, w, psyms...)
    build(expression) = Symbolics.build_function(expression, args...; expression = Val(false))
    return LagrangianDensity(build(L),
        build(Symbolics.derivative(L, u)),
        build(Symbolics.derivative(L, v)),
        build(Symbolics.derivative(L, w)),
        names, params)
end

"""The values of `params` in the order of the density's built functions."""
function parameter_values(density::LagrangianDensity, params)
    map(name -> getfield(params, name), density.names)
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

"""The three derivatives of a density at `(u, v, w)`, as a named tuple."""
function derivatives(density::LagrangianDensity, u, v, w, params = density.params)
    return (u = derivative_u(density, u, v, w, params),
        v = derivative_v(density, u, v, w, params),
        w = derivative_w(density, u, v, w, params))
end

function Base.show(io::IO, density::LagrangianDensity)
    print(io, "LagrangianDensity with parameters ", density.names)
end
