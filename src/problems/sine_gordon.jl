"""
    SineGordon

The nonlinear sine-Gordon equation `c ∂ₜ²u − ∂ₓ²u = −sin(u)` on `x ∈ [0, l]`, with the exact
solution of the travelling kink.

The Lagrangian density is `L = ½(c v² − w²) + 1 + cos(u)`, and the exact solution is

```math
u(t, x) = 4\\arctan(\\exp(γ (x - v t))), \\qquad γ = 1 / \\sqrt{1 - c v²},
```

with `v = ∂ₜu` and `w = ∂ₓu` its derivatives. `γ` is a function of `c` and the kink velocity, so
the parameters are `(c, velocity)` and a kink exists only where `c velocity² < 1`.

```julia
problem = SineGordon.problem(timespan = (0.0, 10.0), timestep = 1.0, xstep = 0.01)
```
"""
module SineGordon

export default_parameters, density, problem, exact_u, exact_v, exact_w, initial_data,
       boundary_data, lorentz_factor

using Symbolics
using MultisymplecticIntegrators

const DEFAULT_PARAMETERS = (c = 4.0, velocity = 0.25)

default_parameters() = DEFAULT_PARAMETERS

"""
    lorentz_factor(params)

`γ = 1 / √(1 − c velocity²)`. Throws `ArgumentError` where the kink has no solution, which is
where `c velocity² ≥ 1`.
"""
function lorentz_factor(params)
    β² = params.c * params.velocity^2
    β² < 1 || throw(ArgumentError("the kink needs c * velocity² < 1, got $β²"))
    return 1 / sqrt(1 - β²)
end

"""
    density(; params = default_parameters())

The Lagrangian density `L = ½(c v² − w²) + 1 + cos(u)`, whose parameter is `c`.
"""
function density(; params = default_parameters())
    u = Symbolics.variable(:u)
    v = Symbolics.variable(:v)
    w = Symbolics.variable(:w)
    c = Symbolics.variable(:c)
    L = (c * v^2 - w^2) / 2 + 1 + cos(u)
    return LagrangianDensity(L, u, v, w; params = (c = params.c,))
end

function exact_u(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    γ = T(lorentz_factor(params))
    return 4 * atan(exp(γ * (x - p.velocity * t)))
end

function exact_v(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    γ = T(lorentz_factor(params))
    ξ = γ * (x - p.velocity * t)
    return -4 * γ * p.velocity * exp(ξ) / (1 + exp(ξ)^2)
end

function exact_w(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    γ = T(lorentz_factor(params))
    ξ = γ * (x - p.velocity * t)
    return 4 * γ * exp(ξ) / (1 + exp(ξ)^2)
end

"""The initial data `(u = …, v = …, w = …)` at time `t`, component-major."""
function initial_data(x::AbstractVector, params = default_parameters(), D::Integer = 1, t = 0.0)
    field_initial_data(exact_u, exact_v, exact_w, x, params, D, t)
end

"""The boundary data of the slab whose two times are `times = (t₀, t₁)`."""
function boundary_data(times, xbreaks::AbstractVector, params = default_parameters(), D::Integer = 1)
    field_boundary_data(exact_u, exact_v, exact_w, times, xbreaks, params, D)
end

"""
    problem(; T = Float64, D = 1, params = default_parameters(), timespan = (0.0, 10.0),
            timestep = 1.0, xspan = (0.0, 1.0), xstep = 0.01)

A [`LagrangianPDEProblem`](@ref) of the sine-Gordon equation. Throws `ArgumentError` where
`c * velocity² ≥ 1`, because the kink solution needs a velocity below the wave speed.
"""
function problem(; T::Type = Float64, D::Integer = 1, params = default_parameters(),
        timespan = (zero(T), T(10)), timestep = T(1.0), xspan = (zero(T), one(T)),
        xstep = T(0.01))
    lorentz_factor(params)
    d = density(; params = params)
    tspan = (T(timespan[1]), T(timespan[2]))
    xsp = (T(xspan[1]), T(xspan[2]))
    h = T(xstep)
    ics = (x, p) -> initial_data(x, p, D, tspan[1])
    bcs = (times, p) -> boundary_data(times, collect(xsp[1]:h:xsp[2]), p, D)
    return LagrangianPDEProblem(d;
        D = D,
        ics_function = ics,
        bcs_function = bcs,
        exact_u = field_exact(exact_u, D),
        exact_v = field_exact(exact_v, D),
        exact_w = field_exact(exact_w, D),
        timespan = tspan,
        timestep = T(timestep),
        xspan = xsp,
        xstep = h,
        params = params)
end

end
