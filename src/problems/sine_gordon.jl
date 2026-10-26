"""
    SineGordon

The nonlinear sine-Gordon equation `c ∂ₜ²u − ∂ₓ²u = −sin(u)` on `x ∈ [0, l]`, with the kink exact
solution that the reference values of part A were recorded from.

The Lagrangian density is `L = ½(c v² − w²) + 1 + cos(u)`, and the exact solution is

```math
u(t, x) = 4\\arctan(\\exp(γ (x - v t))), \\qquad γ = 1 / \\sqrt{1 - c v²},
```

with `v = ∂ₜu` and `w = ∂ₓu` its derivatives. The kink exists where `c v² < 1`, which the
constructor checks.

```julia
problem = SineGordon.problem(timespan = (0.0, 10.0), timestep = 1.0, xstep = 0.01)
```
"""
module SineGordon

export default_parameters, density, problem, exact_u, exact_v, exact_w, initial_data,
       boundary_data

using Symbolics
using MultisymplecticIntegrators

const DEFAULT_PARAMETERS = (c = 4.0, velocity = 0.25,
    γ = 1 / sqrt(1 - 4.0 * 0.25^2))

default_parameters() = DEFAULT_PARAMETERS

"""
    density(; params = default_parameters())

The Lagrangian density `L = ½(c v² − w²) + 1 + cos(u)`, whose parameter is `c`.
"""
function density(; params = default_parameters())
    u, v, w, c = Symbolics.variable(:u), Symbolics.variable(:v), Symbolics.variable(:w),
    Symbolics.variable(:c)
    L = (c * v^2 - w^2) / 2 + 1 + cos(u)
    return LagrangianDensity(L, u, v, w; params = (c = params.c,))
end

function exact_u(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    return 4 * atan(exp(p.γ * (x - p.velocity * t)))
end

function exact_v(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    ξ = p.γ * (x - p.velocity * t)
    return -4 * p.γ * p.velocity * exp(ξ) / (1 + exp(ξ)^2)
end

function exact_w(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    ξ = p.γ * (x - p.velocity * t)
    return 4 * p.γ * exp(ξ) / (1 + exp(ξ)^2)
end

"""The initial data `(u = …, v = …, w = …)` at `t = timespan[1]`."""
function initial_data(x::AbstractVector, params = default_parameters(), D::Integer = 1, t = 0.0)
    field_initial_data(exact_u, exact_v, exact_w, x, params, D, t)
end

"""The boundary data of a slab at time `t`."""
function boundary_data(t, xbreaks::AbstractVector, params = default_parameters())
    field_boundary_data(exact_u, exact_v, exact_w, t, xbreaks, params)
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
    params.c * params.velocity^2 < 1 ||
        throw(ArgumentError("the kink needs c * velocity² < 1, got $(params.c * params.velocity^2)"))
    d = density(; params = params)
    ics = (x, p) -> initial_data(x, p, D, timespan[1])
    bcs = (t, p) -> boundary_data(t, collect(xspan[1]:xstep:xspan[2]), p)
    return LagrangianPDEProblem(d;
        D = D,
        ics_function = ics,
        bcs_function = bcs,
        exact_u = field_exact(exact_u, D),
        exact_v = field_exact(exact_v, D),
        exact_w = field_exact(exact_w, D),
        timespan = (T(timespan[1]), T(timespan[2])),
        timestep = T(timestep),
        xspan = (T(xspan[1]), T(xspan[2])),
        xstep = T(xstep),
        params = params)
end

end
