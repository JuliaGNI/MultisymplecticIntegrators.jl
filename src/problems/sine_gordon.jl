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

export default_parameters, density, problem, solution_u, solution_v, solution_w,
       initial_data,
       side_data, lorentz_factor

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

function solution_u(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    γ = T(lorentz_factor(params))
    return 4 * atan(exp(γ * (x - p.velocity * t)))
end

function solution_v(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    γ = T(lorentz_factor(params))
    ξ = γ * (x - p.velocity * t)
    return -4 * γ * p.velocity * exp(ξ) / (1 + exp(ξ)^2)
end

function solution_w(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    γ = T(lorentz_factor(params))
    ξ = γ * (x - p.velocity * t)
    return 4 * γ * exp(ξ) / (1 + exp(ξ)^2)
end

"""The initial data `(u = …, v = …, w = …)` at time `t`, one `(length(x), D)` matrix per entry."""
function initial_data(x::AbstractVector, params = default_parameters(), D::Integer = 1, t = 0.0)
    field_initial_data(solutions(D), x, params, D, t)
end

"""The exact solution of the kink of `D` components, as [`field_exact`](@ref) lifts it."""
function solutions(D::Integer = 1)
    return (u = field_exact(solution_u, D), v = field_exact(solution_v, D),
        w = field_exact(solution_w, D))
end

"""The side data of the slab at the one time `t`, as [`field_side_data`](@ref) gives it."""
function side_data(t, xspan, params = default_parameters(), D::Integer = 1)
    field_side_data(solutions(D), t, params, D, xspan)
end

"""
    problem(; T = Float64, D = 1, params = default_parameters(), timespan = (0.0, 10.0),
            timestep = 1.0, xspan = (0.0, 1.0), xstep = 0.01)

A [`LagrangianPDEProblem`](@ref) of the sine-Gordon equation. Throws `ArgumentError` where
`c * velocity² ≥ 1`, because the kink solution needs a velocity below the wave speed. Every
parameter is converted to `T`, so that the density and the energy of a `Float32` problem are
evaluated in `Float32`.
"""
function problem(; T::Type = Float64, D::Integer = 1, params = default_parameters(),
        timespan = (zero(T), T(10)), timestep = T(1.0), xspan = (zero(T), one(T)),
        xstep = T(0.01))
    lorentz_factor(params)
    params = map(T, params)
    d = density(; params = params)
    tspan = (T(timespan[1]), T(timespan[2]))
    xsp = (T(xspan[1]), T(xspan[2]))
    s = solutions(D)
    ics = (x, p) -> field_initial_data(s, x, p, D, tspan[1])
    bcs = (t, p) -> field_side_data(s, t, p, D, xsp)
    return LagrangianPDEProblem(d;
        D = D,
        ics_function = ics,
        bcs_function = bcs,
        solutions = s,
        timespan = tspan,
        timestep = T(timestep),
        xspan = xsp,
        xstep = T(xstep),
        params = params)
end

end
