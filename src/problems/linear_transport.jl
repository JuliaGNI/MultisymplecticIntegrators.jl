"""
    LinearTransport

The linear transport equation `∂ₜu + c ∂ₓu = 0` on `x ∈ [-0.5, 0.0]`, whose exact solution is the
initial profile shifted by `c t`.

The Lagrangian density is the doubled field's, `L = ψ (v₁ + c w₁)` on `D = 2`: component 1 is the
transported field `u` and component 2 the adjoint field `ψ`, which carries zero data. The variation
in `ψ` gives the transport equation for `u`, and the variation in `u` the adjoint equation for `ψ`,
so the density carries the dynamics that one field alone could not: it is linear in the velocities,
its Legendre map is degenerate (`p_u = ψ`, `p_ψ = 0`), and the two components' rows carry each
other's equation.

The exact solution of `u` is the Gaussian

```math
u(t, x) = \\frac{1}{2\\sqrt{\\pi}} \\exp\\!\\big(-(x + 2 - ct)^2\\big),
```

with `v = ∂ₜu` and `w = ∂ₓu` its derivatives, and the adjoint field's exact value is zero.

The transport equation needs its inflow value at `x = -0.5`, which
`WeakInitialAndBoundaryConditions` imposes and `CoefficientRestriction` does not, so under the
restriction the error is largest at that end and the converged order is read on `x ≥ -0.5 + c t`.

```julia
problem = LinearTransport.problem(timespan = (0.0, 10.0), timestep = 1.0, xstep = 0.01)
```
"""
module LinearTransport

export default_parameters, density, problem, solution_u, solution_v, solution_w,
       initial_data,
       side_data

using Symbolics
using MultisymplecticIntegrators
import MultisymplecticIntegrators: derivatives

const DEFAULT_PARAMETERS = (c = 0.2,)

default_parameters() = DEFAULT_PARAMETERS

"""
    TransportDensity{P}

The doubled-field density `L = ψ (v₁ + c w₁)`, with `P` the type of its parameters, of which it
keeps `c`. The field is `(u, ψ)`: `u[1]` is the transported field, `u[2]` the adjoint field, and
`v` and `w` are the field's derivatives in time and in space.
"""
struct TransportDensity{P}
    params::P
end

TransportDensity(; params = DEFAULT_PARAMETERS) = TransportDensity((c = params.c,))

function (density::TransportDensity)(u, v, w, params = density.params)
    return u[2] * (v[1] + params.c * w[1])
end

"""
The partial derivatives of the doubled-field density, component by component: `∂L/∂u = (0,
u_t + c u_x)` from the adjoint field, `∂L/∂v = (ψ, 0)` and `∂L/∂w = (c ψ, 0)` from the transported
one.
"""
function derivatives(density::TransportDensity, u, v, w, params = density.params)
    c = params.c
    z = zero(u[2] * c + v[1] + w[1])
    return (u = (z, v[1] + c * w[1]), v = (u[2], z), w = (c * u[2], z))
end

# The partials are affine in the field, so the residual is linear in the coefficients.
MultisymplecticIntegrators.is_linear_density(::TransportDensity) = true

"""
    density(; params = default_parameters())

The Lagrangian density of the doubled field, `L = ψ (v₁ + c w₁)`, whose parameter is the speed `c`.
"""
function density(; params = default_parameters())
    return TransportDensity(; params = params)
end

function solution_u(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    u = T(0.5) * exp(-(x + 2 - p.c * t)^2) / sqrt(T(π))
    return (u, zero(u))
end

function solution_v(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    v = p.c * (2 + x - p.c * t) * exp(-((2 + x - p.c * t)^2)) / sqrt(T(π))
    return (v, zero(v))
end

function solution_w(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    w = -(2 + x - p.c * t) * exp(-((2 + x - p.c * t)^2)) / sqrt(T(π))
    return (w, zero(w))
end

"""The exact solution of the doubled field at time `t`, as [`field_exact`](@ref) lifts it."""
function solutions(D::Integer = 2)
    return (u = field_exact(solution_u, D), v = field_exact(solution_v, D),
        w = field_exact(solution_w, D))
end

"""
    initial_data(x::AbstractVector, params, D = 2, t = 0.0)

The initial data `(u = …, v = …, w = …)` at time `t`, one `(length(x), D)` matrix per entry, of
which the adjoint field's column is zero.
"""
function initial_data(x::AbstractVector, params = default_parameters(), D::Integer = 2, t = 0.0)
    field_initial_data(solutions(D), x, params, D, t)
end

"""The side data of the slab at the one time `t`, as [`field_side_data`](@ref) gives it."""
function side_data(t, xspan, params = default_parameters(), D::Integer = 2)
    field_side_data(solutions(D), t, params, D, xspan)
end

"""
    problem(; T = Float64, params = default_parameters(), timespan = (0.0, 10.0),
            timestep = 1.0, xspan = (-0.5, 0.0), xstep = 0.01)

A [`LagrangianPDEProblem`](@ref) of the doubled field, `D = 2` as the density needs. The default
`xspan` is `(-0.5, 0.0)`, where the initial profile is smooth and nonzero over the whole slab.
Every parameter is converted to `T`, so that the density and the energy of a `Float32` problem are
evaluated in `Float32`.
"""
function problem(; T::Type = Float64, params = default_parameters(),
        timespan = (zero(T), T(10)), timestep = T(1.0), xspan = (T(-0.5), zero(T)),
        xstep = T(0.01))
    params = map(T, params)
    D = 2
    tspan = (T(timespan[1]), T(timespan[2]))
    xsp = (T(xspan[1]), T(xspan[2]))
    s = solutions(D)
    ics = (x, p) -> field_initial_data(s, x, p, D, tspan[1])
    bcs = (t, p) -> field_side_data(s, t, p, D, xsp)
    return LagrangianPDEProblem(density(; params = params);
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
