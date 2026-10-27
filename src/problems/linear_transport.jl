"""
    LinearTransport

The linear transport equation `∂ₜu + c ∂ₓu = 0` on `x ∈ [-0.5, 0.0]`, whose exact solution is the
initial profile shifted by `c t`.

The Lagrangian density is `L = ½u (v + c w)`. It is a null Lagrangian: its Euler–Lagrange
expression vanishes identically, because `∂L/∂u = (v + c w)/2`, `∂ₜ(∂L/∂v) = ∂ₜu/2` and
`∂ₓ(∂L/∂w) = c ∂ₓu/2`. The transport equation is therefore carried by the constraint `v = −c w`,
which the ansatz of the family that integrates this problem imposes, and not by the density alone.
The exact solution is the Gaussian

```math
u(t, x) = \\frac{1}{2\\sqrt{\\pi}} \\exp\\!\\big(-(x + 2 - ct)^2\\big),
```

```julia
problem = LinearTransport.problem(timespan = (0.0, 10.0), timestep = 1.0, xstep = 0.01)
```
"""
module LinearTransport

export default_parameters, density, problem, exact_u, exact_v, exact_w, initial_data,
       boundary_data

using Symbolics
using MultisymplecticIntegrators

const DEFAULT_PARAMETERS = (c = 0.2,)

default_parameters() = DEFAULT_PARAMETERS

"""
    density(; params = default_parameters())

The Lagrangian density `L = ½u (v + c w)`, whose parameter is the speed `c`.
"""
function density(; params = default_parameters())
    u, v, w, c = Symbolics.variable(:u), Symbolics.variable(:v), Symbolics.variable(:w),
    Symbolics.variable(:c)
    L = u * (v + c * w) / 2
    return LagrangianDensity(L, u, v, w; params = (c = params.c,))
end

function exact_u(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    return T(0.5) * exp(-(x + 2 - p.c * t)^2) / sqrt(T(π))
end

function exact_v(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    return p.c * (2 + x - p.c * t) * exp(-((2 + x - p.c * t)^2)) / sqrt(T(π))
end

function exact_w(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    return -(2 + x - p.c * t) * exp(-((2 + x - p.c * t)^2)) / sqrt(T(π))
end

"""The initial data `(u = …, v = …, w = …)` at `t = timespan[1]`."""
function initial_data(x::AbstractVector, params = default_parameters(), D::Integer = 1, t = 0.0)
    field_initial_data(exact_u, exact_v, exact_w, x, params, D, t)
end

"""The boundary data of the slab whose two times are `times = (t₀, t₁)`."""
function boundary_data(times, xbreaks::AbstractVector, params = default_parameters(), D::Integer = 1)
    field_boundary_data(exact_u, exact_v, exact_w, times, xbreaks, params, D)
end

"""
    problem(; T = Float64, D = 1, params = default_parameters(), timespan = (0.0, 10.0),
            timestep = 1.0, xspan = (-0.5, 0.0), xstep = 0.01)

A [`LagrangianPDEProblem`](@ref) of the linear transport equation. The default `xspan` is
`(-0.5, 0.0)`, where the initial profile is smooth and nonzero over the whole slab.
"""
function problem(; T::Type = Float64, D::Integer = 1, params = default_parameters(),
        timespan = (zero(T), T(10)), timestep = T(1.0), xspan = (T(-0.5), zero(T)),
        xstep = T(0.01))
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
