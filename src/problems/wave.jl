"""
    Wave

The scalar wave equation `∂ₜ²u − c² ∂ₓ²u = 0` on `x ∈ [0, l]`, with the standing-wave exact
solution of the two lowest harmonics.

The Lagrangian density is `L = ½v² − ½c²w²`, and the exact solution is

```math
u(t, x) = (A₁\\cos(πct/l) + B₁\\sin(πct/l + π/6)) \\sin(πx/l)
        + (A₂\\cos(2πct/l) + B₂\\sin(2πct/l + π/6)) \\sin(2πx/l),
```

with `v = ∂ₜu` and `w = ∂ₓu` its derivatives.

```julia
problem = Wave.problem(timespan = (0.0, 3.0), timestep = 0.3, xstep = 0.01)
```
"""
module Wave

export default_parameters, density, problem, exact_u, exact_v, exact_w, initial_data,
       boundary_data

using Symbolics
using MultisymplecticIntegrators

const DEFAULT_PARAMETERS = (c = 0.5, A1 = 0.8, A2 = 0.0, B1 = 0.8, B2 = 0.0, l = 1.0)

default_parameters() = DEFAULT_PARAMETERS

"""
    density(; params = default_parameters(), m = 0.0, β = 0.0)

The Lagrangian density of the wave equation, `L = ½v² − ½c²w²`, as a [`LagrangianDensity`](@ref)
whose parameter is `c`. A non-zero mass `m` or coupling `β` adds `−½m²u² − ¼βu⁴` to the
expression, with `m` and `β` fixed at construction.
"""
function density(; params = default_parameters(), m = 0.0, β = 0.0)
    u = Symbolics.variable(:u)
    v = Symbolics.variable(:v)
    w = Symbolics.variable(:w)
    c = Symbolics.variable(:c)
    L = v^2 / 2 - c^2 * w^2 / 2
    m == 0 || (L -= m^2 * u^2 / 2)
    β == 0 || (L -= β * u^4 / 4)
    return LagrangianDensity(L, u, v, w; params = (c = params.c,))
end

function exact_u(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    ω₁ = T(π) * p.c / p.l
    return (p.A1 * cos(ω₁ * t) + p.B1 * sin(ω₁ * t + T(π) / 6)) * sin(T(π) * x / p.l) +
           (p.A2 * cos(2ω₁ * t) + p.B2 * sin(2ω₁ * t + T(π) / 6)) *
           sin(2 * T(π) * x / p.l)
end

function exact_v(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    ω₁ = T(π) * p.c / p.l
    return (-p.A1 * ω₁ * sin(ω₁ * t) + p.B1 * ω₁ * cos(ω₁ * t + T(π) / 6)) *
           sin(T(π) * x / p.l) +
           (-2 * p.A2 * ω₁ * sin(2ω₁ * t) + 2 * p.B2 * ω₁ * cos(2ω₁ * t + T(π) / 6)) *
           sin(2 * T(π) * x / p.l)
end

function exact_w(t, x, params = default_parameters())
    T = promote_type(typeof(t), typeof(x))
    p = convert_parameters(T, params)
    return (p.A1 * cos(T(π) * p.c * t / p.l) +
            p.B1 * sin(T(π) * p.c * t / p.l + T(π) / 6)) * T(π) / p.l *
           cos(T(π) * x / p.l) +
           (p.A2 * cos(2 * T(π) * p.c * t / p.l) +
            p.B2 * sin(2 * T(π) * p.c * t / p.l + T(π) / 6)) * 2 * T(π) / p.l *
           cos(2 * T(π) * x / p.l)
end

"""
    initial_data(x::AbstractVector, params, D = 1, t = 0.0)

The initial data `(u = …, v = …, w = …)` at time `t`, each entry of length `D * length(x)` and
laid out component-major.
"""
function initial_data(x::AbstractVector, params = default_parameters(), D::Integer = 1, t = 0.0)
    field_initial_data(exact_u, exact_v, exact_w, x, params, D, t)
end

"""
    boundary_data(times, xbreaks, params, D = 1)

The boundary data of the slab whose two times are `times = (t₀, t₁)`, as
[`field_boundary_data`](@ref) gives it.
"""
function boundary_data(times, xbreaks::AbstractVector, params = default_parameters(), D::Integer = 1)
    field_boundary_data(exact_u, exact_v, exact_w, times, xbreaks, params, D)
end

"""
    problem(; T = Float64, D = 1, params = default_parameters(), timespan = (0.0, 3.0),
            timestep = 0.3, xspan = (0.0, 1.0), xstep = 0.01)

A [`LagrangianPDEProblem`](@ref) of the wave equation, with the massless density
`L = ½v² − ½c²w²` of which the standing wave is the exact solution. A mass or a quartic coupling is
a keyword of `density` only, because the exact solution holds for neither.
"""
function problem(; T::Type = Float64, D::Integer = 1, params = default_parameters(),
        timespan = (zero(T), T(3)), timestep = T(0.3), xspan = (zero(T), one(T)),
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

"""
    VectorWave

The wave equation for a field `u ∈ ℝᴰ` on `x ∈ [0, l]` with zero Dirichlet data, the density

```math
L = \\tfrac{1}{2}|u_t|^2 - \\tfrac{c^2}{2}|u_x|^2 - \\tfrac{m^2}{2}|u|^2 - \\tfrac{\\beta}{4}|u|^4,
```

the exact solution of the uncoupled case `m = β = 0`, each component a standing wave of `Wave` with
amplitudes of its own, and the angular momentum `J = ∫ (u₁ ∂ₜu₂ − u₂ ∂ₜu₁) dx` of the `D = 2`
field.

`L` is invariant under a rotation of `(u, v, w)`, so `J` is conserved; a numerical integration that
loses it has lost the symmetry, which is the oracle of the vector case. The default parameters give
the two components different phases, so that `J` is not zero.

```julia
problem = VectorWave.problem(timespan = (0.0, 1.0), timestep = 0.1, xstep = 0.05)
```
"""
module VectorWave

export default_parameters, component_parameters, density, vector_density, problem,
       angular_momentum

using MultisymplecticIntegrators
import MultisymplecticIntegrators: derivatives
import ..Wave

const DEFAULT_PARAMETERS = (
    c = 0.5, l = 1.0, A1 = (0.8, 0.8), A2 = (0.0, 0.0), B1 = (0.8, 0.0),
    B2 = (0.0, 0.0))

default_parameters() = DEFAULT_PARAMETERS

"""
    component_parameters(params, i)

The parameters of `Wave` for component `i`: the shared speed `c` and length `l`, and the `i`-th
entry of each of the amplitudes `A1`, `A2`, `B1` and `B2`.
"""
function component_parameters(params, i::Integer)
    return (c = params.c, A1 = params.A1[i], A2 = params.A2[i], B1 = params.B1[i],
        B2 = params.B2[i], l = params.l)
end

"""
    VectorDensity(density, D = 2, β = 0)

The density of a component-major vector field `u ∈ ℝᴰ`: the scalar `density` summed over the `D`
components, minus the quartic coupling `¼β|u|⁴`. The scalar `density` carries no quartic term, so
that the whole is invariant under a rotation of `(u, v, w)`. `u`, `v` and `w` are vectors of the
`D` components' values at one point.
"""
struct VectorDensity{DT, BT}
    density::DT
    D::Int
    β::BT
end

VectorDensity(density; D::Integer = 2, β = 0.0) = VectorDensity(density, Int(D), β)

function (density::VectorDensity)(u, v, w, params)
    length(u) == density.D ||
        throw(ArgumentError("expected $(density.D) components, got $(length(u))"))
    r² = sum(abs2, u)
    return sum(density.density(u[i], v[i], w[i], params) for i in eachindex(u)) -
           oftype(r², density.β) * r²^2 / 4
end

"""
The derivatives of a vector density, component by component: those of the scalar density, and
`−β|u|²uᵢ` from the quartic coupling in `∂L/∂uᵢ`.
"""
function derivatives(density::VectorDensity, u, v, w, params)
    length(u) == density.D ||
        throw(ArgumentError("expected $(density.D) components, got $(length(u))"))
    r² = sum(abs2, u)
    β = oftype(r², density.β)
    each = [derivatives(density.density, u[i], v[i], w[i], params) for i in eachindex(u)]
    return (u = [d.u - β * r² * uᵢ for (d, uᵢ) in zip(each, u)], v = [d.v for d in each],
        w = [d.w for d in each])
end

"""
    density(; params = default_parameters(), m = 0.0)

The scalar density of one component, `½v² − ½c²w² − ½m²u²`. The quartic coupling of the
components is in `vector_density`, because `¼β|u|⁴` is not a sum over the components.
"""
function density(; params = default_parameters(), m = 0.0)
    Wave.density(; params = params, m = m)
end

"""
    vector_density(; params = default_parameters(), D = 2, m = 0.0, β = 0.0)

The vector density of `D` components with the mass `m` and the quartic coupling `β`.
"""
function vector_density(; params = default_parameters(), D::Integer = 2, m = 0.0, β = 0.0)
    VectorDensity(density(; params = params, m = m), Int(D), β)
end

"""
    angular_momentum(u₁, u₂, v₁, v₂, x)

`J = ∫ (u₁ v₂ − u₂ v₁) dx` by the trapezoidal rule on the grid `x`. It is the charge of the
rotation that leaves the density invariant, so an exact solution conserves it.
"""
function angular_momentum(u₁, u₂, v₁, v₂, x::AbstractVector)
    f = @. u₁ * v₂ - u₂ * v₁
    s = zero(eltype(f))
    for i in 1:(length(x) - 1)
        s += (f[i] + f[i + 1]) * (x[i + 1] - x[i]) / 2
    end
    return s
end

"""
    problem(; T = Float64, params = default_parameters(), timespan = (0.0, 3.0),
            timestep = 0.3, xspan = (0.0, 1.0), xstep = 0.01)

A [`LagrangianPDEProblem`](@ref) of `D = length(params.A1)` components, with zero Dirichlet data in
space and the exact solution of the uncoupled case: component `i` is the standing wave of `Wave`
with the parameters `component_parameters(params, i)`. The initial, boundary and exact data are
stacked component-major.
"""
function problem(; T::Type = Float64, params = default_parameters(),
        timespan = (zero(T), T(3)), timestep = T(0.3), xspan = (zero(T), one(T)),
        xstep = T(0.01))
    D = length(params.A1)
    all(name -> length(getfield(params, name)) == D, (:A2, :B1, :B2)) ||
        throw(ArgumentError("A1, A2, B1 and B2 must hold one amplitude per component"))
    tspan = (T(timespan[1]), T(timespan[2]))
    xsp = (T(xspan[1]), T(xspan[2]))
    h = T(xstep)
    grid = collect(xsp[1]:h:xsp[2])
    # `map(vcat, …)` over the components' named tuples stacks each entry component-major.
    ics = (x, p) -> map(vcat,
        (Wave.initial_data(x, component_parameters(p, i), 1, tspan[1]) for i in 1:D)...)
    bcs = (times, p) -> map((sides...) -> map(vcat, sides...),
        (Wave.boundary_data(times, grid, component_parameters(p, i)) for i in 1:D)...)
    exact(f) = (t, x, p) -> mapreduce(i -> f(t, x, component_parameters(p, i)), vcat, 1:D)
    return LagrangianPDEProblem(density(; params = params);
        D = D,
        ics_function = ics,
        bcs_function = bcs,
        exact_u = exact(field_exact(Wave.exact_u, 1)),
        exact_v = exact(field_exact(Wave.exact_v, 1)),
        exact_w = exact(field_exact(Wave.exact_w, 1)),
        timespan = tspan,
        timestep = T(timestep),
        xspan = xsp,
        xstep = h,
        params = params)
end

end
