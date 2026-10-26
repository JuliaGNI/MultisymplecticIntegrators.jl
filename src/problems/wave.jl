"""
    Wave

The scalar wave equation `∂ₜ²u − c² ∂ₓ²u = 0` on `x ∈ [0, l]`, with the standing-wave exact
solution that the reference values of part A were recorded from.

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
whose parameter is `c`. A non-zero mass `m` or coupling `β` adds `−½m²u² − ¼βu⁴`.
"""
function density(; params = default_parameters(), m = 0.0, β = 0.0)
    u, v, w, c = Symbolics.variable(:u), Symbolics.variable(:v), Symbolics.variable(:w),
    Symbolics.variable(:c)
    L = v^2 / 2 - c^2 * w^2 / 2
    m == 0 || (L -= m^2 * u^2 / 2)
    β == 0 || (L -= β * u^4 / 4)
    return LagrangianDensity(L, u, v, w; params = (c = params.c, m = m, β = β))
end

function exact_u(t, x, params = default_parameters())
    ω₁ = π * params.c / params.l
    return (params.A1 * cos(ω₁ * t) + params.B1 * sin(ω₁ * t + π / 6)) *
           sin(π * x / params.l) +
           (params.A2 * cos(2ω₁ * t) + params.B2 * sin(2ω₁ * t + π / 6)) *
           sin(2π * x / params.l)
end

function exact_v(t, x, params = default_parameters())
    ω₁ = π * params.c / params.l
    return (-params.A1 * ω₁ * sin(ω₁ * t) + params.B1 * ω₁ * cos(ω₁ * t + π / 6)) *
           sin(π * x / params.l) +
           (-2 * params.A2 * ω₁ * sin(2ω₁ * t) +
            2 * params.B2 * ω₁ * cos(2ω₁ * t + π / 6)) *
           sin(2π * x / params.l)
end

function exact_w(t, x, params = default_parameters())
    return (params.A1 * cos(π * params.c * t / params.l) +
            params.B1 * sin(π * params.c * t / params.l + π / 6)) * π / params.l *
           cos(π * x / params.l) +
           (params.A2 * cos(2π * params.c * t / params.l) +
            params.B2 * sin(2π * params.c * t / params.l + π / 6)) * 2π / params.l *
           cos(2π * x / params.l)
end

"""
    initial_data(x::AbstractVector, params, D = 1)

The initial data `(u = …, v = …, w = …)` at `t = timespan[1]`, each entry of length
`D * length(x)`, laid out component-major.
"""
function initial_data(x::AbstractVector, params = default_parameters(), D::Integer = 1, t = 0.0)
    u = reduce(vcat, [exact_u(t, xi, params) for _ in 1:D for xi in x])
    v = reduce(vcat, [exact_v(t, xi, params) for _ in 1:D for xi in x])
    w = reduce(vcat, [exact_w(t, xi, params) for _ in 1:D for xi in x])
    return (u = u, v = v, w = w)
end

function exact_u(t, x::AbstractVector, params = default_parameters())
    [exact_u(t, xi, params) for xi in x]
end
function exact_v(t, x::AbstractVector, params = default_parameters())
    [exact_v(t, xi, params) for xi in x]
end
function exact_w(t, x::AbstractVector, params = default_parameters())
    [exact_w(t, xi, params) for xi in x]
end

"""
    boundary_data(t, xbreaks, params)

The four boundary point sets of a slab at physical time `t`, with zero Dirichlet data in space.
"""
function boundary_data(t, xbreaks, params = default_parameters())
    u = exact_u(t, xbreaks, params)
    v = exact_v(t, xbreaks, params)
    w = exact_w(t, xbreaks, params)
    return (bottom = (u = u, v = v, w = w), top = (u = zero(u), v = zero(v), w = zero(w)),
        left = zero(eltype(u)), right = zero(eltype(u)))
end

"""
    problem(; T = Float64, D = 1, params = default_parameters(), timespan = (0.0, 3.0),
            timestep = 0.3, xspan = (0.0, 1.0), xstep = 0.01, m = 0.0, β = 0.0)

A [`LagrangianPDEProblem`](@ref) of the wave equation.
"""
function problem(; T::Type = Float64, D::Integer = 1, params = default_parameters(),
        timespan = (zero(T), T(3)), timestep = T(0.3), xspan = (zero(T), one(T)),
        xstep = T(0.01), m = 0.0, β = 0.0)
    d = density(; params = params, m = m, β = β)
    ics = (x, p) -> initial_data(x, p, D, timespan[1])
    bcs = (t, p) -> boundary_data(t, collect(xspan[1]:xstep:xspan[2]), p)
    return LagrangianPDEProblem(d;
        D = D,
        ics_function = ics,
        bcs_function = bcs,
        exact_u = (t, x, p) -> reduce(vcat, [exact_u(t, xi, p) for _ in 1:D for xi in x]),
        exact_v = (t, x, p) -> reduce(vcat, [exact_v(t, xi, p) for _ in 1:D for xi in x]),
        exact_w = (t, x, p) -> reduce(vcat, [exact_w(t, xi, p) for _ in 1:D for xi in x]),
        timespan = (T(timespan[1]), T(timespan[2])),
        timestep = T(timestep),
        xspan = (T(xspan[1]), T(xspan[2])),
        xstep = T(xstep),
        params = params)
end

end

"""
    VectorWave

The wave equation for a field `u ∈ ℝᴰ` on `x ∈ [0, l]` with zero Dirichlet data, the density

```math
L = \\tfrac{1}{2}|u_t|^2 - \\tfrac{c^2}{2}|u_x|^2 - \\tfrac{m^2}{2}|u|^2 - \\tfrac{\\beta}{4}|u|^4,
```

the exact solution of the uncoupled case `m = β = 0`, each component a standing wave of `Wave`, and
the angular momentum `J = ∫ (u₁ ∂ₜu₂ − u₂ ∂ₜu₁) dx` of the `D = 2` field.

`L` is invariant under a rotation of `u`, so `J` is conserved; a numerical integration that loses
it has lost the symmetry, which is the oracle of the vector case.

```julia
problem = VectorWave.problem(timespan = (0.0, 1.0), timestep = 0.1, xstep = 0.05)
```
"""
module VectorWave

export default_parameters, density, problem, angular_momentum

using LinearAlgebra
using MultisymplecticIntegrators
import MultisymplecticIntegrators: derivatives
import ..Wave

const DEFAULT_PARAMETERS = (c = 0.5, A1 = 0.8, A2 = 0.0, B1 = 0.8, B2 = 0.0, l = 1.0)

default_parameters() = DEFAULT_PARAMETERS

"""
    VectorDensity(density, D = 2)

The density of a component-major vector field: the sum of the scalar `density` over the `D`
components. `u`, `v` and `w` are vectors of the `D` components' values at one point.
"""
struct VectorDensity{DT}
    density::DT
    D::Int
end

VectorDensity(density; D::Integer = 2) = VectorDensity(density, Int(D))

function (density::VectorDensity)(u, v, w, params)
    length(u) == density.D ||
        throw(ArgumentError("expected $(density.D) components, got $(length(u))"))
    return sum(density.density(u[i], v[i], w[i], params) for i in eachindex(u))
end

"""The derivatives of a vector density, component by component."""
function derivatives(density::VectorDensity, u, v, w, params)
    length(u) == density.D ||
        throw(ArgumentError("expected $(density.D) components, got $(length(u))"))
    return (
        u = [derivatives(density.density, u[i], v[i], w[i], params).u for i in eachindex(u)],
        v = [derivatives(density.density, u[i], v[i], w[i], params).v for i in eachindex(u)],
        w = [derivatives(density.density, u[i], v[i], w[i], params).w for i in eachindex(u)])
end

"""The scalar density of one component of the vector wave equation."""
function density(; params = default_parameters(), m = 0.0, β = 0.0)
    Wave.density(; params = params, m = m, β = β)
end

"""The vector density of `D = 2` components."""
function vector_density(; params = default_parameters(), D::Integer = 2, m = 0.0, β = 0.0)
    VectorDensity(density(; params = params, m = m, β = β), D)
end

"""
    angular_momentum(u₁, u₂, v₁, v₂, x)

`J = ∫ (u₁ v₂ − u₂ v₁) dx` by the trapezoidal rule on the grid `x`.
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

A [`LagrangianPDEProblem`](@ref) with `D = 2`, zero Dirichlet data in space and the exact solution
of the uncoupled case.
"""
function problem(; T::Type = Float64, params = default_parameters(),
        timespan = (zero(T), T(3)), timestep = T(0.3), xspan = (zero(T), one(T)),
        xstep = T(0.01))
    return Wave.problem(;
        T = T, D = 2, params = params, timespan = timespan, timestep = timestep,
        xspan = xspan, xstep = xstep)
end

end
