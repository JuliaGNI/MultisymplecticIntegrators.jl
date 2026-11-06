"""
    LagrangianEquation{P} <: GeometricEquation

The equation of a Lagrangian field theory in one space dimension, with parameters of type `P`.

It carries no data of its own: the density lives in the problem, because every problem of this kind
is a different density. The type exists so that a `LagrangianPDEProblem` has the equation type its
`GeometricProblem` supertype needs.
"""
struct LagrangianEquation{P} <: GeometricEquation{NullInvariants, P, NullPeriodicity} end

periodicity(::LagrangianEquation) = NullPeriodicity()

"""
    LagrangianPDEProblem(density; D = 1, ics_function, bcs_function, solutions = nothing,
                         timespan, timestep, xspan, xstep, params = density.params)

A Lagrangian field theory in one space dimension on a space-time slab, with a generic element type.

`density` is a Lagrangian density: a callable `L(u, v, w, params)` with a `derivatives` method
returning its three partial derivatives `∂L/∂u`, `∂L/∂v` and `∂L/∂w`. A
[`LagrangianDensity`](@ref) supplies both for one component at a time; a density that couples the
components of the field, such as the vector density that `VectorWave.vector_density` builds,
supplies them for the `D` components of one point at once. Such a density receives the `D` values
of `u`, `v` and `w` at a point as an indexable collection: the couplings of the slab pass
`NTuple{D}`s to `derivatives`, and [`energy_error`](@ref) passes `Vector`s, or scalars where
`D = 1`, to `derivatives` and to the density itself. Write its methods for any collection of length `D`, not for
`AbstractVector` alone; `derivatives` returns the named tuple `(u = …, v = …, w = …)` of the `D`
values of each partial derivative. The problem's parameters are the
density's own `params` field where it has one, and the `params` keyword otherwise.

`ics_function(x, params)` returns the named tuple `(u = …, v = …, w = …)` of the initial data on the
vector `x` of grid points, each entry a `length(x) × D` matrix whose column `d` holds component `d`.
`bcs_function(t, params)` returns the side data of the slab at the one time `t`: `left` and `right`,
each `(u = …, v = …, w = …)` of `D` vectors, entry `d` component `d`'s value at that end.
`solutions` is `nothing` where the problem has no exact solution, and otherwise the named tuple
`(u = …, v = …, w = …)` of `(t, x, params)` functions that return the `D` components' values at a
point and the component-major vector of length `D * length(x)` at a grid.

`timespan`, `timestep`, `xspan` and `xstep` are converted to the element type `T` of `timespan`.
The functions of the problem take its parameters, never a module-level default.

```julia
problem = LagrangianPDEProblem(density;
    ics_function = (x, p) -> (u = …, v = …, w = …),
    bcs_function = (t, p) -> …,
    timespan = (0.0, 1.0), timestep = 0.1, xspan = (0.0, 1.0), xstep = 0.01)
```
"""
struct LagrangianPDEProblem{T, PT, LDT, ICT, BCT, SOT, ICST} <:
       GeometricProblem{LagrangianEquation{PT}, T, T}
    density::LDT
    params::PT
    D::Int
    ics_function::ICT
    bcs_function::BCT
    solutions::SOT
    ics::ICST
    timespan::NTuple{2, T}
    timestep::T
    xspan::NTuple{2, T}
    xstep::T
end

function LagrangianPDEProblem(density;
        D::Integer = 1,
        ics_function,
        bcs_function,
        solutions = nothing,
        timespan,
        timestep,
        xspan,
        xstep,
        params = density_params(density))
    T = eltype(timespan)
    T <: AbstractFloat || throw(ArgumentError("the element type must be a float, got $T"))
    T === Float16 && throw(ArgumentError("Float16 problems are not supported"))
    D ≥ 1 || throw(ArgumentError("D must be ≥ 1, got $D"))
    tspan = _interval(T, timespan, :timespan)
    xsp = _interval(T, xspan, :xspan)
    h = convert(T, timestep)
    xh = convert(T, xstep)
    h > zero(T) || throw(ArgumentError("timestep must be positive, got $h"))
    xh > zero(T) || throw(ArgumentError("xstep must be positive, got $xh"))
    grid = _xgrid(xsp, xh)
    ics = ics_function(collect(grid), params)
    for name in (:u, :v, :w)
        haskey(ics, name) ||
            throw(ArgumentError("ics_function must return the entries u, v and w, got $(keys(ics))"))
        size(getfield(ics, name)) == (length(grid), D) ||
            throw(ArgumentError("ics.$name is of size $(size(getfield(ics, name))); " *
                                "D = $D and $(length(grid)) grid points need " *
                                "($(length(grid)), $D)"))
    end
    return LagrangianPDEProblem{T, typeof(params), typeof(density), typeof(ics_function),
        typeof(bcs_function), typeof(solutions), typeof(ics)}(
        density, params, Int(D), ics_function, bcs_function, solutions, ics, tspan, h, xsp, xh)
end

"""
    density_params(density)

The parameters of a density: its own `params` field where it has one, and `nothing` otherwise.
"""
density_params(density) = hasproperty(density, :params) ? density.params : nothing

function _interval(::Type{T}, span, name::Symbol) where {T}
    length(span) == 2 || throw(ArgumentError("$name must hold two values, got $span"))
    a, b = convert(T, span[1]), convert(T, span[2])
    isfinite(a) && isfinite(b) || throw(ArgumentError("$name must be finite, got $a to $b"))
    a < b || throw(ArgumentError("$name must increase, got $a to $b"))
    return (a, b)
end

"""
    field_initial_data(solutions, x, params, D, t)

The initial data `(u = …, v = …, w = …)` of `D` components at time `t`: three
`(length(x), D)` matrices, entry `[r, d]` the value of component `d` at the `r`-th point of `x`.
"""
function field_initial_data(solutions, x::AbstractVector, params, D::Integer, t)
    (; u, v, w) = solutions
    across(f) = [_field_component(f(t, xi, params), d) for xi in x, d in 1:D]
    return (u = across(u), v = across(v), w = across(w))
end

# One component of a closed form's value at a point: a number serves every component, and a vector
# holds one entry per component.
_field_component(value::Number, ::Integer) = value
_field_component(value, d::Integer) = value[d]

"""
    convert_parameters(::Type{T}, params)

The parameters of a problem as a named tuple whose entries are all of type `T`, so that a closed
form written with `π` and with the default parameters keeps the element type of its arguments.
"""
convert_parameters(::Type{T}, params) where {T} = map(value -> convert(T, value), params)

"""
    field_exact(f, D)

The exact function of `D` components, `(t, x, params)` returning a `Vector`: the `D` components'
values at a scalar point `x`, and the component-major vector of length `D * length(x)` at a grid.
A closed form that returns a number gives every component that number; one that returns a vector
supplies the components' values.
"""
function field_exact(f, D::Integer)
    return function (t, x, params)
        points = x isa AbstractVector ? x : (x,)
        return vec([_field_component(f(t, xi, params), d) for xi in points, d in 1:D])
    end
end

"""
    field_side_data(solutions, t, params, D, xspan)

The side data of a slab at the one time `t`: `left` and `right`, each `(u = …, v = …, w = …)` of `D`
vectors, entry `d` the exact value of component `d` at that end of `xspan`.
"""
function field_side_data(solutions, t, params, D::Integer, xspan)
    (; u, v, w) = solutions
    side(f, x) = [_field_component(f(t, x, params), d) for d in 1:D]
    left, right = first(xspan), last(xspan)
    return (left = (u = side(u, left), v = side(v, left), w = side(w, left)),
        right = (u = side(u, right), v = side(v, right), w = side(w, right)))
end

datatype(::LagrangianPDEProblem{T}) where {T} = T
timetype(::LagrangianPDEProblem{T}) where {T} = T
equation(::LagrangianPDEProblem{<:Any, PT}) where {PT} = LagrangianEquation{PT}()
arrtype(::LagrangianPDEProblem{T}) where {T} = Vector{T}
equtype(::LagrangianPDEProblem{<:Any, PT}) where {PT} = LagrangianEquation{PT}

timespan(problem::LagrangianPDEProblem) = problem.timespan
timestep(problem::LagrangianPDEProblem) = problem.timestep
parameters(problem::LagrangianPDEProblem) = problem.params
# The state is component-major, as `GeometricSolution` carries it, so the initial state is the
# initial data's columns one after the other.
function initialstate(problem::LagrangianPDEProblem)
    State(problem.timespan[1], map(ics -> AlgebraicVariable(vec(ics)), problem.ics))
end
function periodicity(problem::LagrangianPDEProblem)
    map(_ -> NullPeriodicity(), problem.ics)
end

"""The number of steps a run of the problem takes, which is the number of steps the solution holds."""
function ntime(problem::LagrangianPDEProblem)
    return length(problem.timespan[1]:problem.timestep:problem.timespan[2]) - 1
end

xspan(problem::LagrangianPDEProblem) = problem.xspan
xstep(problem::LagrangianPDEProblem) = problem.xstep

"""The number of components of the field."""
components(problem::LagrangianPDEProblem) = problem.D

"""
    xgrid(problem)

The output grid of the problem: `n = round(Int, Δx / xstep) + 1` equally spaced points from
`xspan[1]` to `xspan[2]`, as a range that ends on the end of the span.
"""
xgrid(problem::LagrangianPDEProblem) = _xgrid(problem.xspan, problem.xstep)

function _xgrid(xsp::NTuple{2, T}, xh::T) where {T}
    n = round(Int, (xsp[2] - xsp[1]) / xh) + 1
    n ≥ 2 || throw(ArgumentError("xstep $xh is too large for xspan $xsp"))
    return range(xsp[1], xsp[2]; length = n)
end

"""
    slab_start(problem, sol)

The start time of the slab that `sol` ends: the solution step's time less one timestep. A solution
step holds its time either as a scalar or as a `GeometricBase.TimeVariable`, whose single entry is
the scalar.
"""
function slab_start(problem::LagrangianPDEProblem, sol)
    return first(sol.t) - timestep(problem)
end

"""The initial conditions of the problem, merged with its initial time."""
function initial_conditions(problem::LagrangianPDEProblem)
    return merge((t = problem.timespan[1],), problem.ics)
end

# The state already carries the field and its two derivatives, so there is no vector field to
# compute from the initial data.
compute_vectorfields!(state, ::LagrangianPDEProblem) = nothing

function Base.show(io::IO, problem::LagrangianPDEProblem{T}) where {T}
    print(io, "LagrangianPDEProblem{$T} with $(problem.D) component(s) on ")
    print(io, "timespan = $(problem.timespan), timestep = $(problem.timestep), ")
    print(io, "xspan = $(problem.xspan), xstep = $(problem.xstep)")
end
