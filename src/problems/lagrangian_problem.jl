"""
    LagrangianEquation{P} <: GeometricEquation

The equation of a Lagrangian field theory in one space dimension, with parameters of type `P`.

It carries no data of its own: the density lives in the problem, because every problem of this kind
is a different density. The type exists so that a `LagrangianPDEProblem` has the equation type its
`GeometricProblem` supertype needs.
"""
struct LagrangianEquation{P} <: GeometricEquation{NullInvariants, P, NullPeriodicity} end

GeometricBase.periodicity(::LagrangianEquation) = NullPeriodicity()

"""
    LagrangianPDEProblem(density; D = 1, ics_function, bcs_function, exact_u, exact_v, exact_w,
                         timespan, timestep, xspan, xstep, params = density.params)

A Lagrangian field theory in one space dimension on a space-time slab, with a generic element type.

`density` is a [`LagrangianDensity`](@ref). `ics_function(x, params)` returns the named tuple
`(u = …, v = …, w = …)` of the initial data on the vector `x` of grid points; each entry is a
vector of length `D * length(x)`, laid out component-major, `θ = [θ¹; θ²; …]`, so that one scalar
ansatz serves every component. `bcs_function(t, params)` returns the boundary data of the slab at
the physical time `t`, at the two ends of `xspan` and across the space nodes. `exact_u`, `exact_v`
and `exact_w` are `(t, x, params)` functions,
or `nothing` where the problem has no exact solution.

`timespan`, `timestep`, `xspan` and `xstep` are converted to the element type `T` of `timespan`.
The functions of the problem take its parameters, never a module-level default.

```julia
problem = LagrangianPDEProblem(density;
    ics_function = (x, p) -> (u = …, v = …, w = …),
    bcs_function = (t, p) -> …,
    timespan = (0.0, 1.0), timestep = 0.1, xspan = (0.0, 1.0), xstep = 0.01)
```
"""
struct LagrangianPDEProblem{T, PT, LDT, ICT, BCT, EUT, EVT, EWT, ICST} <:
       GeometricProblem{LagrangianEquation{PT}, T, T}
    density::LDT
    params::PT
    D::Int
    ics_function::ICT
    bcs_function::BCT
    exact_u::EUT
    exact_v::EVT
    exact_w::EWT
    ics::ICST
    timespan::NTuple{2, T}
    timestep::T
    xspan::NTuple{2, T}
    xstep::T
end

function LagrangianPDEProblem(density::LagrangianDensity;
        D::Integer = 1,
        ics_function,
        bcs_function,
        exact_u = nothing,
        exact_v = nothing,
        exact_w = nothing,
        timespan,
        timestep,
        xspan,
        xstep,
        params = density.params)
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
    xgrid = collect(xsp[1]:xh:xsp[2])
    length(xgrid) ≥ 2 ||
        throw(ArgumentError("xspan with xstep holds $(length(xgrid)) points; two are needed"))
    ics = ics_function(xgrid, params)
    for name in (:u, :v, :w)
        haskey(ics, name) ||
            throw(ArgumentError("ics_function must return the entries u, v and w, got $(keys(ics))"))
        length(getfield(ics, name)) == D * length(xgrid) ||
            throw(ArgumentError("ics.$name has length $(length(getfield(ics, name))); " *
                                "D = $D and $(length(xgrid)) grid points need $(D * length(xgrid))"))
    end
    return LagrangianPDEProblem{T, typeof(params), typeof(density), typeof(ics_function),
        typeof(bcs_function), typeof(exact_u), typeof(exact_v),
        typeof(exact_w), typeof(ics)}(
        density, params, Int(D), ics_function, bcs_function, exact_u, exact_v, exact_w, ics,
        tspan, h, xsp, xh)
end

function _interval(::Type{T}, span, name::Symbol) where {T}
    length(span) == 2 || throw(ArgumentError("$name must hold two values, got $span"))
    a, b = convert(T, span[1]), convert(T, span[2])
    a < b || throw(ArgumentError("$name must increase, got $a to $b"))
    return (a, b)
end

GeometricBase.datatype(::LagrangianPDEProblem{T}) where {T} = T
GeometricBase.timetype(::LagrangianPDEProblem{T}) where {T} = T
GeometricBase.arrtype(::LagrangianPDEProblem{T}) where {T} = Vector{T}
GeometricBase.equtype(::LagrangianPDEProblem{<:Any, PT}) where {PT} = LagrangianEquation{PT}

GeometricBase.timespan(problem::LagrangianPDEProblem) = problem.timespan
GeometricBase.timestep(problem::LagrangianPDEProblem) = problem.timestep
GeometricBase.parameters(problem::LagrangianPDEProblem) = problem.params
function GeometricBase.initialstate(problem::LagrangianPDEProblem)
    State(problem.timespan[1], map(AlgebraicVariable, problem.ics))
end
function GeometricBase.periodicity(problem::LagrangianPDEProblem)
    map(_ -> NullPeriodicity(), problem.ics)
end
function GeometricBase.ntime(problem::LagrangianPDEProblem)
    Int(abs(div(problem.timespan[end] - problem.timespan[1], problem.timestep, RoundUp)))
end

xspan(problem::LagrangianPDEProblem) = problem.xspan
xstep(problem::LagrangianPDEProblem) = problem.xstep
density(problem::LagrangianPDEProblem) = problem.density
ndof(problem::LagrangianPDEProblem) = problem.D

"""The number of components of the field."""
components(problem::LagrangianPDEProblem) = problem.D

"""The vector of grid points in space."""
function xgrid(problem::LagrangianPDEProblem{T}) where {T}
    return collect(problem.xspan[1]:problem.xstep:problem.xspan[2])
end

"""The initial conditions of the problem, merged with its initial time."""
function GeometricEquations.initial_conditions(problem::LagrangianPDEProblem)
    return merge((t = problem.timespan[1],), problem.ics)
end

GeometricEquations.compute_vectorfields!(state, ::LagrangianPDEProblem) = nothing

function GeometricIntegratorsBase._extrapolate!(newsolution, oldsolution, ::LagrangianPDEProblem, extrap)
    nothing
end

function Base.show(io::IO, problem::LagrangianPDEProblem{T}) where {T}
    print(io, "LagrangianPDEProblem{$T} with $(problem.D) component(s) on ")
    print(io, "timespan = $(problem.timespan), timestep = $(problem.timestep), ")
    print(io, "xspan = $(problem.xspan), xstep = $(problem.xstep)")
end
