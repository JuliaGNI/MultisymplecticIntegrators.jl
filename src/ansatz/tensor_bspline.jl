"""
    TensorBSplineAnsatz(; degree, ncells, xspan, x_boundary = Dirichlet(), n = nothing)

A tensor B-spline ansatz of one space-time slab, on `SimpleSplines`.

The slab is the square `[0, 1] × xspan` in the normalised time coordinate `t̂` and the physical
space coordinate `x`. The ansatz approximates the field by

```math
u_h(t̂, x) = \\sum_{i,j} \\hat{u}_{ij} \\, B_i(t̂) \\, B_j(x),
```

with a clamped B-spline basis of degree `degree[1]` on `ncells[1]` uniform cells of `[0, 1]` and a
B-spline basis of degree `degree[2]` on `ncells[2]` uniform cells of `xspan`. `degree` and
`ncells` are given per direction, as a number for both or as a `(time, space)` tuple, so that the
two directions may differ in degree and in cell count.

`x_boundary` is the boundary condition the space basis is built with: `Dirichlet()` recombines the
basis at both ends of `xspan`, so that every function vanishes there, and `Free()` keeps the natural
condition. The ansatz carries a homogeneous `Dirichlet` value in `x` only; an inhomogeneous value is
a coupling's business.

`n` is the number of Gauss–Legendre nodes per cell in each direction, again a number or a
`(time, space)` tuple; the default integrates the products of the basis functions and of their first
derivatives exactly, which is `p + 1` nodes for degree `p`, the count the products of the basis
functions need.

Both degrees are at least 1. The slab core reads the space gradient `∂ₓu` from the same space basis,
and a piecewise-constant degree-0 basis carries none, so a lower degree raises `ArgumentError`.

The ansatz counts the **degree**, which is what `SimpleSplines` takes; a basis of order `k` is one of
degree `k - 1`.

```julia
ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0))
```
"""
struct TensorBSplineAnsatz{
    T, BT <: TensorProductBasis{T, 2}, QB <: TensorProductQuadrature,
    BC <: BoundaryCondition} <: SpaceTimeAnsatz
    degree::Tuple{Int, Int}
    ncells::Tuple{Int, Int}
    xspan::NTuple{2, T}
    x_boundary::BC
    basis::BT
    quadrature::QB
end

function TensorBSplineAnsatz(; degree, ncells, xspan,
        x_boundary::BoundaryCondition = Dirichlet(), n = nothing)
    d = _per_axis(degree, :degree)
    c = _per_axis(ncells, :ncells)
    _check_degree(d)
    all(>=(1), c) || throw(ArgumentError("ncells must be ≥ 1, got $c"))
    T = _coordinate_type(xspan)
    xsp = (convert(T, xspan[1]), convert(T, xspan[2]))
    xsp[1] < xsp[2] || throw(ArgumentError("xspan must increase, got $xsp"))
    tbasis = BSplineBasis(UniformMesh(c[1], zero(T) .. one(T)), d[1])
    xbasis = BSplineBasis(UniformMesh(c[2], xsp[1] .. xsp[2]), d[2], x_boundary)
    basis = TensorProductBasis(tbasis, xbasis)
    order = n === nothing ? map(_default_quadrature_order, d) : _per_axis(n, :n)
    all(>=(1), order) || throw(ArgumentError("n must be ≥ 1, got $order"))
    quadrature = TensorProductQuadrature(basis; nq = order)
    return TensorBSplineAnsatz{T, typeof(basis), typeof(quadrature), typeof(x_boundary)}(
        d, c, xsp, x_boundary, basis, quadrature)
end

# A basis of degree 0 is piecewise constant, and the slab core takes the space gradient `∂ₓu` from
# the same space basis, which such a basis does not carry.
function _check_degree(d::Tuple{Int, Int})
    for (axis, p) in zip((:time, :space), d)
        p ≥ 1 || throw(ArgumentError("degree must be ≥ 1 in the $axis direction, got $d"))
    end
    return nothing
end

_per_axis(x::Integer, ::Symbol) = (Int(x), Int(x))
function _per_axis(x, name::Symbol)
    length(x) == 2 ||
        throw(ArgumentError("$name must be a number or a (time, space) tuple, " *
                            "got $x"))
    return (Int(x[1]), Int(x[2]))
end

function _coordinate_type(xspan)
    T = float(promote_type(map(typeof, xspan)...))
    T <: AbstractFloat ||
        throw(ArgumentError("xspan must hold floating-point values, its type is $T"))
    return T
end

# The smallest Gauss rule that integrates the product of two basis functions of degree `p` and
# of their first derivatives exactly, which is degree `2p` and `2p - 2` respectively.
_default_quadrature_order(p::Integer) = max(cld(2p + 1, 2), 1)

"""The tensor B-spline basis of the ansatz, on `[0, 1] × xspan`."""
basis(ansatz::TensorBSplineAnsatz) = ansatz.basis

"""The tensor Gauss–Legendre quadrature of the ansatz."""
quadrature(ansatz::TensorBSplineAnsatz) = ansatz.quadrature

"""The degree of the ansatz in each direction, `(time, space)`."""
degree(ansatz::TensorBSplineAnsatz) = ansatz.degree

"""The number of cells of the ansatz in each direction, `(time, space)`."""
ncells(ansatz::TensorBSplineAnsatz) = ansatz.ncells

"""The space interval of the slab."""
xspan(ansatz::TensorBSplineAnsatz) = ansatz.xspan

"""The boundary condition the space basis is built with."""
x_boundary(ansatz::TensorBSplineAnsatz) = ansatz.x_boundary

"""The one-dimensional basis of the normalised time coordinate."""
time_basis(ansatz::TensorBSplineAnsatz) = bases(basis(ansatz))[1]

"""The one-dimensional basis of the space coordinate."""
space_basis(ansatz::TensorBSplineAnsatz) = bases(basis(ansatz))[2]

"""The number of basis functions in the time direction."""
ntime_functions(ansatz::TensorBSplineAnsatz) = nbasis(time_basis(ansatz))

"""The number of basis functions in the space direction."""
nspace_functions(ansatz::TensorBSplineAnsatz) = nbasis(space_basis(ansatz))

"""The nodes of the space quadrature, the points the ansatz stores its trace at."""
function space_quadrature_nodes(ansatz::TensorBSplineAnsatz{T}) where {T}
    return quadrature_nodes(quadratures(quadrature(ansatz))[2])
end

"""
    output_grid(ansatz::TensorBSplineAnsatz, xstep)
    output_grid(ansatz::TensorBSplineAnsatz, problem::LagrangianPDEProblem)

The grid the ansatz reports its solution on: `xspan` in `n` equally spaced points, with
`n = round(Int, Δx / xstep) + 1`.

A `xstep` that does not divide `xspan` therefore still ends on the end of the span, and the grid is
built as a `range` rather than as a stepped `:`, which stops short of the end when the step does not
divide the span. A `xstep` that is not a positive finite number raises `ArgumentError` before the
division, and one that leaves fewer than two points raises `ArgumentError` as well.
"""
function output_grid(ansatz::TensorBSplineAnsatz, xstep)
    T = eltype(ansatz.xspan)
    Δx = ansatz.xspan[2] - ansatz.xspan[1]
    step = convert(T, xstep)
    (isfinite(step) && step > zero(T)) ||
        throw(ArgumentError("output_grid needs a positive finite xstep, got $step"))
    n = round(Int, Δx / step) + 1
    n ≥ 2 || throw(ArgumentError("xstep $xstep is too large for xspan $(ansatz.xspan)"))
    return range(ansatz.xspan[1], ansatz.xspan[2]; length = n)
end

function output_grid(ansatz::TensorBSplineAnsatz, problem::LagrangianPDEProblem)
    output_grid(ansatz, xstep(problem))
end

"""
    is_linear_in_coefficients(ansatz)

Whether the ansatz is linear in its coefficients, which is what an `L²` projection can fit.

A tensor B-spline ansatz is; an ansatz whose coefficients enter a nonlinearity is not, and a
[`Projection`](@ref) refuses it at construction.
"""
is_linear_in_coefficients(::TensorBSplineAnsatz) = true

function Base.show(io::IO, ansatz::TensorBSplineAnsatz{T}) where {T}
    print(io,
        "TensorBSplineAnsatz{$T} of degree $(ansatz.degree) on $(ansatz.ncells) cells, ",
        "$(ntime_functions(ansatz)) × $(nspace_functions(ansatz)) functions, ",
        "x ∈ $(ansatz.xspan), $(ansatz.x_boundary)")
end
