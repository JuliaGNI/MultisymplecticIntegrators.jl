"""
    TensorBSplineAnsatz(; degree, ncells, xspan, x_boundary = Dirichlet(), n = nothing)

A tensor B-spline ansatz of one space-time slab, on `SimpleSplines`.

The slab is the rectangle `[0, 1] × xspan` in the normalised time coordinate `t̂` and the physical
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

A grid of `xspan` in `n` equally spaced points, with `n = round(Int, Δx / xstep) + 1`. It is a
public helper that the integrator does not call: the solution is reported on `xgrid(problem)`.

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

# A tensor B-spline ansatz is linear in its coefficients, so a `Projection()` fits it.
is_linear_in_coefficients(::TensorBSplineAnsatz) = true

"""
    check_ansatz(problem, ansatz)

Reject an ansatz that the problem cannot support, when the integrator is built.

A tensor B-spline ansatz must be of the problem's element type and on the problem's space interval.
Each refusal is an `ArgumentError` that names the two values it compared, and the element type is
compared first, so a pair that mismatches in both hears about the type.
"""
function check_ansatz(problem::LagrangianPDEProblem{PT}, a::TensorBSplineAnsatz{T}) where {
        PT, T}
    T === PT || throw(ArgumentError(
        "the ansatz element type $T does not match the problem's $PT"))
    xspan(a) == xspan(problem) || throw(ArgumentError(
        "the ansatz xspan $(xspan(a)) does not match the problem's $(xspan(problem))"))
    return nothing
end

"""
    PointEvaluation{T}

The space basis at a set of points: the points, and the basis functions and their first derivatives
there, a row per point.
"""
struct PointEvaluation{T}
    points::Vector{T}
    value::Matrix{T}
    slope::Matrix{T}
end

function PointEvaluation(xb, points::AbstractVector{T}) where {T}
    nx = nbasis(xb)
    return PointEvaluation{T}(collect(points),
        [evaluate(xb, j, x) for x in points, j in 1:nx],
        [evaluate(xb, j, x, 1) for x in points, j in 1:nx])
end

"""The points of an evaluation, at which the core applies the coupling's transform of the field."""
evaluation_points(e::PointEvaluation) = e.points

"""
    TensorBSplineSlab{ST, T}

The block of a [`TensorBSplineAnsatz`](@ref) in the slab cache: what the ansatz keeps of one slab.

`coeffs` is the coefficient array of the slab, of size `(Nt, Nx, D)`, `residual` its interior weak
form, and `top` the top row of the last solved slab. `tabulation` holds the tables of the slab's
quadrature, which the ansatz owns. `output` and `trace` evaluate the space basis at the output grid
and at the nodes of the space quadrature, and `top_slope` holds the time functions' first derivatives
at the top of the slab.
"""
struct TensorBSplineSlab{ST, T}
    coeffs::Array{ST, 3}
    residual::Array{ST, 3}
    top::Matrix{ST}
    tabulation::SlabTabulation{T}
    output::PointEvaluation{T}
    trace::PointEvaluation{T}
    top_slope::Vector{T}
end

"""
    ansatz_data(ansatz, problem, ST)

The ansatz's block of the slab cache, built once per integrator.
"""
function ansatz_data(a::TensorBSplineAnsatz{T}, problem::LagrangianPDEProblem,
        ::Type{ST}) where {T, ST}
    nt = ntime_functions(a)
    nx = nspace_functions(a)
    coeffs = zeros(ST, nt, nx, components(problem))
    xb = space_basis(a)
    return TensorBSplineSlab{ST, T}(
        coeffs, similar(coeffs), zeros(ST, nx, components(problem)),
        SlabTabulation(quadrature(a)), PointEvaluation(xb, xgrid(problem)),
        PointEvaluation(xb, space_quadrature_nodes(a)),
        [evaluate(time_basis(a), k, one(T), 1) for k in 1:nt])
end

"""
    ansatz_data_type(ansatz, problem, ST)

The type of [`ansatz_data`](@ref)`(ansatz, problem, ST)`, computed without building the block.
"""
function ansatz_data_type(::TensorBSplineAnsatz{T}, ::LagrangianPDEProblem,
        ::Type{ST}) where {T, ST}
    return TensorBSplineSlab{ST, T}
end

"""The normalised time nodes and the space nodes of the slab's quadrature."""
function quadrature_points(s::TensorBSplineSlab)
    return (s.tabulation.time.node, s.tabulation.space.node)
end

"""The evaluation of the space basis at the output grid."""
output_evaluation(s::TensorBSplineSlab) = s.output

"""The evaluation of the space basis at the nodes of the space quadrature."""
trace_evaluation(s::TensorBSplineSlab) = s.trace

"""
    scatter!(coeffs, x, bottom, nf)

Scatter the solver's vector into the coefficient array: the free rows, the last `nf` rows, from
`x`, and the rows below them from `bottom`, the coupling's fixed bottom row, where `nf < Nt`.
"""
function scatter!(coeffs::Array{ST, 3}, x::AbstractVector{ST}, bottom,
        nf::Integer) where {ST}
    nt, nx, D = size(coeffs)
    m = nt - nf
    for d in 1:D
        for j in 1:nx
            m == 0 || (coeffs[1, j, d] = bottom[j, d])
            for a in (m + 1):nt
                coeffs[a, j, d] = x[free_index(a - m, j, d, nf, nx)]
            end
        end
    end
    return nothing
end

"""
    slab_field!(u, v, w, slab, x, bottom, nf, h)

The field of the slab that the solver's vector `x` describes, at the nodes of the slab's quadrature:
the coefficient array from `x` and the coupling's fixed bottom row, and the field and its two
derivatives from it.
"""
function slab_field!(
        u::AbstractArray{ST, 3}, v::AbstractArray{ST, 3}, w::AbstractArray{ST, 3},
        s::TensorBSplineSlab{ST}, x::AbstractVector{ST}, bottom, nf::Integer,
        h) where {ST}
    scatter!(s.coeffs, x, bottom, nf)
    slab_values!(u, v, w, s.coeffs, s.tabulation, h)
    return nothing
end

"""The interior weak form of the density's derivatives `∂u`, `∂v`, `∂w`, into the block."""
function slab_weak_form!(s::TensorBSplineSlab, ∂u, ∂v, ∂w, h)
    interior_weak_form!(s.residual, ∂u, ∂v, ∂w, s.tabulation, h)
    return nothing
end

"""
    free_residual!(b, slab, nf)

The rows of the weak form that the free rows carry, into the solver's layout: the rows of the test
functions `1:nf`, which vanish at the top of the slab.
"""
function free_residual!(b::AbstractVector, s::TensorBSplineSlab, nf::Integer)
    _, nx, D = size(s.residual)
    for d in 1:D, j in 1:nx, k in 1:nf
        b[free_index(k, j, d, nf, nx)] = s.residual[k, j, d]
    end
    return nothing
end

"""Keep the top row of the solved slab, which the coupling and the guess of the next slab read."""
function keep_top!(s::TensorBSplineSlab)
    s.top .= view(s.coeffs, size(s.coeffs, 1), :, :)
    return nothing
end

"""
    top_values!(u, v, w, slab, evaluation, h)

The field and its two derivatives at the top of the slab, `t̂ = 1`, at the points of `evaluation`,
into `u[i, d]`, `v[i, d]` and `w[i, d]`. `∂ₜu` is the time derivative of the normalised coordinate
divided by the slab's `h`, which is the factor the interior weak form carries as well.
"""
function top_values!(u::AbstractMatrix{ST}, v::AbstractMatrix{ST}, w::AbstractMatrix{ST},
        s::TensorBSplineSlab{ST}, e::PointEvaluation, h) where {ST}
    nt, nx, D = size(s.coeffs)
    hh = convert(ST, h)
    fill!(u, zero(ST))
    fill!(v, zero(ST))
    fill!(w, zero(ST))
    for d in 1:D
        for j in 1:nx
            # the time derivative at the top of the slab, per space function: `∂ₜu` is `∂_{t̂}u / h`
            tderiv = zero(ST)
            for k in 1:nt
                tderiv += s.coeffs[k, j, d] * s.top_slope[k]
            end
            tderiv /= hh
            for i in axes(e.value, 1)
                u[i, d] += s.coeffs[nt, j, d] * e.value[i, j]
                w[i, d] += s.coeffs[nt, j, d] * e.slope[i, j]
                v[i, d] += tderiv * e.value[i, j]
            end
        end
    end
    return nothing
end

function Base.show(io::IO, ansatz::TensorBSplineAnsatz{T}) where {T}
    print(io,
        "TensorBSplineAnsatz{$T} of degree $(ansatz.degree) on $(ansatz.ncells) cells, ",
        "$(ntime_functions(ansatz)) × $(nspace_functions(ansatz)) functions, ",
        "x ∈ $(ansatz.xspan), $(ansatz.x_boundary)")
end
