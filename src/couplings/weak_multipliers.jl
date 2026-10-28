"""
    MultiplierBasis

Abstract type of the basis that a weak boundary condition's multiplier is expanded in:
[`BSplineMultiplier`](@ref), the ansatz's own trace basis, or [`LagrangeMultiplier`](@ref), the
nodal basis of the same space. Either way there is one function per spline node of the trace,
thinned by the coupling's `stride`, so the two bases span the same space and differ in their
functions alone.
"""
abstract type MultiplierBasis end

"""The multiplier basis of the ansatz's own trace basis, every `stride`-th function of it."""
struct BSplineMultiplier <: MultiplierBasis end

"""
The multiplier basis of the same space in its nodal functions: the function that is one at a spline
node of the trace basis and zero at the other nodes the thinning keeps, for every `stride`-th node.
"""
struct LagrangeMultiplier <: MultiplierBasis end

function _multiplier_basis(basis::Symbol)
    basis === :bspline && return BSplineMultiplier()
    basis === :lagrange && return LagrangeMultiplier()
    throw(ArgumentError("multiplier_basis must be :bspline or :lagrange, got :$basis"))
end

"""
    MultiplierCoupling <: SlabCoupling

Abstract type of a coupling that imposes the slab's data weakly: the data is a row per multiplier
function, the multiplier is an unknown of the step, and the momentum that leaves a slab is the
multiplier the next slab enters with.

The unknown of the slab is the multiplier at its **top** surface, the momentum the slab hands on; the
multiplier at its bottom surface is the datum, which the previous slab's top multiplier supplies.
"""
abstract type MultiplierCoupling <: SlabCoupling end

"""
    WeakInitialCondition() <: SlabCoupling
    WeakInitialCondition(; multiplier_basis = :bspline) <: SlabCoupling

The initial condition of a slab, imposed weakly.

The bottom row of the coefficient array is free, so all `Nt` rows of the slab are unknowns, and the
slab is closed by one row per multiplier function `ψ_p`,

```math
\\int \\bigl( u(0, x) - u_0(x) \\bigr) \\, \\psi_p(x) \\, dx ,
```

where `u₀` is the problem's initial condition on the first slab and the previous slab's top trace on
every later one. The multiplier at the bottom surface enters the momentum balance as the momentum
that enters the slab: the least-squares fit of `∂L/∂v(u₀, v₀, w₀)` on the first slab, and the
previous slab's top multiplier afterwards.

`multiplier_basis` names the basis of `ψ_p`: `:bspline`, the ansatz's own trace basis, or
`:lagrange`, the nodal basis of that same set of functions.

The coupling imposes no space boundary condition, so the ends of `xspan` are left to the space
basis: the ansatz must take `x_boundary = Dirichlet()`, and every other value throws an
`ArgumentError` that names the condition the coupling needs and the one it got. A `Free()` basis
leaves the ends unconstrained, and the remaining conditions constrain them in a way the interior
weak form does not carry.

```julia
method = SpaceTimeGalerkin(
    TensorBSplineAnsatz(degree = 2, ncells = (2, 20), xspan = (0, 1)),
    WeakInitialCondition())
```
"""
struct WeakInitialCondition{B <: MultiplierBasis} <: MultiplierCoupling
    multiplier_basis::B
end

function WeakInitialCondition(; multiplier_basis::Symbol = :bspline)
    return WeakInitialCondition(_multiplier_basis(multiplier_basis))
end

"""
    WeakInitialAndBoundaryConditions(; stride = 1, multiplier_basis = :bspline) <: SlabCoupling

The initial condition and the space boundary conditions of a slab, imposed weakly.

The initial condition is imposed as [`WeakInitialCondition`](@ref) imposes it. The multipliers at
`x = a` and at `x = b` are unknowns of the step as well, and their rows are

```math
\\int \\bigl( u(t, a) - g_a(t) \\bigr) \\, \\chi_p(t) \\, dt ,
\\qquad
\\int \\bigl( g_b(t) - u(t, b) \\bigr) \\, \\chi_p(t) \\, dt ,
```

where `g_a` and `g_b` are the problem's side traces, evaluated at each node of the time trace, so
that inhomogeneous data is imposed as it stands. The multiplier basis is the ansatz's own trace
basis, every `stride`-th function of it, or the nodal basis of that same set of functions, so that a
slab with `Nt` time functions and `Nx` space functions has `length(1:stride:Nx)` multipliers at the
bottom surface, without its two corner functions at `stride = 1`, and `length(1:stride:Nt)` at each
end of `xspan`.

A `stride` below 1 throws `ArgumentError`. Only `stride = 1` converges: at `stride > 1` the thinned
set of multiplier functions no longer reproduces a constant, and the observed order of the slab
falls with each step of the thinning.

The boundary rows carry an unknown, so the ansatz must take `x_boundary = Free()`; every other value
throws an `ArgumentError` naming the condition the coupling needs and the one it got. A `Dirichlet()`
basis in particular makes the trace vanish at each end of `xspan`, which would leave the rows empty.

```julia
method = SpaceTimeGalerkin(
    TensorBSplineAnsatz(degree = 2, ncells = (2, 20), xspan = (0, 1),
        x_boundary = Free()),
    WeakInitialAndBoundaryConditions())
```
"""
struct WeakInitialAndBoundaryConditions{B <: MultiplierBasis} <: MultiplierCoupling
    stride::Int
    multiplier_basis::B
end

function WeakInitialAndBoundaryConditions(;
        stride::Integer = 1, multiplier_basis::Symbol = :bspline)
    stride ≥ 1 || throw(ArgumentError("stride must be ≥ 1, got $stride"))
    return WeakInitialAndBoundaryConditions(Int(stride), _multiplier_basis(multiplier_basis))
end

"""The spacing of the multiplier functions of a coupling that imposes the initial condition alone."""
multiplier_stride(::WeakInitialCondition) = 1

"""The spacing of the multiplier functions of a coupling that imposes the boundary conditions too."""
multiplier_stride(coupling::WeakInitialAndBoundaryConditions) = coupling.stride

"""Whether the coupling imposes the space boundary conditions, which are the `2 Nμ` multipliers."""
imposes_boundary_conditions(::WeakInitialCondition) = false

imposes_boundary_conditions(::WeakInitialAndBoundaryConditions) = true

"""
    multiplier_counts(coupling, ansatz)

The number of the coupling's multiplier functions: at the bottom surface, `Nλ`, and at each end of
`xspan`, `Nμ`.
"""
function multiplier_counts(coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz)
    return (length(space_multiplier_indices(coupling, ansatz)),
        length(boundary_multiplier_indices(coupling, ansatz)))
end

"""
    space_multiplier_indices(coupling, ansatz)

The functions of the space trace basis that the multiplier of the bottom surface is expanded in.

They are every `stride`-th function of the basis, in either multiplier basis. A coupling that
imposes the space boundary conditions leaves the two corner functions out at `stride = 1`, where
the thinned set is the whole basis and the bottom corners carry an initial row already: a
constraint on the same corner twice would leave the slab rank deficient by two. At `stride ≥ 2`
nothing leaves, because the thinned set is of full rank as it stands.
"""
function space_multiplier_indices(coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz)
    indices = _multiplier_indices(coupling, nspace_functions(ansatz))
    if imposes_boundary_conditions(coupling) && multiplier_stride(coupling) == 1
        return filter(index -> 1 < index < nspace_functions(ansatz), indices)
    end
    return indices
end

"""The functions of the time trace basis that the multipliers of the two ends of `xspan` take."""
function boundary_multiplier_indices(coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz)
    imposes_boundary_conditions(coupling) || return Int[]
    return _multiplier_indices(coupling, ntime_functions(ansatz))
end

# The indices of the trace basis that the multipliers of one surface take, before the corners go.
function _multiplier_indices(coupling::MultiplierCoupling, nbasis::Integer)
    return collect(1:multiplier_stride(coupling):nbasis)
end

free_rows(coupling::MultiplierCoupling, ansatz::SpaceTimeAnsatz) = ntime_functions(ansatz)

"""
    MultiplierData{ST} <: CouplingData

The block of a weak-multiplier coupling in the slab cache.

`initial_multiplier` is the datum of the step, the multiplier of the bottom surface, and
`top_multiplier` its unknown, the multiplier of the top surface; `top_values` is the unknown at the
nodes of the space quadrature, which is the momentum that leaves the slab. `boundary_multiplier`
holds the `2 Nμ` multipliers of the two ends of `xspan` — the first `Nμ` at `x = a`, the rest at
`x = b`.

`multiplier_basis` and `boundary_basis` are the tables those multipliers are expanded in, at the
nodes of the space and of the time quadrature. `projection` reads the bottom multiplier's datum back
into its coefficients, and `boundary_projection` the flux at the two ends of `xspan`; both are
factorised once when the cache is built. `target` and `boundary_target` are the data of the step: the
trace the initial condition is compared with, and the problem's side traces at each node of the time
rule.

`bottom_test` and `top_test` are the test functions' values at the two ends of the time direction,
and `left_space`, `right_space`, `left_slope` and `right_slope` those of the space functions and of
their derivatives at the two ends of `xspan`. The remaining fields are the scratch of the residual,
of the guess and of the boundary data, so that none of them allocates.
"""
struct MultiplierData{ST} <: CouplingData
    initial_multiplier::Matrix{ST}
    top_multiplier::Matrix{ST}
    guess_multiplier::Matrix{ST}
    top_values::Matrix{ST}
    boundary_multiplier::Matrix{ST}
    multiplier_basis::Matrix{ST}
    boundary_basis::Matrix{ST}
    target::Matrix{ST}
    boundary_target::Matrix{ST}
    projection::Matrix{ST}
    boundary_projection::Matrix{ST}
    bottom_test::Vector{ST}
    top_test::Vector{ST}
    left_space::Vector{ST}
    right_space::Vector{ST}
    left_slope::Vector{ST}
    right_slope::Vector{ST}
    top_u::Matrix{ST}
    top_v::Matrix{ST}
    top_w::Matrix{ST}
    momentum::Matrix{ST}
    left_coefficients::Matrix{ST}
    right_coefficients::Matrix{ST}
    left_slopes::Matrix{ST}
    right_slopes::Matrix{ST}
    space_row::Vector{ST}
    time_row::Vector{ST}
    space_work::Vector{ST}
    time_work::Vector{ST}
end

function coupling_data(coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz{T},
        problem::LagrangianPDEProblem, ::Type{ST}) where {T, ST}
    nt = ntime_functions(ansatz)
    nx = nspace_functions(ansatz)
    D = components(problem)
    tb = time_basis(ansatz)
    xb = space_basis(ansatz)
    rules = quadratures(quadrature(ansatz))
    tnodes = quadrature_nodes(rules[1])
    xnodes = quadrature_nodes(rules[2])
    tweights = quadrature_weights(rules[1])
    xweights = quadrature_weights(rules[2])
    basis = coupling.multiplier_basis
    nλ, nμ = multiplier_counts(coupling, ansatz)
    qt = length(tnodes)
    qx = length(xnodes)
    # `top_u`, `top_v` and `top_w` hold a trace at the nodes of the space rule in one call and at the
    # nodes of the time rule in another, so they take the longer of the two rules.
    nq = max(qx, qt)
    λbasis = multiplier_table(
        basis, xb, xnodes, space_multiplier_indices(coupling, ansatz), ST)
    μbasis = multiplier_table(
        basis, tb, tnodes, boundary_multiplier_indices(coupling, ansatz),
        ST)
    # The datum of the bottom multiplier is the quadrature-weighted L² projection wherever the λ
    # basis has lost its corner functions, whose thinned set can no longer read the momentum; the
    # coupling that keeps the whole trace basis keeps the nodal fit its family's entry carries.
    λprojection = imposes_boundary_conditions(coupling) ? least_squares(λbasis, xweights) :
                  nodal_fit(λbasis)
    xs = xspan(ansatz)
    return MultiplierData{ST}(
        zeros(ST, nλ, D),
        zeros(ST, nλ, D),
        zeros(ST, nλ, D),
        zeros(ST, qx, D),
        zeros(ST, 2 * nμ, D),
        λbasis,
        μbasis,
        zeros(ST, qx, D),
        zeros(ST, 2 * qt, D),
        λprojection,
        least_squares(μbasis, tweights),
        ST[evaluate(tb, a, zero(T)) for a in 1:nt],
        ST[evaluate(tb, a, one(T)) for a in 1:nt],
        ST[evaluate(xb, j, xs[1]) for j in 1:nx],
        ST[evaluate(xb, j, xs[2]) for j in 1:nx],
        ST[evaluate(xb, j, xs[1], 1) for j in 1:nx],
        ST[evaluate(xb, j, xs[2], 1) for j in 1:nx],
        zeros(ST, nq, D),
        zeros(ST, nq, D),
        zeros(ST, nq, D),
        zeros(ST, nq, D),
        zeros(ST, nt, D),
        zeros(ST, nt, D),
        zeros(ST, nt, D),
        zeros(ST, nt, D),
        zeros(ST, nx),
        zeros(ST, nt),
        zeros(ST, qx),
        zeros(ST, qt))
end

function coupling_data_type(
        ::MultiplierCoupling, ::TensorBSplineAnsatz, ::LagrangianPDEProblem,
        ::Type{ST}) where {ST}
    return MultiplierData{ST}
end

free_rows(::MultiplierData, ansatz::SpaceTimeAnsatz) = ntime_functions(ansatz)

function coupling_unknowns(coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem)
    nλ, nμ = multiplier_counts(coupling, ansatz)
    return (nλ + 2 * nμ) * components(problem)
end

function coupling_unknowns(data::MultiplierData)
    length(data.initial_multiplier) +
    length(data.boundary_multiplier)
end

"""
    multiplier_table(basis, trace, points, indices, ST)

The multiplier functions of `indices` evaluated at the quadrature `points`, one row per function.

A `BSplineMultiplier` takes the trace basis functions themselves. A `LagrangeMultiplier` takes the
function of that same space that is one at the spline node `nodes(trace)[index]` and zero at the
other nodes `indices` keeps, written over the B-spline functions of `indices`: the two tables span
one space and differ in their functions.
"""
function multiplier_table(::BSplineMultiplier, trace, points::AbstractVector,
        indices::AbstractVector{<:Integer}, ::Type{ST}) where {ST}
    return ST[evaluate(trace, indices[p], points[r])
              for p in eachindex(indices),
    r in eachindex(points)]
end

function multiplier_table(::LagrangeMultiplier, trace, points::AbstractVector,
        indices::AbstractVector{<:Integer}, ::Type{ST}) where {ST}
    ξ = nodes(trace)
    n = length(indices)
    values = ST[evaluate(trace, indices[j], ξ[indices[i]]) for i in 1:n, j in 1:n]
    # the nodal functions over the multiplier space's own functions: `coefficients * valuesᵀ = I`
    return Matrix(transpose(values \ Matrix{ST}(I, n, n))) *
           multiplier_table(BSplineMultiplier(), trace, points, indices, ST)
end

"""
    least_squares(basis, weights)

The operator that reads a trace given at the nodes back into the coefficients of `basis`, by the
quadrature-weighted `L²` projection `(B W Bᵀ)⁻¹ (B W)` of `m`: the coefficients `c` that minimise
`Σᵣ ωᵣ (Σₚ cₚ Bₚᵣ - mᵣ)²`, with `W` the rule's weights, so that `∫ c(x) ψ_p(x) dx = ∫ m(x) ψ_p(x) dx`
for every function of `basis`.

The mass matrix is factored once, here, and the operator is reused by every step. An unweighted
nodal fit of the same values is not this projection: a basis that no longer reproduces the trace —
a thinned one, or one whose corner functions have left — reads the momentum of the datum wrongly,
because the fit minimises the residual at the nodes alone.
"""
function least_squares(basis::AbstractMatrix{ST}, weights) where {ST}
    m = size(basis, 1)
    n = size(basis, 2)
    m == 0 && return zeros(ST, 0, n)
    mass = zeros(ST, m, m)
    for q in 1:m, p in 1:m

        s = zero(ST)
        for r in 1:n
            s += basis[p, r] * convert(ST, weights[r]) * basis[q, r]
        end
        mass[p, q] = s
    end
    load = zeros(ST, m, n)
    for p in 1:m, r in 1:n

        load[p, r] = basis[p, r] * convert(ST, weights[r])
    end
    return mass \ load
end

"""
    nodal_fit(basis)

The operator that reads a trace given at the nodes back into the coefficients of `basis` by the
unweighted least-squares fit `(B Bᵀ)⁻¹ B`.

It is the datum of the family that keeps the whole trace basis: that basis reproduces the momentum
at the nodes, which is what the §A entry of the Dirichlet coupling was generated with. A basis whose
corner functions have left does not, and takes [`least_squares`](@ref) instead.
"""
function nodal_fit(basis::AbstractMatrix{ST}) where {ST}
    n = size(basis, 2)
    m = size(basis, 1)
    m == 0 && return zeros(ST, 0, n)
    return Matrix(adjoint(basis) \ Matrix{ST}(I, n, n))
end

# The load of a trace on a basis: `∫ g(x) ψ_j(x) dx` for every function `j` of the axis.
function spline_load!(load::AbstractVector{ST}, coefficients::AbstractMatrix{ST},
        basis::AbstractMatrix{ST}, axis, d::Integer) where {ST}
    fill!(load, zero(ST))
    for r in axes(basis, 2)
        value = zero(ST)
        for p in axes(basis, 1)
            value += coefficients[p, d] * basis[p, r]
        end
        wz = axis.weight[r] * value
        for l in axes(axis.index, 2)
            j = axis.index[r, l]
            j == 0 && continue
            load[j] += wz * axis.value[r, l]
        end
    end
    return nothing
end

# The load of a multiplier trace on the time basis: `h ∫ g(t̂) φ_a(t̂) dt̂` for every `a`, from the
# multiplier coefficients `p₀ + 1 : p₀ + N` of component `d`.
function time_load!(load::AbstractVector{ST}, coefficients::AbstractMatrix{ST},
        basis::AbstractMatrix{ST}, axis, h::ST, d::Integer, p₀::Integer) where {ST}
    fill!(load, zero(ST))
    for r in axes(basis, 2)
        value = zero(ST)
        for p in axes(basis, 1)
            value += coefficients[p₀ + p, d] * basis[p, r]
        end
        wz = h * axis.weight[r] * value
        for l in axes(axis.index, 2)
            a = axis.index[r, l]
            a == 0 && continue
            load[a] += wz * axis.value[r, l]
        end
    end
    return nothing
end

# Add a load on the space basis to the rows of the free coefficients, weighted by the test
# functions' values at an end of the time direction.
function add_space_load!(b::AbstractVector{ST}, load::AbstractVector{ST},
        tests::AbstractVector{ST}, nf::Integer, nx::Integer, d::Integer, sign::ST) where {ST}
    for a in 1:nf
        s = sign * tests[a]
        s == zero(ST) && continue
        for j in 1:nx
            b[free_index(a, j, d, nf, nx)] += s * load[j]
        end
    end
    return nothing
end

# Add a load on the time basis to the rows of the free coefficients, weighted by the space
# functions' values at an end of `xspan`.
function add_edge_load!(b::AbstractVector{ST}, load::AbstractVector{ST},
        edge::AbstractVector{ST}, nf::Integer, nx::Integer, d::Integer, sign::ST) where {ST}
    for a in 1:nf
        s = sign * load[a]
        s == zero(ST) && continue
        for j in 1:nx
            value = edge[j]
            value == zero(ST) && continue
            b[free_index(a, j, d, nf, nx)] += s * value
        end
    end
    return nothing
end

"""
    coupling_term!(b, x, coupling, data::MultiplierData, cache, ansatz, problem)

The coupling's part of one slab's residual.

The multiplier of the bottom surface enters the momentum balance of the slab with the sign of an
incoming momentum, and the multiplier of the top surface with the sign of an outgoing one, at the
two test functions that do not vanish at those surfaces; the multipliers of the two ends of `xspan`
enter the same way, weighted by the space functions there. After the rows of the free coefficients
comes one row per multiplier function, the boundary data of that function.
"""
function coupling_term!(
        b::AbstractVector{ST}, x::AbstractVector{ST}, coupling::MultiplierCoupling,
        data::MultiplierData{ST}, c::SpaceTimeGalerkinCache{ST}, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem) where {ST}
    nx, D = size(c.bottom)
    nf = ntime_functions(ansatz)
    nλ = size(data.initial_multiplier, 1)
    nμ = size(data.boundary_basis, 1)
    tq = c.tabulation.time
    xq = c.tabulation.space
    h = convert(ST, timestep(problem))
    coupling_unknowns!(data, x, D)
    for d in 1:D
        spline_load!(data.space_row, data.initial_multiplier, data.multiplier_basis, xq, d)
        add_space_load!(b, data.space_row, data.bottom_test, nf, nx, d, one(ST))
        spline_load!(data.space_row, data.top_multiplier, data.multiplier_basis, xq, d)
        add_space_load!(b, data.space_row, data.top_test, nf, nx, d, -one(ST))
    end
    if nμ > 0
        for d in 1:D
            time_load!(
                data.time_row, data.boundary_multiplier, data.boundary_basis, tq, h, d, 0)
            add_edge_load!(b, data.time_row, data.left_space, nf, nx, d, one(ST))
            time_load!(
                data.time_row, data.boundary_multiplier, data.boundary_basis, tq, h, d, nμ)
            add_edge_load!(b, data.time_row, data.right_space, nf, nx, d, -one(ST))
        end
    end
    multiplier_constraints!(b, data, c, nx, D, nf, nλ, nμ, h)
    return nothing
end

"""
    multiplier_constraints!(b, data, cache, nx, D, nf, Nλ, Nμ, h)

The coupling's own rows, one per multiplier function: the boundary data projected on that function.

The initial condition is imposed on the bottom trace of the slab, and the space boundary
conditions on its two ends, over the nodes of the space and of the time quadrature.
"""
function multiplier_constraints!(b::AbstractVector{ST}, data::MultiplierData{ST},
        c::SpaceTimeGalerkinCache{ST}, nx::Integer, D::Integer, nf::Integer,
        nλ::Integer, nμ::Integer, h::ST) where {ST}
    xq = c.tabulation.space
    tq = c.tabulation.time
    qt = nquadrature(tq)
    nt = size(data.left_coefficients, 1)
    base = nf * nx * D
    for d in 1:D
        for r in axes(data.target, 1)
            value = zero(ST)
            for l in axes(xq.index, 2)
                j = xq.index[r, l]
                j == 0 && continue
                value += c.coeffs[1, j, d] * xq.value[r, l]
            end
            data.space_work[r] = xq.weight[r] * (value - data.target[r, d])
        end
        for p in 1:nλ
            s = zero(ST)
            for r in axes(data.multiplier_basis, 2)
                s += data.multiplier_basis[p, r] * data.space_work[r]
            end
            b[base + (d - 1) * nλ + p] = -s
        end
    end
    nμ == 0 && return nothing
    end_coefficients!(data, c.coeffs, nx, nt, D)
    for d in 1:D
        for edge in 1:2
            coefficients = edge == 1 ? data.left_coefficients : data.right_coefficients
            for rt in 1:qt
                value = zero(ST)
                for l in axes(tq.index, 2)
                    a = tq.index[rt, l]
                    a == 0 && continue
                    value += coefficients[a, d] * tq.value[rt, l]
                end
                data.time_work[rt] = h * tq.weight[rt] *
                                     (value - data.boundary_target[(edge - 1) * qt + rt, d])
            end
            for p in 1:nμ
                s = zero(ST)
                for rt in 1:qt
                    s += data.boundary_basis[p, rt] * data.time_work[rt]
                end
                b[base + D * nλ + (edge - 1) * D * nμ + (d - 1) * nμ + p] = -s
            end
        end
    end
    return nothing
end

# The coupling's unknowns, read out of the solver's vector: they sit behind the coefficient block.
function coupling_unknowns!(data::MultiplierData{ST}, x::AbstractVector{ST},
        D::Integer) where {ST}
    nλ = size(data.initial_multiplier, 1)
    nμ = size(data.boundary_basis, 1)
    base = length(x) - (nλ + 2 * nμ) * D
    for d in 1:D, p in 1:nλ

        data.top_multiplier[p, d] = x[base + (d - 1) * nλ + p]
    end
    for d in 1:D, p in 1:nμ

        data.boundary_multiplier[p, d] = x[base + D * nλ + (d - 1) * nμ + p]
        data.boundary_multiplier[nμ + p, d] = x[base + D * nλ + D * nμ + (d - 1) * nμ + p]
    end
    return nothing
end

"""
    coupling_update!(data, x, cache, ansatz, problem)

Read the coupling's unknowns out of the solved vector, and their values at the nodes of the space
quadrature: the momentum that leaves the slab.
"""
function coupling_update!(data::MultiplierData{ST}, x::AbstractVector{ST},
        c::SpaceTimeGalerkinCache{ST}, ansatz::SpaceTimeAnsatz,
        problem::LagrangianPDEProblem) where {ST}
    D = size(data.top_multiplier, 2)
    coupling_unknowns!(data, x, D)
    for d in 1:D, r in axes(data.multiplier_basis, 2)

        value = zero(ST)
        for p in axes(data.multiplier_basis, 1)
            value += data.top_multiplier[p, d] * data.multiplier_basis[p, r]
        end
        data.top_values[r, d] = value
    end
    return nothing
end

# The projected coefficients of a trace at the nodes, written into one column of a block.
function project!(coefficients::AbstractMatrix{ST}, first_row::Integer, column::Integer,
        projection::AbstractMatrix{ST}, values::AbstractVector{ST}) where {ST}
    for p in axes(projection, 1)
        s = zero(ST)
        for r in axes(projection, 2)
            s += projection[p, r] * values[r]
        end
        coefficients[first_row + p - 1, column] = s
    end
    return nothing
end

# The coefficients of the traces at the two ends of `xspan`, over the time basis, and of their
# space derivatives there.
function end_coefficients!(data::MultiplierData{ST}, coeffs::Array{ST, 3},
        nx::Integer, nt::Integer, D::Integer) where {ST}
    for d in 1:D, a in 1:nt

        value = zero(ST)
        right = zero(ST)
        slope = zero(ST)
        right_slope = zero(ST)
        for j in 1:nx
            c = coeffs[a, j, d]
            value += c * data.left_space[j]
            right += c * data.right_space[j]
            slope += c * data.left_slope[j]
            right_slope += c * data.right_slope[j]
        end
        data.left_coefficients[a, d] = value
        data.right_coefficients[a, d] = right
        data.left_slopes[a, d] = slope
        data.right_slopes[a, d] = right_slope
    end
    return nothing
end

# The trace of the guessed coefficients at one end of `xspan`, over the nodes of the time rule, in
# `top_u`, `top_v` and `top_w`, and its momentum `∂L/∂w` there.
function edge_momentum!(data::MultiplierData{ST}, c::SpaceTimeGalerkinCache{ST},
        problem::LagrangianPDEProblem, edge::Integer, h) where {ST}
    tq = c.tabulation.time
    qt = nquadrature(tq)
    D = size(data.top_u, 2)
    hh = convert(ST, h)
    values = edge == 1 ? data.left_coefficients : data.right_coefficients
    slopes = edge == 1 ? data.left_slopes : data.right_slopes
    fill!(data.top_u, zero(ST))
    fill!(data.top_v, zero(ST))
    fill!(data.top_w, zero(ST))
    for d in 1:D
        for rt in 1:qt
            value = zero(ST)
            velocity = zero(ST)
            gradient = zero(ST)
            for l in axes(tq.index, 2)
                a = tq.index[rt, l]
                a == 0 && continue
                value += values[a, d] * tq.value[rt, l]
                velocity += values[a, d] * tq.slope[rt, l]
                gradient += slopes[a, d] * tq.value[rt, l]
            end
            data.top_u[rt, d] = value
            data.top_v[rt, d] = velocity / hh
            data.top_w[rt, d] = gradient
        end
    end
    momentum_load!(data.momentum, problem.density, data.top_u, data.top_v, data.top_w,
        parameters(problem))
    for d in 1:D, rt in 1:qt

        data.time_work[rt] = data.momentum[rt, d]
    end
    return nothing
end

"""
    coupling_guess!(data, cache, ansatz, problem, sol)

The coupling's initial guess of its unknowns, once per slab.

The top multiplier starts at the momentum `∂L/∂v` of the guessed top trace, and the multipliers of
the two ends of `xspan` at the flux `∂L/∂w` of the guessed traces there, each read into the
multiplier basis by the projection the cache factorised. On the first slab the multiplier of the
bottom surface — the momentum the slab enters with — starts at the momentum of the problem's initial
data, `∂L/∂v(u₀, v₀, w₀)`, which is a datum of the step and not a guess of the solve.
"""
function coupling_guess!(data::MultiplierData{ST}, c::SpaceTimeGalerkinCache{ST},
        ansatz::TensorBSplineAnsatz, problem::LagrangianPDEProblem, sol) where {ST}
    nt = ntime_functions(ansatz)
    nx, D = size(c.bottom)
    nf = ntime_functions(ansatz)
    nλ = size(data.initial_multiplier, 1)
    nμ = size(data.boundary_basis, 1)
    h = timestep(problem)
    base = nf * nx * D
    scatter!(c.coeffs, c.x, c.bottom, nt, nf)
    trace!(data.top_u, data.top_v, data.top_w, c, ansatz, h)
    momentum_load!(data.momentum, problem.density, data.top_u, data.top_v, data.top_w,
        parameters(problem))
    for d in 1:D
        for r in axes(data.space_work, 1)
            data.space_work[r] = data.momentum[r, d]
        end
        project!(data.guess_multiplier, 1, d, data.projection, data.space_work)
        for p in 1:nλ
            c.x[base + (d - 1) * nλ + p] = data.guess_multiplier[p, d]
        end
    end
    if nμ > 0
        end_coefficients!(data, c.coeffs, nx, nt, D)
        for edge in 1:2
            edge_momentum!(data, c, problem, edge, h)
            for d in 1:D
                project!(data.boundary_multiplier, (edge - 1) * nμ + 1, d,
                    data.boundary_projection, data.time_work)
                for p in 1:nμ
                    c.x[base + D * nλ + (edge - 1) * D * nμ + (d - 1) * nμ + p] = data.boundary_multiplier[
                        (edge - 1) * nμ + p, d]
                end
            end
        end
    end
    c.steps == 0 && initial_multiplier!(data, problem, ansatz)
    return nothing
end

# The momentum of the problem's initial data, read into the multiplier basis: the datum of the first
# slab's bottom surface.
function initial_multiplier!(data::MultiplierData{ST}, problem::LagrangianPDEProblem,
        ansatz::TensorBSplineAnsatz) where {ST}
    nodes = space_quadrature_nodes(ansatz)
    values = problem.ics_function(collect(nodes), parameters(problem))
    qx = length(nodes)
    D = size(data.top_u, 2)
    fill!(data.top_u, zero(ST))
    fill!(data.top_v, zero(ST))
    fill!(data.top_w, zero(ST))
    for d in 1:D, r in 1:qx

        data.top_u[r, d] = convert(ST, values.u[(d - 1) * qx + r])
        data.top_v[r, d] = convert(ST, values.v[(d - 1) * qx + r])
        data.top_w[r, d] = convert(ST, values.w[(d - 1) * qx + r])
    end
    momentum_load!(data.momentum, problem.density, data.top_u, data.top_v, data.top_w,
        parameters(problem))
    for d in 1:D
        for r in 1:qx
            data.space_work[r] = data.momentum[r, d]
        end
        project!(data.initial_multiplier, 1, d, data.projection, data.space_work)
    end
    return nothing
end

"""
    slab_bottom!(cache, coupling, problem, ansatz, sol)

Fix the data of the slab: the trace the initial condition is compared with — the problem's initial
condition on the first slab, the previous slab's top trace afterwards — the multiplier the slab
enters with, and the problem's side traces at each node of the time rule.

The multiplier of the bottom surface is a datum of the step: the momentum the previous slab handed
out, which the cache keeps as that slab's top multiplier. The multiplier of the top surface is the
step's unknown, and the initial guess of it is no datum.
"""
function slab_bottom!(c::SpaceTimeGalerkinCache{ST, T, <:MultiplierData},
        coupling::MultiplierCoupling, problem::LagrangianPDEProblem,
        ansatz::TensorBSplineAnsatz, sol) where {ST, T}
    data = c.coupling
    if c.steps == 0
        data.target .= c.initial_sample
    else
        data.target .= c.trace_u
        copyto!(data.initial_multiplier, data.top_multiplier)
    end
    boundary_data!(data, problem, c, sol)
    return nothing
end

# The problem's side traces at every node of the time rule: `g_a` in the first `Qt` rows of the
# target and `g_b` in the rest.
function boundary_data!(data::MultiplierData{ST}, problem::LagrangianPDEProblem,
        c::SpaceTimeGalerkinCache{ST}, sol) where {ST}
    nμ = size(data.boundary_basis, 1)
    nμ == 0 && return nothing
    tq = c.tabulation.time
    qt = nquadrature(tq)
    D = size(data.boundary_target, 2)
    h = timestep(problem)
    t₀ = first(sol.t) - h
    for rt in 1:qt
        t = t₀ + h * tq.node[rt]
        values = problem.bcs_function((t, t), parameters(problem))
        for d in 1:D
            data.boundary_target[rt, d] = convert(ST, values.left.u[2 * (d - 1) + 1])
            data.boundary_target[qt + rt, d] = convert(ST, values.right.u[2 * (d - 1) + 1])
        end
    end
    return nothing
end

"""
    check_coupling(problem, coupling::WeakInitialCondition, ansatz)

Refuse every space basis but a `Dirichlet()` one, naming the condition the coupling needs and the
one the ansatz has: the coupling imposes the initial condition alone, so the basis itself must
constrain the two ends of `xspan`. A `Free()` basis leaves them unconstrained, and every other
condition constrains them in a way that the interior weak form does not carry.
"""
function check_coupling(::LagrangianPDEProblem, coupling::WeakInitialCondition,
        ansatz::TensorBSplineAnsatz)
    b = x_boundary(ansatz)
    b isa Dirichlet && return nothing
    throw(ArgumentError("$(nameof(typeof(coupling))) needs a space basis with x_boundary = " *
                        "Dirichlet(); this ansatz has x_boundary = $(nameof(typeof(b)))"))
end

"""
    check_coupling(problem, coupling::WeakInitialAndBoundaryConditions, ansatz)

Refuse every space basis but a `Free()` one, naming the condition the coupling needs and the one the
ansatz has: a `Dirichlet()` basis vanishes at the two ends of `xspan`, so the boundary rows of the
coupling would carry no unknown, and every other condition constrains the ends in a way the weak
rows do not undo.
"""
function check_coupling(::LagrangianPDEProblem, coupling::WeakInitialAndBoundaryConditions,
        ansatz::TensorBSplineAnsatz)
    b = x_boundary(ansatz)
    b isa Free && return nothing
    throw(ArgumentError("$(nameof(typeof(coupling))) imposes the space boundary conditions " *
                        "itself, so it needs a space basis with x_boundary = Free(); this " *
                        "ansatz has x_boundary = $(nameof(typeof(b)))"))
end

"""
    coupling_internal(coupling, ansatz, problem)

The coupling's own state of a step, in the solution step's `internal` named tuple: the multiplier of
the step's top surface.

The ecosystem allocates the named tuple once per solution step and fills it after every step, so the
`top_multiplier` entry holds the coefficients of the last solved slab's top multiplier in the
multiplier basis, which is the momentum the next slab enters with.
"""
function coupling_internal(coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem)
    ST = datatype(problem)
    nλ, _ = multiplier_counts(coupling, ansatz)
    return (top_multiplier = zeros(ST, nλ, components(problem)),)
end

"""
    copy_internal_variables!(sol, cache)

Fill the solution step's internal state with the coupling's own: the multiplier of the last solved
slab's top surface, which is the momentum the slab hands on.
"""
function copy_internal_variables!(sol::SolutionStep,
        c::SpaceTimeGalerkinCache{ST, T, <:MultiplierData}) where {ST, T}
    copyto!(internal(sol).top_multiplier, c.coupling.top_multiplier)
    return nothing
end

"""
    copy_coupling_data!(cache_data, data)

Copy the multiplier the slab enters with from the cache of the problem's element type into the
cache the solver works in.
"""
function copy_coupling_data!(c::MultiplierData, data::MultiplierData)
    c.initial_multiplier .= data.initial_multiplier
    c.target .= data.target
    c.boundary_target .= data.boundary_target
    return nothing
end
