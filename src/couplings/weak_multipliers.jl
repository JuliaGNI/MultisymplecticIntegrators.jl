"""
    MultiplierBasis

Abstract type of the basis that a weak boundary condition's multiplier is expanded in:
[`BSplineMultiplier`](@ref), the ansatz's own trace basis, or [`LagrangeMultiplier`](@ref), the
nodal basis of the trace's quadrature rule.
"""
abstract type MultiplierBasis end

"""The multiplier basis of the ansatz's own trace basis, every `stride`-th function of it."""
struct BSplineMultiplier <: MultiplierBasis end

"""The nodal multiplier basis of the trace's quadrature rule, one function per node."""
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

`multiplier_basis` names the basis of `ψ_p`: `:bspline`, the ansatz's own trace basis, every
`stride`-th function of it, or `:lagrange`, the nodal basis of the space quadrature's nodes.

The coupling leaves the space boundary unconstrained, so an ansatz whose `x_boundary` is `Free()`
has no boundary condition to impose and building the integrator throws `ArgumentError`.

```julia
method = SpaceTimeGalerkin(
    TensorBSplineAnsatz(degree = 2, ncells = (2, 20), xspan = (0.0, 1.0)),
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
basis, every `stride`-th function of it, or the nodal basis of the trace's quadrature rule, so that
a slab with `Nt` time functions and `Nx` space functions has `length(1:stride:Nx)` multipliers at
the bottom surface and `length(1:stride:Nt)` at each end of `xspan`.

A `stride` below 1 throws `ArgumentError`. The boundary rows carry an unknown, so an ansatz whose
`x_boundary` is `Dirichlet()` — its trace vanishes at each end of `xspan` — would leave them empty;
building the integrator with one throws `ArgumentError`.

```julia
method = SpaceTimeGalerkin(
    TensorBSplineAnsatz(degree = 2, ncells = (2, 20), xspan = (0.0, 1.0),
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
stride(::WeakInitialCondition) = 1

"""The spacing of the multiplier functions of a coupling that imposes the boundary conditions too."""
stride(coupling::WeakInitialAndBoundaryConditions) = coupling.stride

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

They are every `stride`-th function of the basis, or one nodal function per node of the space rule.
A coupling that imposes the space boundary conditions leaves the two corner functions out: the space
boundary rows already carry an unknown at each end, so a constraint on the same corner twice would
leave the slab rank deficient by two.
"""
function space_multiplier_indices(coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz)
    indices = _multiplier_indices(coupling, ansatz, nspace_functions(ansatz), 2)
    imposes_boundary_conditions(coupling) || return indices
    return filter(index -> 1 < index < nspace_functions(ansatz), indices)
end

"""The functions of the time trace basis that the multipliers of the two ends of `xspan` take."""
function boundary_multiplier_indices(coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz)
    imposes_boundary_conditions(coupling) || return Int[]
    return _multiplier_indices(coupling, ansatz, ntime_functions(ansatz), 1)
end

# The indices of the trace basis that the multipliers of one surface take, before the corners go.
function _multiplier_indices(
        coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz, nbasis::Integer,
        axis::Integer)
    _multiplier_indices(coupling.multiplier_basis, stride(coupling), nbasis,
        length(quadrature_nodes(quadratures(quadrature(ansatz))[axis])))
end

function _multiplier_indices(::BSplineMultiplier, s::Integer, nbasis::Integer, nnodes::Integer)
    collect(1:s:nbasis)
end

function _multiplier_indices(::LagrangeMultiplier, s::Integer, nbasis::Integer, nnodes::Integer)
    collect(1:nnodes)
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
nodes of the space and of the time quadrature, and `projection` and `boundary_projection` are the
least-squares operators that read a trace at those nodes back into their coefficients, computed once
when the cache is built. `target` and `boundary_target` are the data of the step: the trace the
initial condition is compared with, and the problem's side traces at each node of the time rule.

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
    boundary_momentum::Matrix{ST}
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
    s = stride(coupling)
    basis = coupling.multiplier_basis
    nλ, nμ = multiplier_counts(coupling, ansatz)
    qt = length(tnodes)
    qx = length(xnodes)
    λbasis = multiplier_table(
        basis, xb, xnodes, space_multiplier_indices(coupling, ansatz), ST)
    μbasis = multiplier_table(
        basis, tb, tnodes, boundary_multiplier_indices(coupling, ansatz),
        ST)
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
        least_squares(λbasis),
        least_squares(μbasis),
        ST[evaluate(tb, a, zero(T)) for a in 1:nt],
        ST[evaluate(tb, a, one(T)) for a in 1:nt],
        ST[evaluate(xb, j, xs[1]) for j in 1:nx],
        ST[evaluate(xb, j, xs[2]) for j in 1:nx],
        ST[evaluate(xb, j, xs[1], 1) for j in 1:nx],
        ST[evaluate(xb, j, xs[2], 1) for j in 1:nx],
        zeros(ST, qx, D),
        zeros(ST, qx, D),
        zeros(ST, qx, D),
        zeros(ST, qx, D),
        zeros(ST, qt, D),
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
    multiplier_table(basis, trace, nodes, indices, ST)

The multiplier functions of `indices` evaluated at `nodes`, one row per function.

A `BSplineMultiplier` takes the trace basis functions themselves; a `LagrangeMultiplier` takes the
nodal function of each node of `indices`, whose table is the identity on those nodes.
"""
function multiplier_table(::BSplineMultiplier, trace, nodes::AbstractVector,
        indices::AbstractVector{<:Integer}, ::Type{ST}) where {ST}
    return ST[evaluate(trace, indices[p], nodes[r])
              for p in eachindex(indices),
    r in eachindex(nodes)]
end

function multiplier_table(::LagrangeMultiplier, trace, nodes::AbstractVector,
        indices::AbstractVector{<:Integer}, ::Type{ST}) where {ST}
    return ST[(indices[p] == r ? one(ST) : zero(ST))
              for p in eachindex(indices),
    r in eachindex(nodes)]
end

"""
    least_squares(basis)

The operator that reads a trace given at the nodes back into the coefficients of `basis`, by the
least-squares solve it factors once; `coefficients = atable * values`.
"""
function least_squares(basis::AbstractMatrix{ST}) where {ST}
    n = size(basis, 2)
    m = size(basis, 1)
    m == 0 && return zeros(ST, 0, n)
    identity = zeros(ST, n, n)
    for i in 1:n
        identity[i, i] = one(ST)
    end
    return Matrix(adjoint(basis) \ identity)
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
    coupling_term!(b, x, data, cache, ansatz, problem)

The coupling's part of one slab's residual.

The multiplier of the bottom surface enters the momentum balance of the slab with the sign of an
incoming momentum, and the multiplier of the top surface with the sign of an outgoing one, at the
two test functions that do not vanish at those surfaces; the multipliers of the two ends of `xspan`
enter the same way, weighted by the space functions there. After the rows of the free coefficients
comes one row per multiplier function, the boundary data of that function.
"""
function coupling_term!(
        b::AbstractVector{ST}, x::AbstractVector{ST}, data::MultiplierData{ST},
        c::SpaceTimeGalerkinCache{ST}, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem) where {ST}
    nt = ntime_functions(ansatz)
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
    for d in 1:D, a in 1:nt

        l = zero(ST)
        r = zero(ST)
        for j in 1:nx
            l += c.coeffs[a, j, d] * data.left_space[j]
            r += c.coeffs[a, j, d] * data.right_space[j]
        end
        data.left_coefficients[a, d] = l
        data.right_coefficients[a, d] = r
    end
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

# The least-squares coefficients of a trace at the nodes, written into one column of a block.
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

# The top trace of the guessed coefficients, at the nodes of the space quadrature.
function top_trace!(data::MultiplierData{ST}, c::SpaceTimeGalerkinCache{ST},
        ansatz::TensorBSplineAnsatz, h) where {ST}
    e = c.evaluation
    nt = ntime_functions(ansatz)
    nx, D = size(c.bottom)
    hh = convert(ST, h)
    fill!(data.top_u, zero(ST))
    fill!(data.top_v, zero(ST))
    fill!(data.top_w, zero(ST))
    for d in 1:D
        for j in 1:nx
            # the time derivative at the top of the slab, per space function: `∂ₜu` is `∂_{t̂}u / h`
            tderiv = zero(ST)
            for k in 1:nt
                tderiv += c.coeffs[k, j, d] * e.top_slope[k]
            end
            tderiv /= hh
            for i in axes(data.top_u, 1)
                data.top_u[i, d] += c.coeffs[nt, j, d] * e.trace[i, j]
                data.top_w[i, d] += c.coeffs[nt, j, d] * e.trace_slope[i, j]
                data.top_v[i, d] += tderiv * e.trace[i, j]
            end
        end
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
    top_trace!(data, c, ansatz, h)
    momentum_load!(data.momentum, problem.density, data.top_u, data.top_v, data.top_w,
        parameters(problem))
    for d in 1:D
        for r in axes(data.top_u, 1)
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

Refuse a `Free()` space basis: the coupling imposes the initial condition alone, and a basis that
leaves the ends of `xspan` unconstrained has no boundary condition for the interior weak form to
fall back on.
"""
function check_coupling(::LagrangianPDEProblem, coupling::WeakInitialCondition,
        ansatz::TensorBSplineAnsatz)
    x_boundary(ansatz) isa Free || return nothing
    throw(ArgumentError("$(nameof(typeof(coupling))) needs a space basis that constrains the ends " *
                        "of xspan; this ansatz has x_boundary = Free()"))
end

"""
    check_coupling(problem, coupling::WeakInitialAndBoundaryConditions, ansatz)

Refuse a `Dirichlet()` space basis: every function of it vanishes at the two ends of `xspan`, so the
boundary rows of the coupling would carry no unknown and the slab would be rank deficient.
"""
function check_coupling(::LagrangianPDEProblem, coupling::WeakInitialAndBoundaryConditions,
        ansatz::TensorBSplineAnsatz)
    x_boundary(ansatz) isa Dirichlet && throw(ArgumentError(
        "$(nameof(typeof(coupling))) imposes the space boundary conditions itself; a Dirichlet() " *
        "space basis vanishes at the ends of xspan, so its boundary rows would carry no unknown"))
    return nothing
end

"""
    internal_variables(coupling, ansatz, problem)

The coupling's own state of a step, in the solution step's `internal` named tuple: the multiplier of
the step's top surface, which is the next step's datum.
"""
function coupling_internal(coupling::MultiplierCoupling, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem)
    T = datatype(problem)
    nλ, _ = multiplier_counts(coupling, ansatz)
    return (top_multiplier = zeros(T, nλ, components(problem)),)
end

"""
    copy_internal_variables!(sol, cache)

Carry the multiplier of the step's top surface into the solution step, where the next step's
[`slab_bottom!`](@ref) reads it as its own bottom datum.
"""
function GeometricIntegratorsBase.copy_internal_variables!(
        sol::GeometricIntegratorsBase.SolutionStep,
        c::SpaceTimeGalerkinCache{ST, T, <:MultiplierData}) where {ST, T}
    copyto!(GeometricIntegratorsBase.internal(sol).top_multiplier, c.coupling.top_multiplier)
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
