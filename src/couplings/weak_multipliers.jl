"""Abstract type of the basis that a weak boundary condition's multiplier is expanded in: [`BSplineMultiplier`](@ref) or [`LagrangeMultiplier`](@ref)."""
abstract type MultiplierBasis end
"""The multiplier basis of the ansatz's own trace basis, the functions the coupling keeps."""
struct BSplineMultiplier <: MultiplierBasis end
"""The multiplier basis in nodal functions at the spline nodes the coupling keeps."""
struct LagrangeMultiplier <: MultiplierBasis end
function _multiplier_basis(basis::Symbol)
    basis === :bspline && return BSplineMultiplier()
    basis === :lagrange && return LagrangeMultiplier()
    throw(ArgumentError("multiplier_basis must be :bspline or :lagrange, got :$basis"))
end
"""Abstract type of a coupling that imposes the slab's data weakly with surface multipliers."""
abstract type MultiplierCoupling <: SlabCoupling end
"""
    WeakInitialCondition() <: SlabCoupling
    WeakInitialCondition(; multiplier_basis = :bspline) <: SlabCoupling

The initial condition of a slab, imposed weakly.

The bottom row of the coefficient array is free, so all `Nt` rows of the slab are unknowns, and the
slab is closed by one row per multiplier function `ψ_p`,

```math
\\int \\bigl( u_0(x) - u(0, x) \\bigr) \\, \\psi_p(x) \\, dx ,
```

where `u₀` is the problem's initial condition on the first slab and the previous slab's top trace on
every later one. The multiplier at the bottom surface enters the momentum balance as the momentum
that enters the slab: on the first slab, the quadrature-weighted `L²` projection of
`∂L/∂v(u₀, v₀, w₀)` at the nodes of the space rule onto the multiplier functions, and the previous
slab's top multiplier afterwards.

`multiplier_basis` names the basis of `ψ_p`: `:bspline`, the ansatz's own trace basis, or
`:lagrange`, the nodal basis of that same set of functions. `basis` is an alias of the keyword.

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
function WeakInitialCondition(; basis::Symbol = :bspline, multiplier_basis::Symbol = basis)
    WeakInitialCondition(_multiplier_basis(multiplier_basis))
end
"""
    WeakInitialAndBoundaryConditions(; multiplier_basis = :bspline) <: SlabCoupling

The initial condition and the space boundary conditions of a slab, imposed weakly.

The initial condition is imposed as [`WeakInitialCondition`](@ref) imposes it. The multipliers at
`x = a` and at `x = b` are unknowns of the step as well, and their rows are

```math
\\int \\bigl( g_a(t) - u(t, a) \\bigr) \\, \\chi_p(t) \\, dt ,
\\qquad
\\int \\bigl( g_b(t) - u(t, b) \\bigr) \\, \\chi_p(t) \\, dt ,
```

where `g_a` and `g_b` are the problem's side traces, evaluated at each node of the time rule, so
that inhomogeneous data is imposed as it stands. The multiplier basis is the ansatz's own trace
basis, or the nodal basis of that same set of functions, as `multiplier_basis` (alias `basis`)
names it, so that a slab with `Nt` time functions and `Nx` space functions has `Nx - 2`
multipliers at the bottom surface, without its two corner functions, and `Nt` at each end of
`xspan`.

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
    multiplier_basis::B
end
function WeakInitialAndBoundaryConditions(;
        basis::Symbol = :bspline, multiplier_basis::Symbol = basis)
    WeakInitialAndBoundaryConditions(_multiplier_basis(multiplier_basis))
end
imposes_boundary_conditions(::WeakInitialCondition) = false
imposes_boundary_conditions(::WeakInitialAndBoundaryConditions) = true
function space_multiplier_indices(c::MultiplierCoupling, a::TensorBSplineAnsatz)
    collect((imposes_boundary_conditions(c) ? 2 :
             1):(nspace_functions(a) - imposes_boundary_conditions(c)))
end
function boundary_multiplier_indices(c::MultiplierCoupling, a::TensorBSplineAnsatz)
    imposes_boundary_conditions(c) ? collect(1:ntime_functions(a)) : Int[]
end
function multiplier_counts(c::MultiplierCoupling, a::TensorBSplineAnsatz)
    (length(space_multiplier_indices(c, a)), length(boundary_multiplier_indices(c, a)))
end
free_rows(::MultiplierCoupling, a::SpaceTimeAnsatz) = ntime_functions(a)
const MultiplierTraces{ST} = @NamedTuple{
    bottom_test::Vector{ST}, top_test::Vector{ST}, left_space::Vector{ST}, right_space::Vector{ST},
    left_slope::Vector{ST}, right_slope::Vector{ST}, x_u::Matrix{ST}, x_v::Matrix{ST},
    x_w::Matrix{ST}, x_load::Matrix{ST}, t_u::Matrix{ST}, t_v::Matrix{ST}, t_w::Matrix{ST},
    t_load::Matrix{ST}}
const MultiplierWork{ST} = @NamedTuple{
    target::Matrix{ST}, boundary_target::Matrix{ST}, guess_multiplier::Matrix{ST},
    left_coefficients::Matrix{ST}, right_coefficients::Matrix{ST}, left_slopes::Matrix{ST},
    right_slopes::Matrix{ST}, space_row::Vector{ST}, time_row::Vector{ST},
    space_work::Vector{ST}, time_work::Vector{ST}}
"""The block of a weak-multiplier coupling `C` in the slab cache."""
struct MultiplierData{ST, C <: MultiplierCoupling} <: CouplingData
    coupling::C
    initial_multiplier::Matrix{ST}
    top_multiplier::Matrix{ST}
    top_values::Matrix{ST}
    boundary_multiplier::Matrix{ST}
    multiplier_basis::Matrix{ST}
    boundary_basis::Matrix{ST}
    projection::Matrix{ST}
    boundary_projection::Matrix{ST}
    bottom_load::Matrix{ST}
    traces::MultiplierTraces{ST}
    work::MultiplierWork{ST}
end
function coupling_data(c::MultiplierCoupling, a::TensorBSplineAnsatz{T},
        p::LagrangianPDEProblem, ::Type{ST}) where {T, ST}
    nt, nx, D = ntime_functions(a), nspace_functions(a), components(p)
    tb, xb, rules = time_basis(a), space_basis(a), quadratures(quadrature(a))
    tn, xn = quadrature_nodes(rules[1]), quadrature_nodes(rules[2])
    tw, xw = quadrature_weights(rules[1]), quadrature_weights(rules[2])
    b, (nλ, nμ) = c.multiplier_basis, multiplier_counts(c, a)
    qt, qx, xs = length(tn), length(xn), xspan(a)
    λb = multiplier_table(b, xb, xn, space_multiplier_indices(c, a), ST)
    μb = multiplier_table(b, tb, tn, boundary_multiplier_indices(c, a), ST)
    ev(bs, n, x, d = 0) = ST[evaluate(bs, i, x, d) for i in 1:n]
    zm(m) = zeros(ST, m, D)
    traces = (bottom_test = ev(tb, nt, zero(T)), top_test = ev(tb, nt, one(T)),
        left_space = ev(xb, nx, xs[1]), right_space = ev(xb, nx, xs[2]),
        left_slope = ev(xb, nx, xs[1], 1), right_slope = ev(xb, nx, xs[2], 1),
        x_u = zm(qx), x_v = zm(qx), x_w = zm(qx), x_load = zm(qx),
        t_u = zm(qt), t_v = zm(qt), t_w = zm(qt), t_load = zm(qt))
    work = (target = zm(qx), boundary_target = zm(2 * qt), guess_multiplier = zm(nλ),
        left_coefficients = zm(nt), right_coefficients = zm(nt), left_slopes = zm(nt),
        right_slopes = zm(nt), space_row = zeros(ST, nx), time_row = zeros(ST, nt),
        space_work = zeros(ST, qx), time_work = zeros(ST, qt))
    return MultiplierData{ST, typeof(c)}(c, zm(nλ), zm(nλ), zm(qx), zm(2 * nμ), λb, μb,
        least_squares(λb, xw), least_squares(μb, tw), zm(nx), traces, work)
end
function coupling_data_type(c::MultiplierCoupling, ::TensorBSplineAnsatz,
        ::LagrangianPDEProblem, ::Type{ST}) where {ST}
    MultiplierData{ST, typeof(c)}
end
bottom_datum(::MultiplierData) = nothing
function coupling_unknowns(c::MultiplierCoupling, a::TensorBSplineAnsatz, p::LagrangianPDEProblem)
    sum(multiplier_counts(c, a) .* (1, 2)) * components(p)
end
function multiplier_unknowns(d::MultiplierData)
    (size(d.initial_multiplier, 1) + 2 * size(d.boundary_basis, 1)) *
    size(d.initial_multiplier, 2)
end
@inline function multiplier_index(nfree::Integer, D::Integer, nλ::Integer, nμ::Integer,
        edge::Symbol, d::Integer, i::Integer)
    edge === :top && return nfree + (d - 1) * nλ + i
    edge === :left && return nfree + D * nλ + (d - 1) * nμ + i
    edge === :right && return nfree + D * nλ + D * nμ + (d - 1) * nμ + i
    throw(ArgumentError("multiplier edge must be :top, :left or :right, got $(repr(edge))"))
end
@inline multiplier_index(c::SpaceTimeGalerkinCache, edge::Symbol, d::Integer, i::Integer) = multiplier_index(
    length(c.x) - multiplier_unknowns(c.coupling),
    size(c.coupling.initial_multiplier, 2), size(c.coupling.initial_multiplier, 1),
    size(c.coupling.boundary_basis, 1), edge, d, i)
function multiplier_table(::BSplineMultiplier, tr, pts::AbstractVector,
        idx::AbstractVector{<:Integer}, ::Type{ST}) where {ST}
    ST[evaluate(tr, idx[p], pts[r]) for p in eachindex(idx), r in eachindex(pts)]
end
function multiplier_table(::LagrangeMultiplier, tr, pts::AbstractVector,
        idx::AbstractVector{<:Integer}, ::Type{ST}) where {ST}
    ξ, n = nodes(tr), length(idx)
    V = ST[evaluate(tr, idx[j], ξ[idx[i]]) for i in 1:n, j in 1:n]
    return Matrix(transpose(V \ Matrix{ST}(I, n, n))) *
           multiplier_table(BSplineMultiplier(), tr, pts, idx, ST)
end
function least_squares(basis::AbstractMatrix{ST}, weights) where {ST}
    size(basis, 1) == 0 && return zeros(ST, 0, size(basis, 2))
    load = basis * Diagonal(ST.(weights))
    return (load * transpose(basis)) \ load
end
function basis_load!(load::AbstractVector{ST}, coeffs::AbstractMatrix{ST},
        basis::AbstractMatrix{ST}, axis, scale::ST, d::Integer, p₀::Integer = 0) where {ST}
    fill!(load, zero(ST))
    for r in axes(basis, 2)
        val = zero(ST)
        for p in axes(basis, 1)
            val += coeffs[p₀ + p, d] * basis[p, r]
        end
        wz = scale * axis.weight[r] * val
        for l in axes(axis.index, 2)
            idx = axis.index[r, l]
            idx == 0 || (load[idx] += wz * axis.value[r, l])
        end
    end
    return nothing
end
function add_tensor_load!(b::AbstractVector{ST}, tvec::AbstractVector{ST},
        xvec::AbstractVector{ST}, nf::Integer, nx::Integer, d::Integer, sgn::ST) where {ST}
    # No short circuit on a zero entry: for some element types `==` compares the primal part
    # only, as ForwardDiff's `Dual` before 1.0 does, and a skipped entry loses its derivative.
    # Either vector is a load of the slab, so either can carry one.
    for a in 1:nf
        s = sgn * tvec[a]
        for j in 1:nx
            b[free_index(a, j, d, nf, nx)] += s * xvec[j]
        end
    end
    return nothing
end
@inline function row_dot(A::AbstractMatrix, i::Integer, v::AbstractVector{ST}) where {ST}
    s = zero(ST)
    for j in axes(A, 2)
        s += A[i, j] * v[j]
    end
    return s
end
function trace_constraint!(b::AbstractVector{ST}, work::AbstractVector{ST},
        target::AbstractMatrix{ST}, t_off::Integer, coeffs, axis, scale::ST,
        basis::AbstractMatrix{ST}, nfree::Integer, D::Integer, nλ::Integer, nμ::Integer,
        edge::Symbol, d::Integer) where {ST}
    for r in eachindex(work)
        val = zero(ST)
        for l in axes(axis.index, 2)
            k = axis.index[r, l]
            k == 0 || (val += coeffs[k] * axis.value[r, l])
        end
        work[r] = scale * axis.weight[r] * (val - target[t_off + r, d])
    end
    for p in axes(basis, 1)
        b[multiplier_index(nfree, D, nλ, nμ, edge, d, p)] = -row_dot(basis, p, work)
    end
    return nothing
end
function coupling_term!(
        b::AbstractVector{ST}, x::AbstractVector{ST}, data::MultiplierData{ST},
        c::SpaceTimeGalerkinCache{ST}, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem) where {ST}
    nx, D = size(c.ansatz.top)
    nf = ntime_functions(ansatz)
    nλ, nμ = size(data.initial_multiplier, 1), size(data.boundary_basis, 1)
    tq, xq = c.ansatz.tabulation.time, c.ansatz.tabulation.space
    h, qt = convert(ST, timestep(problem)), nquadrature(tq)
    nt, nfree = size(data.work.left_coefficients, 1), nf * nx * D
    w, tr, λb, μb = data.work, data.traces, data.multiplier_basis, data.boundary_basis
    coupling_unknowns!(data, x, D)
    for d in 1:D
        add_tensor_load!(
            b, tr.bottom_test, view(data.bottom_load, :, d), nf, nx, d, one(ST))
        basis_load!(w.space_row, data.top_multiplier, λb, xq, one(ST), d)
        add_tensor_load!(b, tr.top_test, w.space_row, nf, nx, d, -one(ST))
        if nμ > 0
            basis_load!(w.time_row, data.boundary_multiplier, μb, tq, h, d, 0)
            add_tensor_load!(b, w.time_row, tr.left_space, nf, nx, d, one(ST))
            basis_load!(w.time_row, data.boundary_multiplier, μb, tq, h, d, nμ)
            add_tensor_load!(b, w.time_row, tr.right_space, nf, nx, d, -one(ST))
        end
        trace_constraint!(b, w.space_work, w.target, 0, view(c.ansatz.coeffs, 1, :, d),
            xq, one(ST), λb, nfree, D, nλ, nμ, :top, d)
    end
    nμ == 0 && return nothing
    end_coefficients!(data, c.ansatz.coeffs, nx, nt, D)
    for d in 1:D
        trace_constraint!(b, w.time_work, w.boundary_target, 0,
            view(w.left_coefficients, :, d), tq, h, μb, nfree, D, nλ, nμ, :left, d)
        trace_constraint!(b, w.time_work, w.boundary_target, qt,
            view(w.right_coefficients, :, d), tq, h, μb, nfree, D, nλ, nμ, :right, d)
    end
    return nothing
end
function coupling_unknowns!(data::MultiplierData{ST}, x::AbstractVector{ST}, D::Integer) where {ST}
    nλ, nμ = size(data.initial_multiplier, 1), size(data.boundary_basis, 1)
    nf = length(x) - multiplier_unknowns(data)
    for d in 1:D, p in 1:nλ

        data.top_multiplier[p, d] = x[multiplier_index(nf, D, nλ, nμ, :top, d, p)]
    end
    for d in 1:D, p in 1:nμ

        data.boundary_multiplier[p, d] = x[multiplier_index(nf, D, nλ, nμ, :left, d, p)]
        data.boundary_multiplier[nμ + p, d] = x[multiplier_index(
            nf, D, nλ, nμ, :right, d, p)]
    end
    return nothing
end
function coupling_update!(data::MultiplierData{ST}, x::AbstractVector{ST},
        ::SpaceTimeGalerkinCache{ST}, ::SpaceTimeAnsatz, ::LagrangianPDEProblem) where {ST}
    D, λb = size(data.top_multiplier, 2), data.multiplier_basis
    coupling_unknowns!(data, x, D)
    for d in 1:D, r in axes(λb, 2)

        data.top_values[r, d] = row_dot(transpose(λb), r, view(data.top_multiplier, :, d))
    end
    return nothing
end
function project!(coeffs::AbstractMatrix{ST}, r₀::Integer, proj::AbstractMatrix{ST},
        vals::AbstractMatrix{ST}) where {ST}
    for d in axes(vals, 2), p in axes(proj, 1)

        coeffs[r₀ + p - 1, d] = row_dot(proj, p, view(vals, :, d))
    end
    return nothing
end
function end_coefficients!(data::MultiplierData{ST}, coeffs::Array{ST, 3},
        nx::Integer, nt::Integer, D::Integer) where {ST}
    w, t = data.work, data.traces
    for d in 1:D
        cd = view(coeffs, :, :, d)
        for a in 1:nt
            w.left_coefficients[a, d] = row_dot(cd, a, t.left_space)
            w.right_coefficients[a, d] = row_dot(cd, a, t.right_space)
            w.left_slopes[a, d] = row_dot(cd, a, t.left_slope)
            w.right_slopes[a, d] = row_dot(cd, a, t.right_slope)
        end
    end
    return nothing
end
function edge_flux!(data::MultiplierData{ST}, c::SpaceTimeGalerkinCache{ST},
        problem::LagrangianPDEProblem, edge::Integer, h) where {ST}
    tq, tr = c.ansatz.tabulation.time, data.traces
    qt, D, hh = nquadrature(tq), size(tr.t_u, 2), convert(ST, h)
    vals = edge == 1 ? data.work.left_coefficients : data.work.right_coefficients
    slps = edge == 1 ? data.work.left_slopes : data.work.right_slopes
    for d in 1:D
        for rt in 1:qt
            u, v, w = zero(ST), zero(ST), zero(ST)
            for l in axes(tq.index, 2)
                a = tq.index[rt, l]
                a == 0 && continue
                u += vals[a, d] * tq.value[rt, l]
                v += vals[a, d] * tq.slope[rt, l]
                w += slps[a, d] * tq.value[rt, l]
            end
            tr.t_u[rt, d], tr.t_v[rt, d], tr.t_w[rt, d] = u, v / hh, w
        end
    end
    return flux_load!(
        tr.t_load, problem.density, tr.t_u, tr.t_v, tr.t_w, parameters(problem))
end
function flux_load!(load::AbstractMatrix{ST}, density, u::AbstractMatrix,
        v::AbstractMatrix, w::AbstractMatrix, params) where {ST}
    D = size(u, 2)
    if D > 3
        density isa LagrangianDensity ||
            return _flux_load!(load, density, u, v, w, params, Val(D))
        # the components of a scalar density are independent: each is the case `D == 1`
        for c in 1:D
            flux_load!(view(load, :, c:c), density, view(u, :, c:c), view(v, :, c:c),
                view(w, :, c:c), params)
        end
        return nothing
    end
    for r in axes(u, 1)
        if D == 1
            load[r, 1] = point_derivatives(density, u, v, w, r, params, Val(1)).w[1]
        elseif D == 2
            load[r, 1], load[r, 2] = point_derivatives(density, u, v, w, r, params, Val(2)).w
        elseif D == 3
            dw = point_derivatives(density, u, v, w, r, params, Val(3)).w
            load[r, 1], load[r, 2], load[r, 3] = dw[1], dw[2], dw[3]
        end
    end
    return nothing
end
# The function barrier of `flux_load!` for a coupled density with `D > 3`, as
# `_density_derivatives!` is.
function _flux_load!(load, density, u, v, w, params, ::Val{D}) where {D}
    for r in axes(u, 1)
        dw = point_derivatives(density, u, v, w, r, params, Val(D)).w
        for c in 1:D
            load[r, c] = dw[c]
        end
    end
    return nothing
end
function project_momentum!(dest::AbstractMatrix{ST}, data::MultiplierData{ST},
        problem::LagrangianPDEProblem) where {ST}
    tr = data.traces
    momentum_load!(tr.x_load, problem.density, tr.x_u, tr.x_v, tr.x_w, parameters(problem))
    return project!(dest, 1, data.projection, tr.x_load)
end
function coupling_guess!(data::MultiplierData{ST}, c::SpaceTimeGalerkinCache{ST},
        ansatz::TensorBSplineAnsatz, problem::LagrangianPDEProblem, sol) where {ST}
    nt, (nx, D) = ntime_functions(ansatz), size(c.ansatz.top)
    nλ, nμ = size(data.initial_multiplier, 1), size(data.boundary_basis, 1)
    h, nf, tr = timestep(problem), nt * nx * D, data.traces
    scatter!(c.ansatz.coeffs, c.x, nothing, nt)
    top_values!(tr.x_u, tr.x_v, tr.x_w, c.ansatz, trace_evaluation(c.ansatz), h)
    project_momentum!(data.work.guess_multiplier, data, problem)
    for d in 1:D, p in 1:nλ

        c.x[multiplier_index(nf, D, nλ, nμ, :top, d, p)] = data.work.guess_multiplier[p, d]
    end
    nμ == 0 && return nothing
    end_coefficients!(data, c.ansatz.coeffs, nx, nt, D)
    for (eidx, esym) in ((1, :left), (2, :right))
        edge_flux!(data, c, problem, eidx, h)
        r₀ = (eidx - 1) * nμ
        project!(data.boundary_multiplier, r₀ + 1, data.boundary_projection, tr.t_load)
        for d in 1:D, p in 1:nμ

            c.x[multiplier_index(nf, D, nλ, nμ, esym, d, p)] = data.boundary_multiplier[r₀ + p, d]
        end
    end
    return nothing
end
function slab_bottom!(c::SpaceTimeGalerkinCache{ST}, data::MultiplierData{ST},
        problem::LagrangianPDEProblem, ansatz::TensorBSplineAnsatz, sol) where {ST}
    tr, w = data.traces, data.work
    if c.steps == 0
        w.target .= initial_condition_sample(ST, problem, ansatz)
        ics = problem.ics_function(collect(space_quadrature_nodes(ansatz)), parameters(problem))
        tr.x_u .= ics.u
        tr.x_v .= ics.v
        tr.x_w .= ics.w
        project_momentum!(data.initial_multiplier, data, problem)
    else
        w.target .= c.trace_u
        copyto!(data.initial_multiplier, data.top_multiplier)
    end
    xq = c.ansatz.tabulation.space
    for d in axes(data.initial_multiplier, 2)
        basis_load!(view(data.bottom_load, :, d), data.initial_multiplier,
            data.multiplier_basis, xq, one(ST), d)
    end
    size(data.boundary_basis, 1) == 0 && return nothing
    tq, h = c.ansatz.tabulation.time, timestep(problem)
    qt, D, t₀ = nquadrature(tq), size(w.boundary_target, 2), slab_start(problem, sol)
    for rt in 1:qt
        bcs = problem.bcs_function(t₀ + h * tq.node[rt], parameters(problem))
        for d in 1:D
            w.boundary_target[rt, d] = ST(bcs.left.u[d])
            w.boundary_target[qt + rt, d] = ST(bcs.right.u[d])
        end
    end
    return nothing
end
function check_coupling(::LagrangianPDEProblem, c::MultiplierCoupling, a::TensorBSplineAnsatz)
    b, want = x_boundary(a), imposes_boundary_conditions(c) ? Free : Dirichlet
    b isa want ||
        throw(ArgumentError("$(nameof(typeof(c))) needs x_boundary = $want(), got $(nameof(typeof(b)))"))
    return nothing
end
function coupling_internal(c::MultiplierCoupling, a::TensorBSplineAnsatz, p::LagrangianPDEProblem)
    (top_multiplier = zeros(datatype(p), first(multiplier_counts(c, a)), components(p)),)
end
function copy_internal_variables!(sol::SolutionStep, c::SpaceTimeGalerkinCache{
        <:Any, <:Any, <:MultiplierData})
    (copyto!(internal(sol).top_multiplier, c.coupling.top_multiplier); nothing)
end
function copy_coupling_data!(c::MultiplierData, data::MultiplierData)
    c.initial_multiplier .= data.initial_multiplier
    c.bottom_load .= data.bottom_load
    c.work.target .= data.work.target
    c.work.boundary_target .= data.work.boundary_target
    return nothing
end

# ---- the symbolic ansatz ---------------------------------------------------
#
# A symbolic ansatz fixes no bottom row and holds no coefficient array: its slab carries the
# parameter vector itself, and its field is a closed-form expression. The block's tables, its
# targets and its multipliers are the trace basis's, so every function that builds or reads them
# delegates to the ansatz's trace basis. The residual terms and the guesses read the field at the
# slab's nodes and at its four boundary sets, which the ansatz's block evaluates.

"""The free rows of a symbolic slab are its parameters."""
free_rows(::MultiplierCoupling, a::SymbolicAnsatz) = a.num_params

function coupling_data(c::MultiplierCoupling, a::SymbolicAnsatz,
        p::LagrangianPDEProblem, ::Type{ST}) where {ST}
    return coupling_data(c, a.trace, p, ST)
end

function coupling_data_type(c::MultiplierCoupling, a::SymbolicAnsatz,
        p::LagrangianPDEProblem, ::Type{ST}) where {ST}
    return coupling_data_type(c, a.trace, p, ST)
end

function coupling_unknowns(c::MultiplierCoupling, a::SymbolicAnsatz,
        p::LagrangianPDEProblem)
    return coupling_unknowns(c, a.trace, p)
end

"""
    check_coupling(problem, coupling, ansatz)

Refuse a coupling that a symbolic ansatz cannot take.

A parameter ansatz has no bottom coefficient row, so its interior weak form alone is not the
gradient of the slab's action; the weak initial-and-boundary coupling adds the four boundary
terms that complete it. The refusals of the other couplings name the one that is taken.
"""
function check_coupling(problem::LagrangianPDEProblem, c::MultiplierCoupling,
        a::SymbolicAnsatz)
    return check_coupling(problem, c, a.trace)
end

# A symbolic ansatz has no bottom coefficient row, so a coupling that fixes one has nothing to
# fix and the interior weak form alone is not the gradient of the slab's action.
function check_coupling(::LagrangianPDEProblem, ::CoefficientRestriction,
        ::SymbolicAnsatz)
    throw(ArgumentError("CoefficientRestriction cannot pair with a SymbolicAnsatz; a " *
                        "SymbolicAnsatz takes WeakInitialAndBoundaryConditions"))
end

# The field is free at both ends of `xspan`, so a coupling that imposes the initial condition
# alone leaves those ends unconstrained.
function check_coupling(::LagrangianPDEProblem, ::WeakInitialCondition,
        ::SymbolicAnsatz)
    throw(ArgumentError("WeakInitialCondition cannot pair with a SymbolicAnsatz; a " *
                        "SymbolicAnsatz takes WeakInitialAndBoundaryConditions"))
end

function coupling_internal(c::MultiplierCoupling, a::SymbolicAnsatz,
        p::LagrangianPDEProblem)
    return coupling_internal(c, a.trace, p)
end

function slab_bottom!(c::SpaceTimeGalerkinCache{ST}, data::MultiplierData{ST},
        problem::LagrangianPDEProblem, ansatz::SymbolicAnsatz, sol) where {ST}
    return slab_bottom!(c, data, problem, ansatz.trace, sol)
end

# The multiplier's value at each node of a rule, from its coefficients in the coupling's basis.
function multiplier_values!(value::AbstractMatrix{ST}, coefficients::AbstractMatrix{ST},
        basis::AbstractMatrix{ST}) where {ST}
    fill!(value, zero(ST))
    for d in axes(value, 2), p in axes(coefficients, 1), r in axes(value, 1)
        value[r, d] += coefficients[p, d] * basis[p, r]
    end
    return nothing
end

function coupling_term!(
        b::AbstractVector{ST}, x::AbstractVector{ST}, data::MultiplierData{ST},
        c::SpaceTimeGalerkinCache{ST}, ansatz::SymbolicAnsatz,
        problem::LagrangianPDEProblem) where {ST}
    s = c.ansatz
    np, D = ansatz.num_params, length(ansatz.expr)
    nλ, nμ = size(data.initial_multiplier, 1), size(data.boundary_basis, 1)
    tq, xq = s.tabulation.time, s.tabulation.space
    qt, qx = nquadrature(tq), nquadrature(xq)
    hh = convert(ST, timestep(problem))
    coupling_unknowns!(data, x, D)
    multiplier_values!(s.lambda_bottom, data.initial_multiplier, data.multiplier_basis)
    multiplier_values!(s.lambda_top, data.top_multiplier, data.multiplier_basis)
    multiplier_values!(s.mu_left, view(data.boundary_multiplier, 1:nμ, :),
        data.boundary_basis)
    multiplier_values!(
        s.mu_right, view(data.boundary_multiplier, (nμ + 1):(2 * nμ), :),
        data.boundary_basis)
    # the four boundary loads on the parameter rows
    for i in 1:np
        total = zero(ST)
        for r in 1:qx, d in 1:D

            total += xq.weight[r] * (s.lambda_bottom[r, d] * s.bottom_J[r, d, i] -
                      s.lambda_top[r, d] * s.top_J[r, d, i])
        end
        for q in 1:qt, d in 1:D

            total += hh * tq.weight[q] *
                     (s.mu_left[q, d] * s.left_J[q, d, i] -
                      s.mu_right[q, d] * s.right_J[q, d, i])
        end
        b[i] += total
    end
    # the bottom constraint, in the rows of the top multiplier: the coupling's carried unknown
    for d in 1:D, p in 1:nλ

        value = zero(ST)
        for r in 1:qx
            value += xq.weight[r] * data.multiplier_basis[p, r] *
                     (s.bottom_u[r, d] - data.work.target[r, d])
        end
        b[multiplier_index(np, D, nλ, nμ, :top, d, p)] = -value
    end
    # the two side constraints
    for d in 1:D, p in 1:nμ

        left = zero(ST)
        right = zero(ST)
        for q in 1:qt
            ω = hh * tq.weight[q] * data.boundary_basis[p, q]
            left += ω * (s.left_u[q, d] - data.work.boundary_target[q, d])
            right += ω * (s.right_u[q, d] - data.work.boundary_target[qt + q, d])
        end
        b[multiplier_index(np, D, nλ, nμ, :left, d, p)] = -left
        b[multiplier_index(np, D, nλ, nμ, :right, d, p)] = -right
    end
    return nothing
end

function coupling_guess!(data::MultiplierData{ST}, c::SpaceTimeGalerkinCache{ST},
        ansatz::SymbolicAnsatz, problem::LagrangianPDEProblem, sol) where {ST}
    s = c.ansatz
    np, D = ansatz.num_params, length(ansatz.expr)
    nλ, nμ = size(data.initial_multiplier, 1), size(data.boundary_basis, 1)
    hh, tr = convert(ST, timestep(problem)), data.traces
    s.params .= view(c.x, 1:np)
    top_values!(tr.x_u, tr.x_v, tr.x_w, s, trace_evaluation(s), hh)
    project_momentum!(data.work.guess_multiplier, data, problem)
    for d in 1:D, p in 1:nλ

        c.x[multiplier_index(np, D, nλ, nμ, :top, d, p)] = data.work.guess_multiplier[p, d]
    end
    tq = s.tabulation.time
    for (eidx, edge, end_of_span) in ((1, :left, xspan(ansatz)[1]),
        (2, :right, xspan(ansatz)[2]))
        for q in 1:nquadrature(tq)
            evaluate_at!(s, tq.node[q], convert(ST, end_of_span))
            for d in 1:D
                tr.t_u[q, d] = s.ubuf[d]
                tr.t_v[q, d] = s.utbuf[d] / hh
                tr.t_w[q, d] = s.uxbuf[d]
            end
        end
        flux_load!(tr.t_load, problem.density, tr.t_u, tr.t_v, tr.t_w,
            parameters(problem))
        offset = (eidx - 1) * nμ
        project!(data.boundary_multiplier, offset + 1, data.boundary_projection, tr.t_load)
        for d in 1:D, p in 1:nμ

            c.x[multiplier_index(np, D, nλ, nμ, edge, d, p)] = data.boundary_multiplier[offset + p, d]
        end
    end
    return nothing
end

# ---- the network ansatz ----------------------------------------------------
#
# A network holds its unknowns as a parameter vector, and its field is free at both ends of `xspan`.
# The multiplier tables, their projections and the surfaces they are read at are the trace basis',
# which the network carries, so every function that builds or reads them delegates to that basis.
# The field's rows are its free parameters, weighted by the derivative of the field with respect to
# each of them, at the four surfaces of the slab, which the network's block evaluates on the trace
# basis' rules; its interior rows are evaluated on its own, at a rule of the ansatz's own.

"""The free rows of a network slab are its free parameters."""
free_rows(::MultiplierCoupling, a::NetworkAnsatz) = nfree(a)

function space_multiplier_indices(c::MultiplierCoupling, a::NetworkAnsatz)
    return space_multiplier_indices(c, trace_basis(a))
end

function boundary_multiplier_indices(c::MultiplierCoupling, a::NetworkAnsatz)
    return boundary_multiplier_indices(c, trace_basis(a))
end

function multiplier_counts(c::MultiplierCoupling, a::NetworkAnsatz)
    return multiplier_counts(c, trace_basis(a))
end

function coupling_data(c::MultiplierCoupling, a::NetworkAnsatz,
        p::LagrangianPDEProblem, ::Type{ST}) where {ST}
    return coupling_data(c, trace_basis(a), p, ST)
end

function coupling_data_type(c::MultiplierCoupling, a::NetworkAnsatz,
        p::LagrangianPDEProblem, ::Type{ST}) where {ST}
    return coupling_data_type(c, trace_basis(a), p, ST)
end

function coupling_unknowns(c::MultiplierCoupling, a::NetworkAnsatz,
        p::LagrangianPDEProblem)
    return coupling_unknowns(c, trace_basis(a), p)
end

function coupling_internal(c::MultiplierCoupling, a::NetworkAnsatz,
        p::LagrangianPDEProblem)
    return coupling_internal(c, trace_basis(a), p)
end

"""
    check_coupling(problem, coupling, ansatz)

Refuse a coupling that a network cannot take.

A network holds no bottom coefficient row, so its interior weak form alone is not the gradient of the
slab's action: the weak initial-and-boundary coupling adds the four surface terms that complete it,
and the refusals of the other couplings name the one that is taken. The trace basis' own `x_boundary`
is not checked, because the network builds that basis with `Free()`, the condition the weak
initial-and-boundary coupling needs.
"""
function check_coupling(problem::LagrangianPDEProblem, c::MultiplierCoupling,
        a::NetworkAnsatz)
    return nothing
end

function check_coupling(::LagrangianPDEProblem, ::CoefficientRestriction,
        ::NetworkAnsatz)
    throw(ArgumentError("CoefficientRestriction cannot pair with a NetworkAnsatz; a " *
                        "NetworkAnsatz takes WeakInitialAndBoundaryConditions"))
end

function check_coupling(::LagrangianPDEProblem, ::WeakInitialCondition,
        ::NetworkAnsatz)
    throw(ArgumentError("WeakInitialCondition cannot pair with a NetworkAnsatz; a " *
                        "NetworkAnsatz takes WeakInitialAndBoundaryConditions"))
end

function slab_bottom!(c::SpaceTimeGalerkinCache{ST}, data::MultiplierData{ST},
        problem::LagrangianPDEProblem, ansatz::NetworkAnsatz, sol) where {ST}
    tr, w = data.traces, data.work
    tb = trace_basis(ansatz)
    xt, tt = c.ansatz.trace_tabulation.space, c.ansatz.trace_tabulation.time
    if c.steps == 0
        w.target .= initial_condition_sample(ST, problem, tb)
        ics = problem.ics_function(collect(space_quadrature_nodes(tb)), parameters(problem))
        tr.x_u .= ics.u
        tr.x_v .= ics.v
        tr.x_w .= ics.w
        project_momentum!(data.initial_multiplier, data, problem)
    else
        w.target .= c.trace_u
        copyto!(data.initial_multiplier, data.top_multiplier)
    end
    for d in axes(data.initial_multiplier, 2)
        basis_load!(view(data.bottom_load, :, d), data.initial_multiplier,
            data.multiplier_basis, xt, one(ST), d)
    end
    size(data.boundary_basis, 1) == 0 && return nothing
    h, qt, D = timestep(problem), nquadrature(tt), components(problem)
    t₀ = slab_start(problem, sol)
    for rt in 1:qt
        bcs = problem.bcs_function(t₀ + h * tt.node[rt], parameters(problem))
        for d in 1:D
            w.boundary_target[rt, d] = ST(bcs.left.u[d])
            w.boundary_target[qt + rt, d] = ST(bcs.right.u[d])
        end
    end
    return nothing
end

function coupling_term!(
        b::AbstractVector{ST}, x::AbstractVector{ST}, data::MultiplierData{ST},
        c::SpaceTimeGalerkinCache{ST}, ansatz::NetworkAnsatz,
        problem::LagrangianPDEProblem) where {ST}
    s = c.ansatz
    nf, D = nfree(ansatz), ansatz.D
    nλ, nμ = size(data.initial_multiplier, 1), size(data.boundary_basis, 1)
    tt, xt = s.trace_tabulation.time, s.trace_tabulation.space
    qt, qx = nquadrature(tt), nquadrature(xt)
    hh = convert(ST, timestep(problem))
    coupling_unknowns!(data, x, D)
    multiplier_values!(s.lambda_bottom, data.initial_multiplier, data.multiplier_basis)
    multiplier_values!(s.lambda_top, data.top_multiplier, data.multiplier_basis)
    multiplier_values!(s.mu_left, view(data.boundary_multiplier, 1:nμ, :),
        data.boundary_basis)
    multiplier_values!(s.mu_right, view(data.boundary_multiplier, (nμ + 1):(2nμ), :),
        data.boundary_basis)
    # the four surface loads on the parameter rows
    for k in 1:nf
        total = zero(ST)
        for r in 1:qx, d in 1:D

            total += xt.weight[r] * (s.lambda_bottom[r, d] * s.bottom_J[r, d, k] -
                      s.lambda_top[r, d] * s.top_J[r, d, k])
        end
        for q in 1:qt, d in 1:D

            total += hh * tt.weight[q] *
                     (s.mu_left[q, d] * s.left_J[q, d, k] -
                      s.mu_right[q, d] * s.right_J[q, d, k])
        end
        b[k] += total
    end
    # the bottom constraint, in the rows of the top multiplier: the coupling's carried unknown
    for d in 1:D, p in 1:nλ

        value = zero(ST)
        for r in 1:qx
            value += xt.weight[r] * data.multiplier_basis[p, r] *
                     (s.bottom_u[r, d] - data.work.target[r, d])
        end
        b[multiplier_index(nf, D, nλ, nμ, :top, d, p)] = -value
    end
    # the two side constraints
    for d in 1:D, p in 1:nμ

        left = zero(ST)
        right = zero(ST)
        for q in 1:qt
            ω = hh * tt.weight[q] * data.boundary_basis[p, q]
            left += ω * (s.left_u[q, d] - data.work.boundary_target[q, d])
            right += ω * (s.right_u[q, d] - data.work.boundary_target[qt + q, d])
        end
        b[multiplier_index(nf, D, nλ, nμ, :left, d, p)] = -left
        b[multiplier_index(nf, D, nλ, nμ, :right, d, p)] = -right
    end
    return nothing
end

function coupling_guess!(data::MultiplierData{ST}, c::SpaceTimeGalerkinCache{ST},
        ansatz::NetworkAnsatz, problem::LagrangianPDEProblem, sol) where {ST}
    s = c.ansatz
    nf, D = nfree(ansatz), ansatz.D
    nλ, nμ = size(data.initial_multiplier, 1), size(data.boundary_basis, 1)
    hh, tr = convert(ST, timestep(problem)), data.traces
    top_values!(tr.x_u, tr.x_v, tr.x_w, s, trace_evaluation(s), hh)
    project_momentum!(data.work.guess_multiplier, data, problem)
    for d in 1:D, p in 1:nλ

        c.x[multiplier_index(nf, D, nλ, nμ, :top, d, p)] = data.work.guess_multiplier[p, d]
    end
    nμ == 0 && return nothing
    tt = s.trace_tabulation.time
    for (eidx, edge, end_of_span) in ((1, :left, xspan(ansatz)[1]), (
        2, :right, xspan(ansatz)[2]))
        for q in 1:nquadrature(tt)
            evaluate_at!(s, tt.node[q], convert(ST, end_of_span))
            for d in 1:D
                tr.t_u[q, d] = s.values[d]
                tr.t_v[q, d] = s.values[D + d] / hh
                tr.t_w[q, d] = s.values[2D + d]
            end
        end
        flux_load!(
            tr.t_load, problem.density, tr.t_u, tr.t_v, tr.t_w, parameters(problem))
        offset = (eidx - 1) * nμ
        project!(data.boundary_multiplier, offset + 1, data.boundary_projection, tr.t_load)
        for d in 1:D, p in 1:nμ

            c.x[multiplier_index(nf, D, nλ, nμ, edge, d, p)] = data.boundary_multiplier[offset + p, d]
        end
    end
    return nothing
end
