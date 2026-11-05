"""
    CoefficientRestriction() <: SlabCoupling

The coupling of two consecutive slabs by restriction: the top trace of the previous slab is
restricted to the bottom constraints of the current one.

The bottom trace of a slab is its bottom row of coefficients, so the restriction is the row itself:
the previous slab's top row is this slab's bottom datum, unchanged. The momentum flux goes with it.
The `v` term of the interior weak form leaves one boundary term at the bottom of the slab,
`∫ ∂L/∂v · φ dx` at `t̂ = 0`, and the previous slab's *top* row of the weak form is exactly that
load at its top — so the flux is not re-integrated from a trace but read off the row the previous
slab already computed. At the first slab both come from the problem's initial condition.

The coupling adds no other term, because the interior weak form enforces everything else the slab
needs.
"""
struct CoefficientRestriction <: SlabCoupling end

free_rows(::CoefficientRestriction, ansatz::SpaceTimeAnsatz) = ntime_functions(ansatz) - 1

"""
    RestrictionData{ST, C} <: CouplingData

The block of a coupling that restricts the bottom row of the coefficient array: the coupling, the
bottom row `bottom` of size `(Nx, D)`, which is the datum of the step, and the momentum load `flux`
that enters the slab with it, of the same size.
"""
struct RestrictionData{ST, C <: SlabCoupling} <: CouplingData
    coupling::C
    bottom::Matrix{ST}
    flux::Matrix{ST}
end

function RestrictionData{ST}(coupling::SlabCoupling, nx::Integer, D::Integer) where {ST}
    return RestrictionData{ST, typeof(coupling)}(coupling, zeros(ST, nx, D), zeros(ST, nx, D))
end

function coupling_data(coupling::CoefficientRestriction, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem, ::Type{ST}) where {ST}
    return RestrictionData{ST}(coupling, nspace_functions(ansatz), components(problem))
end

function coupling_data_type(coupling::CoefficientRestriction, ::TensorBSplineAnsatz,
        ::LagrangianPDEProblem, ::Type{ST}) where {ST}
    return RestrictionData{ST, typeof(coupling)}
end

bottom_datum(data::RestrictionData) = data.bottom

function slab_bottom!(c::SpaceTimeGalerkinCache, data::RestrictionData,
        problem::LagrangianPDEProblem, ansatz::TensorBSplineAnsatz, sol)
    if c.steps == 0
        initial_trace!(data, c, problem, ansatz)
    else
        data.bottom .= c.ansatz.top
    end
    return nothing
end

"""
    coupling_term!(b, x, data::RestrictionData, cache, ansatz, problem)

The momentum load of the coupling on the row of the bottom test function, `∫ ∂L/∂v · φ₁ ψ_j dx` at
`t̂ = 0`, which the previous slab's top weak form already computed. Every other row is the interior
weak form's alone, so no other entry changes.
"""
function coupling_term!(b::AbstractVector{ST}, x, data::RestrictionData{ST},
        c::SpaceTimeGalerkinCache{ST}, ansatz::SpaceTimeAnsatz,
        problem::LagrangianPDEProblem) where {ST}
    nx, D = size(data.flux)
    nf = free_rows(coupling(data), ansatz)
    for d in 1:D, j in 1:nx

        b[free_index(1, j, d, nf, nx)] += data.flux[j, d]
    end
    return nothing
end

# The restriction has no unknowns of its own to guess.
coupling_guess!(::RestrictionData, cache, ansatz, problem, sol) = nothing

# The momentum load that leaves the solved slab is the top row of its weak form, which the next
# slab enters with.
function coupling_update!(data::RestrictionData, x, c::SpaceTimeGalerkinCache,
        ansatz::TensorBSplineAnsatz, problem)
    data.flux .= view(c.ansatz.residual, ntime_functions(ansatz), :, :)
    return nothing
end

function copy_coupling_data!(c::RestrictionData, data::RestrictionData)
    c.bottom .= data.bottom
    c.flux .= data.flux
    return nothing
end

"""
    initial_trace!(data, cache, problem, ansatz)

The bottom datum of the first slab — the problem's initial condition, interpolated on the space
basis — and the momentum load that goes with it, `∫ ∂L/∂v(u₀, v₀, w₀) φ_j dx` at the nodes of the
space quadrature.
"""
function initial_trace!(data::RestrictionData{ST}, c::SpaceTimeGalerkinCache{ST},
        problem::LagrangianPDEProblem, ansatz::TensorBSplineAnsatz{T}) where {ST, T}
    xb = space_basis(ansatz)
    nx = nspace_functions(ansatz)
    D = components(problem)
    nodes = collect(spline_nodes(xb))
    values = problem.ics_function(nodes, parameters(problem))
    interpolation = [evaluate(xb, j, x) for x in nodes, j in 1:nx]
    for d in 1:D
        data.bottom[:, d] .= interpolation \ view(values.u, ((d - 1) * nx + 1):(d * nx))
    end
    xq = collect(space_quadrature_nodes(ansatz))
    sampled = problem.ics_function(xq, parameters(problem))
    u0 = [sampled.u[(d - 1) * length(xq) + r] for r in eachindex(xq), d in 1:D]
    v0 = [sampled.v[(d - 1) * length(xq) + r] for r in eachindex(xq), d in 1:D]
    w0 = [sampled.w[(d - 1) * length(xq) + r] for r in eachindex(xq), d in 1:D]
    momentum = similar(u0, ST)
    momentum_load!(momentum, problem.density, u0, v0, w0, parameters(problem))
    boundary_form!(data.flux, momentum, c.ansatz.tabulation)
    return nothing
end

"""The momentum `∂L/∂v` of a trace at the quadrature nodes."""
function momentum_load!(load::AbstractMatrix{ST}, density, u::AbstractMatrix,
        v::AbstractMatrix, w::AbstractMatrix, params) where {ST}
    D = size(u, 2)
    if D == 1
        for r in axes(u, 1)
            d = point_derivatives(density, u, v, w, r, params, Val(1))
            load[r, 1] = d.v[1]
        end
    elseif D == 2
        for r in axes(u, 1)
            d = point_derivatives(density, u, v, w, r, params, Val(2))
            for c in 1:2
                load[r, c] = d.v[c]
            end
        end
    elseif D == 3
        for r in axes(u, 1)
            d = point_derivatives(density, u, v, w, r, params, Val(3))
            for c in 1:3
                load[r, c] = d.v[c]
            end
        end
    else
        for r in axes(u, 1)
            d = point_derivatives(density, u, v, w, r, params, Val(D))
            for c in 1:D
                load[r, c] = d.v[c]
            end
        end
    end
    return nothing
end
