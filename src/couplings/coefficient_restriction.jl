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

"""
    flux_load(coupling, cache, j, d)

The coupling's contribution to the row of the bottom test function: the momentum flux that enters
the slab at `t̂ = 0`, at space function `j` and component `d`.
"""
function flux_load(::CoefficientRestriction, cache::SpaceTimeGalerkinCache, j::Integer, d::Integer)
    cache.flux[j, d]
end

function slab_bottom!(c::SpaceTimeGalerkinCache, ::CoefficientRestriction,
        problem::LagrangianPDEProblem, ansatz::TensorBSplineAnsatz, sol)
    if c.steps == 0
        initial_trace!(c, problem, ansatz)
    else
        c.bottom .= c.top
    end
    return nothing
end

"""
    initial_trace!(cache, problem, ansatz)

The bottom datum of the first slab — the problem's initial condition, interpolated on the space
basis — and the momentum load that goes with it, `∫ ∂L/∂v(u₀, v₀, w₀) φ_j dx` at the nodes of the
space quadrature.
"""
function initial_trace!(c::SpaceTimeGalerkinCache{ST}, problem::LagrangianPDEProblem,
        ansatz::TensorBSplineAnsatz{T}) where {ST, T}
    xb = space_basis(ansatz)
    nx = nspace_functions(ansatz)
    D = components(problem)
    nodes = collect(spline_nodes(xb))
    values = problem.ics_function(nodes, parameters(problem))
    interpolation = [evaluate(xb, j, x) for x in nodes, j in 1:nx]
    for d in 1:D
        c.bottom[:, d] .= interpolation \ view(values.u, ((d - 1) * nx + 1):(d * nx))
    end
    xq = collect(space_quadrature_nodes(ansatz))
    sampled = problem.ics_function(xq, parameters(problem))
    u0 = [sampled.u[(d - 1) * length(xq) + r] for r in eachindex(xq), d in 1:D]
    v0 = [sampled.v[(d - 1) * length(xq) + r] for r in eachindex(xq), d in 1:D]
    w0 = [sampled.w[(d - 1) * length(xq) + r] for r in eachindex(xq), d in 1:D]
    momentum = similar(u0, ST)
    momentum_load!(momentum, problem.density, u0, v0, w0, parameters(problem))
    boundary_form!(c.flux, momentum, c.tabulation)
    return nothing
end

"""The momentum `∂L/∂v` of the initial data, one component at a time."""
function momentum_load!(load::AbstractMatrix{ST}, density::LagrangianDensity,
        u::AbstractMatrix, v::AbstractMatrix, w::AbstractMatrix, params) where {ST}
    for d in axes(u, 2), r in axes(u, 1)

        load[r, d] = derivative_v(density, u[r, d], v[r, d], w[r, d], params)
    end
    return nothing
end

"""The momentum `∂L/∂v` of the initial data, the components of a point at once."""
function momentum_load!(load::AbstractMatrix{ST}, density, u::AbstractMatrix,
        v::AbstractMatrix, w::AbstractMatrix, params) where {ST}
    for r in axes(u, 1)
        d = derivatives(density, view(u, r, :), view(v, r, :), view(w, r, :), params)
        for c in axes(u, 2)
            load[r, c] = d.v[c]
        end
    end
    return nothing
end
