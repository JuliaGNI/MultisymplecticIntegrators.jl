"""
    energy_density(density, u, v, w, params = density_params(density))

The Hamiltonian density of a Lagrangian density, `v ∂L/∂v − L`.

For a density `L(u, v, w)` whose field has velocity `v = ∂ₜu` and slope `w = ∂ₓu`, the quantity
conserved by the continuous problem is the integral of this density over space. It is read off the
density's own API — the partial derivative with respect to `v` and the value — so it costs no new
formula per problem, and it is the right notion for a density that couples the components of a
vector field as well, where `u`, `v` and `w` are the vectors of the `D` components at one point
and the product is their inner product.
"""
function energy_density(density, u, v, w, params = density_params(density))
    return dot(v, derivatives(density, u, v, w, params).v) - density(u, v, w, params)
end

"""
    space_quadrature_weights(ansatz)

The weights of the ansatz's space quadrature: the measure the trace `(u, v, w)` of a step is stored
on, and the one that integrates the energy density over `x`.
"""
function space_quadrature_weights(ansatz::TensorBSplineAnsatz)
    return quadrature_weights(quadratures(quadrature(ansatz))[2])
end

"""
    energy_error(problem, method, traces)
    energy_error(problem, ansatz, traces)

The relative deviation of a run's conserved energy from its value at the first stored step: one
value per stored step, the first of which is zero.

`traces` is the second value [`integrate`](@ref) returns. The energy of a step is
[`energy_density`](@ref) integrated over the space quadrature at the nodes the trace is stored on,
which is why the rule is reached through the ansatz rather than through the problem: a
[`LagrangianPDEProblem`](@ref) holds the space interval and not the rule on it. The deviation is
relative to the first step and not to the initial time, because the trace of a step exists only at
the top of the slab it solved, and no trace of `t₀` is stored.

A run whose first energy vanishes — an all-zero trace, for one — has no relative scale, and its
deviation is the absolute one.

A run that stored no trace raises an `ArgumentError`: the energy of a step is a function of its
trace, and no other value of the run carries the velocity and the slope at the quadrature nodes.
"""
function energy_error(problem::LagrangianPDEProblem, method::SpaceTimeGalerkin, traces)
    return energy_error(problem, ansatz(method), traces)
end

function energy_error(problem::LagrangianPDEProblem{T}, ansatz::TensorBSplineAnsatz,
        traces) where {T}
    isempty(traces) &&
        throw(ArgumentError("the run stored no trace, so it has no energy; `integrate` returns " *
                            "an empty `traces` for a run that took no step"))
    weights = space_quadrature_weights(ansatz)
    energies = Vector{T}(undef, length(traces))
    for (n, trace) in enumerate(traces)
        energies[n] = step_energy(problem, weights, trace, n)
    end
    reference = first(energies)
    return iszero(reference) ? abs.(energies .- reference) :
           abs.((energies .- reference) ./ reference)
end

# The energy of one trace, as the quadrature sum of the energy density at its nodes. A trace with
# `D` components is component-major, one block of `nq` entries per component.
function step_energy(problem::LagrangianPDEProblem{T}, weights::AbstractVector,
        trace, n::Integer) where {T}
    trace === nothing &&
        throw(ArgumentError("the run stored no trace at step $n; the energy of a step needs " *
                            "the trace `(u, v, w)` at the nodes of the space quadrature"))
    D = problem.D
    nq = length(trace.u) ÷ D
    length(weights) == nq ||
        throw(ArgumentError("the trace holds $(nq) nodes per component and the quadrature " *
                            "$(length(weights))"))
    density = problem.density
    params = problem.params
    energy = zero(T)
    for j in 1:nq
        energy += weights[j] * energy_density(density,
            node_values(trace.u, j, nq, D),
            node_values(trace.v, j, nq, D),
            node_values(trace.w, j, nq, D), params)
    end
    return energy
end

# One component's value at node `j`: the entry itself for a scalar field, and the `D` components of
# that node for a vector one.
@inline function node_values(values::AbstractVector{T}, j::Integer, nq::Integer,
        D::Integer) where {T}
    D === 1 && return values[j]
    return T[values[(d - 1) * nq + j] for d in 1:D]
end
