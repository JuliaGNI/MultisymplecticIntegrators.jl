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
    space_quadrature_weights(ansatz)

The weights of the ansatz's space quadrature, which a [`SymbolicAnsatz`](@ref) takes from the
trace basis that owns its rule.
"""
space_quadrature_weights(ansatz::SymbolicAnsatz) = space_quadrature_weights(ansatz.trace)

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
trace, and no other value of the run carries the velocity and the slope at the quadrature nodes. A
trace whose `u`, `v` and `w` hold different numbers of entries, or whose length is not a whole
number of the problem's components, raises one as well rather than being read at the nodes it
happens to hold.
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
    # `Val(problem.D)` is built once for the run and carried into the quadrature behind one function
    # barrier, so the `D` values of a node are a tuple whose length is a compile-time constant. The
    # barrier is given the density and the parameters rather than the problem: a call whose `Val` is
    # built from a runtime `Int` is dynamic, and boxing the problem in it costs more than it saves.
    step_energies!(
        energies, problem.density, problem.params, weights, traces, Val(problem.D))
    return relative_deviation(energies)
end

"""
    energy_error(problem, ansatz::SymbolicAnsatz, traces)

The relative energy deviation of a symbolic run, read from the traces through the trace basis'
space rule, which is the rule the trace is stored on.
"""
function energy_error(problem::LagrangianPDEProblem, ansatz::SymbolicAnsatz, traces)
    energy_error(problem, ansatz.trace, traces)
end

# The deviation of an energy series from its first entry: relative where that entry is a scale, and
# absolute where it vanishes, so that an all-zero first trace still has a series.
function relative_deviation(energies)
    reference = first(energies)
    return iszero(reference) ? abs.(energies .- reference) :
           abs.((energies .- reference) ./ reference)
end

"""
    energy_error(problem, method::BoxScheme, traces)

The relative deviation of a box scheme run's energy from its value at the first stored step, one
value per stored step and the first of which is zero.

The deviation is the ansatz method's rule, and the quadrature is the method's own: a box scheme's
trace is stored at the nodes of the output grid, and its energy is [`box_energies`](@ref)' series —
the density's Hamiltonian at the mean of each box's two nodes, summed over the boxes with the boxes'
widths as weights, which is the midpoint rule [`box_quadrature_weights`](@ref) gives. The energy of
the first step is therefore the closed form's energy at that step up to the scheme's own error,
which is second order in the time step and in the space step.
"""
function energy_error(problem::LagrangianPDEProblem, method::BoxScheme, traces)
    return relative_deviation(box_energies(problem, traces))
end

"""
    error(problem, method, result) -> Union{Vector, Nothing}

The error of every stored step of a run against the problem's exact solution, as the largest
absolute deviation over the components and the nodes of the output grid, and `nothing` where the
problem has none.

`result` is what [`integrate`](@ref) returns, or a record of a run that holds its `solution` and its
`traces`: the field is read from the solution because that is where every method writes it, on the
output grid the problem fixes, and the steps counted are the ones the run stored. A run that stopped
early ends at its last stored step, and the steps of the grid that it never computed are not read.

This is the error against the continuum and not against another run of the code under test: the
closed form is the problem's own. A problem without one is measured against a finer run of the same
method instead, which the matrix's runner does.
"""
function error(problem::LagrangianPDEProblem{T}, method::SpaceTimeMethod, result) where {T}
    problem.solutions === nothing && return nothing
    solution = result.solution
    nodes = collect(xgrid(problem))
    steps = min(ntime(solution), length(result.traces))
    errors = Vector{T}(undef, steps)
    for n in 1:steps
        closed = problem.solutions.u(solution.t[n], nodes, problem.params)
        state = solution.u[n]
        worst = zero(T)
        for i in eachindex(state)
            worst = max(worst, abs(state[i] - closed[i]))
        end
        errors[n] = worst
    end
    return errors
end

"""
    noether_charge(problem, method, traces) -> Union{Vector, Nothing}

The rotation charge of every stored step, `J = ∫ (u₁ v₂ − u₂ v₁) dx` integrated by the method's own
rule — at the nodes the trace of a slab method is stored on, and at the mean of each box's two nodes
for the box scheme — and `nothing` for a density without the symmetry.

The charge is conserved by a density that a rotation of the field's two components leaves
invariant, and the package's `VectorWave` is the problem with it: its density is the sum of the two
components' `½v² − ½c²w²`, a function of the lengths of `u`, `v` and `w` alone. Whether the symmetry
holds is decided on the density's values, never on the component count: `LinearTransport`'s doubled
field has two components and is not invariant, because its density reads one component against the
other's velocity and slope.

The trace of a step exists only at the top of the slab it solved, so the first entry is the charge
of the first stored step and not of the initial time.
"""
function noether_charge(problem::LagrangianPDEProblem{T}, method::SpaceTimeMethod,
        traces) where {T}
    has_rotation_symmetry(problem) || return nothing
    weights = trace_quadrature_weights(problem, method)
    nq = trace_nodes_per_component(method, weights)
    charges = Vector{T}(undef, length(traces))
    for (n, trace) in enumerate(traces)
        trace === nothing &&
            throw(ArgumentError("the run stored no trace at step $n; the charge of a step needs " *
                                "the trace `(u, v, w)` at the nodes of the space quadrature"))
        length(trace.u) == 2nq ||
            throw(ArgumentError("the trace of step $n holds $(length(trace.u)) entries of `u`, " *
                                "and the charge of two components needs $(2nq)"))
        charges[n] = T(trace_charge(method, weights, trace, nq))
    end
    return charges
end

# The number of nodes of one component of a trace, which the method's quadrature is read on: one
# node per weight for a slab method, and the ends of the boxes for the box scheme.
trace_nodes_per_component(::SpaceTimeMethod, weights) = length(weights)

trace_nodes_per_component(::BoxScheme, weights) = length(weights) + 1

# The rotation charge of one stored trace, `u₁ v₂ − u₂ v₁` integrated by the method's own rule: a
# slab method reads the trace at its nodes, one weight each.
function trace_charge(::SpaceTimeMethod, weights::AbstractVector, trace, nq::Integer)
    charge = zero(eltype(trace.u))
    for j in 1:nq
        charge += weights[j] * (trace.u[j] * trace.v[nq + j] - trace.u[nq + j] * trace.v[j])
    end
    return charge
end

# The box scheme's charge takes the rule its energy takes: one weight per box, and the rotation
# density at the mean of the box's two nodes.
function trace_charge(::BoxScheme, weights::AbstractVector, trace, nq::Integer)
    charge = zero(eltype(trace.u))
    for j in 1:(nq - 1)
        charge += weights[j] * (box_mean(trace.u[j], trace.u[j + 1]) *
                   box_mean(trace.v[nq + j], trace.v[nq + j + 1]) -
                   box_mean(trace.u[nq + j], trace.u[nq + j + 1]) *
                   box_mean(trace.v[j], trace.v[j + 1]))
    end
    return charge
end

# The weights of the rule a method's charge is integrated by: the ansatz's space quadrature for a
# slab method, one weight per node of the trace, and the box scheme's midpoint rule, one weight per
# box.
function trace_quadrature_weights(problem::LagrangianPDEProblem, method::SpaceTimeGalerkin)
    return space_quadrature_weights(ansatz(method))
end

function trace_quadrature_weights(problem::LagrangianPDEProblem, ::BoxScheme)
    box_quadrature_weights(problem)
end

"""
    has_rotation_symmetry(problem) -> Bool

Whether the density of a two-component problem is invariant under a rotation of the field's
components, which is the symmetry [`noether_charge`](@ref) is conserved by.

The test is on the density's values: the density at a probe point and at the same point with `u`,
`v` and `w` rotated by `π/4` must agree. A density of one component at a time is summed over the two
components, as the energy quadrature applies it.
"""
function has_rotation_symmetry(problem::LagrangianPDEProblem{T}) where {T}
    components(problem) == 2 || return false
    density = problem.density
    params = problem.params
    u = (T(7) / 10, T(3) / 10)
    v = (T(1) / 2, T(-9) / 10)
    w = (T(-2) / 5, T(3) / 5)
    angle = T(π) / 4
    cosine = cos(angle)
    sine = sin(angle)
    rotate(x) = (cosine * x[1] - sine * x[2], sine * x[1] + cosine * x[2])
    value = point_density(density, u, v, w, params)
    turned = point_density(density, rotate(u), rotate(v), rotate(w), params)
    return abs(turned - value) ≤ 64 * eps(T) * max(one(T), abs(value))
end

# The density of one point of a two-component field: each component's density for a density of one
# component at a time, and the coupled value for a density that takes both components at once.
function point_density(density::LagrangianDensity, u, v, w, params)
    return density(u[1], v[1], w[1], params) + density(u[2], v[2], w[2], params)
end

point_density(density, u, v, w, params) = density(u, v, w, params)

"""
    benchmark_panels(runs) -> (; cost, time)

The data of the matrix's two comparison panels, from the archives of the runs: one series per
method, of the archived cost and of the archived wall time against the archived error.

A series is a named tuple `(label, x, y)`: `label` the method's name, `x` the value the panel's own
axis carries and `y` the error of the run's last stored step. The cost panel pairs the error with
the number of scalar unknowns one of the run's solves handled, and the time panel pairs it with the
run's measured wall time. The two are never exchanged, which is the whole point of drawing both.

A run whose error is not finite has no point on a logarithmic axis, and it is left out of both
series rather than drawn as a gap: the tables carry it, with the status that says why it has no
error. A run with no stored step is left out with it, and neither makes the panel refuse to be
drawn.
"""
function benchmark_panels(runs)
    ordered = sort(collect(runs); by = run -> string(get(run, "stem", "")))
    cost = NamedTuple{(:label, :x, :y), Tuple{String, Float64, Float64}}[]
    time = copy(cost)
    for run in ordered
        errors = get(run, "error", Float64[])
        isempty(errors) && continue
        value = last(errors)
        isfinite(value) || continue
        rounds = get(run, "cost", Float64[])
        isempty(rounds) && continue
        seconds = get(run, "wall_time", NaN)
        label = string(get(run, "method", ""))
        push!(cost, (label = label, x = Float64(last(rounds)), y = Float64(value)))
        isfinite(seconds) &&
            push!(time, (label = label, x = Float64(seconds), y = Float64(value)))
    end
    return (; cost = group_series(cost), time = group_series(time))
end

# One series per method, in the order the methods first appear in the sorted archives.
function group_series(points)
    labels = String[]
    for point in points
        point.label in labels || push!(labels, point.label)
    end
    return [(label = label,
                x = [point.x for point in points if point.label == label],
                y = [point.y for point in points if point.label == label])
            for label in labels]
end

# The energy of every trace of a run, one `step_energy` each.
function step_energies!(energies::AbstractVector, density, params, weights::AbstractVector,
        traces, ::Val{D}) where {D}
    for (n, trace) in enumerate(traces)
        energies[n] = step_energy(density, params, weights, trace, n, Val(D))
    end
    return energies
end

# The energy of one trace, as the quadrature sum of the energy density at its nodes, one weight per
# node.
function step_energy(density, params, weights::AbstractVector, trace, n::Integer,
        ::Val{D}) where {D}
    nq = trace_node_count(trace, n, Val(D))
    length(weights) == nq ||
        throw(ArgumentError("the trace holds $(nq) nodes per component and the quadrature " *
                            "$(length(weights))"))
    energy = zero(eltype(trace.u))
    for j in 1:nq
        energy += weights[j] * node_energy_density(density, trace, j, nq, Val(D), params)
    end
    return energy
end

# The number of nodes per component of one stored trace. A trace with `D` components is
# component-major, one block of `nq` entries per component, and the same number of entries in `u`,
# `v` and `w`: a trace of another shape is refused before it is read, because a floored node count
# would drop the surplus entries in silence.
function trace_node_count(trace, n::Integer, ::Val{D}) where {D}
    trace === nothing &&
        throw(ArgumentError("the run stored no trace at step $n; the energy of a step needs " *
                            "the trace `(u, v, w)` at the nodes of the space quadrature"))
    entries = length(trace.u)
    entries == length(trace.v) == length(trace.w) ||
        throw(ArgumentError("the trace of step $n holds $(entries), $(length(trace.v)) and " *
                            "$(length(trace.w)) entries of `u`, `v` and `w`; a trace holds the " *
                            "same number of each"))
    nq, remainder = divrem(entries, D)
    iszero(remainder) ||
        throw(ArgumentError("the trace of step $n holds $(entries) entries, which is not a " *
                            "whole number of the problem's $(D) components"))
    return nq
end

# A density of one component at a time is applied to each component of the node, which is the split
# the slab's own residual makes between a `LagrangianDensity` and a density that couples the
# components of the field.
@inline function node_energy_density(density::LagrangianDensity, trace, j::Integer,
        nq::Integer, ::Val{D}, params) where {D}
    energy = zero(eltype(trace.u))
    for d in 1:D
        i = (d - 1) * nq + j
        energy += energy_density(density, trace.u[i], trace.v[i], trace.w[i], params)
    end
    return energy
end

# A density that takes the `D` components of one point at once gets them as a tuple, and its own
# derivative with respect to `v` is the tuple that closes the inner product.
@inline function node_energy_density(density, trace, j::Integer, nq::Integer, ::Val{D},
        params) where {D}
    return energy_density(density,
        node_values(trace.u, j, nq, Val(D)),
        node_values(trace.v, j, nq, Val(D)),
        node_values(trace.w, j, nq, Val(D)), params)
end

# One component's value at node `j`: the entry itself for a scalar field, and the `D` components of
# that node for a vector one, as a tuple rather than a heap `Vector` — the density reads the `D`
# values of one point and holds none of them, so a container per node is an allocation the
# quadrature does not need. `D` arrives as a `Val` so that the tuple's length is the same constant
# the caller was compiled for.
@inline function node_values(values::AbstractVector, j::Integer, nq::Integer,
        ::Val{D}) where {D}
    D === 1 && return values[j]
    return ntuple(d -> values[(d - 1) * nq + j], Val(D))
end

"""
    box_energies(problem, traces) -> Vector

The energy of every stored step of a box scheme run: the density's Hamiltonian read at the mean of
each box's two nodes and summed over the boxes with the boxes' widths as weights,
`Σ Δx · H(½(zⱼ + zⱼ₊₁))`, which is the midpoint rule.

This is the series [`energy_error`](@ref)'s box method reports the deviation of, and it is the one
value of the run a caller can hold against the density's closed form: at the initial level the trace
is the problem's own initial data, so the sum there is the Hamiltonian at `t₀` and not a
discretisation of it.
"""
function box_energies(problem::LagrangianPDEProblem{T}, traces) where {T}
    isempty(traces) &&
        throw(ArgumentError("the run stored no trace, so it has no energy; `integrate` returns " *
                            "an empty `traces` for a run that took no step"))
    energies = Vector{T}(undef, length(traces))
    box_step_energies!(energies, problem.density, problem.params,
        box_quadrature_weights(problem), traces, Val(problem.D))
    return energies
end

# The energy of every trace of a run under the midpoint rule over the boxes of the output grid.
function box_step_energies!(
        energies::AbstractVector, density, params, weights::AbstractVector,
        traces, ::Val{D}) where {D}
    for (n, trace) in enumerate(traces)
        energies[n] = box_step_energy(density, params, weights, trace, n, Val(D))
    end
    return energies
end

# The energy of one trace under the midpoint rule: one weight per box, and the density at the mean of
# the box's two nodes, which is a state the nodes themselves do not hold on a density that is not
# linear in `(u, v, w)`.
function box_step_energy(density, params, weights::AbstractVector, trace, n::Integer,
        ::Val{D}) where {D}
    nq = trace_node_count(trace, n, Val(D))
    length(weights) == nq - 1 ||
        throw(ArgumentError("the trace holds $(nq) nodes per component and the box quadrature " *
                            "$(length(weights)) boxes"))
    energy = zero(eltype(trace.u))
    for j in 1:(nq - 1)
        energy += weights[j] * box_energy_density(density, trace, j, nq, Val(D), params)
    end
    return energy
end

# A density of one component at a time is applied to each component of the box's mean state, as the
# node rule applies it to each component of a node.
@inline function box_energy_density(density::LagrangianDensity, trace, j::Integer,
        nq::Integer, ::Val{D}, params) where {D}
    energy = zero(eltype(trace.u))
    for d in 1:D
        i = (d - 1) * nq + j
        energy += energy_density(density,
            box_mean(trace.u[i], trace.u[i + 1]), box_mean(trace.v[i], trace.v[i + 1]),
            box_mean(trace.w[i], trace.w[i + 1]), params)
    end
    return energy
end

# A density that takes the `D` components of one point at once gets them as a tuple, as the node rule
# hands them to it.
@inline function box_energy_density(density, trace, j::Integer, nq::Integer, ::Val{D},
        params) where {D}
    return energy_density(density,
        box_node_values(trace.u, j, nq, Val(D)),
        box_node_values(trace.v, j, nq, Val(D)),
        box_node_values(trace.w, j, nq, Val(D)), params)
end

# One component's value at the mean of the two nodes of box `j`: the mean itself for a scalar field,
# and the `D` means as a tuple for a vector one, for the reason `node_values` gives.
@inline function box_node_values(values::AbstractVector, j::Integer, nq::Integer,
        ::Val{D}) where {D}
    D === 1 && return box_mean(values[j], values[j + 1])
    return ntuple(d -> box_mean(values[(d - 1) * nq + j], values[(d - 1) * nq + j + 1]), Val(D))
end

# The mean of two node values: the midpoint rule's point, which is no node of the grid.
@inline box_mean(a, b) = (a + b) / 2

"""
    solution_data(problem, result) -> (t, x, u)

The solution of a run as the plain values a figure is drawn from: the step times including the
problem's initial time, the output grid, and one matrix per component of the field, of size
`length(t) × length(x)`. A run that `integrate` stopped early ends at its last stored trace.

`result` is what [`integrate`](@ref) returns. A figure needs vectors and not a solution object: a
solution is tied to the version of `GeometricSolutions` that built it, so an archived run could not
be drawn without pinning that version, while these three values survive a round trip through a file.
"""
function solution_data(problem::LagrangianPDEProblem{T}, result) where {T}
    solution = result.solution
    D = problem.D
    x = collect(xgrid(problem))
    nx = length(x)
    steps = min(ntime(solution), length(result.traces))
    t = T[solution.t[n] for n in 0:steps]
    # A step's state is component-major: the `nx` entries of component `d` follow those of `d - 1`.
    u = [T[solution.u[n][(d - 1) * nx + j] for n in 0:steps, j in 1:nx] for d in 1:D]
    return t, x, u
end
