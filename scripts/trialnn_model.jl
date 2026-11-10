# The TrialNN family, built: its registry entry, the problem, the ansatz and the method of one
# entry, and the archive of a run.
#
# The driver includes this file through its `*_model.jl` glob, so this is the only place the
# family's constructor calls are written. The family is the network ansatz under the
# hard-constraint lifting coupling, which imposes the slab's data on the field by construction: the
# old `TrialNN_PDE_int` brought back, on the seam the other families share.
#
# `SimpleSplines` counts the degree where the old constructor took an order, so an entry's `degree`
# is one less than the `k` of the old record it reproduces.

using MultisymplecticIntegrators

"""
The TrialNN family: the space-time Galerkin integrator on a `NetworkAnsatz` under
[`HardConstraintLifting`](@ref), on the wave equation and on sine-Gordon. An entry's `S` is the
number of hidden units, and `degree` and `ncells` the resolution of the trace basis whose functions
the coupling's tables are built on.

`stem` names the archive and the figure; `label` is what a report, a legend and a heading show; and
`steps` is the number of steps the entry runs, which its `timespan` and `timestep` must agree with.
"""
const TRIALNN = (
    (stem = "wave-trialnn-h0.05", label = "TrialNN, Wave, h = 0.05",
        equation = :wave, timespan = (0.0, 0.25), timestep = 0.05, xspan = (0.0, 1.0),
        xstep = 0.05, degree = 3, ncells = (2, 5), S = 10, steps = 5),
    (stem = "sine-gordon-trialnn-h0.05", label = "TrialNN, sine-Gordon, h = 0.05",
        equation = :sine_gordon, timespan = (0.0, 0.15), timestep = 0.05, xspan = (
            0.0, 1.0),
        xstep = 0.05, degree = 3, ncells = (2, 5), S = 10, steps = 3))

"""
    trialnn_problem(entry; T = Float64)

The problem of a registry entry: the wave equation or sine-Gordon, at the entry's time span, time
step and space step.
"""
function trialnn_problem(entry; T::Type = Float64)
    span = map(T, entry.timespan)
    xspan = map(T, entry.xspan)
    entry.equation === :wave && return Wave.problem(T = T, timespan = span,
        timestep = T(entry.timestep), xspan = xspan, xstep = T(entry.xstep))
    entry.equation === :sine_gordon && return SineGordon.problem(T = T, timespan = span,
        timestep = T(entry.timestep), xspan = xspan, xstep = T(entry.xstep))
    throw(ArgumentError("the TrialNN family has no equation `$(entry.equation)`"))
end

"""
    trialnn_ansatz(entry; T = Float64)

The ansatz of a registry entry: a network of `S` hidden units and one output, whose trace basis is
the entry's degree and cells.
"""
function trialnn_ansatz(entry; T::Type = Float64)
    return NetworkAnsatz(S = entry.S, xspan = map(T, entry.xspan), degree = entry.degree,
        ncells = entry.ncells)
end

"""
    trialnn_method(entry; T = Float64)

The method of a registry entry: its ansatz, the hard-constraint lifting, and the previous slab's
field fitted by the greedy dictionary, which falls back to the initial condition at the first slab.
"""
function trialnn_method(entry; T::Type = Float64)
    return SpaceTimeGalerkin(trialnn_ansatz(entry; T = T), HardConstraintLifting();
        initial_guess = FitTo(PreviousSlab(); method = OrthogonalGreedy()))
end

"""
    trialnn_run(entry; T = Float64) -> Dict{String, Any}

The registry's `run` for the TrialNN family: integrate one entry, report the steps it took and what
its solve did, and return its archive under the keys of the schema in `scripts/README.md`.
`store_run!` adds the `"stem"`.
"""
function trialnn_run(entry; T::Type = Float64)
    problem = trialnn_problem(entry; T = T)
    method = trialnn_method(entry; T = T)
    seconds = @elapsed result = integrate(problem, method)
    stats = result.stats
    report(entry.label, "$(length(result.traces)) steps in $(round(seconds; digits = 2)) s")
    stats === nothing ||
        report(entry.label, "$(stats.iterations) iterations, converged $(stats.converged)")
    t, x, u = solution_data(problem, result)
    return Dict{String, Any}(
        "kind" => "solution",
        "label" => entry.label,
        "problem_label" => entry.equation === :wave ? "Wave" : "SineGordon",
        "timestep" => Float64(entry.timestep),
        "t" => t,
        "x" => x,
        "u" => u)
end

register_family!(:trialnn, TRIALNN; run = trialnn_run)
