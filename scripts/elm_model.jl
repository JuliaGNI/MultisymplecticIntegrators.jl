# The ELM family, built: its registry entry, the problem, the ansatz and the method of one entry,
# and the archive of a run.
#
# The driver includes this file through its `*_model.jl` glob; it is the only place the family's
# constructor calls are written, so a change to the ansatz or the guess of the family is one edit
# and not four. The family is the network ansatz in `:last_layer` mode, whose hidden layer the
# initial guess trains and whose output layer the solver determines, on the weak
# initial-and-boundary coupling that a network takes.

using MultisymplecticIntegrators

"""
The ELM family: the space-time Galerkin integrator on a network ansatz in `:last_layer` mode, for
the wave equation. An entry's `S` is the number of hidden units, and `epochs` and `learning_rate`
are those of the [`TrainedHiddenLayer`](@ref) fit that seeds every slab.

`stem` names the archive and the figure; `label` is what a report, a legend and a heading show.
"""
const ELM = (
    (stem = "wave-elm-h0.05", label = "ELM, h = 0.05",
        timespan = (0.0, 0.25), timestep = 0.05, xspan = (0.0, 1.0), xstep = 0.05,
        degree = 3, ncells = (2, 5), S = 20, epochs = 20, learning_rate = 1e-5),
    (stem = "wave-elm-h0.025", label = "ELM, h = 0.025",
        timespan = (0.0, 0.25), timestep = 0.025, xspan = (0.0, 1.0), xstep = 0.05,
        degree = 3, ncells = (2, 5), S = 20, epochs = 20, learning_rate = 1e-5))

"""
    elm_problem(entry; T = Float64)

The wave problem of a registry entry: its time span, its time step and its space step.
"""
function elm_problem(entry; T::Type = Float64)
    return Wave.problem(
        T = T, timespan = map(T, entry.timespan), timestep = T(entry.timestep),
        xspan = map(T, entry.xspan), xstep = T(entry.xstep))
end

"""
    elm_ansatz(entry; T = Float64)

The ansatz of a registry entry: a network of `S` hidden units and one output in `:last_layer` mode,
on the trace basis of its degree and cells.
"""
function elm_ansatz(entry; T::Type = Float64)
    return NetworkAnsatz(S = entry.S, mode = :last_layer, xspan = map(T, entry.xspan),
        degree = entry.degree, ncells = entry.ncells)
end

"""
    elm_method(entry; T = Float64)

The method of a registry entry: its ansatz, the coupling a network takes, and an initial condition
fitted by a trained hidden layer.
"""
function elm_method(entry; T::Type = Float64)
    return SpaceTimeGalerkin(elm_ansatz(entry; T = T), WeakInitialAndBoundaryConditions();
        initial_guess = FitTo(InitialCondition();
            method = TrainedHiddenLayer(epochs = entry.epochs,
                learning_rate = T(entry.learning_rate))))
end

"""
    elm_run(entry; T = Float64) -> Dict{String, Any}

The registry's `run` for the ELM family: integrate one entry, report the steps it took, and return
its archive under the keys of the schema in `scripts/README.md`. `store_run!` adds the `"stem"`.
"""
function elm_run(entry; T::Type = Float64)
    problem = elm_problem(entry; T = T)
    method = elm_method(entry; T = T)
    seconds = @elapsed result = integrate(problem, method)
    report(entry.label,
        "$(length(result.traces)) steps in $(round(seconds; digits = 2)) s")
    t, x, u = solution_data(problem, result)
    return Dict{String, Any}(
        "kind" => "solution",
        "label" => entry.label,
        "problem_label" => "Wave",
        "timestep" => Float64(entry.timestep),
        "t" => t,
        "x" => x,
        "u" => u)
end

register_family!(:elm, ELM; run = elm_run)
