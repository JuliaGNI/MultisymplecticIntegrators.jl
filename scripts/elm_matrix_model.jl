# The ELM family as the matrix's rows: the network ansatz in `:last_layer` mode with a trained hidden
# layer, on the wave and the sine-Gordon equation, at three time steps each.
#
# The driver includes this file through its `*_model.jl` glob, before `elm_model.jl`, so the rows
# are plain values and the method is built in a function body by the ELM family's own `elm_method`,
# which exists once every file has loaded. The problem is this file's own, because the family's
# `elm_problem` builds `Wave` alone.

using MultisymplecticIntegrators

"""The problems of the family, as the `problem` of a row, and their display names."""
const ELM_PROBLEMS = ((problem = :wave, label = "Wave"),
    (problem = :sine_gordon, label = "SineGordon"))

"""The time steps of one problem, as the `resolution` of a row."""
const ELM_STEPS = (0.1, 0.05, 0.025)

const ELM_TIMESPAN = (0.0, 0.2)
const ELM_XSPAN = (0.0, 1.0)
const ELM_XSTEP = 0.05

"""
The ELM rows: every problem at every time step, 6 in all. A row's `parameters` are those of the ELM
family's entries: the network's width, the epochs and the learning rate of the trained fit, and the
trace basis' degree and cells.

A stem carries `matrix`, because the run family's `wave-elm-h0.05` and `wave-elm-h0.025` already
name two archives of the same problem, method and time step.

A row's `solver` is `f_abstol = 0` and the relative tolerance that the ELM family measured on its
run of `Wave`, `trained_f_reltol(Float64)` of `test/helpers/testsetup.jl`.
"""
const ELM_MATRIX = [(stem = "$(held.problem)-elm-matrix-h$(timestep)",
                        label = "$(held.problem), elm, h = $(timestep)",
                        method = :elm, problem = held.problem, problem_label = held.label,
                        tag = :matrix, resolution = (timestep = timestep,),
                        parameters = (
                            S = 20, epochs = 20, learning_rate = 1e-5, degree = 3,
                            ncells = (2, 5)),
                        solver = (f_abstol = 0.0, f_reltol = 1e-2))
                    for held in ELM_PROBLEMS, timestep in ELM_STEPS]

"""
    elm_matrix_problem(entry)

The problem of a row: the wave or the sine-Gordon equation, on the row's time step and the family's
space step.
"""
function elm_matrix_problem(entry)
    build = entry.problem === :wave ? Wave.problem :
            entry.problem === :sine_gordon ? SineGordon.problem :
            throw(ArgumentError("the ELM family has no problem `$(entry.problem)`; it takes " *
                                ":wave and :sine_gordon"))
    return build(timespan = ELM_TIMESPAN, timestep = entry.resolution.timestep,
        xspan = ELM_XSPAN, xstep = ELM_XSTEP)
end

"""
    elm_matrix_method(entry)

The method of a row: the ELM family's ansatz, guess and settings, from the row's parameters.
"""
elm_matrix_method(entry) = elm_method(merge(entry.parameters, (xspan = ELM_XSPAN,)))

"""
    elm_matrix_benchmark(entry) -> Dict{String, Any}

The registry's `benchmark` for the ELM family: build the row's problem and method, run it, and
return its record for the archive of the `"benchmark"` kind.
"""
function elm_matrix_benchmark(entry)
    benchmark_run(elm_matrix_problem(entry), elm_matrix_method(entry), entry)
end

register_family!(:elm_matrix, ELM_MATRIX; benchmark = elm_matrix_benchmark)
