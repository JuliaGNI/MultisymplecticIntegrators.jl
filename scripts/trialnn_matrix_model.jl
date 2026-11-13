# The TrialNN family as the matrix's rows: the network ansatz under the hard-constraint lifting, on
# the wave and the sine-Gordon equation, at three time steps each.
#
# The driver includes this file through its `*_model.jl` glob, before `trialnn_model.jl`, so the
# rows are plain values and the method is built in a function body by the TrialNN family's own
# `trialnn_method`, which exists once every file has loaded. The problem is the matrix's
# `matrix_problem` of `experiments.jl`, because the family's `trialnn_problem` reads an `equation`
# field and not the row's `problem`.

using MultisymplecticIntegrators

"""The problems of the family, as the `problem` of a row."""
const TRIALNN_PROBLEMS = (:wave, :sine_gordon)

"""The time steps of one problem, as the `resolution` of a row."""
const TRIALNN_STEPS = (0.1, 0.05, 0.025)

"""
The TrialNN rows: every problem at every time step, 6 in all. A row's `parameters` are those of the
TrialNN family's entries: the network's width and the trace basis' degree and cells.

A stem carries `matrix`, because the run family's `wave-trialnn-h0.05` already names an archive of
the same problem, method and time step.

A row's `solver` is `f_abstol = 0` and the network family's relative tolerance,
`network_f_reltol(Float64)` of `test/helpers/testsetup.jl`: the TrialNN ansatz is that family's
`NetworkAnsatz`, and the TrialNN family measured no tolerance of its own (K58 in `KNOWN_ISSUES.md`).
"""
const TRIALNN_MATRIX = [(stem = "$(problem)-trialnn-matrix-h$(timestep)",
                            label = "$(problem), trialnn, h = $(timestep)",
                            method = :trialnn, problem,
                            problem_label = matrix_problem_label(problem),
                            tag = :matrix, resolution = (timestep = timestep,),
                            parameters = (S = 10, degree = 3, ncells = (2, 5)),
                            solver = (f_abstol = 0.0, f_reltol = 1e-4))
                        for problem in TRIALNN_PROBLEMS, timestep in TRIALNN_STEPS]

"""
    trialnn_matrix_problem(entry)

The problem of a row: the wave or the sine-Gordon equation, on the row's time step and the matrix's
spans and space step.
"""
function trialnn_matrix_problem(entry)
    matrix_problem("TrialNN family", entry.problem, TRIALNN_PROBLEMS;
        timestep = entry.resolution.timestep)
end

"""
    trialnn_matrix_method(entry)

The method of a row: the TrialNN family's ansatz, coupling and guess, from the row's parameters.
"""
function trialnn_matrix_method(entry)
    trialnn_method(merge(entry.parameters, (xspan = MATRIX_XSPAN,)))
end

"""
    trialnn_matrix_benchmark(entry) -> Dict{String, Any}

The registry's `benchmark` for the TrialNN family: build the row's problem and method, run it, and
return its record for the archive of the `"benchmark"` kind.
"""
function trialnn_matrix_benchmark(entry)
    benchmark_run(trialnn_matrix_problem(entry), trialnn_matrix_method(entry), entry)
end

register_family!(:trialnn_matrix, TRIALNN_MATRIX; benchmark = trialnn_matrix_benchmark)
