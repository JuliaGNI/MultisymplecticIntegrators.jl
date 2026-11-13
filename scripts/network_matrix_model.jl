# The multiplier network as the matrix's rows: the space-time Galerkin integrator on a network
# ansatz in both of its modes, under the weak initial-and-boundary conditions and continued from the
# previous slab, on the three example problems a network takes, at three time steps each.
#
# The driver includes this file through its `*_model.jl` glob. It builds nothing and runs nothing at
# top level, so `run_family.jl` includes it beside the figure families without paying for a solve.

using MultisymplecticIntegrators

"""
The two modes of the network, as the `method` of a row: every parameter free, and the output layer
alone free with the hidden layer held at its start.
"""
const NETWORK_METHODS = ((method = :network_all, mode = :all),
    (method = :network_last_layer, mode = :last_layer))

"""The problems of the family, as the `problem` of a row, and their display names."""
const NETWORK_PROBLEMS = ((problem = :wave, label = "Wave"),
    (problem = :sine_gordon, label = "SineGordon"),
    (problem = :vector_wave, label = "VectorWave"))

"""The time steps of one method and problem, as the `resolution` of a row."""
const NETWORK_STEPS = (0.1, 0.05, 0.025)

const NETWORK_TIMESPAN = (0.0, 0.2)
const NETWORK_XSPAN = (0.0, 1.0)
const NETWORK_XSTEP = 0.05

"""
The network's rows: both modes on every problem at every time step, 18 in all. A row's `parameters`
are the network's width, its mode, the trace basis' degree and cells, and the number of the field's
components, `2` on `VectorWave`.

A row's `solver` is `f_abstol = 0` and the relative tolerance that the network family measured on
its affine control, `network_f_reltol(Float64)` of `test/ansatz/network.jl`.
"""
const NETWORK_MATRIX = [(stem = "$(held.problem)-$(network.method)-h$(timestep)",
                            label = "$(held.problem), $(network.method), h = $(timestep)",
                            method = network.method, problem = held.problem,
                            problem_label = held.label, tag = :matrix,
                            resolution = (timestep = timestep,),
                            parameters = (S = 10, mode = network.mode, degree = 3,
                                ncells = (2, 5), D = held.problem === :vector_wave ? 2 : 1),
                            solver = (f_abstol = 0.0, f_reltol = 1e-4))
                        for network in NETWORK_METHODS, held in NETWORK_PROBLEMS,
timestep in NETWORK_STEPS]

"""
    network_matrix_problem(entry)

The problem of a row: the example problem it names, on the row's time step and the family's space
step.
"""
function network_matrix_problem(entry)
    build = entry.problem === :wave ? Wave.problem :
            entry.problem === :sine_gordon ? SineGordon.problem :
            entry.problem === :vector_wave ? VectorWave.problem :
            throw(ArgumentError("the network family has no problem `$(entry.problem)`; it " *
                                "takes :wave, :sine_gordon and :vector_wave"))
    return build(timespan = NETWORK_TIMESPAN, timestep = entry.resolution.timestep,
        xspan = NETWORK_XSPAN, xstep = NETWORK_XSTEP)
end

"""
    network_matrix_method(entry)

The method of a row: the network ansatz of the row's parameters under the weak initial-and-boundary
conditions, each slab started from the previous one.
"""
function network_matrix_method(entry)
    held = entry.parameters
    ansatz = NetworkAnsatz(S = held.S, mode = held.mode, xspan = NETWORK_XSPAN,
        degree = held.degree, ncells = held.ncells, D = held.D)
    return SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
        initial_guess = Continuation())
end

"""
    network_matrix_benchmark(entry) -> Dict{String, Any}

The registry's `benchmark` for the network family: build the row's problem and method, run it, and
return its record for the archive of the `"benchmark"` kind.
"""
function network_matrix_benchmark(entry)
    benchmark_run(network_matrix_problem(entry), network_matrix_method(entry), entry)
end

register_family!(:network_matrix, NETWORK_MATRIX; benchmark = network_matrix_benchmark)
