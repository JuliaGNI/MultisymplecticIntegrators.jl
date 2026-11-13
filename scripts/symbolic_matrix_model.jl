# The symbolic family as the matrix's rows: the closed-form ansatz of the standing wave of `Wave`, at
# the wave's period, over two periods, at three space resolutions of its trace basis.
#
# The driver includes this file through its `*_model.jl` glob, before `symbolic_model.jl`, so the
# rows are plain values and the problem and the method are built in function bodies by the symbolic
# family's own `symbolic_problem` and `symbolic_method`, which exist once every file has loaded.

using MultisymplecticIntegrators

"""
The space cells of the trace basis, one row each: the three space resolutions of the family.
"""
const SYMBOLIC_CELLS = ((4, 4), (4, 8), (4, 16))

"""
The symbolic rows: the standing wave at the time step `4.0`, which is its period `2l / c` with
`l = 1` and `c = 0.5`, at three space resolutions of the degree-2 trace basis.

The solver keywords are those of the symbolic family: `f_abstol = 0` and the relative tolerance
`1e-12` that `symbolic_f_reltol(Float64)` names, written as a literal because `symbolic_model.jl`
loads after this file.
"""
const SYMBOLIC_MATRIX = [(stem = "wave-symbolic-matrix-c$(ncells[2])",
                             label = "wave, symbolic, $(ncells[2]) space cells",
                             method = :symbolic, problem = :wave, problem_label = "Wave",
                             tag = :matrix,
                             resolution = (timestep = 4.0, ncells = ncells),
                             parameters = (degree = 2, ncells = ncells),
                             solver = (f_abstol = 0.0, f_reltol = 1e-12))
                         for ncells in SYMBOLIC_CELLS]

# The settings of one row in the shape the symbolic family's constructors read.
function symbolic_matrix_settings(entry)
    return (timestep = entry.resolution.timestep, xspan = (0.0, 1.0), xstep = 0.05,
        degree = entry.parameters.degree, ncells = entry.parameters.ncells)
end

"""
    symbolic_matrix_problem(entry)

The problem of a row: `Wave` over two periods at the row's time step, as the symbolic family
builds it.
"""
symbolic_matrix_problem(entry) = symbolic_problem(symbolic_matrix_settings(entry))

"""
    symbolic_matrix_method(entry)

The method of a row: the symbolic family's ansatz on the row's trace basis, under the coupling a
symbolic ansatz takes.
"""
symbolic_matrix_method(entry) = symbolic_method(symbolic_matrix_settings(entry))

"""
    symbolic_matrix_benchmark(entry) -> Dict{String, Any}

The registry's `benchmark` for the symbolic family: build the row's problem and method, run it, and
return its record for the archive of the `"benchmark"` kind.
"""
function symbolic_matrix_benchmark(entry)
    benchmark_run(symbolic_matrix_problem(entry), symbolic_matrix_method(entry), entry)
end

register_family!(:symbolic_matrix, SYMBOLIC_MATRIX; benchmark = symbolic_matrix_benchmark)
