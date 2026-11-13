# The regularisation scan of the multiplier network: the network family's rows on `Wave` at one time
# step, both modes, each solved at four values of the solver's `regularization_factor`.
#
# The driver includes this file through its `*_model.jl` glob, after `network_matrix_model.jl`, so
# the network family's `NETWORK_METHODS` exists when the rows are built, and a row's problem and
# method are that family's `network_matrix_problem` and `network_matrix_method`. It builds nothing
# and runs nothing at top level.

using MultisymplecticIntegrators

"""
The values of the solver's `regularization_factor` that the scan runs, as the `λ` of a row.
SimpleSolvers' `Newton` adds `λ` to the diagonal of the whole slab Jacobian before each
factorisation, unscaled; `0.0` is the network family's own solve.
"""
const NETWORK_REGULARISATION_FACTORS = (0.0, 1e-8, 1e-5, 1e-2)

"""
The scan's rows: both modes of the network on `Wave` at `h = 0.05`, at every value of
`NETWORK_REGULARISATION_FACTORS`, 8 in all. A row's `parameters` are the network family's, with the
row's `regularization_factor` beside them.

A row's `solver` is `f_abstol = 0`, the relative tolerance that the network family measured on its
affine control, `1e-4` of `network_f_reltol(Float64)` in `test/helpers/testsetup.jl`, and the row's
`regularization_factor`.
"""
const NETWORK_REGULARISATION = [(stem = "wave-$(network.method)-h0.05-lambda$(λ)",
                                    label = "wave, $(network.method), h = 0.05, λ = $(λ)",
                                    method = network.method, problem = :wave,
                                    problem_label = matrix_problem_label(:wave),
                                    tag = :matrix, resolution = (timestep = 0.05,),
                                    parameters = (S = 10, mode = network.mode, degree = 3,
                                        ncells = (2, 5), D = 1, regularization_factor = λ),
                                    solver = (f_abstol = 0.0, f_reltol = 1e-4,
                                        regularization_factor = λ))
                                for network in NETWORK_METHODS,
λ in NETWORK_REGULARISATION_FACTORS]

"""
    network_regularisation_benchmark(entry) -> Dict{String, Any}

The registry's `benchmark` for the scan: build the row's problem and method as the network family
does, run it with the row's solver keywords, and return its record for the archive of the
`"benchmark"` kind.
"""
function network_regularisation_benchmark(entry)
    benchmark_run(network_matrix_problem(entry), network_matrix_method(entry), entry)
end

register_family!(:network_regularisation, NETWORK_REGULARISATION;
    benchmark = network_regularisation_benchmark)
