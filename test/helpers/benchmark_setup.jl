# The shared setup of the benchmark test files `test/integration/benchmark*.jl`: the scripts and the
# helpers they include, the modules of the rows whose child ends on a code or a signal of its own,
# and the rows, the problems and the differences the files build. Each file includes it after its
# own `using` lines.

# The scripts are no part of the package, so the whole harness is tested through its files. The
# reference records are included before the runner, which loads them itself only when its caller has
# not, and the driver then includes the archive layer, the registry, the matrix's runner and every
# model file in the order it includes them itself.
include(joinpath(@__DIR__, "..", "helpers", "reference_values.jl"))
include(joinpath(@__DIR__, "..", "..", "scripts", "run_benchmark.jl"))
include(joinpath(@__DIR__, "..", "helpers", "problem_generator.jl"))
include(joinpath(@__DIR__, "..", "helpers", "testsetup.jl"))

# The module a copy of the driver is included into, for the one row whose child process ends on an
# exit code the tree's own registry cannot produce: the parent reads the child's code, so the row
# that ends on a code of its own needs a registry, and therefore a copy, of its own.
module ExitingRow end

# The same, for the row whose child the operating system kills: the parent reads the signal, and the
# arm does not exist on Windows.
module SignalledRow end

# One row's metadata, which a family's rows carry and a test builds for a run of its own. Its
# `problem_label` is the display name the families' rows carry, from the registry's
# `matrix_problem_label`.
function row(stem, method, problem, tag, resolution; parameters = (k = 1,), kwargs...)
    merge(
        (stem = stem, label = stem, method = method, problem = problem,
            problem_label = matrix_problem_label(problem),
            tag = tag, resolution = resolution, parameters = parameters),
        (; kwargs...))
end

# A problem whose density is the same but whose runs have no exact solution: the matrix measures
# such a run against the finest run of its method instead.
function unsolved_problem(base, xstep; timestep = base.timestep)
    return LagrangianPDEProblem(base.density; D = base.D, ics_function = base.ics_function,
        bcs_function = base.bcs_function, timespan = base.timespan,
        timestep, xspan = base.xspan, xstep, params = base.params)
end

# The largest absolute difference of two archived runs' fields at the times the coarse run stored,
# on the nodes their grids share — computed here, from the archives, and not by the code under
# test. An archive's `"error"` holds one entry per stored step, so the comparison is per step too.
# `by_index` compares the two runs at the same step index instead, the reading the archive must not
# make once the resolutions refine the time step as well.
function archived_difference(coarse, fine; by_index = false)
    shared = [searchsortedfirst(fine["x"], node) for node in coarse["x"]]
    rows = by_index ? collect(1:(coarse["steps"] + 1)) :
           [searchsortedfirst(fine["t"], time) for time in coarse["t"]]
    return [maximum(abs(coarse["u"][d][n + 1, j] - fine["u"][d][rows[n + 1], k])
            for d in eachindex(coarse["u"]), (j, k) in enumerate(shared))
            for n in 1:coarse["steps"]]
end
