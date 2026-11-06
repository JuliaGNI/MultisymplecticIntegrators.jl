# The spline family, built: its registry entries, the problem and the method of one of them, the
# archive of a run, and the drift of a run against the exact solution.
#
# The driver includes this file through its `*_model.jl` glob, and the two scans and the profile
# script include it by name after the archive layer and the registry; it is the only place the
# family's constructor calls are written, so a change to the ansatz or the coupling of the family is
# one edit and not four.

using MultisymplecticIntegrators

"""
The spline family: the space-time Galerkin integrator on a tensor B-spline ansatz, for the wave
equation, at a ladder of time steps. An entry's `coupling` is `:weak`, which imposes the initial and
boundary data with multipliers, or `:restriction`, which fixes the bottom time row of each slab.

`stem` names the archive and the figure; `label` is what a report, a legend and a heading show.
"""
const SPLINE = (
    (stem = "wave-spline-h0.2", label = "spline, h = 0.2",
        timespan = (0.0, 1.0), timestep = 0.2, xspan = (0.0, 1.0), xstep = 0.05,
        degree = 2, ncells = (1, 10), n = 3, coupling = :restriction),
    (stem = "wave-spline-h0.1", label = "spline, h = 0.1",
        timespan = (0.0, 1.0), timestep = 0.1, xspan = (0.0, 1.0), xstep = 0.05,
        degree = 2, ncells = (1, 20), n = 3, coupling = :restriction),
    (stem = "wave-spline-h0.05", label = "spline, h = 0.05",
        timespan = (0.0, 1.0), timestep = 0.05, xspan = (0.0, 1.0), xstep = 0.05,
        degree = 2, ncells = (2, 20), n = 3, coupling = :restriction))

"""
    spline_problem(entry)

The wave problem of a registry entry: its time span, its time step and its space step.
"""
function spline_problem(entry)
    return Wave.problem(timespan = entry.timespan, timestep = entry.timestep,
        xspan = entry.xspan, xstep = entry.xstep)
end

"""
    spline_method(entry)

The space-time Galerkin method of a registry entry: the tensor B-spline ansatz at its degree, cell
count and quadrature order, and the coupling the entry names.

The two couplings the family runs are named; any other symbol raises an `ArgumentError` naming it,
because a mistyped coupling that runs as another one is a result about the wrong method.
"""
function spline_method(entry)
    ansatz = TensorBSplineAnsatz(degree = entry.degree, ncells = entry.ncells,
        xspan = entry.xspan, n = entry.n)
    coupling = entry.coupling === :weak ? WeakInitialAndBoundaryConditions() :
               entry.coupling === :restriction ? CoefficientRestriction() :
               throw(ArgumentError("unknown coupling `$(entry.coupling)`; the spline family " *
                                   "takes :weak or :restriction"))
    return SpaceTimeGalerkin(ansatz, coupling)
end

"""
    run_entry(entry) -> (problem, method, result, seconds)

Integrate one registry entry, and report how long the run took.
"""
function run_entry(entry)
    problem = spline_problem(entry)
    method = spline_method(entry)
    seconds = @elapsed result = integrate(problem, method)
    return problem, method, result, seconds
end

"""
    solution_archive(entry, problem, result) -> Dict{String, Any}

One run as the archive `figures.jl` draws: the solution on the output grid, under the keys of the
schema in `scripts/README.md`. `store_run!` adds the `"stem"` that names the archive and its figure.
"""
function solution_archive(entry, problem, result)
    t, x, u = solution_data(problem, result)
    return Dict{String, Any}(
        "kind" => "solution",
        "label" => entry.label,
        "problem_label" => "Wave",
        "timestep" => entry.timestep,
        "t" => t,
        "x" => x,
        "u" => u)
end

"""
    spline_run(entry) -> Dict{String, Any}

The registry's `run` for the spline family: integrate one entry, report the steps it took and the
largest relative energy error over them, and return its archive.

The energy error is the run's own series and not a key of the archive: it is a diagnostic of the
run, read once here, and a stored copy could only disagree with it.
"""
function spline_run(entry)
    problem, method, result, _ = run_entry(entry)
    report(entry.label, "$(length(result.traces)) steps")
    report_error("maximum energy error", maximum(energy_error(problem, method, result.traces)))
    return solution_archive(entry, problem, result)
end

"""
    phase_drift(problem, result)

The error of a run against the exact solution at the last step it stored, relative to the amplitude
of the exact field on the output grid.

The exact solution is a closed form of the problem, so this is measured against the continuum and
not against a second run of the code under test. The step is the last one the run stored and not
the last of the output grid: a run that stopped early never computed the later ones, and reading
them would report the distance between the solution and a step it does not have.

A run that stored no trace raises an `ArgumentError`, as `energy_error` does: no other value of the
run carries the field at a time.
"""
function phase_drift(problem, result)
    solution = result.solution
    k = length(result.traces)
    k > 0 ||
        throw(ArgumentError("the run stored no trace, so it has no drift to measure; `integrate` " *
                            "returns an empty `traces` for a run that took no step"))
    x = collect(xgrid(problem))
    exact = problem.exact_u(solution.t[k], x, problem.params)
    return maximum(abs.(solution.u[k] .- exact)) / maximum(abs.(exact))
end

register_family!(:spline, SPLINE; run = spline_run)
