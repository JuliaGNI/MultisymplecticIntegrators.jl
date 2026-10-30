# The spline family, built: the problem and the method of a registry entry, the archive of a run,
# and the drift of a run against the exact solution.
#
# `include`d by the spline driver and by the two scans, each of which also loads the archive layer
# and the registry; this file is the only place the family's constructor calls are written, so a
# change to the ansatz or the coupling of the family is one edit and not four.

using MultisymplecticIntegrators
using GeometricBase: ntime

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
"""
function spline_method(entry)
    ansatz = TensorBSplineAnsatz(degree = entry.degree, ncells = entry.ncells,
        xspan = entry.xspan, n = entry.n)
    coupling = entry.coupling === :weak ? WeakInitialAndBoundaryConditions() :
               CoefficientRestriction()
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
    solution_archive(entry, problem, method, result) -> Dict{String, Any}

One run as the archive `figures.jl` draws: the solution on the output grid, and the relative energy
error over the stored steps.
"""
function solution_archive(entry, problem, method, result)
    t, x, u = solution_data(problem, result)
    energy = energy_error(problem, method, result.traces)
    return Dict{String, Any}(
        "kind" => "solution",
        "label" => entry.label,
        "problem_label" => "Wave",
        "timestep" => entry.timestep,
        "final_time" => entry.timespan[2],
        "dimension" => length(u),
        "t" => t,
        "x" => x,
        "u" => u,
        "energy_t" => t[(end - length(energy) + 1):end],
        "energy_error" => energy)
end

"""
    phase_drift(problem, result)

The error of a run against the exact solution at its final time, relative to the amplitude of the
exact field on the output grid.

The exact solution is a closed form of the problem, so this is measured against the continuum and
not against a second run of the code under test.
"""
function phase_drift(problem, result)
    solution = result.solution
    n = ntime(solution)
    x = collect(xgrid(problem))
    exact = problem.exact_u(solution.t[n], x, problem.params)
    return maximum(abs.(solution.u[n] .- exact)) / maximum(abs.(exact))
end
