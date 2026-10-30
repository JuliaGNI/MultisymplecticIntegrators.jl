# The phase-drift scan: the error of a run against the exact solution at its final time, at a
# ladder of ansatz resolutions, with the time step held fixed. It prints a table and writes nothing.
#
#   julia --project=scripts scripts/spline_phase_drift_scan.jl
#
# The exact solution is a closed form of the wave problem, so the drift is measured against the
# continuum. A run whose error stops falling as the resolution rises has hit something other than
# the space-time discretisation — a Newton tolerance, or a coupling that does not carry the datum.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))
include(joinpath(@__DIR__, "spline_model.jl"))

using Printf

function main(args)
    parse_arguments(args)
    template = smallest_entry(SPLINE)

    banner("phase drift: $(length(PHASE_ANSATZE)) resolution(s), h = $(template.timestep)")
    @printf("%6s %10s %3s %10s %10s %8s\n",
        "degree", "ncells", "n", "drift", "energy", "seconds")
    for (degree, ncells, n) in PHASE_ANSATZE
        entry = merge(template, (degree = degree, ncells = ncells, n = n))
        problem, method, result, seconds = run_entry(entry)
        drift = phase_drift(problem, result)
        energy = maximum(energy_error(problem, method, result.traces))
        @printf("%6d %10s %3d %10.3e %10.3e %8.2f\n",
            degree, string(ncells), n, drift, energy, seconds)
    end
    return 0
end

exit(main(ARGS))
