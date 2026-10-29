# The energy scan: the relative energy error of the wave problem over time, at a ladder of time
# steps, as one archive of the `"energy"` kind — which is the comparison `figures.jl` draws.
#
#   julia --project=scripts scripts/spline_energy_scan.jl
#   julia --project=scripts scripts/spline_energy_scan.jl --steps 0.2,0.1
#
# The energy is the density's Hamiltonian at the nodes of the space quadrature, and the error is
# relative to the first stored step of each run (see `energy_error`).

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))
include(joinpath(@__DIR__, "spline_model.jl"))

function main(args)
    _, options = parse_arguments(args, ("--steps",))
    steps = option_steps(options, ENERGY_STEPS)
    template = smallest_entry(SPLINE)

    labels = String[]
    times = Vector{Vector{Float64}}()
    errors = Vector{Vector{Float64}}()
    banner("energy scan: $(length(steps)) step(s) into $(RUNS_DIR[])")
    for timestep in steps
        entry = merge(template, (
            timestep = timestep, stem = "wave-spline-energy-h$(timestep)"))
        problem, method, result, seconds = run_entry(entry)
        series = energy_error(problem, method, result.traces)
        push!(labels, "h = $(timestep)")
        push!(times, Float64[result.solution.t[n] for n in 1:length(series)])
        push!(errors, collect(Float64, series))
        report(labels[end], "$(length(series)) steps, $(round(seconds; digits = 2)) s")
        report_error("maximum energy error", maximum(series))
    end

    data = Dict{String, Any}(
        "kind" => "energy",
        "title" => "Energy error against time, wave equation, spline ansatz",
        "labels" => labels,
        "t" => times,
        "errors" => errors)
    report_path("archive", store_run!("wave-spline-energy", data))
    return 0
end

exit(main(ARGS))
