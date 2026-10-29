# The renderer: glob `runs/`, draw every archive as the kind it says it is, and write the figure
# under `results/`. This is the only place a figure is saved.
#
#   julia --project=scripts scripts/figures.jl                    # every archive
#   julia --project=scripts scripts/figures.jl wave-spline-h0.1   # one archive
#   julia --project=scripts scripts/figures.jl --results-dir talk/figures
#
# Loading `CairoMakie` is what activates the plotting extension; the archive layer needs neither it
# nor the package, which is why a figure can be redrawn without a solve.

include(joinpath(@__DIR__, "archives.jl"))

using CairoMakie
using MultisymplecticIntegrators
import MultisymplecticIntegrators.Diagnostics as Diagnostics

function main(args)
    positional, _ = parse_arguments(args)
    mkpath(RESULTS_DIR[])
    runs = load_runs()
    isempty(positional) || (runs = filter(run -> get(run, "stem", "") in positional, runs))
    isempty(runs) && (println("figures.jl: no archive under $(RUNS_DIR[])"); return 0)

    banner("figures: $(length(runs)) archive(s) into $(RESULTS_DIR[])")
    failures = 0
    for data in runs
        stem = get(data, "stem", "")
        try
            # The kind decides the figure, and an archive of a kind this renderer cannot draw is
            # refused: a figure guessed at would be drawn from the wrong keys.
            kind = get(data, "kind", nothing)
            figure = if kind == "solution"
                Diagnostics.plot_solution(data)
            elseif kind == "energy"
                Diagnostics.plot_energy_error(data)
            else
                throw(ArgumentError("the archive `$(stem)` has kind `$(kind)`, and this renderer " *
                                    "draws \"solution\" and \"energy\""))
            end
            path = joinpath(RESULTS_DIR[], stem * ".pdf")
            save(path, figure)
            report_path("figure", path)
        catch exception
            failures += 1
            report(stem, "FAILED: $(failure_message(exception))")
        end
    end
    return failures == 0 ? 0 : 1
end

exit(main(ARGS))
