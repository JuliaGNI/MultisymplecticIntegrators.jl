# The renderer: glob `runs/`, draw every archive as the kind it declares, and write the figure under
# `results/`. This is the only place a figure is saved.
#
#   julia --project=scripts scripts/figures.jl                    # every archive
#   julia --project=scripts scripts/figures.jl wave-spline-h0.2   # one archive
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
        # The extension picks the figure from the archive's kind and names it from its stem; a kind
        # it cannot draw is refused there rather than drawn from the wrong keys.
        drawn = try
            Diagnostics.figures(data)
        catch exception
            failures += 1
            report(stem, "FAILED: $(failure_message(exception))")
            continue
        end
        for (name, figure) in drawn
            path = joinpath(RESULTS_DIR[], name * ".pdf")
            save(path, figure)
            report_path("figure", path)
        end
    end
    return failures == 0 ? 0 : 1
end

exit(main(ARGS))
