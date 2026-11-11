# The renderer: glob `runs/`, draw every archive as the kind it declares, and write the figure under
# `results/<stem>.pdf`. This is the only place a figure is saved.
#
#   julia --project=scripts scripts/figures.jl                    # every archive
#   julia --project=scripts scripts/figures.jl wave-spline-h0.2   # one archive
#   julia --project=scripts scripts/figures.jl --results-dir talk/figures
#
# Loading `CairoMakie` is what activates the plotting extension; the archive layer needs neither it
# nor the package, which is why a figure can be redrawn without a solve.
#
# A `"benchmark"` archive is the one kind the extension does not draw: the matrix's comparison is of
# several runs, and the extension draws one archive at a time. Its panels are drawn here, from the
# data `Diagnostics.benchmark_panels` gives, and one Markdown table per problem is written beside
# them. The panels are drawn per problem — the runs of one problem compared by method — and each
# archive's own PDF marks its own run among them, so that a figure per archive says what it is.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))
include(joinpath(@__DIR__, "benchmark.jl"))

using CairoMakie
using MultisymplecticIntegrators
import MultisymplecticIntegrators.Diagnostics as Diagnostics

"""
    benchmark_figure(problem, runs; own = nothing)

The two panels of one problem's comparison: the error of each run against its cost on the left, and
against its wall time on the right, both on a logarithmic scale and with one series per method.

`own` is the stem of the archive the figure is drawn for, and its point carries a marker of its own,
so that each archive's PDF says which of the points is its run. Nothing else here decides what the
figure looks like: the theme is the extension's, `Diagnostics.plot_theme`.
"""
function benchmark_figure(problem, runs; own = nothing)
    panels = benchmark_panels(runs)
    figure = Figure(; size = (1000, 420))
    with_theme(Diagnostics.plot_theme()) do
        panels_of = (
            (series = panels.cost, title = "error against cost",
                xlabel = "cost (scalar unknowns per solve)", field = "cost"),
            (series = panels.time, title = "error against time",
                xlabel = "wall time (s)", field = "wall_time"))
        for (position, panel) in enumerate(panels_of)
            ax = Axis(figure[1, position]; xscale = log10, yscale = log10,
                title = "$(problem): $(panel.title)", xlabel = panel.xlabel, ylabel = "error")
            for series in panel.series
                isempty(series.x) && continue
                scatterlines!(ax, series.x, series.y; label = series.label)
            end
            mine = only_mine(runs, own, panel.field)
            mine === nothing || scatter!(ax, [mine[1]], [mine[2]]; marker = :star5,
                markersize = 18, label = "this run")
            isempty(panel.series) || axislegend(ax)
        end
    end
    return figure
end

# The point of the archive the figure is drawn for, and `nothing` where it has none to show.
function only_mine(runs, own, field)
    own === nothing && return nothing
    for run in runs
        get(run, "stem", "") == own || continue
        errors = get(run, "error", Float64[])
        isempty(errors) && return nothing
        value = last(errors)
        isfinite(value) || return nothing
        abscissa = field == "cost" ?
                   (isempty(get(run, "cost", Float64[])) ? NaN : last(run["cost"])) :
                   get(run, "wall_time", NaN)
        isfinite(abscissa) || return nothing
        return (Float64(abscissa), Float64(value))
    end
    return nothing
end

function main(args)
    positional, _ = parse_arguments(args)
    mkpath(RESULTS_DIR[])
    runs, unreadable = load_runs()
    isempty(positional) || (runs = filter(run -> get(run, "stem", "") in positional, runs))
    # A named stem that matches no archive counts as a failure, so that a mistyped stem does not
    # pass for a figure that was drawn. So does an archive that cannot be read, whether it is named
    # or not: the other archives are still drawn.
    unmatched = setdiff(positional, (get(run, "stem", "") for run in runs))
    isempty(unmatched) ||
        println("figures.jl: no archive under $(RUNS_DIR[]) for $(join(unmatched, ", "))")
    failures = length(unmatched) + length(unreadable)
    if isempty(runs)
        isempty(positional) && println("figures.jl: no archive under $(RUNS_DIR[])")
        return failures == 0 ? 0 : 1
    end

    banner("figures: $(length(runs)) archive(s) into $(RESULTS_DIR[])")
    # The matrix's tables are one per problem, over every benchmark archive of that problem, so they
    # are written once, before the figures and from the whole selection.
    benchmarks = [run for run in runs if get(run, "kind", "") == "benchmark"]
    if !isempty(benchmarks)
        for (problem, text) in benchmark_tables(benchmarks)
            path = joinpath(RESULTS_DIR[], problem * ".md")
            atomic_write(path) do temporary
                write(temporary, text)
            end
            report_path("table", path)
        end
    end
    for data in runs
        stem = get(data, "stem", "")
        path = joinpath(RESULTS_DIR[], stem * ".pdf")
        drawn = try
            if get(data, "kind", "") == "benchmark"
                problem = string(get(data, "problem", ""))
                others = [run
                          for run in benchmarks
                          if string(get(run, "problem", "")) == problem]
                benchmark_figure(problem, others; own = stem)
            else
                # The extension picks the figure from the archive's kind; a kind it cannot draw is
                # refused there rather than drawn from the wrong keys.
                Diagnostics.figures(data)
            end
        catch exception
            failures += 1
            report(stem, "FAILED: $(failure_message(exception))")
            continue
        end
        # The stem names the file, and the figure is written through a sibling temporary so that a
        # failed save leaves the earlier one whole.
        atomic_write(path) do temporary
            save(temporary, drawn)
        end
        report_path("figure", path)
    end
    return failures == 0 ? 0 : 1
end

exit(main(ARGS))
