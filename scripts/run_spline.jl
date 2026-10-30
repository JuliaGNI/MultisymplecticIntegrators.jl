# The spline family's driver: integrate the registry entries of the family, archive every run under
# `runs/`, and print what each cost.
#
#   julia --project=scripts scripts/run_spline.jl                 # the smallest entry
#   julia --project=scripts scripts/run_spline.jl wave-spline-h0.1
#   julia --project=scripts scripts/run_spline.jl --runs-dir figures/runs
#
# With no entry name it runs the cheapest entry of the registry, which is what answers "does this
# family run" in the least time. An archive is one JLD2 file per run, and `figures.jl` draws it.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))
include(joinpath(@__DIR__, "spline_model.jl"))

function main(args)
    positional, _ = parse_arguments(args)
    entries = family_entries(:spline)
    selected = isempty(positional) ? (smallest_entry(entries),) :
               Tuple(entry_by_stem(entries, stem) for stem in positional)

    banner("spline: $(length(selected)) run(s) into $(RUNS_DIR[])")
    failures = 0
    for entry in selected
        try
            problem, method, result, seconds = run_entry(entry)
            data = solution_archive(entry, problem, method, result)
            path = store_run!(entry.stem, data)
            report(entry.label, "$(length(result.traces)) steps, $(round(seconds; digits = 2)) s")
            report_error("maximum energy error", maximum(data["energy_error"]))
            report_path("archive", path)
        catch exception
            failures += 1
            report(entry.stem, "FAILED: $(failure_message(exception))")
        end
    end
    return failures == 0 ? 0 : 1
end

exit(main(ARGS))
