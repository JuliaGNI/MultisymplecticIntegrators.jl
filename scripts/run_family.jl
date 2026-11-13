# The one driver: run the entries of a family, archive every run under `runs/`, and print what each
# cost.
#
#   julia --project=scripts scripts/run_family.jl spline                   # the smallest entry
#   julia --project=scripts scripts/run_family.jl spline wave-spline-h0.1  # the entry named
#   julia --project=scripts scripts/run_family.jl --runs-dir talk/runs spline
#
# A family is a `*_model.jl` file beside this one: it registers its entries and the function that
# runs one of them, and this file neither knows nor names any of them. Every model file is included
# here in sorted order, so a new family is a new file and no edit to the driver.
#
# With no entry named it runs the cheapest entry of the family, which is what answers "does this
# family run" in the least time, or the entry the family names as its default where the cheapest is
# not the one to run. An archive is one JLD2 file per run, and `figures.jl` draws it.
#
# A run that throws is reported on **stderr** with its stem and its backtrace and archives nothing —
# the stem says which of the selected entries failed, the frame that threw says where the family
# broke, and an archive of a failed run is worse than none — and `main` then returns 1.
#
# The file is includable: `main` returns the exit code and `exit` runs only when this file is the
# program, so a caller can register a family of its own and run the real driver over it.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))

for file in sort(filter(endswith("_model.jl"), readdir(@__DIR__)))
    include(joinpath(@__DIR__, file))
end

"""
    main(args) -> Int

Run the family named by the first positional argument, over the entries its stems name or else its
[`default_entry`](@ref), and return 0 iff every selected entry was archived.
"""
function main(args)
    positional, _ = parse_arguments(args)
    isempty(positional) &&
        throw(ArgumentError("no family named; this driver takes a family name and then the stems " *
                            "of its entries"))
    name = Symbol(first(positional))
    stems = positional[2:end]
    entries = family_entries(name)
    selected = isempty(stems) ? (default_entry(name),) :
               Tuple(entry_by_stem(entries, stem) for stem in stems)
    runner = family_run(name)
    runner === nothing &&
        throw(ArgumentError("the family `$(name)` registers no `run`; its rows are the matrix's " *
                            "and `run_benchmark.jl` runs them"))

    banner("$(name): $(length(selected)) run(s) into $(RUNS_DIR[])")
    failures = 0
    for entry in selected
        try
            started = time_ns()
            data = runner(entry)
            seconds = (time_ns() - started) / 1.0e9
            path = store_run!(entry.stem, data)
            report(entry.label, "$(round(seconds; digits = 2)) s")
            report_path("archive", path)
        catch exception
            failures += 1
            println(stderr, "$(entry.stem) FAILED: ",
                sprint(showerror, exception, catch_backtrace()))
        end
    end
    return failures == 0 ? 0 : 1
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && exit(main(ARGS))
