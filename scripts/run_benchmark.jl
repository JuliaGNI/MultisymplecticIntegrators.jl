# The matrix's driver: run the rows of a selection, one row per child process, and archive each.
#
#   julia --project=scripts scripts/run_benchmark.jl --tag smoke
#   julia --project=scripts scripts/run_benchmark.jl --method box --problem wave
#   julia --project=scripts scripts/run_benchmark.jl wave-box-h0.1 wave-box-h0.05
#   julia --project=scripts scripts/run_benchmark.jl --runs-dir talk/runs --tag smoke
#
# The rows come from the registry that `run_family.jl` reads, through `select_rows`: a row is an
# entry of a family that registers a `benchmark`, selected by its method, its problem or its tag, or
# named by its stem. An unknown name is refused before any row runs, so a mistyped tag cannot pass
# for a selection that archived nothing.
#
# **One child process per row.** Every row is integrated by a process of its own, started from
# `Base.julia_cmd()` with this script's own active project, so that no row's compilation flatters the
# next one's time and two rows of one invocation archive two different process ids. The child does
# the measuring; this file only starts it, and writes an archive of its own for a row whose child
# ended without one.
#
# **A failed row is a result.** A row that throws, or that ends on a step which did not converge, is
# archived with its status and the stats of the steps before it: the driver continues with the
# remaining rows and returns 0 when every selected row has an archive. That is the difference
# between this driver and `run_family.jl`, which archives nothing for a run that throws and exits 1.
#
# The timing is the child's, and it excludes the first step of the run: `julia-performance`'s rule is
# that a measured number names its conditions, and the archive records them beside it — the load
# average at the process's start, the two thread counts, the Julia version and the `--check-bounds`
# setting, which `Pkg.test()` forces to `yes` and which the matrix therefore never takes its numbers
# from.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))
include(joinpath(@__DIR__, "benchmark.jl"))

for file in sort(filter(endswith("_model.jl"), readdir(@__DIR__)))
    include(joinpath(@__DIR__, file))
end

"""
    record_row!(row) -> String

Run one row in this process, archive it, and return the path of its archive.

This is the child's half of the driver. The row's family supplies the whole of it: its `benchmark`
builds the row's problem and method, integrates them and returns the record, so the driver names no
ansatz, no coupling and no family, and a family that arrives later adds a model file and no line
here. A row whose run throws still archives, because `benchmark_run` catches the throw and records
it; a row whose record cannot be built at all lets the exception out, and the parent then writes the
failed archive itself.
"""
function record_row!(row)
    entry = row.entry
    data = row.benchmark(entry)
    return store_run!(entry.stem, benchmark_archive(entry.stem, data))
end

"""
    child_command(stem) -> Cmd

The command that runs one row in a process of its own: this script, this Julia, the active project,
and the row's stem.
"""
function child_command(stem)
    return `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) --runs-dir $(RUNS_DIR[]) --one $(stem)`
end

# The rows a selection names: the registry's own selection when no stem is given, and the named
# stems otherwise, each of which the registry must hold.
function selected_rows(positional, options)
    method = haskey(options, "--method") ? Symbol(options["--method"]) : nothing
    problem = haskey(options, "--problem") ? Symbol(options["--problem"]) : nothing
    tag = haskey(options, "--tag") ? Symbol(options["--tag"]) : nothing
    rows = select_rows(; method, problem, tag)
    isempty(positional) && return rows
    stems = Set(row.entry.stem for row in rows)
    named = String[]
    for stem in positional
        stem in stems ||
            throw(ArgumentError("the selection holds no row `$(stem)`; it holds " *
                                join(sort!(collect(stems)), ", ")))
        push!(named, stem)
    end
    return [row for row in rows if row.entry.stem in named]
end

"""
    run_row(row) -> Bool

Run one selected row in a child process of its own and return whether it now has an archive.

The child's archive is what the parent reads: a row whose child wrote one is reported with its path,
and a row whose child ended without one is archived by the parent as a failure carrying the exit code
or the signal that ended the child, as the process reported it. Either way the row has an archive,
which is what `main` counts.

This is the piece of the driver a test drives with a row its child's registry does not hold, which is
the case the fallback exists for.
"""
function run_row(row)
    stem = row.entry.stem
    path = archive_path(stem)
    rm(path; force = true)
    code, signal = try
        process = run(ignorestatus(child_command(stem)))
        (Int(process.exitcode), Int(process.termsignal))
    catch raised
        report(stem, "FAILED to start: $(failure_message(raised))")
        (1, 0)
    end
    if isfile(path)
        report_path("archive", path)
        return true
    end
    reason = signal == 0 ?
             (code == 0 ? "the child process wrote no archive" :
              "the child process exited with code $(code)") :
             "the child process was killed by signal $(signal)"
    store_run!(stem, benchmark_archive(stem, failed_archive(row.entry, reason)))
    report_path("archive", path)
    report(stem, "failed, archived by the driver")
    return true
end

"""
    main(args) -> Int

Run the selection `args` names, one child process per row, and return 0 iff every selected row has
an archive.

The flags are `--method`, `--problem` and `--tag`, each a row field's value, with `--runs-dir` and
`--results-dir` beside them; the positional arguments are the stems to run. `--one <stem>` is the
child's own form, and runs that one row in this process.
"""
function main(args)
    positional, options = parse_arguments(args, ("--method", "--problem", "--tag", "--one"))
    if haskey(options, "--one")
        stem = options["--one"]
        rows = [row for row in select_rows() if row.entry.stem == stem]
        isempty(rows) &&
            throw(ArgumentError("the registry holds no row `$(stem)`; the selection cannot run it"))
        record_row!(first(rows))
        return 0
    end
    rows = selected_rows(positional, options)
    isempty(rows) && throw(ArgumentError("the selection names no row; the registry holds " *
                        join(sort!([row.entry.stem for row in select_rows()]), ", ")))
    banner("benchmark: $(length(rows)) row(s) into $(RUNS_DIR[])")
    archived = 0
    for row in rows
        run_row(row) && (archived += 1)
    end
    # The runs of a problem without an exact solution are compared with the finest run of their
    # method, which needs the archives of the whole selection: the comparison is made here, once,
    # and the archives it changes are written again through `store_run!`.
    runs, _ = load_runs()
    for data in attach_references!(runs)
        store_run!(data["stem"], data)
        report(data["stem"], "reference $(data["reference_stem"])")
    end
    return archived == length(rows) ? 0 : 1
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && exit(main(ARGS))
