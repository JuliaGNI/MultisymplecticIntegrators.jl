# The shared layer under the driver and every other script in this directory: where output goes,
# how a run is archived, how arguments are parsed, and how progress is printed.
#
# No *work* happens at top level, so a script can `include` this and pay only the load. The two
# output-directory `Ref`s do execute, so including it twice in one session — a REPL that loads two
# scripts — resets a `--runs-dir` already set. Each script includes it once.
#
# It depends on `JLD2` and `Printf` and on nothing else, which is what lets a renderer that needs no
# solver load it cheaply: `figures.jl` draws archives of plain vectors and never touches an
# integrator.

using JLD2
using Printf

# ---- where output goes -------------------------------------------------------

const RUNS_DIR = Ref(joinpath(@__DIR__, "..", "runs"))
const RESULTS_DIR = Ref(joinpath(@__DIR__, "..", "results"))

# ---- arguments ---------------------------------------------------------------

const COMMON_OPTIONS = ("--runs-dir", "--results-dir")

"""
    parse_arguments(args, extra = ()) -> (positional, options)

Split `args` into positional names and `--flag value` pairs, accepting `--runs-dir` and
`--results-dir` everywhere plus whatever `extra` a driver adds, and **rejecting anything else**.

One parser rather than one per driver, and the rejection is the reason: a mistyped flag that is
accepted silently has a driver fall back to its default and look like it worked.

Sets `RUNS_DIR` and `RESULTS_DIR` as a side effect, so a driver calls this before it touches either.
"""
function parse_arguments(args, extra = ())
    known = (COMMON_OPTIONS..., extra...)
    options = Dict{String, String}()
    positional = String[]

    i = firstindex(args)
    while i ≤ lastindex(args)
        argument = args[i]
        if startswith(argument, "--")
            argument in known || throw(ArgumentError(
                "unknown option `$(argument)`; this script takes " * join(known, ", ")))
            i < lastindex(args) || throw(ArgumentError("`$(argument)` needs a value"))
            options[argument] = args[i + 1]
            i += 2
        else
            push!(positional, argument)
            i += 1
        end
    end

    haskey(options, "--runs-dir") && (RUNS_DIR[] = options["--runs-dir"])
    haskey(options, "--results-dir") && (RESULTS_DIR[] = options["--results-dir"])

    return positional, options
end

"""
    option_steps(options, default) -> Tuple

The `--steps h1,h2,…` value, or `default`.
"""
function option_steps(options, default)
    haskey(options, "--steps") || return default
    return Tuple(parse.(Float64, split(options["--steps"], ",")))
end

# ---- archives ----------------------------------------------------------------
#
# One JLD2 file per run, under `runs/`, named after its stem. The archive's keys are the schema
# documented in `scripts/README.md`: `"kind"` says what shape of figure the run draws and `"stem"`
# names the archive and its figure, and a reader globs the directory and needs no registry to know
# what it is looking at.

archive_path(stem) = joinpath(RUNS_DIR[], stem * ".jld2")

"""
    atomic_write(write::Function, path::AbstractString)

Write one file through a temporary beside it and rename that over the destination, and return
`path`.

The temporary is a sibling, so `rename` is a rename within one file system, and it carries the
destination's extension, so a writer that picks its format from the file name — a PDF, say — writes
the format the destination asks for. Its name starts with a dot, so a reader that takes every file
of that extension for a finished result — `load_runs` for the archives — skips it, and a
temporary left behind by a writer that was killed is not mistaken for one. A writer that throws
leaves the destination as it was and its temporary is removed, so a failed write cannot truncate an
earlier result into half a file. The writer's exception is rethrown, because the caller is what
decides whether a failed write is fatal.

    atomic_write(path) do temporary
        save(temporary, figure)
    end
"""
function atomic_write(write::Function, path::AbstractString)
    directory = dirname(abspath(path))
    mkpath(directory)
    temporary = joinpath(directory, "." * basename(tempname(directory)) * splitext(path)[2])
    try
        write(temporary)
        Base.Filesystem.rename(temporary, path)
    finally
        rm(temporary; force = true)
    end
    return path
end

"""
    store_run!(stem, data) -> String

Write one run's archive and return its path.

A data set without a `"kind"` is refused: a renderer globbing `runs/` cannot tell what shape of
figure it draws, and a figure guessed at is worse than none. The `"stem"` is the argument's, so
that the archive and its figure are named in one place.

A stem that `load_runs` would not find is refused: an empty one, which names `.jld2`; one that starts
with a dot, which `load_runs` skips as the mark of an `atomic_write` temporary; and one that holds a
path separator, which names a file below `runs/`, where `load_runs` does not look. An archive stored
under any of them would be written and never drawn.

`jldopen` with explicit writes rather than `jldsave(path; data...)`: splatting a `Dict{String, Any}`
into keyword arguments requires `Symbol` keys, and these keys are deliberately strings — they are
data, read back by name, not identifiers.
"""
function store_run!(stem::String, data::Dict{String, Any})
    (isempty(stem) || startswith(stem, ".") || basename(stem) != stem) &&
        throw(ArgumentError(
            "the stem `$(stem)` must be a file name that does not start with a dot: `load_runs` reads " *
            "the files directly under `runs/` and skips a dotted name as a temporary, so this archive " *
            "would be written and never drawn."))
    haskey(data, "kind") ||
        throw(ArgumentError("the archive for `$(stem)` has no \"kind\"; a renderer globbing " *
                            "`runs/` cannot tell what shape of figure it draws."))
    data["stem"] = stem
    path = archive_path(stem)
    # Written through a sibling temporary so that an interrupted write leaves the earlier archive
    # whole; `atomic_write` names that temporary with a leading dot, which `load_runs` skips.
    atomic_write(path) do temporary
        jldopen(temporary, "w") do file
            for (key, value) in data
                file[key] = value
            end
        end
    end
    return path
end

load_run(stem::String) = load(archive_path(stem))

# A file `store_run!` has finished writing: an archive whose name does not start with a dot, which
# is how `atomic_write` names its temporary.
is_written_archive(file) = endswith(file, ".jld2") && !startswith(file, ".")

"""
    load_runs() -> (runs::Vector{Dict}, unreadable::Vector{String})

Every archive under `runs/`, sorted by file name, and the names of the files that could not be read.

Read from the directory rather than enumerated from a registry: the directory is the record of what
exists, and a registry that has drifted from the stem a driver wrote produces a figure that silently
is not drawn. A file that fails to load is reported with a warning and left out of `runs`, so one
corrupt archive does not stop the figures of the others; the caller counts `unreadable` as failures.
A name that starts with a dot is skipped: `atomic_write` names its temporary that way, so a
temporary whose writer was killed is not read as a corrupt archive.
"""
function load_runs()
    runs = Dict{String, Any}[]
    unreadable = String[]
    isdir(RUNS_DIR[]) || return runs, unreadable
    for file in sort(filter(is_written_archive, readdir(RUNS_DIR[])))
        data = try
            load(joinpath(RUNS_DIR[], file))
        catch exception
            push!(unreadable, file)
            @warn "cannot read the archive $(file): $(failure_message(exception))"
            continue
        end
        # The file name is the stem, by construction in `store_run!`.
        get!(data, "stem", chop(file; tail = length(".jld2")))
        push!(runs, data)
    end
    return runs, unreadable
end

# ---- reporting ---------------------------------------------------------------

report(label, value) = @printf("  %-46s %s\n", label, value)

report_error(label, value) = @printf("  %-46s %.3e\n", label, value)

banner(text) = println("\n", text, "\n", repeat("-", length(text)))

"""
    failure_message(exception) -> String

An exception as one line of report output: its own message, first line only and capped.

The type alone is not enough — `ArgumentError` and `KeyError` carry the argument and the key, and
those are how an archive written by an older revision fails to draw. First line and capped because a
`MethodError` prints its whole candidate list.
"""
function failure_message(exception)
    text = first(split(sprint(showerror, exception), '\n'))
    return length(text) > 160 ? first(text, 160) * "…" : text
end

"""
    report_path(label, path)

Print a path relative to the repository root rather than to the script, so that a redirected
`--runs-dir` does not print `../../../../…`.
"""
function report_path(label, path)
    root = normpath(joinpath(@__DIR__, ".."))
    shown = startswith(normpath(path), root) ? relpath(path, root) : path
    return report(label, shown)
end
