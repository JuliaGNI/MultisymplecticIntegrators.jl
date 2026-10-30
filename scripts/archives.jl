# The shared layer under every driver in this directory: where output goes, how a run is archived,
# how arguments are parsed, and how progress is printed.
#
# No *work* happens at top level, so a driver can `include` this and pay only the load. The two
# output-directory `Ref`s do execute, so including it twice in one session — a REPL that loads two
# drivers — resets a `--runs-dir` already set. Each driver includes it once.
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
# One JLD2 file per run, under `runs/`. The keys are flat and named for what they hold rather than
# for the variable that happened to hold it, so a reader needs no per-run knowledge to know what an
# archive contains.
#
# Two keys carry the schema itself:
#
#   "kind"    what shape of figure this run draws
#   "stem"    the figure's name, so that a renderer can glob `runs/` and needs no registry to know
#             what it is looking at or what to call the output
#
# Everything else is per kind, and `figures.jl` is what reads it back. Stored for a `"solution"`
# run:
#
#   "label", "problem_label"      the run's identity
#   "timestep", "final_time", "dimension"
#   "t", "x", "u"                 the step times, the output grid, and one `length(t) × length(x)`
#                                 matrix of the field per component
#   "energy_t", "energy_error"    the relative energy error over the stored steps
#
# and for an `"energy"` run: "title", "labels", "t" and "errors", with one time vector and one error
# vector per series.

archive_path(stem) = joinpath(RUNS_DIR[], stem * ".jld2")

"""
    store_run!(stem, data) -> String

Write one run's archive and return its path.

A data set without a `"kind"` is refused: a renderer globbing `runs/` cannot tell what shape of
figure it draws, and a figure guessed at is worse than none.

`jldopen` with explicit writes rather than `jldsave(path; data...)`: splatting a `Dict{String, Any}`
into keyword arguments requires `Symbol` keys, and these keys are deliberately strings — they are
data, read back by name, not identifiers.
"""
function store_run!(stem::String, data::Dict{String, Any})
    haskey(data, "kind") ||
        throw(ArgumentError("the archive for `$(stem)` has no \"kind\"; a renderer globbing " *
                            "`runs/` cannot tell what shape of figure it draws."))
    mkpath(RUNS_DIR[])
    data["stem"] = stem
    path = archive_path(stem)
    # Written to a file of its own beside the archive and renamed over it, so that an interrupted
    # write leaves the earlier archive whole and two writers of one stem never share a file;
    # `load_runs` reads only `.jld2` files and skips the temporary one. `rename` replaces the
    # destination in one step, where `mv(...; force = true)` removes it first on Julia 1.11.
    temporary = tempname(RUNS_DIR[]; cleanup = false)
    try
        jldopen(temporary, "w") do file
            for (key, value) in data
                file[key] = value
            end
        end
        Base.Filesystem.rename(temporary, path)
    finally
        rm(temporary; force = true)
    end
    return path
end

load_run(stem::String) = load(archive_path(stem))

"""
    load_runs() -> (runs::Vector{Dict}, unreadable::Vector{String})

Every archive under `runs/`, sorted by file name, and the names of the files that could not be read.

Read from the directory rather than enumerated from a registry: the directory is the record of what
exists, and a registry that has drifted from the stem a driver wrote produces a figure that silently
is not drawn. A file that fails to load is reported with a warning and left out of `runs`, so one
corrupt archive does not stop the figures of the others; the caller counts `unreadable` as failures.
"""
function load_runs()
    runs = Dict{String, Any}[]
    unreadable = String[]
    isdir(RUNS_DIR[]) || return runs, unreadable
    for file in sort(filter(endswith(".jld2"), readdir(RUNS_DIR[])))
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
