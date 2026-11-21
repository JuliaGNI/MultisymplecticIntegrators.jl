# The convergence tables and the long-time energy figure of the manual, written into
# `docs/src/assets/`, which the docs build reads and never recomputes:
#
#   julia --project=scripts scripts/manual_verification.jl              # into docs/src/assets/
#   julia --project=scripts scripts/manual_verification.jl --out <dir>  # into <dir>
#
# It writes three files and nothing else: `convergence_tables.md`, one table per B-spline coupling
# and one for the box scheme; `long_time_energy.svg`, the figure; and `long_time_energy.md`, the
# figure's caption. Each table and the caption name the commit the script runs on and the Julia
# version, so the script refuses to run while `src/`, `test/helpers/` or this file differ from that
# commit.
#
# The ladders, the runs and the long-time settings are those of `test/helpers/manufactured.jl`,
# which `test/verification/manufactured.jl` and `test/verification/long_time_energy.jl` assert on.
# An observed order is `log₂(e₁ / e₂)` of two consecutive rungs of a halving ladder, and the energy
# error is `|E_h − E_*| / E_*(0)`.

using CairoMakie
using MultisymplecticIntegrators
using Printf
import MultisymplecticIntegrators.Diagnostics as Diagnostics

include(joinpath(@__DIR__, "..", "test", "helpers", "manufactured.jl"))

const ROOT = dirname(@__DIR__)

"""
    parse_arguments(args)

The output directory: `docs/src/assets/` of the repository, or the directory that `--out` names.
"""
function parse_arguments(args)
    out = joinpath(ROOT, "docs", "src", "assets")
    i = 1
    while i ≤ length(args)
        args[i] == "--out" || throw(ArgumentError("unknown argument $(args[i])"))
        i < length(args) || throw(ArgumentError("--out needs a directory"))
        out = args[i + 1]
        i += 2
    end
    return out
end

"""
    source_commit()

The short hash of the commit the script runs on. The tables are that commit's only if the code they
come from is, so the script refuses a tree whose `src/`, `test/helpers/` or script differ from it.
"""
function source_commit()
    paths = ["src", joinpath("test", "helpers"), relpath(@__FILE__, ROOT)]
    changed = readchomp(Cmd(`git status --porcelain -- $paths`; dir = ROOT))
    isempty(changed) ||
        error("src/, test/helpers/ or the script differ from the commit:\n$(changed)")
    return readchomp(Cmd(`git rev-parse --short HEAD`; dir = ROOT))
end

# A ladder's runs, refused unless every step of every run converged and each run took the steps
# that its rung asks for.
function checked(runs, steps, label)
    all(run -> run.converged, runs) || error("$(label): a run did not converge")
    [run.steps for run in runs] == steps ||
        error("$(label): the runs took $([run.steps for run in runs]) steps, not $(steps)")
    return runs
end

# One row of a table: the ladder's three values, each beside the order to the next rung.
function ladder_row(cells, values, expected)
    orders = observed_orders(values)
    numbers = [@sprintf("%.4e", values[1]), @sprintf("%.4e", values[2]),
        @sprintf("%.3f", orders[1]), @sprintf("%.4e", values[3]),
        @sprintf("%.3f", orders[2])]
    return "| " * join([cells; string(expected); numbers], " | ") * " |"
end

const VALUE_COLUMNS = "| value 1 | value 2 | order 1–2 | value 3 | order 2–3 |"

# The ladder cells of the tables, which name each ladder by its rungs.
const JOINT_CELL = "joint `nx = $(join(JOINT_LADDER, ", "))`, `h = 1/nx`"
const STEPS_CELL = "self-difference `h = $(join(SELF_DIFFERENCE_STEPS, ", "))`"
const BOX_SPACE_CELL = "space `nx = $(join(BOX_SPACE_LADDER, ", "))`, " *
                       "`h = 1/$(round(Int, 1 / BOX_SPACE_STEP))`"

function coupling_table(name, coupling, boundary, stamp)
    rows = String[]
    for p in (1, 2), D in (1, 2)

        label = "$(name), p = $(p), D = $(D)"
        runs = checked(
            [slab_run(coupling, boundary, p, 1 / nx, nx, D) for nx in JOINT_LADDER],
            collect(JOINT_LADDER), "$(label), joint ladder")
        errors = [run.error for run in runs]
        push!(rows, ladder_row([JOINT_CELL, string(p), string(D)], errors, p + 1))
        nx = SLAB_SELF_DIFFERENCE_NX
        runs = checked(
            [slab_run(coupling, boundary, p, h, nx, D) for h in SELF_DIFFERENCE_STEPS],
            [10, 20, 40, 80], "$(label), self-difference ladder")
        cells = ["$(STEPS_CELL), `nx = $(nx)`", string(p), string(D)]
        push!(rows, ladder_row(cells, self_differences(runs, D), 2p))
    end
    return """
    **$(name).** The `L²` error of the top trace against `u*` at `t = $(TEND)` on the joint ladder,
    expected order `p + 1`, and the `L²` difference of consecutive runs on the self-difference
    ladder, expected order `2p` in `h`. $(stamp)

    | ladder | p | D | expected $(VALUE_COLUMNS)
    |:--|--:|--:|--:|--:|--:|--:|--:|--:|
    $(join(rows, "\n"))
    """
end

function box_table(stamp)
    rows = String[]
    for D in (1, 2)
        steps = round(Int, TEND / BOX_SPACE_STEP)
        runs = checked([box_run(BOX_SPACE_STEP, nx, D) for nx in BOX_SPACE_LADDER],
            fill(steps, length(BOX_SPACE_LADDER)), "box scheme, D = $(D), space ladder")
        errors = [run.error for run in runs]
        push!(rows, ladder_row([BOX_SPACE_CELL, string(D)], errors, 2))
        nx = BOX_SELF_DIFFERENCE_NX
        runs = checked([box_run(h, nx, D) for h in SELF_DIFFERENCE_STEPS],
            [10, 20, 40, 80], "box scheme, D = $(D), self-difference ladder")
        cells = ["$(STEPS_CELL), `nx = $(nx)`", string(D)]
        push!(rows, ladder_row(cells, self_differences(runs, D), 2))
    end
    return """
    **The box scheme.** The `L²` error of the last level against `u*` at the grid nodes at
    `t = $(TEND)` on the space ladder, and the `L²` difference of consecutive runs on the
    self-difference ladder, expected order `2` in both. $(stamp)

    | ladder | D | expected $(VALUE_COLUMNS)
    |:--|--:|--:|--:|--:|--:|--:|--:|
    $(join(rows, "\n"))
    """
end

# The relative energy error of the four ten-period runs of `Wave`, one series each, at the times
# `h, 2h, …, tend`.
function long_time_series()
    (; h, tend, nx, p, box_xstep) = LONG_TIME
    steps = round(Int, tend / h)
    labels, errors = String[], Vector{Float64}[]
    for (name, coupling, boundary) in COUPLINGS
        run = wave_run(coupling, boundary, p, h, nx, steps)
        length(run.result.traces) == steps && all_steps_converge(run.result, run.int) ||
            error("long-time run of $(name): not $(steps) converged steps")
        push!(labels, "$(name), p = $(p)")
        push!(errors, energy_errors(slab_run_energies(run, h)...))
    end
    run = box_wave_run(h, tend, box_xstep)
    length(run.result.traces) == steps && all_steps_converge(run.result, run.int) ||
        error("long-time run of the box scheme: not $(steps) converged steps")
    push!(labels, "box scheme")
    push!(errors, energy_errors(box_run_energies(run, h)...))
    return h .* (1:steps), errors, labels
end

function main(args)
    out = parse_arguments(args)
    commit = source_commit()
    stamp = "Run on commit `$(commit)` with Julia $(VERSION)."

    tables = [coupling_table(name, coupling, boundary, stamp)
              for (name, coupling, boundary) in COUPLINGS]
    push!(tables, box_table(stamp))
    times, errors, labels = long_time_series()

    (; h, tend, nx, p, box_xstep) = LONG_TIME
    caption = """
    `long_time_energy.svg`: the relative energy error `|E_h − E_*| / E_*(0)` of `Wave` at its defaults over ten periods,
    `t ∈ (0, $(tend)]`, at `h = $(h)`: the three B-spline couplings at `p = $(p)` on `nx = $(nx)`
    space cells, and the box scheme at `xstep = $(box_xstep)`. $(stamp)
    """

    mkpath(out)
    write(joinpath(out, "convergence_tables.md"), join(tables, "\n"))
    Diagnostics.plot_energy_error(times, errors; labels = labels,
        path = joinpath(out, "long_time_energy.svg"))
    write(joinpath(out, "long_time_energy.md"), caption)
    return out
end

# Run as a script, not when included.
if abspath(PROGRAM_FILE) == @__FILE__
    println("wrote ", main(ARGS))
end
