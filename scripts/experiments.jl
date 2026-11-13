# The registry: what a driver turns into a run.
#
# An entry is plain data and no work happens at this file's top level, so a driver can `include`
# this and pay only the load. The problem and the method are built from the entry by the family's
# own model file, which registers the family here; this file holds the registry, the ladders of the
# spline scans and one family, the fixed-boundary network's, with its entries and constructors.
#
# An entry is a named tuple with at least `stem`, which names its archive and its figure, and
# `label`, which a report and a heading show; the rest of its fields are the family's own.
#
# A row of the benchmark matrix is an entry whose family registers a `benchmark`, and it names four
# fields more: `method`, `problem`, `tag` and `resolution`. The driver selects on the first three
# and archives the fourth, so that the coarsest-to-finest comparison of one method reads a
# resolution as data instead of parsing a stem.

# The registry itself: one record per family name. It is a `Dict` and not a constant per family,
# because a constant is an edit to this file for every family that arrives.
const FAMILIES = Dict{Symbol, NamedTuple}()

# The fields a benchmark row adds to the entry every family's entry already has.
const ROW_FIELDS = (:method, :problem, :tag, :resolution)

"""
    register_family!(name, entries; run = nothing, benchmark = nothing) -> Symbol

Add the family `name` to the registry: its `entries`, the `run` that turns one of them into the
archive [`store_run!`](@ref) writes, and the `benchmark` that turns one of them into a row of the
matrix [`benchmark_run`](@ref) runs.

The registry is append-only. A family is registered once, by the model file that builds it, and a
second registration of the same name is refused with an `ArgumentError` naming it: two model files
that claim one name are a mistake to report, not a claim to resolve in favour of the later one.

A family registers its `run` for the figure archives, its `benchmark` for the matrix's rows, or
both; one that registers neither is refused with an `ArgumentError` naming it, because neither
driver could do anything with it.

An entry of a family that registers a `benchmark` is a row, and a row names `method`, `problem`,
`tag` and `resolution`. One that lacks any of them is refused with an `ArgumentError` naming the
stem and the missing field: a row the driver cannot select on is not a row of the matrix.

A stem names one archive under `runs/` for every family, so a stem that `entries` repeats, or that
a registered family already holds, is refused with an `ArgumentError` naming it: the later run would
replace the earlier archive.

`default` is the stem of the entry a driver given no stem runs, for a family whose cheapest entry
is not that entry; [`default_entry`](@ref) reads it. A `default` that names none of `entries` is
refused with an `ArgumentError` naming it.

`run(entry)` returns the entry's archive as a `Dict{String, Any}` whose `"kind"` says what shape of
figure it draws; `store_run!` adds the `"stem"`. `benchmark(entry)` integrates the row and returns
its record as a `Dict{String, Any}` of the `"benchmark"` kind, which
[`scripts/README.md`](README.md) documents.
"""
function register_family!(name::Symbol, entries; run = nothing, benchmark = nothing,
        default = nothing)
    haskey(FAMILIES, name) &&
        throw(ArgumentError("the family `$(name)` is already registered; a family is " *
                            "registered once, by its model file"))
    (run === nothing && benchmark === nothing) &&
        throw(ArgumentError("the family `$(name)` registers neither a `run` nor a `benchmark`; " *
                            "`run_family.jl` and `run_benchmark.jl` would both skip it"))
    # A tuple, so that no entry arrives after the check.
    entries = Tuple(entries)
    taken = Set(entry.stem for family in values(FAMILIES) for entry in family.entries)
    for entry in entries
        entry.stem in taken &&
            throw(ArgumentError("the stem `$(entry.stem)` names two entries; a stem names one " *
                                "archive under `runs/`"))
        push!(taken, entry.stem)
        benchmark === nothing && continue
        for field in ROW_FIELDS
            haskey(entry, field) ||
                throw(ArgumentError("the row `$(entry.stem)` of the family `$(name)` has no " *
                                    "`$(field)`; a row of the matrix names method, problem, tag " *
                                    "and resolution"))
        end
    end
    default === nothing || any(entry -> entry.stem == default, entries) ||
        throw(ArgumentError("the default `$(default)` of the family `$(name)` names none of " *
                            "its entries"))
    FAMILIES[name] = (
        entries = entries, run = run, benchmark = benchmark, default = default)
    return name
end

"""
    family_entries(name) -> Tuple

The entries of the family `name`, and an `ArgumentError` naming a name the registry does not hold.
"""
family_entries(name::Symbol) = family(name).entries

"""
    family_run(name) -> Function

The function that turns one entry of the family `name` into its archive, and an `ArgumentError`
naming a name the registry does not hold.
"""
family_run(name::Symbol) = family(name).run

"""
    family_benchmark(name) -> Function

The function that runs one row of the family `name`, and an `ArgumentError` naming a name the
registry does not hold.
"""
family_benchmark(name::Symbol) = family(name).benchmark

"""
    default_entry(name)

The entry of the family `name` that a driver given no stem runs: the entry the family names as its
`default`, and else its [`smallest_entry`](@ref).
"""
function default_entry(name::Symbol)
    held = family(name)
    held.default === nothing && return smallest_entry(held.entries)
    return entry_by_stem(held.entries, held.default)
end

"""
    select_rows(; method = nothing, problem = nothing, tag = nothing) -> Vector

The rows of the matrix whose `method`, `problem` and `tag` equal every filter given, in stem order.

The rows are the entries of the families that register a `benchmark`, and a family that registers a
`run` alone contributes none: the two drivers read one registry, and a family of the figure archives
is not a row of the matrix. A filter left at `nothing` selects on nothing, and two filters together
select their intersection rather than their union.

A filter that no row matches raises an `ArgumentError` naming it and the values the rows hold, and
it does so before any row runs: a mistyped tag that selected nothing would otherwise be reported as
a run that archived nothing.

```julia
select_rows(tag = :smoke)
select_rows(method = :box, problem = :wave)
```
"""
function select_rows(; method = nothing, problem = nothing, tag = nothing)
    filters = (method = method, problem = problem, tag = tag)
    rows = [(family = name, entry = entry, benchmark = FAMILIES[name].benchmark)
            for name in sort!(collect(keys(FAMILIES)))
            for entry in FAMILIES[name].entries
            if FAMILIES[name].benchmark !== nothing]
    # A filter is checked against every row of the matrix, not against the rows the other filters
    # already left: `select_rows(method = :typo, tag = :smoke)` names the method, which is the one
    # that is wrong.
    for (field, value) in pairs(filters)
        value === nothing && continue
        held = [getfield(row.entry, field) for row in rows]
        any(h -> h == value, held) ||
            throw(ArgumentError("no row of the matrix has the $(field) `$(value)`; the registry " *
                                "holds $(join(sort!(unique!(held)), ", "))"))
    end
    selected = [row
                for row in rows
                if all(getfield(row.entry, field) == value
    for (field, value) in pairs(filters) if value !== nothing)]
    return sort!(selected; by = row -> row.entry.stem)
end

function family(name::Symbol)
    haskey(FAMILIES, name) || throw(ArgumentError(
        "unknown family `$(name)`; the registry holds " *
        (isempty(FAMILIES) ? "no family yet" : join(sort!(collect(keys(FAMILIES))), ", "))))
    return FAMILIES[name]
end

"""
The time steps of the energy scan: the ladder whose relative energy errors one figure compares.

A scan rather than a registry entry, because its runs are not figures of their own — the archive it
writes is the comparison.
"""
const ENERGY_STEPS = (0.4, 0.2, 0.1)

"""
The ansatz resolutions of the phase-drift scan, as `(degree, ncells, n)` per entry.

The drift of a run is the error of the field against the exact solution at the final time, so the
ladder is one of resolution rather than of time step: the time step of every entry is the same.
"""
const PHASE_ANSATZE = ((2, (2, 10), 3), (2, (2, 20), 3), (2, (2, 40), 3))

"""
    smallest_entry(entries)

The cheapest entry of a family: the largest time step, and among equals the coarsest output grid.

This is what a driver runs when it is given no entry name, because it is the one that answers "does
this family run at all" in the least time. An entry that names neither a time step nor a space step
gives `0.0` for each of them, which is what a family of one entry needs: the entry it holds is the
one selected.
"""
function smallest_entry(entries)
    isempty(entries) && throw(ArgumentError("the registry holds no entry of this family"))
    key(entry) = (get(entry, :timestep, 0.0), get(entry, :xstep, 0.0))
    return last(sort(collect(entries); by = key))
end

"""
    entry_by_stem(entries, stem)

The entry named `stem`, which is the name of its archive and of its figure.
"""
function entry_by_stem(entries, stem)
    for entry in entries
        entry.stem == stem && return entry
    end
    throw(ArgumentError("the registry holds no entry `$(stem)`; it holds " *
                        join((entry.stem for entry in entries), ", ")))
end

# ---- the problems of the matrix ------------------------------------------------------

# This section is here, and not in a model file, because the model files call it at their top level
# and the driver includes them after this file.
using MultisymplecticIntegrators: LinearTransport, SineGordon, VectorWave, Wave

"""
The example problems a row of the matrix names, by the symbol its `problem` holds: each is the
module whose `problem` builds it.
"""
const MATRIX_PROBLEMS = (wave = Wave, vector_wave = VectorWave, sine_gordon = SineGordon,
    linear_transport = LinearTransport)

# The time span, the field's span and the space step that `matrix_problem` gives a row's problem
# unless its family names its own.
const MATRIX_TIMESPAN = (0.0, 0.2)
const MATRIX_XSPAN = (0.0, 1.0)
const MATRIX_XSTEP = 0.05

"""
    matrix_problem_label(problem) -> String

The display name of the problem `problem`, which a row archives and a table is named by: the name of
its module in [`MATRIX_PROBLEMS`](@ref), and the symbol itself for a problem the matrix does not
hold.
"""
function matrix_problem_label(problem::Symbol)
    haskey(MATRIX_PROBLEMS, problem) || return string(problem)
    return string(nameof(MATRIX_PROBLEMS[problem]))
end

"""
    matrix_problem(family, problem, takes; timespan = MATRIX_TIMESPAN, xspan = MATRIX_XSPAN,
                   xstep = MATRIX_XSTEP, keywords...)

The example problem `problem` of a row of `family`, built by its module's `problem` with the spans,
the space step and the other `keywords` given.

`takes` is the problems the family takes. A problem outside them raises an `ArgumentError` naming
the family and what it takes: a row of a problem its family cannot run is a mistake to report, not a
row to build.
"""
function matrix_problem(family, problem::Symbol, takes; timespan = MATRIX_TIMESPAN,
        xspan = MATRIX_XSPAN, xstep = MATRIX_XSTEP, keywords...)
    problem in takes ||
        throw(ArgumentError("the $(family) has no problem `$(problem)`; it takes " *
                            join((":$(name)" for name in takes), ", ")))
    return MATRIX_PROBLEMS[problem].problem(; timespan, xspan, xstep, keywords...)
end

# ---- the fixed-boundary network ------------------------------------------------------

# The names the family's rows and runner build, taken from their owners so that this section loads
# under any driver that includes the registry.
using MultisymplecticIntegrators: Continuation, FitTo, FixedBoundary, FixedBoundaryNetwork,
                                  LevenbergMarquardt, Predictor, SpaceTimeGalerkin,
                                  integrate
using SimpleSolvers: Newton

"""
The branches of the fixed-boundary family: the dead branch — the previous level continued and solved
by Newton in `(α, β, γ)` — and the live branch's four combinations of the bubble's form and its
solve, each started by the predictor.
"""
const FIXED_BOUNDARY_BRANCHES = (
    (name = "dead", form = :abg, solve = :newton),
    (name = "live-abg-joint", form = :abg, solve = :joint),
    (name = "live-tsr-joint", form = :tsr, solve = :joint),
    (name = "live-abg-projected", form = :abg, solve = :projected),
    (name = "live-tsr-projected", form = :tsr, solve = :projected))

# The problems of the family, as the symbol a row selects on and the name its stem carries.
const FIXED_BOUNDARY_PROBLEMS = ((problem = :wave, name = "wave"),
    (problem = :sine_gordon, name = "sinegordon"))
const FIXED_BOUNDARY_STEPS = (0.1, 0.05, 0.025)

"""
    fixed_boundary_solver_method(solve)

The solver method of a branch's solve: `Newton()` for `:newton`, Levenberg–Marquardt for `:joint`,
and Levenberg–Marquardt with the bubble's output weights eliminated for `:projected`.
"""
function fixed_boundary_solver_method(solve::Symbol)
    solve === :newton && return Newton()
    solve === :joint && return LevenbergMarquardt()
    solve === :projected && return LevenbergMarquardt(variable_projection = true)
    throw(ArgumentError("the fixed-boundary family has no solve `$(solve)`; it takes " *
                        ":newton, :joint or :projected"))
end

# One row of the matrix: a problem of the family, a time step and a branch.
function fixed_boundary_row(held, timestep, branch)
    return (stem = "$(held.name)-fixed-boundary-$(branch.name)-h$(timestep)",
        label = "$(held.name), $(branch.name), h = $(timestep)",
        method = Symbol("fixed_boundary_", replace(branch.name, "-" => "_")),
        problem = held.problem, problem_label = matrix_problem_label(held.problem), tag = :matrix,
        resolution = (timestep = timestep,),
        parameters = (form = branch.form, solve = branch.solve, Sx = 8, St = 10,
            steps = round(Int, 0.2 / timestep), T = :Float64),
        solver = (solver = fixed_boundary_solver_method(branch.solve),))
end

"""
The fixed-boundary family's rows of the matrix: `Wave` and `SineGordon` at three time steps, on the
dead branch and on the four combinations of the bubble's form and its solve.

A row's `method` is its branch, its `resolution` its time step, and its `parameters` the bubble's
form, the solve, the two networks' widths, the `steps` it integrates — the time step's ladder up to
`t = 0.2`, the final time of every other family's rows of `Wave` and `SineGordon` but the symbolic
family's, so that the rows of one problem compare errors at one time — and the name of its element
type. Its `solver` is the solver method of its branch, which the matrix's runner hands to the
integrator.
"""
const FIXED_BOUNDARY_MATRIX = vec([fixed_boundary_row(held, timestep, branch)
                                   for held in FIXED_BOUNDARY_PROBLEMS,
timestep in FIXED_BOUNDARY_STEPS, branch in FIXED_BOUNDARY_BRANCHES])

"""
The fixed-boundary family's runs: the two entries of the plateau probe.

`steps` is what an entry integrates, and the extended-precision entry takes one step. `f_reltol` is
the tolerance an entry names for its solver: the extended-precision entry names `big"1e-8"`, because
the family measures none at `BigFloat`, and the other names none, so its solver is built at the
tolerance the family measures at its own element type.

An entry's `solve` is its branch's solve, the symbol a row holds in its `parameters`; a row's
`solver` is the solver method that solve names.

The two entries share their time step, so the cheapest entry is no answer to which one a driver
given no stem runs: the family names its `default`, [`FIXED_BOUNDARY_DEFAULT`](@ref), the `Float64`
entry, and not the `BigFloat` entry, which cannot run (K53 in `KNOWN_ISSUES.md`).
"""
const FIXED_BOUNDARY = (
    (stem = "wave-fixed-boundary-plateau-extended",
        label = "wave, plateau, extended precision",
        problem = :wave, timestep = 0.025, steps = 1, form = :tsr, solve = :joint,
        Sx = 8, St = 10, T = BigFloat, f_reltol = big"1e-8"),
    (stem = "wave-fixed-boundary-plateau-h0.025", label = "wave, plateau, h = 0.025",
        problem = :wave, timestep = 0.025, steps = 40, form = :tsr, solve = :joint,
        Sx = 12, St = 16, T = Float64, f_reltol = nothing))

"""
The stem of the plateau entry that `run_family.jl fixed_boundary` runs when it is given no stem.
"""
const FIXED_BOUNDARY_DEFAULT = "wave-fixed-boundary-plateau-h0.025"

# The settings of either kind of entry, as one named tuple: a row of the matrix holds them in its
# `resolution` and its `parameters`, and an entry of the plateau runs at its top level.
function fixed_boundary_settings(entry)
    haskey(entry, :resolution) || return entry
    held = entry.parameters
    return (
        problem = entry.problem, timestep = entry.resolution.timestep, steps = held.steps,
        form = held.form, solve = held.solve, Sx = held.Sx, St = held.St,
        T = getfield(Base, held.T))
end

"""
    fixed_boundary_problem(entry)

The problem of a fixed-boundary row or entry: `Wave` for the problem `:wave` and `SineGordon` for
`:sine_gordon`, at its time step and over the time span of its `steps` steps.
"""
function fixed_boundary_problem(entry)
    settings = fixed_boundary_settings(entry)
    T = settings.T
    timestep = T(settings.timestep)
    return matrix_problem("fixed-boundary family", settings.problem, (:wave, :sine_gordon);
        T = T, timespan = (zero(T), T(settings.steps) * timestep),
        timestep = timestep, xspan = (zero(T), one(T)), xstep = timestep)
end

"""
    fixed_boundary_method(entry)

The method of a fixed-boundary row or entry, at its own element type: its ansatz, the
fixed-boundary coupling, and the guess its branch starts from.
"""
function fixed_boundary_method(entry)
    settings = fixed_boundary_settings(entry)
    T = settings.T
    ansatz = FixedBoundaryNetwork(Sx = settings.Sx, St = settings.St,
        bubble_form = settings.form, xspan = (zero(T), one(T)), ncells = 20, nq = (6, 4))
    guess = settings.solve === :newton ? Continuation() : FitTo(Predictor())
    return SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = guess)
end

"""
    fixed_boundary_benchmark(entry) -> Dict{String, Any}

The registry's `benchmark` for the fixed-boundary family's rows: build the row's problem and
method, run it, and return its record for the archive of the `"benchmark"` kind.
"""
function fixed_boundary_benchmark(entry)
    benchmark_run(fixed_boundary_problem(entry), fixed_boundary_method(entry), entry)
end

"""
    fixed_boundary_solver_keywords(entry)

The solver keywords of a fixed-boundary entry: the tolerance the entry names, and nothing else.

An entry that names none is solved at the tolerance the family measures at its element type, which
the family resolves when the integrator is built; the extended-precision entry names `big"1e-8"`,
because the family measures no tolerance at `BigFloat`. The entry's driver and the build-only
testset both build their keywords here, so an entry's `f_reltol` reaches its solver by one path.
"""
function fixed_boundary_solver_keywords(entry)
    entry.f_reltol === nothing && return (;)
    return (; f_reltol = entry.f_reltol)
end

"""
    fixed_boundary_run(entry) -> Dict{String, Any}

The registry's `run` for the fixed-boundary family: integrate one entry with the solver its branch
names and return the archive of its per-step record.

The solver is `Newton()` on the dead branch, and Levenberg–Marquardt on the live one — joint, or
with the bubble's output weights eliminated. The convergence flags are the run's own record and not
an assertion: no live-branch variant of this method converges, and an entry that did not converge is
a result of the probe.
"""
function fixed_boundary_run(entry)
    problem = fixed_boundary_problem(entry)
    method = fixed_boundary_method(entry)
    solvermethod = fixed_boundary_solver_method(entry.solve)
    seconds = @elapsed result = integrate(problem, method; solver = solvermethod,
        fixed_boundary_solver_keywords(entry)..., verbosity = 0)
    stats = result.stats
    report(entry.label, "$(length(result.traces)) steps, " *
                        "$(count(stats.converged)) converged")
    report_error("worst relative residual",
        maximum(stats.residuals ./ max.(stats.initial_residuals, eps(entry.T))))
    t, x, u = solution_data(problem, result)
    return Dict{String, Any}(
        "kind" => "solution",
        "label" => entry.label,
        "problem_label" => string(entry.problem),
        "timestep" => Float64(entry.timestep),
        "t" => t,
        "x" => x,
        "u" => u,
        "iterations" => stats.iterations,
        "residuals" => stats.residuals,
        "initial_residuals" => stats.initial_residuals,
        "converged" => stats.converged,
        "seconds" => seconds)
end

register_family!(:fixed_boundary, FIXED_BOUNDARY; run = fixed_boundary_run,
    default = FIXED_BOUNDARY_DEFAULT)
register_family!(:fixed_boundary_matrix, FIXED_BOUNDARY_MATRIX; benchmark = fixed_boundary_benchmark)
