# The registry: what a driver turns into a run.
#
# An entry is plain data and no work happens at this file's top level, so a driver can `include`
# this and pay only the load. The problem and the method are built from the entry by the family's
# own model file, which registers the family here; this file holds no family's entries and no
# constructor, only the registry and the ladders of the spline scans.
#
# An entry is a named tuple with at least `stem`, which names its archive and its figure, and
# `label`, which a report and a heading show; the rest of its fields are the family's own.

# The registry itself: one record per family name. It is a `Dict` and not a constant per family,
# because a constant is an edit to this file for every family that arrives.
const FAMILIES = Dict{Symbol, NamedTuple}()

"""
    register_family!(name, entries; run) -> Symbol

Add the family `name` to the registry: its `entries`, and the `run` that turns one of them into the
archive [`store_run!`](@ref) writes.

The registry is append-only. A family is registered once, by the model file that builds it, and a
second registration of the same name is refused with an `ArgumentError` naming it: two model files
that claim one name are a mistake to report, not a claim to resolve in favour of the later one.

A stem names one archive under `runs/` for every family, so a stem that `entries` repeats, or that
a registered family already holds, is refused with an `ArgumentError` naming it: the later run would
replace the earlier archive.

`run(entry)` returns the entry's archive as a `Dict{String, Any}` whose `"kind"` says what shape of
figure it draws; `store_run!` adds the `"stem"`.
"""
function register_family!(name::Symbol, entries; run::Function)
    haskey(FAMILIES, name) &&
        throw(ArgumentError("the family `$(name)` is already registered; a family is " *
                            "registered once, by its model file"))
    # A tuple, so that no entry arrives after the check.
    entries = Tuple(entries)
    taken = Set(entry.stem for family in values(FAMILIES) for entry in family.entries)
    for entry in entries
        entry.stem in taken &&
            throw(ArgumentError("the stem `$(entry.stem)` names two entries; a stem names one " *
                                "archive under `runs/`"))
        push!(taken, entry.stem)
    end
    FAMILIES[name] = (entries = entries, run = run)
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

# ---- the fixed-boundary network ------------------------------------------------------

# The names the family's rows and runner build, taken from their owners so that this section loads
# under any driver that includes the registry.
using MultisymplecticIntegrators: Continuation, FitTo, FixedBoundary, FixedBoundaryNetwork,
                                 LevenbergMarquardt, Predictor, SineGordon,
                                 SpaceTimeGalerkin, Wave, integrate
using SimpleSolvers: Newton

"""
The branches of the fixed-boundary family: the dead branch — the previous level continued and solved
by Newton in `(α, β, γ)` — and the live branch's four combinations of the bubble's form and its
solve, each started by the predictor.
"""
const FIXED_BOUNDARY_BRANCHES = (
    (name = "dead", form = :abg, solver = :newton),
    (name = "live-abg-joint", form = :abg, solver = :joint),
    (name = "live-tsr-joint", form = :tsr, solver = :joint),
    (name = "live-abg-projected", form = :abg, solver = :projected),
    (name = "live-tsr-projected", form = :tsr, solver = :projected))

const FIXED_BOUNDARY_PROBLEMS = ("wave", "sinegordon")
const FIXED_BOUNDARY_STEPS = (0.1, 0.05, 0.025)

"""
The fixed-boundary family: `Wave` and `SineGordon` at three time steps, on the dead branch and on
the four combinations of the bubble's form and its solve, and the two rows of the plateau probe.

`steps` is what a row integrates: the matrix's rows take the time step's ladder up to `t = 1`, so
that the three resolutions of one problem compare one span, and the extended-precision row of the
plateau probe takes one step.
"""
const FIXED_BOUNDARY = vcat(
    vec([(stem = "$(problem)-fixed-boundary-$(branch.name)-h$(step)",
          label = "$(problem), $(branch.name), h = $(step)", problem, step,
          steps = round(Int, 1 / step), form = branch.form, solver = branch.solver,
          Sx = 8, St = 10, T = Float64)
         for problem in FIXED_BOUNDARY_PROBLEMS, step in FIXED_BOUNDARY_STEPS,
         branch in FIXED_BOUNDARY_BRANCHES]),
    [
        (stem = "wave-fixed-boundary-plateau-h0.025", label = "wave, plateau, h = 0.025",
            problem = "wave", step = 0.025, steps = 40, form = :tsr, solver = :joint,
            Sx = 12, St = 16, T = Float64),
        (stem = "wave-fixed-boundary-plateau-extended",
            label = "wave, plateau, extended precision",
            problem = "wave", step = 0.025, steps = 1, form = :tsr, solver = :joint,
            Sx = 8, St = 10, T = BigFloat)])

"""
    fixed_boundary_problem(entry)

The problem of a fixed-boundary entry: its time step, and the time span of its `steps` steps.
"""
function fixed_boundary_problem(entry)
    T = entry.T
    problem = entry.problem === "wave" ? Wave.problem : SineGordon.problem
    timestep = T(entry.step)
    return problem(
        T = T, timespan = (zero(T), T(entry.steps) * timestep),
        timestep = timestep, xspan = (zero(T), one(T)), xstep = timestep)
end

"""
    fixed_boundary_method(entry; T = entry.T)

The method of a fixed-boundary entry: its ansatz, the fixed-boundary coupling, and the guess its
branch starts from.
"""
function fixed_boundary_method(entry)
    T = entry.T
    ansatz = FixedBoundaryNetwork(Sx = entry.Sx, St = entry.St, bubble_form = entry.form,
        xspan = (zero(T), one(T)), ncells = 20, nq = (6, 4))
    guess = entry.solver === :newton ? Continuation() : FitTo(Predictor())
    return SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = guess)
end

"""
    fixed_boundary_run(entry) -> Dict{String, Any}

The registry's `run` for the fixed-boundary family: integrate one entry with the solver its branch
names and return the archive of its per-step record.

The solver is `Newton()` on the dead branch, and Levenberg–Marquardt on the live one — joint, or
with the bubble's output weights eliminated. The convergence flags are the run's own record and not
an assertion: no live-branch variant of this method converges, and a row that did not converge is a
result of the comparison.
"""
function fixed_boundary_run(entry)
    problem = fixed_boundary_problem(entry)
    method = fixed_boundary_method(entry)
    solvermethod = entry.solver === :newton ? Newton() :
                   entry.solver === :projected ?
                   LevenbergMarquardt(variable_projection = true) :
                   LevenbergMarquardt()
    seconds = @elapsed result = integrate(problem, method; solver = solvermethod, verbosity = 0)
    stats = result.stats
    report(entry.label, "$(length(result.traces)) steps, " *
                        "$(count(stats.converged)) converged")
    report_error("worst relative residual",
        maximum(stats.residuals ./ max.(stats.initial_residuals, eps(entry.T))))
    t, x, u = solution_data(problem, result)
    return Dict{String, Any}(
        "kind" => "solution",
        "label" => entry.label,
        "problem_label" => entry.problem,
        "timestep" => Float64(entry.step),
        "t" => t,
        "x" => x,
        "u" => u,
        "iterations" => stats.iterations,
        "residuals" => stats.residuals,
        "initial_residuals" => stats.initial_residuals,
        "converged" => stats.converged,
        "seconds" => seconds)
end

register_family!(:fixed_boundary, FIXED_BOUNDARY; run = fixed_boundary_run)
