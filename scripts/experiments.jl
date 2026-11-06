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
