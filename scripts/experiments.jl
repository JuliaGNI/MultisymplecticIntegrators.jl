# The registry: what a driver turns into a run.
#
# An entry is plain data — the problem, its time step, its space grid and the ansatz resolution —
# and no work happens at its top level, so a driver can `include` this and pay only the load. The
# problem and the method are built from the entry by the driver that loads the package
# (`spline_model.jl` does it for the spline family); this file holds neither.
#
# A family whose part has not merged holds an empty tuple rather than an entry that cannot run:
# `run_network.jl` and `run_symbolic.jl` say which part fills theirs.

"""
The spline family: the space-time Galerkin integrator on a tensor B-spline ansatz, for the wave
equation, at a ladder of time steps. An entry's `coupling` is `:weak`, which imposes the initial and
boundary data with multipliers, or `:restriction`, which fixes the bottom time row of each slab.

`stem` names the archive and the figure; `label` is what a legend and a title show.
"""
const SPLINE = (
    (stem = "wave-spline-h0.2", label = "spline, h = 0.2",
        timespan = (0.0, 1.0), timestep = 0.2, xspan = (0.0, 1.0), xstep = 0.05,
        degree = 2, ncells = (1, 10), n = 3, coupling = :restriction),
    (stem = "wave-spline-h0.1", label = "spline, h = 0.1",
        timespan = (0.0, 1.0), timestep = 0.1, xspan = (0.0, 1.0), xstep = 0.05,
        degree = 2, ncells = (1, 20), n = 3, coupling = :restriction),
    (stem = "wave-spline-h0.05", label = "spline, h = 0.05",
        timespan = (0.0, 1.0), timestep = 0.05, xspan = (0.0, 1.0), xstep = 0.05,
        degree = 2, ncells = (2, 20), n = 3, coupling = :restriction))

"""The network family: empty, because part F adds its ansatz and its guess and with them its entries."""
const NETWORK = ()

"""The symbolic family: empty, because part G adds its ansatz and with it its entries."""
const SYMBOLIC = ()

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

"""The entries of one family, by name: `:spline`, `:network` or `:symbolic`."""
function family_entries(family::Symbol)
    family === :spline && return SPLINE
    family === :network && return NETWORK
    family === :symbolic && return SYMBOLIC
    throw(ArgumentError("unknown family `$(family)`; the registry holds :spline, :network and " *
                        ":symbolic"))
end

"""
    smallest_entry(entries)

The cheapest entry of a family: the largest time step, and among equals the coarsest ansatz.

This is what a driver runs when it is given no entry name, because it is the one that answers "does
this family run at all" in the least time.
"""
function smallest_entry(entries)
    isempty(entries) && throw(ArgumentError("the registry holds no entry of this family"))
    return last(sort(collect(entries); by = entry -> (entry.timestep, entry.xstep)))
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
