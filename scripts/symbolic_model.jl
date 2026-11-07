# The symbolic family, built: its registry entries, the problem and the method of one of them,
# and the archive of a run.
#
# The driver includes this file through its `*_model.jl` glob; it is the only place the family's
# constructor calls are written, so a change to the ansatz or the coupling is one edit.

using MultisymplecticIntegrators
using Symbolics

"""
The symbolic family: the space-time Galerkin integrator on a closed-form ansatz, for the standing
wave of the `Wave` problem, at the wave's period.

An entry's `timestep` is the period of the standing wave of its parameters, so every slab carries
the same function of the slab's normalised time and the ansatz is the exact solution of each of
them. `stem` names the archive and the figure; `label` is what a report shows.
"""
const SYMBOLIC = (
    (stem = "wave-symbolic-h4", label = "symbolic, h = 4",
    timestep = 4.0, xspan = (0.0, 1.0), xstep = 0.05,
    degree = 2, ncells = (4, 4)),)

"""
    symbolic_f_reltol(::Type{T})

The relative tolerance the symbolic family's steps are solved to, measured on the family's affine
control: its two steps reach `2.6e-16` and `1.6e-16` of their initial residual in `Float64` and
`6.1e-8` and `6.9e-8` of it in `Float32`. `1e-12` and `1e-6` sit above both.
"""
symbolic_f_reltol(::Type{Float64}) = 1e-12
symbolic_f_reltol(::Type{Float32}) = 1.0f-6
function symbolic_f_reltol(::Type{T}) where {T}
    throw(ArgumentError("the symbolic family has no measured tolerance for element type $T"))
end

"""
    symbolic_expression(entry; T = Float64)

The six-parameter standing wave of `Wave`, converted from physical time to the slab's normalised
time by `t = timestep · t̂`, with the symbols it is written in.

The parameters are `[A₁, B₁, A₂, B₂, l, c]`: the amplitudes and the phase of the two harmonics, the
length of the interval and the wave speed.
"""
function symbolic_expression(entry; T::Type = Float64)
    @variables t τ x p[1:6]
    physical = (p[1] * cos(π * p[6] * t / p[5]) +
                p[2] * sin(π * p[6] * t / p[5] + π / 6)) * sin(π * x / p[5]) +
               (p[3] * cos(2 * π * p[6] * t / p[5]) +
                p[4] * sin(2 * π * p[6] * t / p[5] + π / 6)) * sin(2 * π * x / p[5])
    return Symbolics.substitute(physical, Dict(t => T(entry.timestep) * τ)), τ, x, p
end

"""
    symbolic_ansatz(entry; T = Float64)

The symbolic ansatz of a registry entry: its expression in the slab's normalised time, the
six-parameter start `[A₁, B₁, A₂, B₂, l, c]` and the entry's trace basis.
"""
function symbolic_ansatz(entry; T::Type = Float64)
    expression, τ, x, p = symbolic_expression(entry; T = T)
    return SymbolicAnsatz(expression, τ, x, p;
        xspan = (T(entry.xspan[1]), T(entry.xspan[2])),
        start = T[0.8, 0.8, 0.0, 0.0, 1.0, 0.5],
        degree = entry.degree, ncells = entry.ncells)
end

"""
    symbolic_problem(entry)
    symbolic_problem(entry; T = Float64)

The wave problem of a registry entry: its time span, which is two of its periods, its time step and
its space step.
"""
function symbolic_problem(entry; T::Type = Float64)
    return Wave.problem(T = T, timespan = (zero(T), T(2 * entry.timestep)),
        timestep = T(entry.timestep), xspan = (T(entry.xspan[1]), T(entry.xspan[2])),
        xstep = T(entry.xstep))
end

"""
    symbolic_method(entry)
    symbolic_method(entry; T = Float64)

The method of a registry entry: its ansatz under the coupling a symbolic ansatz takes.
"""
function symbolic_method(entry; T::Type = Float64)
    return SpaceTimeGalerkin(symbolic_ansatz(entry; T = T),
        WeakInitialAndBoundaryConditions())
end

"""
    symbolic_run(entry; T = Float64) -> Dict{String, Any}

The registry's `run` for the symbolic family: integrate one entry, report the steps it took and
the largest relative energy error over them, and return its archive.
"""
function symbolic_run(entry; T::Type = Float64)
    problem = symbolic_problem(entry; T = T)
    method = symbolic_method(entry; T = T)
    result = integrate(problem, method; f_abstol = zero(T),
        f_reltol = symbolic_f_reltol(T))
    report(entry.label, "$(length(result.traces)) steps")
    report_error("maximum energy error", maximum(energy_error(problem, method, result.traces)))
    t, x, u = solution_data(problem, result)
    return Dict{String, Any}(
        "kind" => "solution",
        "label" => entry.label,
        "problem_label" => "Wave",
        "timestep" => entry.timestep,
        "t" => t,
        "x" => x,
        "u" => u)
end

register_family!(:symbolic, SYMBOLIC; run = symbolic_run)
