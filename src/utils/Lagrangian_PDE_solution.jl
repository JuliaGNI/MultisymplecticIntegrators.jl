mutable struct LPDE_solution{TT, ST, PT, IT} #<: AbstractPDESolution
    step::Int
    t::TT
    ntime::Int

    s::ST

    params::PT
    internal::IT
    function LPDE_solution(t, ics::NamedTuple, ntime::Int, params::PT;
            internal::IT = nothing, kwargs...) where {IT, PT}
        step = 1

        s = map(v -> (v, ntuple(_ -> zeros(size(v)...), ntime)...), ics)

        return new{typeof(t), typeof(s), typeof(params), typeof(internal)}(
            step, t, ntime, s, params, internal)
    end
end

function Base.show(io::IO, sol::LPDE_solution)
    print(io, "\n Lagrangian PDE Solution with:\n")
    print(io, "   Current step: $(sol.step) / $(sol.ntime) \n")
    print(io, "   Current time step size: $(sol.t) \n")
    print(io, "   State fields: $(keys(sol.s)) \n")
    print(io, "   Parameters: $(keys(sol.params)) \n")
    print(io, "   Has internal variables: $(sol.internal !== nothing) \n")
end

function LPDE_solution(problem::LPDEProblem; internal = nothing)
    ntime = round(Int, (problem.timespan[2] - problem.timespan[1]) / problem.timestep)
    t = problem.timestep
    LPDE_solution(t, problem.ics, ntime, problem.params; internal)
end

ntime(sol::LPDE_solution) = sol.ntime

function Base.getindex(sol::LPDE_solution, n::Int)
    @assert n ≥ 0
    @assert n ≤ ntime(sol)

    return map(v -> v[n + 1], sol.s)
end
