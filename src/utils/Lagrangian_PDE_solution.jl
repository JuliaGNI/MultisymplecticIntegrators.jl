mutable struct LPDE_solution{TT,ST,PT,IT} #<: AbstractPDESolution
    step::Int
    t::TT
    ntime::Int

    s::ST

    params::PT
    internal::IT
    function LPDE_solution(t,ics::NamedTuple,ntime::Int,params::PT;internal::IT = nothing, kwargs...) where {IT,PT}
        step = 1

        s = map(v -> (v, ntuple(_ -> zeros(size(v)...), ntime)...), ics)

        return new{typeof(t),typeof(s),typeof(params),typeof(internal)}(step, t, ntime, s, params, internal)
    end
end

function LPDE_solution(problem::PDEProblem;internal = nothing)
    ntime = Int((problem.tspan[2] - problem.tspan[1]) / problem.tstep)
    t = problem.tstep
    LPDE_solution(t,problem.ics, ntime, problem.params;internal)
end

ntime(sol::LPDE_solution) = sol.ntime

function Base.getindex(sol::LPDE_solution, n::Int)
    @assert n ≥ 0
    @assert n ≤ ntime(sol)

    return map(v -> v[n], sol.s)
end