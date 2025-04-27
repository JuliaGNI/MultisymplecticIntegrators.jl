mutable struct LPDE_solution{TT,ST,PT,IT} <: AbstractPDESolution
    current_step::Int
    t::TT
    ntime::Int

    sol::ST

    params::PT
    internal::IT
    function LPDE_solution(t,ics::NamedTuple,ntime::Int,params::PT,internal::IT;kwargs...) where {IT,PT}
        current_step = 1

        sol = map(v -> (v, ntuple(_ -> zeros(size(v)...), ntime)...), ics)

        return new{typeof(t),typeof(sol),typeof(params),typeof(internal)}(current_step, t, ntime, sol, params, internal)
    end
end

function LPDE_solution(problem::PDEProblem)
    ntime = Int((problem.tspan[2] - problem.tspan[1]) / problem.tstep)
    t = problem.tstep
    LPDE_solution(t,problem.ics_values, ntime, problem.params, problem.internal)
end

ntime(sol::LPDE_solution) = sol.ntime