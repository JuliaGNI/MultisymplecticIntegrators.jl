mutable struct LPDE_solution{TT,ST,PT,IT} <: GeometricPDESolution
    current_step::Int
    current_time::TT
    ntime::Int

    sol::ST

    params::PT
    internal::IT
    function LPDE_solution(ics::NamedTuple,ntime::Int,params::PT; internal::IT;kwargs...) where {IT,PT}
        sol = map(v -> (v, ntuple(_ -> zeros(size(v)...), ntime)...), ics)

        current_step = 1
        current_time = 0.0

        return new{typeof(current_time),typeof(sol),typeof(params),typeof(internal)}(current_step, current_time, ntime, sol, params, internal)
    end
end

function LPDE_solution(problem::PDEProblem)
    ntime = div(problem.tspan[2] - problem.tspan[1], problem.tstep)
    @assert ntime isa Int

    LPDE_solution(problem.ics_values, ntime, problem.params; problem.internal)
end

ntime(sol::LPDE_solution) = sol.ntime