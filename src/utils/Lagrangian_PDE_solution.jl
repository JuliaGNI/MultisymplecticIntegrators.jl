struct LPDE_solution{TT,ST,PT,IT} <: GeometricPDESolution
    current_step::Int
    current_time::TT

    sol::ST

    params::PT
    internal::IT
    function LPDE_solution(ics::NamedTuple,ntime::Int,params::PT; internal::IT;kwargs...) where {IT,PT}
        sol = map(v -> (v, ntuple(_ -> zeros(size(v)...), ntime)...), ics)

        current_step = 1
        current_time = 0.0

        return new{typeof(current_time),typeof(sol),typeof(params),typeof(internal)}(current_step, current_time, sol, params, internal)
    end
end