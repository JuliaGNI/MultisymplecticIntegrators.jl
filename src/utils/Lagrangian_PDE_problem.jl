struct LPDEProblem <: PDEProblem
    lagrangian_system
    D::Int
    ics_function::Function
    bcs_function::Function
    
    ics_values::NamedTuple

    tspan::Tuple{Float64, Float64}
    tstep::Float64

    xspan::Tuple{Float64, Float64}
    xstep::Float64

    params
    internal

    exact_u
    function LPDEProblem(lag_sys,ics_function,bcs_function,ics_values,tspan, tstep, xspan, xstep, params,exact_u;internal = nothing)
        new(
            lag_sys,
            lag_sys.D,
            ics_function,
            bcs_function,
            ics_values,
            tspan,
            tstep,
            xspan,
            xstep,
            params,
            internal,
            exact_u
        )
    end
end

datatype(problem::LPDEProblem) = eltype(problem.ics_values.u)
timestep(problem::LPDEProblem) = problem.tstep
spacestep(problem::LPDEProblem) = problem.xstep