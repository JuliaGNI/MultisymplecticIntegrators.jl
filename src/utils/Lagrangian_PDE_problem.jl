struct LPDEProblem <: PDEProblem
    lagrangian_system
    ics_function::Function
    bcs_function::Function

    tspan::Tuple{Float64, Float64}
    tstep::Float64

    xstep::Float64
    xspan::Vector{Tuple{Float64, Float64}}

    params
    function LPDEProblem(lag_sys,ics_function,bcs_function,tspan, tstep, xspan, xstep, params)
        new(
            lagrangian_system = lag_sys,
            ics_function = ics_function,
            bcs_function = bcs_function,
            tspan = tspan,
            tstep = tstep,
            xspan = xspan,
            xstep = xstep,
            params = params
        )
    end
end