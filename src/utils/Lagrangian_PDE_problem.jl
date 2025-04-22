struct LPDEProblem <: PDEProblem
    lagrangian_system
    tspan::Tuple{Float64, Float64}
    tstep::Float64
    xspan::Vector{Tuple{Float64, Float64}}

    
end