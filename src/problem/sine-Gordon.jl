"""
    Nonlinear Sine-Gordon equation
    c* u_tt - u_xx = - sin(u) = d(V(u))/du = d(1+cos(u)) / du
"""



module SineGordon
    using Parameters: @unpack
    using EulerLagrange
    using LinearAlgebra
    using Symbolics

    const D = 1
    const DX = 1

    const tstep = 0.01
    const tspan = (0.0, 1.0)
    const xspan = (0.0, 1.0)

    const c = 4.0 # wave speed square
    const velotity = 1.0
    const γ = 1/sqrt(1-velocity^2/c)

    const default_parameters = (
        c=c, 
    )

    @variables t x 
    exact_u = 4 * atan(exp(γ*(x - velocity*t))) # kink solution
    exact_v = Symbolics.derivative(exact_u, t)
    exact_w = Symbolics.derivative(exact_u, x)

    exact_u = Symbolics.eval(Symbolics.build_function(exact_u,t,x))
    exact_v = Symbolics.eval(Symbolics.build_function(exact_v,t,x))
    exact_w = Symbolics.eval(Symbolics.build_function(exact_w,t,x))
    
    function exact_solution(t::Float64,x::Float64)
        (u = exact_u(t,x), v = exact_v(t,x), w = exact_w(t,x))
    end

    function exact_solution(t::Vector{Float64},x::Float64)
        (u = [exact_u(ti,x) for ti in t], v = [exact_v(ti,x) for ti in t], w = [exact_w(ti,x) for ti in t])
    end

    function exact_solution(t::Float64,x::Vector{Float64})
        (u = [exact_u(t,xi) for xi in x], v = [exact_v(t,xi) for xi in x], w = [exact_w(t,xi) for xi in x])
    end

    function exact_solution(t::Vector{Float64},x::Vector{Float64})
        (u = [exact_u(ti,xi) for ti in t, xi in x], 
         v = [exact_v(ti,xi) for ti in t, xi in x], 
         w = [exact_w(ti,xi) for ti in t, xi in x])
    end

    function lagrangian(t, x, u, v, w, params)
        @unpack c = params
        1 / 2 * (c * v[1]^2 - w[1,1]^2) + (1 + cos(u[1]))
    end

    function hamiltonian(t, x, u, v, w, params)
        @unpack c = params
        1 / 2 * (c * v[1]^2 + w[1]^2) - (1 + cos(u[1]))
    end

    function lpdeproblem(u₀=u₀, v₀=v₀, w₀=w₀; tspan=tspan, tstep=tstep, xspan=xspan, xstep=xstep, parameters=default_parameters)
        @assert tstep < xstep "tstep must be less than xstep for stability"
        @assert tspan[1] < tspan[2] "tspan must be increasing"
        @assert xspan[1] < xspan[2] "xspan must be increasing"

        t,x,U,V,W = lagrangianPDE_variables(1, 1) # U,V,W does not have t,x dependence 
        sparams = symbolize(parameters)
        lag_sys = LagrangianPDESystem(lagrangian(t,x,U,V,W, sparams), t,x,U,V,W, sparams)
        #TODO LPDEProblem(lag_sys, tspan, tstep, xspan=xspan, xstep=xstep, u₀, v₀, w₀; v̄=θ̇, parameters=parameters)
    end

    export lagrangian, hamiltonian
end


sol = [exact_u(ti,xi) for ti in 0:0.1:1, xi in 0:0.1:2]