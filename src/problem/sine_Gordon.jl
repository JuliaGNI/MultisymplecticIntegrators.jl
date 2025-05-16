"""
    Nonlinear Sine-Gordon equation
    c* u_tt - u_xx = - sin(u) = d(V(u))/du = d(1+cos(u)) / du
"""


module SineGordon

    export lagrangian, hamiltonian, initial_condition, boundary_condition, lpdeproblem

    using Parameters: @unpack
    using LinearAlgebra
    using Symbolics
    using MultiSymplectic

    const D = 1
    const DX = 1

    const tstep = 1.0
    const tspan = (0.0, 10.0)

    const xstep = 0.01
    const xspan = (0., 1.0)

    const c = 4.0 # wave speed square
    const velocity = 1.0
    const γ = 1 / sqrt(1 - velocity^2 / c)

    const default_parameters = (
        c=4.0,
        velocity = 1.0,
        γ = 1 / sqrt(1 - velocity^2 / c),
    )

    function exact_u(t,x;params = default_parameters)
        @unpack c, velocity, γ = params
        4 * atan(exp(γ * (x - velocity * t)))
    end

    function exact_v(t,x;params = default_parameters)
        @unpack c, velocity, γ = params
        (-4 * γ * velocity * exp(γ * (x - velocity * t))) / (1 + exp(γ * (x - velocity * t))^2)
    end

    function exact_w(t,x;params = default_parameters)
        @unpack c, velocity, γ = params
        (4 * γ * exp(γ * (x - velocity * t))) / (1 + exp(γ * (x - velocity * t))^2)
    end

    function exact_solution(t::Float64, x::Float64)
        (u=exact_u(t, x), v=exact_v(t, x), w=exact_w(t, x))
    end

    function exact_solution(t::Vector{Float64}, x::Float64)
        (u=[exact_u(ti, x) for ti in t], v=[exact_v(ti, x) for ti in t], w=[exact_w(ti, x) for ti in t])
    end

    function exact_solution(t::Float64, x::Vector{Float64})
        (u=[exact_u(t, xi) for xi in x], v=[exact_v(t, xi) for xi in x], w=[exact_w(t, xi) for xi in x])
    end

    function exact_solution(t::Vector{Float64}, x::Vector{Float64})
        (u=[exact_u(ti, xi) for ti in t, xi in x],
        v=[exact_v(ti, xi) for ti in t, xi in x],
        w=[exact_w(ti, xi) for ti in t, xi in x])
    end

    function initial_condition(x::Float64)
        u₀ = exact_u(tspan[1], x)
        v₀ = exact_v(tspan[1], x)
        w₀ = exact_w(tspan[1], x)
        return (u=u₀, v=v₀, w=w₀)
    end

    function initial_condition(x::Vector{Float64})

        u₀ = [exact_u(tspan[1], xi) for xi in x]
        v₀ = [exact_v(tspan[1], xi) for xi in x]
        w₀ = [exact_w(tspan[1], xi) for xi in x]
        return (u=u₀, v=v₀, w=w₀)
    end

    function left_boundary_condition(t::Float64,xspan::Tuple)
        u₀ = exact_u(t, xspan[1])
        v₀ = exact_v(t, xspan[1])
        w₀ = exact_w(t, xspan[1])
        return (u=u₀, v=v₀, w=w₀)
    end

    function left_boundary_condition(t::Vector{Float64},xspan::Tuple)
        u₀ = [exact_u(ti, xspan[1]) for ti in t]
        v₀ = [exact_v(ti, xspan[1]) for ti in t]
        w₀ = [exact_w(ti, xspan[1]) for ti in t]
        return (u=u₀, v=v₀, w=w₀)
    end

    function right_boundary_condition(t::Float64,xspan::Tuple)
        u₀ = exact_u(t, xspan[2])
        v₀ = exact_v(t, xspan[2])
        w₀ = exact_w(t, xspan[2])
        return (u=u₀, v=v₀, w=w₀)
    end

    function right_boundary_condition(t::Vector{Float64},xspan::Tuple)
        u₀ = [exact_u(ti, xspan[2]) for ti in t]
        v₀ = [exact_v(ti, xspan[2]) for ti in t]
        w₀ = [exact_w(ti, xspan[2]) for ti in t]
        return (u=u₀, v=v₀, w=w₀)
    end

    function boundary_condition(t,xspan)
        bc₀ = left_boundary_condition(t,xspan)
        bc₁ = right_boundary_condition(t,xspan)
        return (bc₀=bc₀, bc₁=bc₁)
    end


    function lagrangian(t, x, u, v, w, params)
        @unpack c = params
        1 / 2 * (c * v[1]^2 - w[1]^2) + (1 + cos(u[1]))
    end

    function hamiltonian(t, x, u, v, w, params)
        @unpack c = params
        1 / 2 * (c * v[1]^2 + w[1]^2) - (1 + cos(u[1]))
    end

    function lpdeproblem(; lagrangian_function=lagrangian, initial_condition_function=initial_condition, boundary_condition_function=boundary_condition, tspan=tspan, tstep::Float64=tstep, xspan::Tuple=xspan, xstep::Float64=xstep, params=default_parameters)
        @unpack c = params
        # @assert tstep^2 < c * xstep^2 "tstep^2 < c*xstep^2 must hold for CFL condition"
        @assert tspan[1] < tspan[2] "tspan must be increasing"
        @assert xspan[1] < xspan[2] "xspan must be increasing"

        t, x, U, V, W = LPDE_variables(1, 1) # U,V,W does not have t,x dependence 
        lag_sys = LPDESystem(lagrangian_function(t, x, U, V, W, params), t, x, U, V, W, params)

        x_nodes = collect(xspan[1]:xstep:xspan[2])
        ics = initial_condition_function(x_nodes)

        LPDEProblem(lag_sys, initial_condition_function, boundary_condition_function, ics, tspan, tstep, xspan, xstep, params)
    end

end


