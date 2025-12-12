"""
    Linear Tansport Problem
    u_t + c(t,x) u_x = 0
    where c(t,x) is a speed of propagation, in our case, c is constant c = 0.02.
    Since when we want to represent the solution as a neural network exactly, the parameters dependend on c.
    Given initial condition u(0,x) = u₀(x), we can solve this problem analytically, u(t,x) = u₀(x - c * t).
    
"""

using MultiSymplectic:ELM_PDE_int

module LinearTransport
 
    export lagrangian, hamiltonian, initial_condition, boundary_condition, lpdeproblem

    using Parameters: @unpack
    using LinearAlgebra
    using Symbolics
    using MultiSymplectic
    using BSplineKit
    using AbstractNeuralNetworks
    using Zygote

    const D = 1
    const DX = 1

    const tstep = 1.0
    const tspan = (0.0, 10.0)

    const xstep = 0.01
    # const xspan = (0., 1.0) # for Exact solution 1 ,2 
    const xspan = (-0.5, 0.0) # for Exact solution 3

    const c = 0.2

    const default_parameters = (
        c = 0.2,
    )

    relu2(x) = max(0, x) ^2

    # Exact solution 1
    # The Initial condition is either set as a BSpline Basis function, or a network approximation of the Basis function.\
    # B = BSplineBasis(BSplineOrder(3), [0, 0, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1, 1, 1],augment = Val(false))
    # function exact_u(t,x;params = default_parameters)
    #     @unpack c, = params
    #     B[4](x-c*t)
    # end

    # function exact_v(t,x;params = default_parameters)
    #     @unpack c, = params
    #     -c * B[4](x - c*t, Derivative(1))
    # end

    # function exact_w(t,x;params = default_parameters)
    #     @unpack c, = params
    #     B[4](x - c * t, Derivative(1))
    # end

    # Exact solution 2
    # The parameters of the neural network are set to approximate the above BSpline Basis B[4], from OGA
    # NN = Chain(
    #     Dense(2, 8, relu2),
    #     Dense(8, 1, identity, use_bias=false)
    # )
    # PNN = NeuralNetwork(NN)
    # PNN.params.L1.W[:,2] .= 1.0 
    # PNN.params.L1.W[:,1] .= - c
    # PNN.params.L1.b[:] = [-0.0000, -0.2930, -0.5664, -0.1328, -0.7207, -0.3965, -0.4893, -0.6426]
    # PNN.params.L2.W[:] = [0.3275,   52.7569,   -6.1713,   -1.7977,    6.4468, -153.8942,137.0191,  -34.2571]

    # function exact_u(t,x;params = default_parameters)
    #     @unpack c, = params
    #     PNN([t,x])[1]
    # end

    # function exact_v(t,x;params = default_parameters)
    #     Zygote.gradient(t -> PNN([t, x])[1], t)[1]    
    # end

    # function exact_w(t,x;params = default_parameters)
    #     Zygote.gradient(x -> PNN([t, x])[1], x)[1]    
    # end

    # Exact solution 3 : Specific example for symbolic integrator.
    function exact_u(t,x;params = default_parameters)
        @unpack c = params
        0.5 * exp(-(x+2 - c*t)^2) / sqrt(π)   
    end

    function exact_v(t,x;params = default_parameters)
        @unpack c = params
        c*(2 + x - c*t)*exp(-((2 + x - c*t)^2)) / sqrt(π)
    end

    function exact_w(t,x;params = default_parameters)
        @unpack c = params
        -(2 + x - c*t)*exp(-((2 + x - c*t)^2)) / sqrt(π)
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
        1 / 2 * u[1] *(v[1] + c * w[1])
    end

    function hamiltonian(t, x, u, v, w, params)
        @unpack c = params
        -1/2 * c * u[1] * w[1]
    end

    function eqn_residual(u,v,w,params)
        @unpack c = params
        return v .- c .* w
    end

    function lpdeproblem(; lagrangian_function=lagrangian, initial_condition_function=initial_condition, boundary_condition_function=boundary_condition, tspan=tspan, tstep::Float64=tstep, xspan::Tuple=xspan, xstep::Float64=xstep, params=default_parameters,
        exact_u_func=exact_u,least_squares_assemble = problem_matrix_assemble)
        @unpack c = params
        # @assert tstep^2 < c * xstep^2 "tstep^2 < c*xstep^2 must hold for CFL condition"
        @assert tspan[1] < tspan[2] "tspan must be increasing"
        @assert xspan[1] < xspan[2] "xspan must be increasing"

        t, x, U, V, W = LPDE_variables(1, 1) # U,V,W does not have t,x dependence 
        lag_sys = LPDESystem(lagrangian_function(t, x, U, V, W, params), t, x, U, V, W, params)

        x_nodes = collect(xspan[1]:xstep:xspan[2])
        ics = initial_condition_function(x_nodes)

        LPDEProblem(lag_sys, initial_condition_function, boundary_condition_function, ics, tspan, tstep, xspan, xstep, params, exact_u_func, least_squares_assemble)
    end

    function problem_matrix_assemble(int::PDEIntegrator{<:ELM_PDE_int})
        local C = cache(int)
        local RT = int.method.RT
        local RX = int.method.RX

        C.system_matrix[1:RT * RX, :] = C.v_basis_quad_values .+ c * C.w_basis_quad_values
        C.system_matrix[RT * RX + 1:RT * RX + RX, :] = C.ut₀_basis_quad_values
        C.system_matrix[RT * RX + RX + 1:RT * RX + RX + RT, :] = C.ux₀_basis_quad_values
        C.system_matrix[RT * RX + RX + RT + 1:RT * RX + RX + 2 * RT, :] = C.ux₁_basis_quad_values

        C.system_rhs[RT * RX + 1:RT * RX + RX] = C.init_condition_t₀[1,:]'
        C.system_rhs[RT * RX + RX + 1:RT * RX + RX + RT] = C.boundary_condition_x₀[1,:]'
        C.system_rhs[RT * RX + RX + RT + 1:RT * RX + RX + 2 * RT] = C.boundary_condition_x₁[1,:]'
    end

end


