"""
    Wave equation
    u_tt -  c^2 * u_xx = 0
"""


module Wave

    export lagrangian, hamiltonian, initial_condition, boundary_condition, lpdeproblem

    using Parameters: @unpack
    using LinearAlgebra
    using Symbolics
    using MultiSymplectic
    using Infiltrator
    using JLD2

    const D = 1
    const DX = 1

    const timestep = 0.3
    const timespan = (0.0, 3.0)

    const xstep = 0.01
    const xspan = (0.0,1.0)
    const c = 0.5 # wave speed square

    const default_parameters = (
        c=0.5,
        A1 = 0.8,
        A2 = 0.0,
        B1 = 0.8,
        B2 = 0.0,
        l = 1.0,
    )

    function exact_u(t,x;params = default_parameters)
        @unpack c, A1, A2, B1, B2, l = params
        (A1 * cos((pi*c*t)/l) + B1 * sin((pi*c*t)/l + pi/6)) * sin((pi*x)/l) + (A2 * cos((2*pi*c*t)/l) + B2 * sin((2*pi*c*t)/l + pi/6)) * sin((2*pi*x)/l)
    end

    function exact_v(t,x;params = default_parameters)
        @unpack c, A1, A2, B1, B2, l = params
        (-A1*c*pi*sin((pi*x) / l)*sin((c*pi*t) / l) - (2//1)*A2*c*pi*sin((2*pi*c*t) / l)*sin((2*pi*x) / l) + B1*c*pi*sin((pi*x) / l)*cos(((1//6)*pi*l + c*pi*t) / l) + (2//1)*B2*c*pi*cos(((1//6)*pi*l + (2//1)*c*pi*t) / l)*sin((2*pi*x) / l)) / l
    end

    function exact_w(t,x;params = default_parameters)
        @unpack c, A1, A2, B1, B2, l = params
        (A1*pi*cos((pi*x) / l)*cos((c*pi*t) / l) + (2//1)*A2*pi*cos((2*pi*c*t) / l)*cos((2*pi*x) / l) + B1*pi*cos((pi*x) / l)*sin(((1//6)*pi*l + c*pi*t) / l) + (2//1)*B2*pi*sin(((1//6)*pi*l + (2//1)*c*pi*t) / l)*cos((2*pi*x) / l)) / l
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
        u₀ = exact_u(timespan[1], x)
        v₀ = exact_v(timespan[1], x)
        w₀ = exact_w(timespan[1], x)
        return (u=u₀, v=v₀, w=w₀)
    end

    function initial_condition(x::Vector{Float64})

        u₀ = [exact_u(timespan[1], xi) for xi in x]
        v₀ = [exact_v(timespan[1], xi) for xi in x]
        w₀ = [exact_w(timespan[1], xi) for xi in x]
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

    function boundary_condition(t,xspan::Tuple)
        bc₀ = left_boundary_condition(t,xspan)
        bc₁ = right_boundary_condition(t,xspan)
        return (bc₀=bc₀, bc₁=bc₁)
    end

    # Lagrangian and Hamiltonian density
    function lagrangian(t, x, u, v, w, params)
        @unpack c, A1, A2, B1, B2, l = params
        1 / 2 * (v[1]^2 - c^2 * w[1]^2)
    end

    function hamiltonian(t, x, u, v, w, params)
        @unpack c, A1, A2, B1, B2, l = params
        1 / 2 * (v[1]^2 + c^2 * w[1]^2)  
    end

    function lpdeproblem(; lagrangian_function=lagrangian, initial_condition_function=initial_condition, boundary_condition_function=boundary_condition, timespan=timespan, timestep::Float64=timestep, xspan::Tuple=xspan, xstep::Float64=xstep, params=default_parameters,
        exact_u = exact_u, exact_v = exact_v, exact_w = exact_w,least_squares_assemble = problem_matrix_assemble)
        # @assert timestep^2 < c * xstep^2 "timestep^2 < c*xstep^2 must hold for CFL condition"
        @assert timespan[1] < timespan[2] "timespan must be increasing"
        @assert xspan[1] < xspan[2] "xspan must be increasing"

        t, x, U, V, W = LPDE_variables(1, 1) # U,V,W does not have t,x dependence 
        lag_sys = LPDESystem(lagrangian_function(t, x, U, V, W, params), t, x, U, V, W, params)

        x_nodes = collect(xspan[1]:xstep:xspan[2])
        ics = initial_condition_function(x_nodes)

        LPDEProblem(lag_sys, initial_condition_function, boundary_condition_function, ics, timespan, timestep, xspan, xstep, params,exact_u,exact_v,exact_w,least_squares_assemble)
    end

    function problem_matrix_assemble(C,int::PDEIntegrator{<:ELM_PDE_int},sol)
        local RT = int.method.RT
        local RX = int.method.RX
        local NP = int.method.basis.NP
        local u_basis = int.method.basis.u_basis
        local show_status = int.method.show_status
        local h = int.problem.timestep
        local tn = sol.t - h
        local xspan = int.problem.xspan
        local x_domain = xspan[2] - xspan[1]
        local exact_u = int.problem.exact_u
        local t_quad_nodes = int.method.time_quadrature.nodes
        local x_quad_nodes = int.method.spatial_quadrature.nodes

        utt_basis_quad_values = zeros(RT*RX, NP);
        uxx_basis_quad_values = zeros(RT*RX, NP);
        for rt in 1:RT
            for rx in 1:RX
                    hess = vector_hessian(u_basis,[t_quad_nodes[rt], xspan[1] + x_domain * x_quad_nodes[rx]])
                    utt_basis_quad_values[(rt-1)*RX + rx, :] = hess[:,1,1]
                    uxx_basis_quad_values[(rt-1)*RX + rx, :] = hess[:,2,2]
            end
        end

        C.system_matrix[1:RT * RX, :] = utt_basis_quad_values -  c^2 * uxx_basis_quad_values
        C.system_matrix[RT * RX + 1:RT * RX + RX, :] = C.ut₀_basis_quad_values
        C.system_matrix[RT * RX + RX + 1:RT * RX + RX + RT, :] = C.ux₀_basis_quad_values
        C.system_matrix[RT * RX + RX + RT + 1:RT * RX + RX + 2 * RT, :] = C.ux₁_basis_quad_values

        C.system_rhs[RT * RX + 1:RT * RX + RX] = C.init_condition_t₀[1,:]'
        C.system_rhs[RT * RX + RX + 1:RT * RX + RX + RT] = C.boundary_condition_x₀[1,:]'
        C.system_rhs[RT * RX + RX + RT + 1:RT * RX + RX + 2 * RT] = C.boundary_condition_x₁[1,:]'
    
        C.x[1:NP] = C.system_matrix \ C.system_rhs


        if show_status
            @show C.x[1:NP]
            elm_pred = zeros(Float64, RT, RX)
            truth = zeros(Float64, RT, RX)
            for rt in 1:RT
                for rx in 1:RX
                    elm_pred[rt, rx] = sum(C.x[1:NP] .* C.u_basis_quad_values[1, :, rt, rx])
                    truth[rt, rx] = exact_u(tn + h * t_quad_nodes[rt], xspan[1] + x_domain * x_quad_nodes[rx])
                end
            end
            @show maximum(abs.(elm_pred - truth))
            # u_func.params[keys(u_func.params)[end]].W[:] = C.x[1:NP]

            record_results = Dict()
            record_results["max_error"] = maximum(abs.(elm_pred - truth))
            record_results["u_basis_params"] = u_basis.params
            record_results["x"] = C.x[1:NP]
            JLD2.save("LS_initial_guess_results.jld2", record_results)

        end
    end

end


