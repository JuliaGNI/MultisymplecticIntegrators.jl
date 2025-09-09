"""
    Implementation of the PDE integrator with ELM method.
    Boundary condition and initial condition are imposed with least square method.
"""

struct ELM_PDE_int{BT<:AbstractPDEBasis} <: PDEMethod
    basis::BT
    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights

    function ELM_PDE_int(basis; RT::Int=6, RX::Int=8)
        if RT == 128
            t_quadrature = GaussQuadrature128()
        elseif RT == 64
            t_quadrature = GaussQuadrature64()
        else
            t_quadrature = QuadratureRules.GaussLegendreQuadrature(RT)
        end

        if RX == 128
            x_quadrature = GaussQuadrature128()
        elseif RX == 64
            x_quadrature = GaussQuadrature64()
        else
            x_quadrature = QuadratureRules.GaussLegendreQuadrature(RX)
        end

        dimensions = [RT, RX]
        grid_matrix, grid_weights = construct_quadrature_grid(dimensions)

        new{typeof(basis)}(basis,
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights)
    end
end


default_solver(::ELM_PDE_int) = Newton()

struct ELM_PDE_intCache{ST,RT,RX,D,NP} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    DX = dimension of input of u,v,w, i.e. number of spatial dimensions
    NP = number of parameters in the expression
    """
    x::Vector{ST}

    u_basis_quad_values::Array{ST} # (NP, RT, RX)
    v_basis_quad_values::Array{ST} # (NP, RT, RX)
    w_basis_quad_values::Array{ST} # (NP, RT, RX)

    ut₀_basis_quad_values::Array{ST}
    ux₀_basis_quad_values::Array{ST}
    ux₁_basis_quad_values::Array{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ut₀_quad_values::Matrix{ST} # bottom boundary, i.e. t = 0
    ux₀_quad_values::Matrix{ST} # left boundary, i.e. x = 0
    ux₁_quad_values::Matrix{ST} # right boundary, i.e. x = L

    init_condition_t₀::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}

    system_matrix::Array{ST}
    system_rhs::Vector{ST}
    function ELM_PDE_intCache{ST,RT,RX,D,NP}() where {ST,RT,RX,D,NP}
        x = zeros(ST, NP) # in ELM, x is just the output layer parameters

        u_basis_quad_values = zeros(ST, D, RT * RX, NP)
        v_basis_quad_values = zeros(ST, D, RT * RX, NP)
        w_basis_quad_values = zeros(ST, D, RT * RX, NP)

        ut₀_basis_quad_values = zeros(ST, D, RX, NP) # bottom boundary, i.e. t = 0
        ux₀_basis_quad_values = zeros(ST, D, RT, NP) # left boundary, i.e. x = 0
        ux₁_basis_quad_values = zeros(ST, D, RT, NP) # right boundary, i.e. x = L

        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)


        ut₀_quad_values = zeros(ST, D, RX) # bottom boundary, i.e. t = 0
        ux₀_quad_values = zeros(ST, D, RT) # left boundary, i.e. x = 0
        ux₁_quad_values = zeros(ST, D, RT) # right boundary, i.e. x = L

        init_condition_t₀ = zeros(ST, D, RX)
        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)

        system_matrix = zeros(ST, RT * RX +  RX + 2* RT, NP)
        system_rhs = zeros(ST, RT * RX +  RX + 2* RT)
        new(x,
            u_basis_quad_values, v_basis_quad_values, w_basis_quad_values,
            ut₀_basis_quad_values, ux₀_basis_quad_values, ux₁_basis_quad_values,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ut₀_quad_values, ux₀_quad_values, ux₁_quad_values,
            init_condition_t₀, boundary_condition_x₀, boundary_condition_x₁,
            system_matrix, system_rhs)
    end
end

nlsolution(cache::ELM_PDE_intCache) = cache.x

function Cache{ST}(problem::PDEProblem, int::ELM_PDE_int; kwargs...) where {ST}
    ELM_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.NP}(; kwargs...)
end

@inline GeometricIntegrators.Integrators.CacheType(ST, problem::PDEProblem, int::ELM_PDE_int) = ELM_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.NP}

@inline function Base.getindex(c::ELM_PDE_intCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

prior_initial_guess!(C, sol, int::PDEIntegrator{<:ELM_PDE_int}) = nothing

function initialize_bcs_ics!(sol,int::PDEIntegrator{<:ELM_PDE_int})
    local C = cache(int)
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local D = int.problem.D 
    local RT = int.method.RT
    local ic_fun = int.problem.ics_function
    local bc_fun = int.problem.bcs_function
    local current_step = sol.current_step
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]

    for d in 1:D
        # println("update initial condition, current time = ", sol.t, "the initial condition is at time = ", sol.t - timestep(int))

        if current_step ==1 
            C.init_condition_t₀[d,:] .= ic_fun(xspan[1] .+ x_domain .* x_quad_nodes).u
        else
            # println("sol.internal.x[current_step-1][1:NP] = " , sol.internal.x[current_step-1][1:NP])
            for i in eachindex(C.init_condition_t₀[d,:])
                C.init_condition_t₀[d,i] = sol.internal.end_quad[current_step-1][i]
            end
            # println("initial condition = " , C.init_condition_t₀[d,:])
        end

        for i in 1:RT
            C.boundary_condition_x₀[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.u
            C.boundary_condition_x₁[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.u
        end
        # println("left boundary condition = " , C.boundary_condition_x₀[d,:])
    end

end

function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:ELM_PDE_int}) where {ST}
    local v_basis_func = int.method.basis.v
    local w_basis_func = int.method.basis.w
    local u_basis_func = int.method.basis.u
    local grid_matrix = int.method.grid_matrix
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local nn_params = int.method.basis.u.params

    local ∂L∂U = int.problem.lagrangian_system.functions.∂L∂U
    local ∂L∂V = int.problem.lagrangian_system.functions.∂L∂V
    local ∂L∂W = int.problem.lagrangian_system.functions.∂L∂W
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local C = cache(int,ST)

    # Load values based on cache size
    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                idx = (rt - 1) * RX + rx
                C.u_basis_quad_values[d, idx, :] = u_basis_func([grid_matrix[rt, rx][1], xspan[1] + x_domain* grid_matrix[rt, rx][2]], nn_params)
                C.v_basis_quad_values[d, idx, :] = v_basis_func([grid_matrix[rt, rx][1], xspan[1] + x_domain* grid_matrix[rt, rx][2]], nn_params)
                C.w_basis_quad_values[d, idx, :] = w_basis_func([grid_matrix[rt, rx][1], xspan[1] + x_domain* grid_matrix[rt, rx][2]], nn_params)
            end
        end

        for j in 1:RX
            C.ut₀_basis_quad_values[d,j, :] = u_basis_func([0.0, xspan[1] + x_domain * x_quad_nodes[j]], nn_params) # bottom 
        end

        for i in 1:RT
            C.ux₀_basis_quad_values[d,i, :] = u_basis_func([t_quad_nodes[i], xspan[1]], nn_params)
            C.ux₁_basis_quad_values[d,i, :] = u_basis_func([t_quad_nodes[i], xspan[2]], nn_params)
        end
    end

    (x == cache(int).x && eltype(x) == eltype(cache(int).x)) ? post_initial_guess!(cache(int),sol,int) : nothing

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                idx = (rt - 1) * RX + rx
                C.u_quad_values[d, rt, rx] = sum(C.u_basis_quad_values[d, idx, :] .* x)
                C.v_quad_values[d, rt, rx] = sum(C.v_basis_quad_values[d, idx, :] .* x)
                C.w_quad_values[d, rt, rx] = sum(C.w_basis_quad_values[d, idx, :] .* x)
            end
        end
    end

    for d in 1:D
        for j in 1:RX
            C.ut₀_quad_values[d,j] = sum(C.ut₀_basis_quad_values[d,j, :] .* x)
        end

        for i in 1:RT
            C.ux₀_quad_values[d,i] =  sum(C.ux₀_basis_quad_values[d,i, :] .* x)
            C.ux₁_quad_values[d,i] =  sum(C.ux₁_basis_quad_values[d,i, :] .* x)
        end
    end

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], nn_params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], nn_params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], nn_params)
            end
        end 
    end




end


function post_initial_guess!(C, sol, int::PDEIntegrator{<:ELM_PDE_int})
    local lsq_assemble = int.problem.least_squares_assemble 
    local NP = int.method.basis.NP
    # Assemble the system matrix and rhs for least squares
    local cache_ = cache(int)

    lsq_assemble(int)
    function elm_lsq!(du,x,p)
        du= cache_.system_matrix * x .- cache_.system_rhs
    end

    prob = NonlinearLeastSquaresProblem(
        NonlinearFunction(elm_lsq!, resid_prototype = zeros(3)), zeros(NP), cache_)
    ls_sol = solve(prob)
    @show ls_sol
    C.x[:] = ls_sol.u
end

function residual!(b::Vector{ST}, sol, int::PDEIntegrator{<:ELM_PDE_int}) where {ST}
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX
    local NP = int.method.basis.NP

    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]

    for d in 1:D 
        for p in 1:NP
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    idx = (rt - 1) * RX + rx
                    z +=  quad_b[rt,rx] * 
                        ( x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx] * C.u_basis_quad_values[d,idx,p]
                        + x_domain                 * C.∂L∂V_quad_values[d,rt,rx] * C.v_basis_quad_values[d,idx,p]
                        + x_domain * timestep(int) * C.∂L∂W_quad_values[d,rt,rx] * C.w_basis_quad_values[d,idx,p])
                end
            end
            b[p] = -z
        end
    end
    
    # for d in 1:D
    #     for rx in 1:RX
    #         b[NP + (d - 1) * RX + rx] = C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx]
    #     end
    # end
    # for d in 1:D
    #     for rt in 1:RT
    #         b[NP+ D * RX+(d-1)*RT+rt]= C.ux₀_quad_values[d,rt] - C.boundary_condition_x₀[d,rt]
    #     end
    # end
    # for d in 1:D
    #     for rt in 1:RT
    #         b[NP+ D * RX+ D * RT + (d-1)*RT+rt]= C.ux₁_quad_values[d,rt] - C.boundary_condition_x₁[d,rt]
    #     end
    # end

end

function update!(sol_struct, int::PDEIntegrator{<:ELM_PDE_int})
    local D = int.problem.D
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local x = nlsolution(int)
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local v_basis_func = int.method.basis.v
    local w_basis_func = int.method.basis.w
    local u_basis_func = int.method.basis.u
    local nn_params = int.method.basis.u.params
    local RX = int.method.RX
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    x_nodes = collect(xspan[1]:xstep:xspan[2])
    for d in 1:D
        for i in eachindex(x_nodes)
            sol_struct.sol.u[sol_struct.current_step][i] = sum(u_basis_func([sol_struct.t, x_nodes[i]],nn_params) .* x)
            sol_struct.sol.v[sol_struct.current_step][i] = sum(v_basis_func([sol_struct.t, x_nodes[i]],nn_params) .* x)
            sol_struct.sol.w[sol_struct.current_step][i] = sum(w_basis_func([sol_struct.t, x_nodes[i]],nn_params) .* x)
        end
    end

    for d in 1:D
        for rx in 1:RX
            sol_struct.internal.end_quad[sol_struct.current_step] .= sum(u_basis_func([1.0, xspan[1] + x_domain* x_quad_nodes[rx]], nn_params) .* x)
        end
    end

    # copy internal variables from cache to solution
    sol_struct.internal.x[sol_struct.current_step] .= cache(int).x 
    sol_struct.t += int.problem.tstep
    # println("In the end of update! function, time = ", sol_struct.t)
end

function internal_variables(int::PDEIntegrator{<:ELM_PDE_int},problem::PDEProblem)
    local x = cache(int).x
    local init_condition_t₀ = cache(int).init_condition_t₀
    ntime = Int((problem.tspan[2] - problem.tspan[1]) / problem.tstep)
    xx = (x, ntuple( _ -> zeros(size(x)...), ntime)...)
    end_quad = (init_condition_t₀, ntuple( _ -> zeros(size(init_condition_t₀)...), ntime)...)

    return (x = xx, end_quad = end_quad)
end