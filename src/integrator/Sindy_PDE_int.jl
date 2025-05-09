struct Sindy_PDE_Integrator{T,MVT,LT,BT<:AbstractPDEBasis} <: PDEMethod

    symbolic_expr_basis::BT
    time_quadrature::QuadratureRule{T}
    RT::Int # Number of quadrature points in time

    spatial_quadrature::QuadratureRule{T}
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions
    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights
    
    k_μ::Int
    μ₀_t::MVT
    μ₁_t::MVT

    k_λ₀_x::Int
    λ₀_x::LT

    init_w::Vector{T}
    mλ₀_x
    mμ_t
    function Sindy_PDE_Integrator(basis,init_w::Vector{T};RT::Int = 6,RX::Int = 8,xspan::Tuple = (0.,1.0),tstep::Float64 = 1.0, k_μ::Int = 4,k_λ₀_x::Int = 4,μ::Symbol = :Spline,λ::Symbol= :Spline) where {T}
        t_quadrature = QuadratureRules.GaussLegendreQuadrature(RT)
        x_quadrature = QuadratureRules.GaussLegendreQuadrature(RX)

        dimensions = [RT,RX]  
        grid_matrix, grid_weights = construct_quadrature_grid(dimensions)

        μ₀_t = Lagrangian_multiplier(μ,k_μ,tstep .* t_quadrature.nodes)
        μ₁_t = Lagrangian_multiplier(μ,k_μ,tstep .* t_quadrature.nodes)
        λ₀_x = Lagrangian_multiplier(λ,k_λ₀_x,xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes)

        mλ₀_x = zeros(RX, RX)
        mμ_t = zeros(RT, RT)

        for i in 1:RX
            mλ₀_x[i,:] = λ₀_x.b[i].(xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes)
        end
    
        for i in 1:RT
            mμ_t[i,:] = μ₀_t.b[i].(tstep .* t_quadrature.nodes)
        end

        new{T,typeof(μ₀_t),typeof(λ₀_x),typeof(basis)}(basis, t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            k_μ, μ₀_t, μ₁_t,
            k_λ₀_x,λ₀_x, #λ₁_x,
            init_w,
            mλ₀_x, mμ_t)
    end 
end

default_solver(::Sindy_PDE_Integrator) = Newton()

struct Sindy_PDE_IntegratorCache{ST,RT,RX,D,NP} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    DX = dimension of input of u,v,w, i.e. number of spatial dimensions
    NP = number of parameters in the expression
    """
    x::Vector{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ∂u∂P_quad_values
    ∂v∂P_quad_values
    ∂w∂P_quad_values

    λ₀_x_coes::Matrix{ST}

    μ₀_t_coes::Matrix{ST}
    μ₁_t_coes::Matrix{ST}

    λ₀_quad_values::Matrix{ST} 
    μ₀_quad_values::Matrix{ST}
    μ₁_quad_values::Matrix{ST} 

    ∂u∂P_t₀_quad_values
    ∂u∂P_t₁_quad_values
    ∂u∂P_x₀_quad_values
    ∂u∂P_x₁_quad_values

    ut₀_quad_values::Matrix{ST} # bottom boundary, i.e. t = 0
    ut₁_quad_values::Matrix{ST} # top boundary, i.e. t = T
    vt₀_quad_values::Matrix{ST}
    vt₁_quad_values::Matrix{ST}
    wt₀_quad_values::Matrix{ST}
    wt₁_quad_values::Matrix{ST}

    ux₀_quad_values::Matrix{ST} # left boundary, i.e. x = 0
    ux₁_quad_values::Matrix{ST} # right boundary, i.e. x = L
    vx₀_quad_values::Matrix{ST}
    vx₁_quad_values::Matrix{ST}
    wx₀_quad_values::Matrix{ST}
    wx₁_quad_values::Matrix{ST}

    mλ₀_x::Matrix{ST} # λ₀_x values at quadrature points
    mμ_t::Matrix{ST} # μ₀_t values at quadrature points

    tem_P::Vector{Vector{ST}} # temporary storage for P values

    init_condition_t₀::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}


    function Sindy_PDE_IntegratorCache{ST,RT,RX,D,NP}(P_sizes) where {ST,RT,RX,D,NP}
        # x = zeros(ST, NP + 2 * D * RX + 2* D * DX * RT ) # TODO: how to deal with RX being a vector/
        x = zeros(ST,NP + D * RX +2* D * RT) # params, λ₀_x_coes,μ₀_t_coes,μ₁_t_coes
        # TODO:consider when DX is a vector
        
        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ∂u∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,P_sizes)
        ∂v∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,P_sizes)
        ∂w∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,P_sizes)
        
        λ₀_x_coes = zeros(ST, D, RX)

        μ₀_t_coes = zeros(ST, D, RT)
        μ₁_t_coes = zeros(ST, D, RT)

        λ₀_quad_values = zeros(ST, D, RX) 
        μ₀_quad_values = zeros(ST, D, RT) 
        μ₁_quad_values = zeros(ST, D, RT) 

        ∂u∂P_t₀_quad_values = create_boundary_derivative_vector(ST, D, RX,P_sizes) 
        ∂u∂P_t₁_quad_values = create_boundary_derivative_vector(ST, D, RX,P_sizes) 
        ∂u∂P_x₀_quad_values = create_boundary_derivative_vector(ST, D, RT,P_sizes) 
        ∂u∂P_x₁_quad_values = create_boundary_derivative_vector(ST, D, RT,P_sizes) 

        ut₀_quad_values = zeros(ST,D,RX) # bottom boundary, i.e. t = 0
        ut₁_quad_values = zeros(ST,D,RX) # top boundary, i.e. t = T
        vt₀_quad_values = zeros(ST,D,RX)
        vt₁_quad_values = zeros(ST,D,RX)
        wt₀_quad_values = zeros(ST,D,RX)
        wt₁_quad_values = zeros(ST,D,RX)

        ux₀_quad_values = zeros(ST,D,RT) # left boundary, i.e. x = 0
        ux₁_quad_values = zeros(ST,D,RT) # right boundary, i.e. x = L
        vx₀_quad_values = zeros(ST,D,RT)
        vx₁_quad_values = zeros(ST,D,RT)
        wx₀_quad_values = zeros(ST,D,RT)
        wx₁_quad_values = zeros(ST,D,RT)

        mλ₀_x = zeros(ST,RX, RX)
        mμ_t = zeros(ST,RT, RT)

        tem_P = create_tem_vector(ST, D, P_sizes)

        init_condition_t₀ = zeros(ST, D, RX)

        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)

        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂P_quad_values, ∂v∂P_quad_values, ∂w∂P_quad_values,
            λ₀_x_coes, μ₀_t_coes,μ₁_t_coes, 
            λ₀_quad_values, μ₀_quad_values, μ₁_quad_values,
            ∂u∂P_t₀_quad_values, ∂u∂P_t₁_quad_values, ∂u∂P_x₀_quad_values, ∂u∂P_x₁_quad_values,
            ut₀_quad_values, ut₁_quad_values,vt₀_quad_values, vt₁_quad_values,wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values,vx₀_quad_values, vx₁_quad_values,wx₀_quad_values, wx₁_quad_values,
            mλ₀_x, mμ_t,
            tem_P,
            init_condition_t₀, 
            boundary_condition_x₀, boundary_condition_x₁)
    end
end

nlsolution(cache::Sindy_PDE_IntegratorCache) = cache.x

function Cache{ST}(problem::PDEProblem, int::Sindy_PDE_Integrator; kwargs...) where {ST}
    Sindy_PDE_IntegratorCache{ST,int.RT,int.RX,problem.D,int.symbolic_expr_basis.NP}(int.symbolic_expr_basis.P_sizes; kwargs...)
end

#{ST,RT,RX,D,NP}(P_sizes) where {ST,RT,RX,D,NP}
@inline GeometricIntegrators.Integrators.CacheType(ST, problem::PDEProblem, int::Sindy_PDE_Integrator) = Sindy_PDE_IntegratorCache{ST,int.RT,int.RX,problem.D,int.symbolic_expr_basis.NP}

@inline function Base.getindex(c::Sindy_PDE_IntegratorCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end


function prior_initial_guess!(C,sol_struct,int::PDEIntegrator{<:Sindy_PDE_Integrator})
    local P_sizes = int.method.symbolic_expr_basis.P_sizes
    local init_w = int.method.init_w
    local internal = sol_struct.internal
    local current_step = sol_struct.current_step

    start_idx = 1
    for (d,P_size) in enumerate(P_sizes)
        if current_step == 1
            C.x[start_idx:start_idx+P_size-1]= init_w[:]
        else
            C.x[start_idx:start_idx+P_size-1] .= internal.x[current_step-1][start_idx:start_idx+P_size-1]
        end

        start_idx += P_size
    end
end

function initialize_bcs_ics!(sol,int::PDEIntegrator{<:Sindy_PDE_Integrator})
    local C = cache(int)
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local u = int.method.symbolic_expr_basis.u # f = f(Parameters,t,x)
    local D = int.problem.D 
    local RT = int.method.RT
    local ic_fun = int.problem.ics_function
    local bc_fun = int.problem.bcs_function
    local current_step = sol.current_step
    local NP = int.method.symbolic_expr_basis.NP

    for d in 1:D
        # println("update initial condition, current time = ", sol.t, "the initial condition is at time = ", sol.t - timestep(int))

        if current_step ==1 
            C.init_condition_t₀[d,:] .= ic_fun(x_quad_nodes).u
        else
            # println("sol.internal.x[current_step-1][1:NP] = " , sol.internal.x[current_step-1][1:NP])

            for i in eachindex(C.init_condition_t₀[d,:])
                C.init_condition_t₀[d,i] = u[d]([sol.internal.x[current_step-1][1:NP]],sol.t - timestep(int),x_quad_nodes[i])
            end
            # println("initial condition = " , C.init_condition_t₀[d,:])
        end

        for i in 1:RT
            C.boundary_condition_x₀[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i]).bc₀.u
            C.boundary_condition_x₁[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i]).bc₁.u
        end
        # println("left boundary condition = " , C.boundary_condition_x₀[d,:])
    end

end

function post_initial_guess!(C,sol_struct,int::PDEIntegrator{<:Sindy_PDE_Integrator},int_method::Sindy_PDE_Integrator{T,MVT,LT,BT}) where {T,MVT<:BSplineDirichlet{T},LT<:BSplineDirichlet{T},BT}
    local NP = int_method.symbolic_expr_basis.NP
    local RT = int_method.RT
    local RX = int_method.RX
    local D = int.problem.D 
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params
    local xspan = int.problem.xspan
    local x_quad_nodes = int_method.spatial_quadrature.nodes
    local k_λ₀_x = int_method.k_λ₀_x
    local t_quad_nodes = int_method.time_quadrature.nodes
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    # for d in 1:D
    #     for i in 1:RX
    #         C.x[NP + (d - 1) * RX + i] = lag_sys.∂L∂V[d](C.ut₀_quad_values[d,i], C.vt₀_quad_values[d,i], C.wt₀_quad_values[d,i], params)
    #     end

    #     for i in 1:RT
    #         C.x[NP+ D * RX+(d-1)*RT+i] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,i], C.vx₀_quad_values[d,i], C.wx₀_quad_values[d,i], params)
    #         C.x[NP+ D * RX+ D * RT + (d-1)*RT+i] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,i], C.vx₁_quad_values[d,i], C.wx₁_quad_values[d,i], params)
    #     end
    # end

    for d in 1:D
        tem_t₀_∂L∂V = zeros(RX)
        for i in 1:RX
            tem_t₀_∂L∂V[i] = lag_sys.∂L∂V[d](C.ut₀_quad_values[d,i], C.vt₀_quad_values[d,i], C.wt₀_quad_values[d,i], params)
        end
    
        tem_x = xspan[1] .+ x_domain .* x_quad_nodes
        λ₀_x = interpolate(tem_x, tem_t₀_∂L∂V, BSplineOrder(k_λ₀_x))
        for i in 1:RX
            C.x[NP + (d - 1) * RX + i] = λ₀_x.spline.coefs[i]
        end

        tem_x₀_∂L∂W = zeros(RT)
        for i in 1:RT
            tem_x₀_∂L∂W[i] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,i], C.vx₀_quad_values[d,i], C.wx₀_quad_values[d,i], params)
        end

        tem_t = sol_struct.t .- timestep(int) .+ timestep(int) .* t_quad_nodes
        μ₀_t = interpolate(tem_t, tem_x₀_∂L∂W, BSplineOrder(int_method.k_μ))
        for i in 1:RT
            C.x[NP + D * RX + (d - 1) * RT + i] = μ₀_t.spline.coefs[i]
        end

        tem_x₁_∂L∂W = zeros(RT)
        for i in 1:RT
            tem_x₁_∂L∂W[i] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,i], C.vx₁_quad_values[d,i], C.wx₁_quad_values[d,i], params)
        end

        μ₁_t = interpolate(tem_t, tem_x₁_∂L∂W, BSplineOrder(int_method.k_μ))
        for i in 1:RT
            C.x[NP+ D * RX + D * RT + (d-1)*RT+i] = μ₁_t.spline.coefs[i]
        end

    end
end

function post_initial_guess!(C,sol_struct,int::PDEIntegrator{<:Sindy_PDE_Integrator},int_method::Sindy_PDE_Integrator{T,MVT,LT,BT}) where {T,MVT<:Lagrange,LT<:Lagrange,BT}
    local NP = int_method.symbolic_expr_basis.NP
    local RT = int_method.RT
    local RX = int_method.RX
    local D = int.problem.D 
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params

    # for d in 1:D
    #     C.x[NP+1:NP+RX] = lag_sys.∂L∂V[d].(C.ut₀_quad_values[d,:], C.vt₀_quad_values[d,:], C.wt₀_quad_values[d,:], params)
    #     C.x[NP+RX+1:NP+RX+RT] = lag_sys.∂L∂W[d].(C.ux₀_quad_values[d,:], C.vx₀_quad_values[d,:], C.wx₀_quad_values[d,:], params)
    #     C.x[NP+RX+RT+1:NP+RX+2*RT] = lag_sys.∂L∂W[d].(C.ux₁_quad_values[d,:], C.vx₁_quad_values[d,:], C.wx₁_quad_values[d,:], params)
    # end

    for d in 1:D
        for i in 1:RX
            C.x[NP + (d - 1) * RX + i] = lag_sys.∂L∂V[d](C.ut₀_quad_values[d,i], C.vt₀_quad_values[d,i], C.wt₀_quad_values[d,i], params)
        end

        for i in 1:RT
            C.x[NP+ D * RX+(d-1)*RT+i] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,i], C.vx₀_quad_values[d,i], C.wx₀_quad_values[d,i], params)
            C.x[NP+ D * RX+ D * RT + (d-1)*RT+i] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,i], C.vx₁_quad_values[d,i], C.wx₁_quad_values[d,i], params)
        end
    end
end

function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:Sindy_PDE_Integrator}) where {ST}
    local C = cache(int,ST)

    local P_sizes = int.method.symbolic_expr_basis.P_sizes
    local NP = int.method.symbolic_expr_basis.NP
    local RT = int.method.RT
    local RX = int.method.RX
    local D = int.problem.D

    local ∂L∂U = int.problem.lagrangian_system.functions.∂L∂U
    local ∂L∂V = int.problem.lagrangian_system.functions.∂L∂V
    local ∂L∂W = int.problem.lagrangian_system.functions.∂L∂W

    local u = int.method.symbolic_expr_basis.u # f = f(Parameters,t,x)
    local v = int.method.symbolic_expr_basis.v
    local w = int.method.symbolic_expr_basis.w

    local ∂u∂P = int.method.symbolic_expr_basis.∂u∂P
    local ∂v∂P = int.method.symbolic_expr_basis.∂v∂P
    local ∂w∂P = int.method.symbolic_expr_basis.∂w∂P

    local grid_matrix = int.method.grid_matrix
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]

    local params = int.problem.lagrangian_system.params
    local mλ₀_x = int.method.mλ₀_x
    local mμ_t = int.method.mμ_t


    start_idx = 1
    # for (d,P_size) in enumerate(P_sizes)
    C.tem_P[1][:] = x[start_idx:start_idx+P_sizes[1]-1]
        # start_idx += P_size
    # end

    # interior values at quadrature points
    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = (u[d])(C.tem_P, sol.t - timestep(int) + timestep(int)* grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2])
                C.v_quad_values[d, i, j] = (v[d])(C.tem_P, sol.t - timestep(int) + timestep(int)* grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2])
                C.w_quad_values[d, i, j] = (w[d])(C.tem_P, sol.t - timestep(int) + timestep(int)* grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2])
            end
        end
    end

    for d in 1:D
        for p in 1:P_sizes[d]
            for i in 1:RT
                for j in 1:RX#TODO what if RX is a Vector
                    C.∂u∂P_quad_values[d][i, j, p] = ∂u∂P[p](C.tem_P, sol.t - timestep(int) + timestep(int)* grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2])
                    C.∂v∂P_quad_values[d][i, j, p] = ∂v∂P[p](C.tem_P, sol.t - timestep(int) + timestep(int)* grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2])
                    C.∂w∂P_quad_values[d][i, j, p] = ∂w∂P[p](C.tem_P, sol.t - timestep(int) + timestep(int)* grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2])
                end
            end
        
            for rx in 1:RX
                C.∂u∂P_t₀_quad_values[d][rx,p] = ∂u∂P[p](C.tem_P, sol.t - timestep(int),xspan[1] + x_domain* x_quad_nodes[rx])
                C.∂u∂P_t₁_quad_values[d][rx,p] = ∂u∂P[p](C.tem_P, sol.t                ,xspan[1] + x_domain* x_quad_nodes[rx])
            end
            for rt in 1:RT
                C.∂u∂P_x₀_quad_values[d][rt,p] = ∂u∂P[p](C.tem_P, sol.t - timestep(int) + timestep(int)* t_quad_nodes[rt], xspan[1])
                C.∂u∂P_x₁_quad_values[d][rt,p] = ∂u∂P[p](C.tem_P, sol.t - timestep(int) + timestep(int)* t_quad_nodes[rt], xspan[2])
            end
        end
    end

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], params)
            end
        end 
    end

    # boundary values at quadrature points
    for d in 1:D
        for j in 1:RX
            C.ut₀_quad_values[d,j] = u[d](C.tem_P,sol.t - timestep(int),xspan[1] + x_domain* x_quad_nodes[j]) # bottom 
            C.ut₁_quad_values[d,j] = u[d](C.tem_P,sol.t,                xspan[1] + x_domain* x_quad_nodes[j]) # top

            C.vt₀_quad_values[d,j] = v[d](C.tem_P,sol.t - timestep(int),xspan[1] + x_domain* x_quad_nodes[j])
            C.vt₁_quad_values[d,j] = v[d](C.tem_P,sol.t,xspan[1] + x_domain* x_quad_nodes[j])

            C.wt₀_quad_values[d,j] = w[d](C.tem_P,sol.t - timestep(int),xspan[1] + x_domain* x_quad_nodes[j])
            C.wt₁_quad_values[d,j] = w[d](C.tem_P,sol.t,xspan[1] + x_domain* x_quad_nodes[j])
        end

        for i in 1:RT
            C.ux₀_quad_values[d,i] = u[d](C.tem_P,sol.t - timestep(int) + timestep(int) .* t_quad_nodes[i],xspan[1])
            C.ux₁_quad_values[d,i] = u[d](C.tem_P,sol.t - timestep(int) + timestep(int) .* t_quad_nodes[i],xspan[2])
            C.vx₀_quad_values[d,i] = v[d](C.tem_P,sol.t - timestep(int) + timestep(int) .* t_quad_nodes[i],xspan[1])
            C.vx₁_quad_values[d,i] = v[d](C.tem_P,sol.t - timestep(int) + timestep(int) .* t_quad_nodes[i],xspan[2])
            C.wx₀_quad_values[d,i] = w[d](C.tem_P,sol.t - timestep(int) + timestep(int) .* t_quad_nodes[i],xspan[1])
            C.wx₁_quad_values[d,i] = w[d](C.tem_P,sol.t - timestep(int) + timestep(int) .* t_quad_nodes[i],xspan[2])
        end

    end

    (x == cache(int).x && eltype(x) == eltype(cache(int).x)) ? post_initial_guess!(cache(int),sol,int,int.method) : nothing

    for d in 1:D
        C.λ₀_x_coes[d,:] = x[NP+1:NP+RX]
        C.μ₀_t_coes[d,:] = x[NP+RX+1:NP+RX+RT]
        C.μ₁_t_coes[d,:] = x[NP+RX+RT+1:NP+RX+2*RT]
    end

    for d in 1:D
        for rx in 1:RX
            C.λ₀_quad_values[d,rx] = sum([C.λ₀_x_coes[d,i] * mλ₀_x[i,rx]  for i in 1:RX])
        end

        for rt in 1:RT
            C.μ₀_quad_values[d,rt] = sum([C.μ₀_t_coes[d,i]*mμ_t[i,rt] for i in 1:RT])
            C.μ₁_quad_values[d,rt] = sum([C.μ₁_t_coes[d,i]*mμ_t[i,rt] for i in 1:RT])
        end
    end
end


function residual!(b::Vector{ST}, sol, int::PDEIntegrator{IT}) where {ST,T, MVT <: Spline,LT  <: Spline,BT,IT <: Sindy_PDE_Integrator{T, MVT, LT, BT}}
    local D = int.problem.D 
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX
    local P_sizes = int.method.symbolic_expr_basis.P_sizes
    local NP = int.method.symbolic_expr_basis.NP

    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local brx = int.method.spatial_quadrature.weights
    local brt = int.method.time_quadrature.weights
    local mλ₀_x = int.method.mλ₀_x
    local mμ_t = int.method.mμ_t

    current_idx = 1
    for d in 1:D 
        for p in 1:P_sizes[d]
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z += timestep(int) * quad_b[rt,rx]* x_domain *
                        (C.∂L∂U_quad_values[d,rt,rx] * C.∂u∂P_quad_values[d][rt, rx,p]
                        + C.∂L∂V_quad_values[d,rt,rx] * C.∂v∂P_quad_values[d][rt, rx,p]
                        + C.∂L∂W_quad_values[d,rt,rx] * C.∂w∂P_quad_values[d][rt, rx,p])
                end
            end
            for rx in 1:RX
                z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * C.∂u∂P_t₀_quad_values[d][rx,p]) #- C.λ₁_quad_values[d,rx] * C.∂u∂P_t₁_quad_values[d, rx]
            end
            for rt in 1:RT
                z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * C.∂u∂P_x₀_quad_values[d][rt,p] - C.μ₁_quad_values[d,rt] * C.∂u∂P_x₁_quad_values[d][rt,p])
            end
            b[current_idx] = z # TODO: check the sign
            current_idx += 1
        end
    end

    @assert current_idx == NP + 1 "Wrong indexing in residual computation"

    for d in 1:D
        for i in 1:RX
            z = zero(ST)
            for rx in 1:RX
                z += x_domain * brx[rx] * mλ₀_x[i,rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
            end
            b[NP + (d - 1) * RX + i] = z
        end
    end

    for d in 1:D
        for i in 1:RT
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) *brt[rt] * mμ_t[i,rt] *(C.ux₀_quad_values[d,rt] - C.boundary_condition_x₀[d,rt])
            end
            b[NP + D * RX + (d - 1) * RT + i] = z
        end
    end

    for d in 1:D
        for i in 1:RT
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) *brt[rt] * mμ_t[i,rt] *(C.boundary_condition_x₁[d,rt] - C.ux₁_quad_values[d,rt])
            end
            b[NP + D * RX + D * RT + (d - 1) * RT + i] = z
        end
    end

end


function residual!(b::Vector{ST}, sol, int::PDEIntegrator{IT}) where {ST,T, MVT <: Lagrange,LT  <: Lagrange,BT,IT <: Sindy_PDE_Integrator{T, MVT, LT, BT}}
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX
    local P_sizes = int.method.symbolic_expr_basis.P_sizes
    local NP = int.method.symbolic_expr_basis.NP

    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local brx = int.method.spatial_quadrature.weights
    local brt = int.method.time_quadrature.weights

    current_idx = 1
    for d in 1:D 
        for p in 1:P_sizes[d]
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z += timestep(int) * quad_b[rt,rx]* x_domain *
                        (C.∂L∂U_quad_values[d,rt,rx] * C.∂u∂P_quad_values[d][rt, rx,p]
                        + C.∂L∂V_quad_values[d,rt,rx] * C.∂v∂P_quad_values[d][rt, rx,p]
                        + C.∂L∂W_quad_values[d,rt,rx] * C.∂w∂P_quad_values[d][rt, rx,p])
                end
            end
            for rx in 1:RX
                z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * C.∂u∂P_t₀_quad_values[d][rx,p]) #- C.λ₁_quad_values[d,rx] * C.∂u∂P_t₁_quad_values[d, rx]
            end
            for rt in 1:RT
                z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * C.∂u∂P_x₀_quad_values[d][rt,p] - C.μ₁_quad_values[d,rt] * C.∂u∂P_x₁_quad_values[d][rt,p])
            end
            b[current_idx] = -z # TODO: check the sign
            current_idx += 1
        end
    end

    @assert current_idx == NP + 1 "Wrong indexing in residual computation"

    for d in 1:D
        for rx in 1:RX
            b[NP + (d - 1) * RX + rx] = x_domain * brx[rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
        end
    end

    for d in 1:D
        for rt in 1:RT
            b[NP+ D * RX+(d-1)*RT+rt]= timestep(int) *brt[rt]*(C.ux₀_quad_values[d,rt] - C.boundary_condition_x₀[d,rt])
        end
    end

    for d in 1:D
        for rt in 1:RT
            b[NP+ D * RX+ D * RT + (d-1)*RT+rt]= timestep(int) *brt[rt]*(C.ux₁_quad_values[d,rt] - C.boundary_condition_x₁[d,rt])
        end
    end

end



function update!(sol_struct, int::PDEIntegrator{<:Sindy_PDE_Integrator})
    local D = int.problem.D
    local u = int.method.symbolic_expr_basis.u # f = f(Parameters,t,x)
    local v = int.method.symbolic_expr_basis.v
    local w = int.method.symbolic_expr_basis.w
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local C = cache(int)
    local NP = int.method.symbolic_expr_basis.NP

    x_nodes = collect(xspan[1]:xstep:xspan[2])
    tem_p = [C.x[1:NP]]

    # println("In update! function, step = ", sol_struct.current_step)
    # println("In update! function, time = ", sol_struct.t)

    for d in 1:D
        for i in eachindex(x_nodes)
            sol_struct.sol.u[sol_struct.current_step][i] = u[d](tem_p,sol_struct.t, x_nodes[i])
            sol_struct.sol.v[sol_struct.current_step][i] = v[d](tem_p,sol_struct.t, x_nodes[i])
            sol_struct.sol.w[sol_struct.current_step][i] = w[d](tem_p,sol_struct.t, x_nodes[i])
        end
    end

    # copy internal variables from cache to solution
    sol_struct.internal.x[sol_struct.current_step] .= cache(int).x 

    sol_struct.t += int.problem.tstep
    # println("In the end of update! function, time = ", sol_struct.t)
end

function create_interior_quadrature_points_derivative_mat(ST::Type, RT::Int,RX::Int, D::Int,P_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, RT, RX, P_sizes[d]))
    end
    return mat
end

function create_interior_quadrature_points_derivative_mat(ST::Type, RT::Int,RX::Int, D::Int, DX::Int,P_sizes::Vector{Int})
    mat = Array{Array{ST}}(undef,D,DX)

    for d in 1:D
        for dx in 1:DX
            mat[d,dx] = zeros(ST, RT, RX, P_sizes[d])
        end
    end

    return mat
end

function create_boundary_derivative_vector(ST::Type, D::Int,R::Int,P_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, R, P_sizes[d]))
    end
    return mat
end

function create_tem_vector(ST::Type, D::Int,P_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, P_sizes[d]))
    end
    return mat
end

function internal_variables(int::PDEIntegrator{<:Sindy_PDE_Integrator},problem::PDEProblem)
    local x = cache(int).x
    ntime = Int((problem.tspan[2] - problem.tspan[1]) / problem.tstep)
    xx = (x, ntuple( _ -> zeros(size(x)...), ntime)...)
    return (x = xx,)
end