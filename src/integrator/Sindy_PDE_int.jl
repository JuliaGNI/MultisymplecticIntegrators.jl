struct Sindy_PDE_Integrator{T,RT,basisType<:AbstractPDEBasis} <: PDEMethod
    symbolic_expr_basis
    time_quadrature::QuadratureRule{T}
    RT::Int # Number of quadrature points in time

    spatial_quadrature::QuadratureRule{T}
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions
    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights
    
    μ₀_t::Lagrange
    μ₁_t::Lagrange

    λ₀_x::Lagrange
    # λ₁_x::Lagrange

    init_w::Vector{T}
    nstages::Int # ?

    function Sindy_PDE_Integrator(basis, RT::Int, RX::Int, init_w::Vector{T};nstages::Int=10) where {T}
        t_quadrature = QuadratureRules.GaussLegendreQuadrature(RT)
        x_quadrature = QuadratureRules.GaussLegendreQuadrature(RX)

        dimensions = [RT,RX]  # Number of quadrature points in each dimension
        grid_matrix, grid_weights = construct_quadrature_grid(dimensions)

        μ₀_t = CompactBasisFunctions.Lagrange(t_quadrature.nodes)
        μ₁_t = CompactBasisFunctions.Lagrange(t_quadrature.nodes)

        λ₀_x = CompactBasisFunctions.Lagrange(x_quadrature.nodes)
        # λ₁_x = CompactBasisFunctions.Lagrange(x_quadrature.nodes)

        new{T,RT,typeof(basis)}(basis, t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            μ₀_t, μ₁_t,
            λ₀_x, #λ₁_x,
            init_w, nstages)
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
    λ₁_x_coes::Matrix{ST}

    μ₀_t_coes::Matrix{ST}
    μ₁_t_coes::Matrix{ST}

    λ₀_quad_values::Matrix{ST} 
    λ₁_quad_values::Matrix{ST} 
    μ₀_quad_values::Matrix{ST}
    μ₁_quad_values::Matrix{ST} 

    ∂u∂P_t₀_quad_values::Matrix{ST}
    ∂u∂P_t₁_quad_values::Matrix{ST}
    ∂u∂P_x₀_quad_values::Matrix{ST}
    ∂u∂P_x₁_quad_values::Matrix{ST}

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

    tem_P::Vector{Vector{ST}} # temporary storage for P values

    init_condition_t₀::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}


    function Sindy_PDE_IntegratorCache{ST,RT,RX,D,NP}(P_sizes) where {ST,RT,RX,D,NP}
        # x = zeros(ST, NP + 2 * D * RX + 2* D * DX * RT ) # TODO: how to deal with RX being a vector/
        x = zeros(ST,NP + D * RX +2* D * RT) # params,μ₁,μ₂,λ₁,λ₂
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
        λ₁_x_coes = zeros(ST, D, RX)
        μ₀_t_coes = zeros(ST, D, RT)
        μ₁_t_coes = zeros(ST, D, RT)

        λ₀_quad_values = zeros(ST, D, RX) 
        λ₁_quad_values = zeros(ST, D, RX) 
        μ₀_quad_values = zeros(ST, D, RT) 
        μ₁_quad_values = zeros(ST, D, RT) 

        ∂u∂P_t₀_quad_values = zeros(ST, D, RX) 
        ∂u∂P_t₁_quad_values = zeros(ST, D, RX)
        ∂u∂P_x₀_quad_values = zeros(ST, D, RT) 
        ∂u∂P_x₁_quad_values = zeros(ST, D, RT)

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

        tem_P = create_boundary_derivative_vector(ST, D, P_sizes)

        init_condition_t₀ = zeros(ST, D, RX)

        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)

        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂P_quad_values, ∂v∂P_quad_values, ∂w∂P_quad_values,
            λ₀_x_coes, λ₁_x_coes, μ₀_t_coes,μ₁_t_coes, 
            λ₀_quad_values, λ₁_quad_values, μ₀_quad_values, μ₁_quad_values,
            ∂u∂P_t₀_quad_values, ∂u∂P_t₁_quad_values, ∂u∂P_x₀_quad_values, ∂u∂P_x₁_quad_values,
            ut₀_quad_values, ut₁_quad_values,vt₀_quad_values, vt₁_quad_values,wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values,vx₀_quad_values, vx₁_quad_values,wx₀_quad_values, wx₁_quad_values,
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


function prior_initial_guess!(C,sol_struct,int)
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


function post_initial_guess!(C,sol_struct,int)
    local NP = int.method.symbolic_expr_basis.NP
    local RT = int.method.RT
    local RX = int.method.RX
    local D = int.problem.D #TODO D should from the problem be used
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

    println("Initial guess",C.x)
end

function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:Sindy_PDE_Integrator}) where {ST}
    local C = cache(int,ST)

    local P_sizes = int.method.symbolic_expr_basis.P_sizes
    local NP = int.method.symbolic_expr_basis.NP
    local RT = int.method.RT
    local RX = int.method.RX
    local D = int.problem.D #TODO D should from the problem be used

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

    local λ₀_x = int.method.λ₀_x
    local μ₀_t = int.method.μ₀_t
    local μ₁_t = int.method.μ₁_t
    local params = int.problem.lagrangian_system.params


    prior_initial_guess!(cache(int),sol,int)

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
                    C.∂u∂P_quad_values[d][i, j, p] = ∂u∂P[d](C.tem_P, sol.t - timestep(int) + timestep(int)* grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2])
                    C.∂v∂P_quad_values[d][i, j, p] = ∂v∂P[d](C.tem_P, sol.t - timestep(int) + timestep(int)* grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2])
                    C.∂w∂P_quad_values[d][i, j, p] = ∂w∂P[d](C.tem_P, sol.t - timestep(int) + timestep(int)* grid_matrix[i, j][1], xspan[1] + x_domain* grid_matrix[i, j][2])
                end
            end
        end

        for rx in 1:RX
            C.∂u∂P_t₀_quad_values[d, rx] = ∂u∂P[d](C.tem_P, sol.t - timestep(int),xspan[1] + x_domain* x_quad_nodes[rx])
            C.∂u∂P_t₁_quad_values[d, rx] = ∂u∂P[d](C.tem_P, sol.t                ,xspan[1] + x_domain* x_quad_nodes[rx])
        end
        for rt in 1:RT
            C.∂u∂P_x₀_quad_values[d, rt] = ∂u∂P[d](C.tem_P, sol.t - timestep(int) + timestep(int)* t_quad_nodes[rt], xspan[1])
            C.∂u∂P_x₁_quad_values[d, rt] = ∂u∂P[d](C.tem_P, sol.t - timestep(int) + timestep(int)* t_quad_nodes[rt], xspan[2])
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
            C.ut₀_quad_values[d,j] = u[d](C.tem_P,sol.t - timestep(int),x_quad_nodes[j])
            C.ut₁_quad_values[d,j] = u[d](C.tem_P,sol.t,x_quad_nodes[j]) 
            C.vt₀_quad_values[d,j] = v[d](C.tem_P,sol.t - timestep(int),x_quad_nodes[j])
            C.vt₁_quad_values[d,j] = v[d](C.tem_P,sol.t,x_quad_nodes[j])
            C.wt₀_quad_values[d,j] = w[d](C.tem_P,sol.t - timestep(int),x_quad_nodes[j])
            C.wt₁_quad_values[d,j] = w[d](C.tem_P,sol.t,x_quad_nodes[j])
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

    for d in 1:D
        C.λ₀_x_coes[d,:] = x[NP+1:NP+RX]
        C.μ₀_t_coes[d,:] = x[NP+RX+1:NP+RX+RT]
        C.μ₁_t_coes[d,:] = x[NP+RX+RT+1:NP+RX+2*RT]
    end

    for d in 1:D
        for rx in 1:RX
            C.λ₀_quad_values[d,rx] = sum([C.λ₀_x_coes[d,i]*λ₀_x.b[i](x_quad_nodes[i]) for i in 1:RX])
        end

        for rt in 1:RT
            C.μ₀_quad_values[d,rt] = sum([C.μ₀_t_coes[d,i]*μ₀_t.b[i](t_quad_nodes[i]) for i in 1:RT])
            C.μ₁_quad_values[d,rt] = sum([C.μ₁_t_coes[d,i]*μ₁_t.b[i](t_quad_nodes[i]) for i in 1:RT])
        end
    end
    
    post_initial_guess!(cache(int),sol,int)


end


function residual!(b::Vector{ST}, sol, int::PDEIntegrator{<:Sindy_PDE_Integrator}) where {ST}
    local D = int.problem.D #TODO D should from the problem be used
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
                z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * C.∂u∂P_t₀_quad_values[d, rx]) #- C.λ₁_quad_values[d,rx] * C.∂u∂P_t₁_quad_values[d, rx]
            end
            for rt in 1:RT
                z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * C.∂u∂P_x₀_quad_values[d, rt] - C.μ₁_quad_values[d,rt] * C.∂u∂P_x₁_quad_values[d, rt])
            end
            b[current_idx] = -z
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
    x_nodes = collect(xspan[1]:xstep:xspan[2])

    sol_struct.current_step += 1
    sol_struct.t += int.problem.tstep

    for d in 1:D
        for i in eachindex(x_nodes)
            sol_struct.sol.u[sol_struct.current_step][i] = u[d](C.tem_P,sol_struct.t, x_nodes[i])
            sol_struct.sol.v[sol_struct.current_step][i] = v[d](C.tem_P,sol_struct.t, x_nodes[i])
            sol_struct.sol.w[sol_struct.current_step][i] = w[d](C.tem_P,sol_struct.t, x_nodes[i])
        end
    end

    # copy internal variables from cache to solution
    sol_struct.internal.x[sol_struct.current_step] .= cache(int).x 
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

function create_boundary_derivative_vector(ST::Type, D::Int,P_sizes::Vector{Int})
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