struct Sindy_PDE_Integrator{T,RT,basisType<:Basis{T}}
    symbolic_expr_basis
    time_quadrature::QuadratureRule{T,NNODES}
    RT::Int # Number of quadrature points in time


    spatial_quadrature::Vector{QuadratureRule{T}}
    RX::Vector{Int} # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions
    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ...]
    grid_weights # Quadrature weights

    μ₁_t::Lagrange
    μ₂_t::Lagrange

    λ₁_x::Vector{Lagrange}
    λ₂_x::Vector{Lagrange}

    init_w::Vector{Vector{T}}
    nstages::Int

    function Sindy_PDE_Integrator(basis, RT::Int, RX::Int, init_w::Vector{Vector{T}};
        nstages::Int=10) where {T}
        @assert length(RX) == length(init_w)

        t_quadrature = QuadratureRules.GaussLegendreQuadrature(RT)
        x_quadratures = [QuadratureRules.GaussLegendreQuadrature(RX[i]) for i in eachindex(RX)]

        dimensions = [RT,RX...]  # Number of quadrature points in each dimension
        grid_matrix, grid_weights = construct_quadrature_grid(dimensions)

        μ₁_t = CompactBasisFunctions.Lagrange(RT)
        μ₂_t = CompactBasisFunctions.Lagrange(RT)

        λ₁_x = [CompactBasisFunctions.Lagrange(RX[i] for i in eachindex(RX))]
        λ₂_x = [CompactBasisFunctions.Lagrange(RX[i] for i in eachindex(RX))]

        new{T,RT,typeof(basis)}(basis, t_quadrature, RT,
            x_quadratures, RX,
            grid_matrix, grid_weights,
            μ₁_t, μ₂_t,
            λ₁_x, λ₂_x,
            init_w, nstages)
    end
end

struct Sindy_PDE_IntegratorCache{ST,RT,RX,D,DX,NP}
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

    λ₁_x_coes::Vector{ST}
    λ₂_x_coes::Vector{ST}

    μ₁_t_coes::Matrix{ST}
    μ₂_t_coes::Matrix{ST}

    tem_P::Vector{Vector{ST}} # temporary storage for P values
    function Sindy_PDE_IntegratorCache{ST,RT,RX,D,DX,NP}(P_sizes) where {ST,RT,RX,D,DX,NP}
        x = zeros(ST, NP + 2 * D * RX + 2* D * DX * RT ) # TODO: how to deal with RX being a vector

        u_quad_values = zeros(ST, D, RT, RX...)
        v_quad_values = zeros(ST, D, RT, RX...)
        w_quad_values = zeros(ST, D, DX, RT, RX...)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX...)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX...)
        ∂L∂W_quad_values = zeros(ST, D, DX, RT, RX...)

        ∂u∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,P_sizes)
        ∂v∂p_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,P_sizes)
        ∂w∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,DX,P_sizes)

        λ₁_x_coes = zeros(ST, D, DX, RX...)
        λ₂_x_coes = zeros(ST, D, DX, RX...)
        μ₁_t_coes = zeros(ST, D, DX, RT)
        μ₂_t_coes = zeros(ST, D, DX, RT)

        u₁_quad_value = zeros(ST,D,RT) # left boundary, i.e. t = 0
        u₂_quad_value = zeros(ST,D,RT) # right boundary, i.e. t = T
        v₁_quad_value = zeros(ST,D,RT)
        v₂_quad_value = zeros(ST,D,RT)
        w₁_quad_value = zeros(ST,D,DX,RT)
        w₂_quad_value = zeros(ST,D,DX,RT)

        u_bottom_quad_value = zeros(ST,D,RX) # bottom boundary, i.e. x = 0
        u_top_quad_value = zeros(ST,D,RX) # top boundary, i.e. x = L
        v_bottom_quad_value = zeros(ST,D,RX)
        v_top_quad_value = zeros(ST,D,RX)
        w_bottom_quad_value = zeros(ST,D,DX,RX...)
        w_top_quad_value = zeros(ST,D,DX,RX...)



        tem_P = create_boundary_derivative_vector(ST, D, P_sizes)
        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂P_quad_values, ∂v∂p_quad_values, ∂w∂P_quad_values,
            λ₁_x_coes, λ₂_x_coes,
            μ₁_t_coes, μ₂_t_coes,
            tem_P)
    end
end

struct LPDEhistory
    u_expr::Vector{Num}
end

function Cache{ST}(problem, int::Sindy_PDE_Integrator; kwargs...) where {ST}
    Sindy_PDE_IntegratorCache{ST,int.RT,int.RX,problem.D,problem.DX,int.symbolic_expr_basis.NP}(; kwargs...)
end

function initial_guess(cache,lag_sys,init_w,int)
    local x = cache.x
    local P_sizes = int.symbolic_expr_basis.P_sizes
    local NP = int.symbolic_expr_basis.NP
    local RT = int.RT
    local RX = int.RX

    local u = int.symbolic_expr_basis.u
    local v = int.symbolic_expr_basis.v
    local w = int.symbolic_expr_basis.w

    start_idx = 1
    for (d,P_size) in enumerate(P_sizes)
        x[start_idx:start_idx+P_size-1]= init_w[d][:]
        start_idx += P_size
    end

    temp_u = u()

    x[NP+1:NP+RT] = lag_sys.∂L∂V()
end


function components(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:PR_Integrator}, lagrangian_system) where {ST}
    local C = cache(int)

    local P_sizes = int.symbolic_expr_basis.P_sizes
    local NP = int.symbolic_expr_basis.NP
    local RT = int.RT
    local RX = int.RX
    local D = SineGordon.D #TODO D should from the problem be used
    local DX = SineGordon.DX # same as D

    local ∂L∂U = lagrangian_system.codes.∂L∂U
    local ∂L∂V = lagrangian_system.codes.∂L∂V
    local ∂L∂W = lagrangian_system.codes.∂L∂W

    local u = int.symbolic_expr_basis.u
    local v = int.symbolic_expr_basis.v
    local w = int.symbolic_expr_basis.w

    local ∂u∂P = int.symbolic_expr_basis.∂u∂P
    local ∂v∂P = int.symbolic_expr_basis.∂v∂P
    local ∂w∂P = int.symbolic_expr_basis.∂w∂P

    local grid_matrix = int.grid_matrix

    @assert DX == length(RX)
    @assert D == length(P_sizes)

    start_idx = 1
    for (d,P_size) in enumerate(P_sizes)
        C.tem_P[d][:] = x[start_idx:start_idx+P_size-1]
        start_idx += P_size
    end

    C.λ₁_x_coes = x[NP+1:NP+RT]
    C.λ₂_x_coes = x[NP+RT+1:NP+2*RT]
    for dx in 1:DX
        C.μ₁_t_coes[dx, :] = x[NP+2*RT+(dx-1)*RX[dx]+1:NP+2*RT+dx*RX[dx]]
        C.μ₂_t_coes[dx, :] = x[NP+2*RT+(DX+dx-1)*RX[dx]+1:NP+2*RT+(DX+dx)*RX[dx]]
    end
    

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = u[d](tem_P[d], grid_matrix[i, j][1], grid_matrix[i, j][2])
                C.v_quad_values[d, i, j] = v[d](tem_P[d], grid_matrix[i, j][1], grid_matrix[i, j][2])
                for dx in 1:DX
                    C.w_quad_values[d, dx, i, j] = w[d,dx](tem_P[d], grid_matrix[i, j][1], grid_matrix[i, j][2])
                end
            end
        end
    end

    for d in 1:D
        for p in 1:P_sizes[d]
            for i in 1:RT
                for j in 1:RX#TODO what if RX is a Vector
                    C.∂u∂P_quad_values[d][p, i, j] = ∂u∂P[d](tem_P[d], grid_matrix[i, j][1], grid_matrix[i, j][2], p)
                    C.∂v∂P_quad_values[d][p, i, j] = ∂v∂P[d](tem_P[d], grid_matrix[i, j][1], grid_matrix[i, j][2], p)
                    for dx in 1:DX
                        C.∂w∂P_quad_values[d,dx][p, i, j] = ∂w∂P[d,dx](tem_P[d], grid_matrix[i, j][1], grid_matrix[i, j][2], p)
                    end
                end
            end
        end
    end

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], SineGordon.default_parameters)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], SineGordon.default_parameters)
                for dx in 1:DX
                    C.∂L∂W_quad_values[d, dx, i, j] = ∂L∂W[d,dx](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], SineGordon.default_parameters)
                end
            end
        end 
    end


end


function create_interior_quadrature_points_derivative_mat(ST::Type, RT::Int,RX::Vector{Int}, D::Int,P_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, RT, RX..., P_sizes[d]))
    end
    return mat
end

function create_interior_quadrature_points_derivative_mat(ST::Type, RT::Int,RX::Vector{Int}, D::Int, DX::Int,P_sizes::Vector{Int})
    mat = Array{Array{ST}}(undef,D,DX)

    for d in 1:D
        for dx in 1:DX
            mat[d,dx] = zeros(ST, RT, RX..., P_sizes[d])
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
