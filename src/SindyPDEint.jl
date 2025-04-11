struct Sindy_PDE_Integrator{T,RT,basisType<:Basis{T}}
    symbolic_expr_basis
    time_quadrature::QuadratureRule{T,NNODES}
    RT::Int # Number of quadrature points in time


    spatial_quadrature::Vector{QuadratureRule{T}}
    RX::Vector{Int} # Number of quadrature points in each spatial dimension
    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ...]
    grid_weights # Quadrature weights

    μ₁_t::Lagrange
    μ₂_t::Lagrange

    λ₁_x::Vector{Lagrange}
    λ₂_x::Vector{Lagrange}

    init_w::Vector{Vector{T}}
    nstages::Int

    function Sindy_PDE_Integrator(basis, RT::Int, RX::Vector{Int}, init_w::Vector{Vector{T}};
        nstages::Int=10) where {T}

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
    function Sindy_PDE_IntegratorCache{ST,RT,RX,D,DX,NP}() where {ST,RT,RX,D,DX,NP}
        x = zeros(ST, NP + 2 * RT + 2 * DX * RX)

        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, DX, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, DX, RT, RX)

        ∂u∂P_quad_values = zeros(ST, D, NP)
        ∂v∂p_quad_values = zeros(ST, D, NP)
        ∂w∂P_quad_values = zeros(ST, D, DX, NP)

        λ₁_x_coes = zeros(ST, RT)
        λ₂_x_coes = zeros(ST, RT)

        μ₁_t_coes = zeros(ST, DX, RX)
        μ₂_t_coes = zeros(ST, DX, RX)
        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂P_quad_values, ∂v∂p_quad_values, ∂w∂P_quad_values,
            λ₁_x_coes, λ₂_x_coes,
            μ₁_t_coes, μ₂_t_coes)
    end
end

struct LPDEhistory
    u_expr::Vector{Num}
end
function Cache{ST}(problem, int::Sindy_PDE_Integrator; kwargs...) where {ST}
    Sindy_PDE_IntegratorCache{ST,int.RT,int.RX,problem.D,problem.DX,int.symbolic_expr_basis.NP}(; kwargs...)
end


function components(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:PR_Integrator}, lagrangian_system) where {ST}
    local C = cache(int)
    local P = int.init_w
    local P_sizes = int.symbolic_expr_basis.P_sizes

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

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = u[d](P[d], grid_matrix[i, j][1], grid_matrix[i, j][2])
                C.v_quad_values[d, i, j] = v[d](P[d], grid_matrix[i, j][1], grid_matrix[i, j][2])
                for dx in 1:DX
                    C.w_quad_values[d, dx, i, j] = w[d,dx](P[d], grid_matrix[i, j][1], grid_matrix[i, j][2])
                end
            end
        end
    end

    for d in 1:D
        for n in 1:P_sizes[d]

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


function create_interior_quadrature_points_derivative_mat(ST::Type, RT::Int,RX::Vector{Int}, D::Int,W_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, RT, RX..., W_sizes[d]))
    end
end

function create_interior_quadrature_points_derivative_mat(ST::Type, RT::Int,RX::Vector{Int}, D::Int, DX::Int,W_sizes::Vector{Int})
    mat = []
    for d in 1:D
        push!(mat, zeros(ST, RT, RX..., W_sizes[d]))
    end
end
