struct Sindy_PDE_Integrator{T,RT,basisType<:Basis{T}}
    symbolic_expr_basis
    time_quadrature::QuadratureRule{T,NNODES}
    RT::Int # Number of quadrature points in time


    spatial_quadrature::Vector{QuadratureRule{T}}
    RX::Vector{Int} # Number of quadrature points in each spatial dimension
    grid_matrix # Quadrature grid points
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

        dimensions = [RX..., RT]  # Number of quadrature points in each dimension
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



struct Sindy_PDE_IntegratorCache{ST,RT,RX,D,DX,NΘ}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    DX = dimension of input of u,v,w, i.e. number of spatial dimensions
    NΘ = number of parameters in the expression
    """
    x::Vector{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ∂u∂θ_mat::Matrix{ST}
    ∂v∂θ_mat::Matrix{ST}
    ∂w∂θ_mat::Array{ST}

    λ₁_x_coes::Vector{ST}
    λ₂_x_coes::Vector{ST}

    μ₁_t_coes::Matrix{ST}
    μ₂_t_coes::Matrix{ST}
    function Sindy_PDE_IntegratorCache{ST,RT,RX,D,DX,NΘ}() where {ST,RT,RX,D,DX,NΘ}
        x = zeros(ST, NΘ + 2 * RT + 2 * DX * RX)

        u_quad_values = zeros(ST, D, DX, RX, RT)
        v_quad_values = zeros(ST, D, DX, RX, RT)
        w_quad_values = zeros(ST, D, DX, RX, RT)

        ∂L∂U_quad_values = zeros(ST, D, RX, RT)
        ∂L∂V_quad_values = zeros(ST, D, RX, RT)
        ∂L∂W_quad_values = zeros(ST, D, DX, RX, RT)

        ∂u∂P_mat = zeros(ST, D, NΘ)
        ∂v∂p_mat = zeros(ST, D, NΘ)
        ∂w∂P_mat = zeros(ST, D, DX, NΘ)

        λ₁_x_coes = zeros(ST, RT)
        λ₂_x_coes = zeros(ST, RT)

        μ₁_t_coes = zeros(ST, DX, RX)
        μ₂_t_coes = zeros(ST, DX, RX)
        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂P_mat, ∂v∂p_mat, ∂w∂P_mat,
            λ₁_x_coes, λ₂_x_coes,
            μ₁_t_coes, μ₂_t_coes)
    end
end

struct LPDEhistory
    u_expr::Vector{Num}
end
function Cache{ST}(problem, int::Sindy_PDE_Integrator; kwargs...) where {ST}
    Sindy_PDE_IntegratorCache{ST,int.RT,int.RX,problem.D,problem.DX,int.symbolic_expr_basis.Nθ}(; kwargs...)
end


function components(x::AbstractVector{ST}, sol, params, int::GeometricIntegrator{<:PR_Integrator}, lagrangian_system) where {ST}
    local C = cache(int)
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

    for dx in 1:DX
        for i in 1:RX
            for j in 1:RT
                C.u_quad_values[dx, i, j] = u[d](rand(6), grid_matrix[i, j][2], grid_matrix[i, j][1])
                C.v_quad_values[dx, i, j] = v[d](rand(6), grid_matrix[i, j][2], grid_matrix[i, j][1])
                for d in 1:D
                    C.w_quad_values[d,dx, i, j] = w[d,dx](rand(6), grid_matrix[i, j][2], grid_matrix[i, j][1])
                end
            end
        end
    end

    for i in 1:RX
        for j in 1:RT
            C.∂L∂U_quad_values[i, j] = ∂L∂U[1](u_quad_values[i, j], v_quad_values[i, j], w_quad_values[i, j], SineGordon.default_parameters)
            C.∂L∂V_quad_values[i, j] = ∂L∂V[1](u_quad_values[i, j], v_quad_values[i, j], w_quad_values[i, j], SineGordon.default_parameters)
            C.∂L∂W_quad_values[i, j] = ∂L∂W[1](u_quad_values[i, j], v_quad_values[i, j], w_quad_values[i, j], SineGordon.default_parameters)
        end
    end



end