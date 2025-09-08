"""
    Implementation of the PDE integrator with ELM method.
    Boundary condition and initial condition are imposed with least square method.
"""

struct ELM_PDE_int{T,BT<:AbstractPDEBasis,IPMT<:InitialParametersMethod} <: PDEMethod
    basis::BT
    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights

    initial_with_PINN::Bool
    initial_guess_method::IPMT # :LSGD or :GroundTruth

    function ELM_PDE_int(basis; RT::Int=6, RX::Int=8, initial_with_PINN=true, initial_guess_method::IPMT=:LSGD()) where {T,IPMT}
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

        new{T,typeof(basis),typeof(initial_guess_method)}(basis,
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            initial_with_PINN, initial_guess_method)
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

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

    ∂u∂P_quad_values
    ∂v∂P_quad_values
    ∂w∂P_quad_values

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

    init_condition_t₀::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}

    sol_params
    function ELM_PDE_intCache{ST,RT,RX,D,NP}(network_arch) where {ST,RT,RX,D,NP}
        x = zeros(ST, NP) # in ELM, x is just the output layer parameters
        
        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ∂u∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT, RX, D, [NP])
        ∂v∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT, RX, D, [NP])
        ∂w∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT, RX, D, [NP])


        ∂u∂P_t₀_quad_values = create_boundary_derivative_vector(ST, D, RX, [NP])
        ∂u∂P_t₁_quad_values = create_boundary_derivative_vector(ST, D, RX, [NP])
        ∂u∂P_x₀_quad_values = create_boundary_derivative_vector(ST, D, RT, [NP])
        ∂u∂P_x₁_quad_values = create_boundary_derivative_vector(ST, D, RT, [NP])

        ut₀_quad_values = zeros(ST, D, RX) # bottom boundary, i.e. t = 0
        ut₁_quad_values = zeros(ST, D, RX) # top boundary, i.e. t = T
        vt₀_quad_values = zeros(ST, D, RX)
        vt₁_quad_values = zeros(ST, D, RX)
        wt₀_quad_values = zeros(ST, D, RX)
        wt₁_quad_values = zeros(ST, D, RX)

        ux₀_quad_values = zeros(ST, D, RT) # left boundary, i.e. x = 0
        ux₁_quad_values = zeros(ST, D, RT) # right boundary, i.e. x = L
        vx₀_quad_values = zeros(ST, D, RT)
        vx₁_quad_values = zeros(ST, D, RT)
        wx₀_quad_values = zeros(ST, D, RT)
        wx₁_quad_values = zeros(ST, D, RT)

        init_condition_t₀ = zeros(ST, D, RX)

        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)

        sol_params = network_cache_create(network_arch, ST)

        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂P_quad_values, ∂v∂P_quad_values, ∂w∂P_quad_values,
            ∂u∂P_t₀_quad_values, ∂u∂P_t₁_quad_values, ∂u∂P_x₀_quad_values, ∂u∂P_x₁_quad_values,
            ut₀_quad_values, ut₁_quad_values, vt₀_quad_values, vt₁_quad_values, wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values, vx₀_quad_values, vx₁_quad_values, wx₀_quad_values, wx₁_quad_values,
            init_condition_t₀,
            boundary_condition_x₀, boundary_condition_x₁,
            sol_params)
    end
end

nlsolution(cache::ELM_PDE_intCache) = cache.x

function Cache{ST}(problem::PDEProblem, int::ELM_PDE_int; kwargs...) where {ST}
    ELM_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.NP}(int.basis.network_arch; kwargs...)
end

#{ST,RT,RX,D,NP}(NP) where {ST,RT,RX,D,NP}
@inline GeometricIntegrators.Integrators.CacheType(ST, problem::PDEProblem, int::ELM_PDE_int) = ELM_PDE_intCache{ST,int.RT,int.RX,problem.D,int.basis.NP}

@inline function Base.getindex(c::ELM_PDE_intCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:ELM_PDE_int}) where {ST}
    local x = C.x


end