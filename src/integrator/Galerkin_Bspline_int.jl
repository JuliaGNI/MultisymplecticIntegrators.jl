struct Galerkin_Bspline_Integrator{T,MVT,LT,BT<:AbstractPDEBasis} <: PDEMethod
    basis::BT
    
    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions
    
    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights
    
    k_μ::Int
    μ₀_t::MVT
    μ₁_t::MVT

    k_λ₀_x::Int
    λ₀_x::LT

    mλ₀_x # λ₀_x evaluated at quadrature points
    mμ_t

    u_collocation_mat
    v_collocation_mat
    w_collocation_mat

    ut₀_basis_quad_values
    ut₁_basis_quad_values
    vt₀_basis_quad_values
    vt₁_basis_quad_values
    wt₀_basis_quad_values
    wt₁_basis_quad_values

    ux₀_basis_quad_values
    ux₁_basis_quad_values
    vx₀_basis_quad_values
    vx₁_basis_quad_values
    wx₀_basis_quad_values
    wx₁_basis_quad_values

    function Galerkin_Bspline_Integrator(basis; RT::Int = 16,RX::Int = 32,xspan::Tuple = (0.,1.0),tstep::T = 1.0, k_μ::Int = 4,k_λ₀_x::Int = 4,
        μ::Symbol = :Lagrange,λ::Symbol= :Lagrange) where {T}

        if RT ==128 
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

        dimensions = [RT,RX]  
        grid_matrix, grid_weights = construct_quadrature_grid(dimensions)

        S = basis.S
        u_collocation_matrix = zeros(S, RT, RX)
        v_collocation_matrix = zeros(S, RT, RX)
        w_collocation_matrix = zeros(S, RT, RX)
        for rt in 1:RT
            for rx in 1:RX
                # tt = tstep .* t_quadrature.nodes[rt]
                # xx = xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes[rx]
                tt = t_quadrature.nodes[rt]
                xx = x_quadrature.nodes[rx]
                u_collocation_matrix[:,rt,rx] = spline2D_coeff_derivatives((basis.Basis_t, basis.Basis_x), (tt, xx))
                v_collocation_matrix[:,rt,rx] = spline2D_coeff_derivatives_time((basis.Basis_t, basis.Basis_x), (tt, xx))
                w_collocation_matrix[:,rt,rx] = spline2D_coeff_derivatives_space((basis.Basis_t, basis.Basis_x), (tt, xx))
            end
        end

        ut₀_basis_quad_values = zeros(S,RX)
        ut₁_basis_quad_values = zeros(S,RX)
        vt₀_basis_quad_values = zeros(S,RX)
        vt₁_basis_quad_values = zeros(S,RX)
        wt₀_basis_quad_values = zeros(S,RX)
        wt₁_basis_quad_values = zeros(S,RX)

        ux₀_basis_quad_values = zeros(S,RT)
        ux₁_basis_quad_values = zeros(S,RT)
        vx₀_basis_quad_values = zeros(S,RT)
        vx₁_basis_quad_values = zeros(S,RT)
        wx₀_basis_quad_values = zeros(S,RT)
        wx₁_basis_quad_values = zeros(S,RT)

        for rx in 1:RX
            xx = x_quadrature.nodes[rx]
            ut₀_basis_quad_values[:,rx] = spline2D_coeff_derivatives((basis.Basis_t, basis.Basis_x),(0.0 ,xx))
            ut₁_basis_quad_values[:,rx] = spline2D_coeff_derivatives((basis.Basis_t, basis.Basis_x),(1.0 ,xx)) # top
            vt₀_basis_quad_values[:,rx] = spline2D_coeff_derivatives_time((basis.Basis_t, basis.Basis_x),(0.0 ,xx))
            vt₁_basis_quad_values[:,rx] = spline2D_coeff_derivatives_time((basis.Basis_t, basis.Basis_x),(1.0 ,xx))   
            wt₀_basis_quad_values[:,rx] = spline2D_coeff_derivatives_space((basis.Basis_t, basis.Basis_x),(0.0 ,xx))
            wt₁_basis_quad_values[:,rx] = spline2D_coeff_derivatives_space((basis.Basis_t, basis.Basis_x),(1.0 ,xx))
        end

        for i in 1:RT
            tt = t_quadrature.nodes[i]
            ux₀_basis_quad_values[:,i] = spline2D_coeff_derivatives((basis.Basis_t, basis.Basis_x),(tt,xspan[1]))
            ux₁_basis_quad_values[:,i] = spline2D_coeff_derivatives((basis.Basis_t, basis.Basis_x),(tt,xspan[2]))
            vx₀_basis_quad_values[:,i] = spline2D_coeff_derivatives_time((basis.Basis_t, basis.Basis_x),(tt,xspan[1]))
            vx₁_basis_quad_values[:,i] = spline2D_coeff_derivatives_time((basis.Basis_t, basis.Basis_x),(tt,xspan[2]))
            wx₀_basis_quad_values[:,i] = spline2D_coeff_derivatives_space((basis.Basis_t, basis.Basis_x),(tt,xspan[1]))
            wx₁_basis_quad_values[:,i] = spline2D_coeff_derivatives_space((basis.Basis_t, basis.Basis_x),(tt,xspan[2]))
        end


        # Construct Lagrangian multipliers
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

        new{T,typeof(μ₀_t),typeof(λ₀_x),typeof(basis)}(basis, 
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            k_μ, μ₀_t, μ₁_t,
            k_λ₀_x,λ₀_x, #λ₁_x,
            mλ₀_x, mμ_t,
            u_collocation_matrix, v_collocation_matrix, w_collocation_matrix,
            ut₀_basis_quad_values,ut₁_basis_quad_values,
            vt₀_basis_quad_values,vt₁_basis_quad_values,
            wt₀_basis_quad_values,wt₁_basis_quad_values,
            ux₀_basis_quad_values,ux₁_basis_quad_values,
            vx₀_basis_quad_values,vx₁_basis_quad_values,
            wx₀_basis_quad_values,wx₁_basis_quad_values
            )
    end 
end

default_solver(::Galerkin_Bspline_Integrator) = Newton()

struct Galerkin_Bspline_IntegratorCache{ST,RT,RX,D,S} <: PDEIntegratorCache{ST,D}
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

    λ₀_x_coes::Matrix{ST}
    μ₀_t_coes::Matrix{ST}
    μ₁_t_coes::Matrix{ST}

    λ₀_quad_values::Matrix{ST} 
    μ₀_quad_values::Matrix{ST}
    μ₁_quad_values::Matrix{ST} 

    ∂u∂P_t₀_quad_values::Array{ST}
    ∂u∂P_t₁_quad_values::Array{ST}
    ∂u∂P_x₀_quad_values::Array{ST}
    ∂u∂P_x₁_quad_values::Array{ST}

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
    function Galerkin_Bspline_IntegratorCache{ST,RT,RX,D,S}() where {ST,RT,RX,D,S}
        # x = zeros(ST, NP + 2 * D * RX + 2* D * DX * RT ) # TODO: how to deal with RX being a vector/
        x = zeros(ST,S + D * RX +2* D * RT) # params, λ₀_x_coes,μ₀_t_coes,μ₁_t_coes
        # TODO:consider when DX is a vector
        # x = zeros(ST,S)
        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        λ₀_x_coes = zeros(ST, D, RX)
        μ₀_t_coes = zeros(ST, D, RT)
        μ₁_t_coes = zeros(ST, D, RT)

        λ₀_quad_values = zeros(ST, D, RX) 
        μ₀_quad_values = zeros(ST, D, RT) 
        μ₁_quad_values = zeros(ST, D, RT) 

        ∂u∂P_t₀_quad_values = zeros(ST,D,RX,S)
        ∂u∂P_t₁_quad_values = zeros(ST,D,RX,S)
        ∂u∂P_x₀_quad_values = zeros(ST,D,RT,S)
        ∂u∂P_x₁_quad_values = zeros(ST,D,RT,S)

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

        init_condition_t₀ = zeros(ST, D, RX)
        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)

        
        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            λ₀_x_coes, μ₀_t_coes,μ₁_t_coes, 
            λ₀_quad_values, μ₀_quad_values, μ₁_quad_values,
            ∂u∂P_t₀_quad_values, ∂u∂P_t₁_quad_values, ∂u∂P_x₀_quad_values, ∂u∂P_x₁_quad_values,
            ut₀_quad_values, ut₁_quad_values,vt₀_quad_values, vt₁_quad_values,wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values,vx₀_quad_values, vx₁_quad_values,wx₀_quad_values, wx₁_quad_values,
            init_condition_t₀, 
            boundary_condition_x₀, boundary_condition_x₁,
            )
    end
end

nlsolution(cache::Galerkin_Bspline_IntegratorCache) = cache.x

function Cache{ST}(problem::PDEProblem, int::Galerkin_Bspline_Integrator; kwargs...) where {ST}
    Galerkin_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.S}(; kwargs...)
end

#{ST,RT,RX,D,NP}(NP) where {ST,RT,RX,D,NP}
@inline GeometricIntegrators.Integrators.CacheType(ST, problem::PDEProblem, int::Galerkin_Bspline_Integrator) = Galerkin_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.S}

@inline function Base.getindex(c::Galerkin_Bspline_IntegratorCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end


function prior_initial_guess!(C,sol,int::PDEIntegrator{<:Galerkin_Bspline_Integrator})
    local exact_u = int.problem.exact_u
    local xs = int.method.basis.xs
    local ts = int.method.basis.ts # t \in [0, 1.0]
    local Cx = int.method.basis.Collocation_x
    local Ct = int.method.basis.Collocation_t
    local S = int.method.basis.S
    local h = timestep(int) 
    local a = int.problem.xspan[1]
    local b = int.problem.xspan[2]
    local x_domain = b - a

    fdata = exact_u.(h .* ts, a .+ x_domain .* xs') # fdata[i,j] = exact_u(h .* ts[i], xs[j])

    # 2D B-spline coefficients (output)
    coefs = similar(fdata)
    # @infiltrate

    # Solve linear systems
    for j ∈ eachindex(xs)
        @views ldiv!(coefs[:, j], Ct, fdata[:, j])
    end
    for i ∈ eachindex(ts)
        @views ldiv!(Cx, coefs[i, :])
    end

    C.x[1:S] = reshape(coefs, :, 1)
end

function post_initial_guess!(C,sol_struct,int::PDEIntegrator{<:Galerkin_Bspline_Integrator},int_method::Galerkin_Bspline_Integrator{T,MVT,LT,BT}) where {T,MVT<:BSplineDirichlet{T},LT<:BSplineDirichlet{T},BT}
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
    local S = int_method.basis.S
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
            C.x[S + (d - 1) * RX + i] = λ₀_x.spline.coefs[i]
        end

        tem_x₀_∂L∂W = zeros(RT)
        for i in 1:RT
            tem_x₀_∂L∂W[i] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,i], C.vx₀_quad_values[d,i], C.wx₀_quad_values[d,i], params)
        end

        tem_t = sol_struct.t .- timestep(int) .+ timestep(int) .* t_quad_nodes
        μ₀_t = interpolate(tem_t, tem_x₀_∂L∂W, BSplineOrder(int_method.k_μ))
        for i in 1:RT
            C.x[S + D * RX + (d - 1) * RT + i] = μ₀_t.spline.coefs[i]
        end

        tem_x₁_∂L∂W = zeros(RT)
        for i in 1:RT
            tem_x₁_∂L∂W[i] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,i], C.vx₁_quad_values[d,i], C.wx₁_quad_values[d,i], params)
        end

        μ₁_t = interpolate(tem_t, tem_x₁_∂L∂W, BSplineOrder(int_method.k_μ))
        for i in 1:RT
            C.x[S+ D * RX + D * RT + (d-1)*RT+i] = μ₁_t.spline.coefs[i]
        end

    end
end

function post_initial_guess!(C,sol_struct,int::PDEIntegrator{<:Galerkin_Bspline_Integrator},int_method::Galerkin_Bspline_Integrator{T,MVT,LT,BT}) where {T,MVT<:Lagrange,LT<:Lagrange,BT}
    local S = int_method.basis.S
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
            C.x[S + (d - 1) * RX + i] = lag_sys.∂L∂V[d](C.ut₀_quad_values[d,i], C.vt₀_quad_values[d,i], C.wt₀_quad_values[d,i], params)
        end

        for i in 1:RT
            C.x[S + D * RX + (d - 1) * RT + i] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,i], C.vx₀_quad_values[d,i], C.wx₀_quad_values[d,i], params)
            C.x[S + D * RX + D * RT + (d - 1) * RT + i] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,i], C.vx₁_quad_values[d,i], C.wx₁_quad_values[d,i], params)
        end
    end
end

function initialize_bcs_ics!(sol,int::PDEIntegrator{<:Galerkin_Bspline_Integrator})
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
            nothing
            # # println("sol.internal.x[current_step-1][1:NP] = " , sol.internal.x[current_step-1][1:NP])
            # sol_params = (L1=(W=zeros(S, 2), b=zeros(S)),L2=(W=zeros(1, S),))
            # sol_params.L1.W[:] = u[d].params.L1.W[:]
            # sol_params.L1.b[:] = u[d].params.L1.b[:]
            # sol_params.L2.W[:] = sol.internal.x[current_step-1][1:NP]
            # @show sol_params
            # # sol_params = NeuralNetworkParameters(reconstruct_params(sol.internal.x[current_step-1][1:NP], u[d].params))
            # for i in eachindex(C.init_condition_t₀[d,:])
            #     C.init_condition_t₀[d,i] = u[d]([1.0,xspan[1] + x_domain * x_quad_nodes[i]], sol_params)[1]
            # end
            # println("initial condition = " , C.init_condition_t₀[d,:])
        end

        for i in 1:RT
            C.boundary_condition_x₀[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.u
            C.boundary_condition_x₁[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.u
        end
        # println("left boundary condition = " , C.boundary_condition_x₀[d,:])
    end

end

function components!(x::AbstractVector{ST}, sol, int::PDEIntegrator{<:Galerkin_Bspline_Integrator}) where {ST}
    local C = cache(int,ST)
    local h = timestep(int)
    local RT = int.method.RT
    local RX = int.method.RX
    local D = int.problem.D
    local S = int.method.basis.S

    local ∂L∂U = int.problem.lagrangian_system.functions.∂L∂U
    local ∂L∂V = int.problem.lagrangian_system.functions.∂L∂V
    local ∂L∂W = int.problem.lagrangian_system.functions.∂L∂W

    local grid_matrix = int.method.grid_matrix
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]

    local lag_params = int.problem.lagrangian_system.params
    local mλ₀_x = int.method.mλ₀_x
    local mμ_t = int.method.mμ_t
    
    # interior values at quadrature points
    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = sum(x[1:S] .* int.method.u_collocation_mat[:, i, j])
                C.v_quad_values[d, i, j] = sum(x[1:S] .* int.method.v_collocation_mat[:, i, j]) / h 
                C.w_quad_values[d, i, j] = sum(x[1:S] .* int.method.w_collocation_mat[:, i, j]) / x_domain
            end
        end
    end

    # @show C.u_quad_values[1,1,:]
    # @show C.v_quad_values[1,1,:]
    # @show C.w_quad_values[1,1,:]

    # local exact_u = int.problem.exact_u
    # local exact_v = int.problem.exact_v
    # local exact_w = int.problem.exact_w
    # local quad_x_nodes = int.method.spatial_quadrature.nodes
    # local quad_t_nodes = int.method.time_quadrature.nodes
    # local h = timestep(int)
    
    # @show exact_u.(h * quad_t_nodes[1], quad_x_nodes)
    # @show exact_v.(h * quad_t_nodes[1], quad_x_nodes)
    # @show exact_w.(h * quad_t_nodes[1], quad_x_nodes)

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], lag_params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], lag_params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], lag_params)
            end
        end 
    end

    # boundary values at quadrature points
    for d in 1:D
        for j in 1:RX
            C.ut₀_quad_values[d,j] = sum(x[1:S] .* int.method.ut₀_basis_quad_values[:,j])
            C.ut₁_quad_values[d,j] = sum(x[1:S] .* int.method.ut₁_basis_quad_values[:,j])

            C.vt₀_quad_values[d,j] = sum(x[1:S] .* int.method.vt₀_basis_quad_values[:,j]) / h
            C.vt₁_quad_values[d,j] = sum(x[1:S] .* int.method.vt₁_basis_quad_values[:,j]) / h

            C.wt₀_quad_values[d,j] = sum(x[1:S] .* int.method.wt₀_basis_quad_values[:,j]) / x_domain
            C.wt₁_quad_values[d,j] = sum(x[1:S] .* int.method.wt₁_basis_quad_values[:,j]) / x_domain
        end

        for i in 1:RT
            C.ux₀_quad_values[d,i] = sum(x[1:S] .* int.method.ux₀_basis_quad_values[:,i])
            C.ux₁_quad_values[d,i] = sum(x[1:S] .* int.method.ux₁_basis_quad_values[:,i])
            C.vx₀_quad_values[d,i] = sum(x[1:S] .* int.method.vx₀_basis_quad_values[:,i])/ h
            C.vx₁_quad_values[d,i] = sum(x[1:S] .* int.method.vx₁_basis_quad_values[:,i])/ h
            C.wx₀_quad_values[d,i] = sum(x[1:S] .* int.method.wx₀_basis_quad_values[:,i])/ x_domain 
            C.wx₁_quad_values[d,i] = sum(x[1:S] .* int.method.wx₁_basis_quad_values[:,i])/ x_domain
        end

    end

    (x == cache(int).x && eltype(x) == eltype(cache(int).x)) ? post_initial_guess!(cache(int),sol,int,int.method) : nothing

    for d in 1:D
        C.λ₀_x_coes[d,:] = x[S+1:S+RX]
        C.μ₀_t_coes[d,:] = x[S+RX+1:S+RX+RT]
        C.μ₁_t_coes[d,:] = x[S+RX+RT+1:S+RX+2*RT]
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
    # @infiltrate
end

function residual!(b::Vector{ST}, sol, int::PDEIntegrator{IT}) where {ST,T, MVT <: BSplineDirichlet,LT  <: BSplineDirichlet,BT,IT <: Galerkin_Bspline_Integrator{T, MVT, LT, BT}}
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX
    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local brx = int.method.spatial_quadrature.weights
    local brt = int.method.time_quadrature.weights
    local mλ₀_x = int.method.mλ₀_x
    local mμ_t = int.method.mμ_t
    local S = int.method.basis.S
    local u_coll_mat = int.method.u_collocation_mat
    local v_coll_mat = int.method.v_collocation_mat
    local w_coll_mat = int.method.w_collocation_mat
    local ut₀_basis_quad_values = int.method.ut₀_basis_quad_values
    local ux₀_basis_quad_values = int.method.ux₀_basis_quad_values
    local ux₁_basis_quad_values = int.method.ux₁_basis_quad_values

    current_idx = 1
    for d in 1:D 
        for p in 1:S
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z +=  quad_b[rt,rx] * 
                        ( x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx] * u_coll_mat[p, rt, rx]
                        + x_domain                 * C.∂L∂V_quad_values[d,rt,rx] * v_coll_mat[p, rt, rx]
                        + timestep(int)            * C.∂L∂W_quad_values[d,rt,rx] * w_coll_mat[p, rt, rx])
                end
            end
            for rx in 1:RX
                z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * ut₀_basis_quad_values[p,rx]) #- C.λ₁_quad_values[d,rx] * C.∂u∂P_t₁_quad_values[d, rx]
            end
            for rt in 1:RT
                z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * ux₀_basis_quad_values[p,rt] - C.μ₁_quad_values[d,rt] * ux₁_basis_quad_values[p,rt])
            end
            b[current_idx] = z 
            current_idx += 1
        end
    end
    # @infiltrate 
    @assert current_idx == S + 1 "Wrong indexing in residual computation"

    for d in 1:D
        for i in 1:RX
            z = zero(ST)
            for rx in 1:RX
                z += x_domain * brx[rx] * mλ₀_x[i,rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
            end
            b[S + (d - 1) * RX + i] = z
        end
    end

    for d in 1:D
        for i in 1:RT
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) *brt[rt] * mμ_t[i,rt] *(C.ux₀_quad_values[d,rt] - C.boundary_condition_x₀[d,rt])
            end
            b[S + D * RX + (d - 1) * RT + i] = z
        end
    end

    for d in 1:D
        for i in 1:RT
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) *brt[rt] * mμ_t[i,rt] *(C.boundary_condition_x₁[d,rt] - C.ux₁_quad_values[d,rt])
            end
            b[S + D * RX + D * RT + (d - 1) * RT + i] = z
        end
    end
    @infiltrate
end


function residual!(b::Vector{ST}, sol, int::PDEIntegrator{IT}) where {ST,T, MVT <: Lagrange,LT  <: Lagrange,BT,IT <: Galerkin_Bspline_Integrator{T, MVT, LT, BT}}
    local D = int.problem.D 
    local RT = int.method.RT
    local RX = int.method.RX

    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local brx = int.method.spatial_quadrature.weights
    local brt = int.method.time_quadrature.weights
    local u_coll_mat = int.method.u_collocation_mat
    local v_coll_mat = int.method.v_collocation_mat
    local w_coll_mat = int.method.w_collocation_mat
    local S = int.method.basis.S
    
    current_idx = 1
    for d in 1:D 
        for p in 1:S
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z +=  quad_b[rt,rx] * 
                        ( x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx] * u_coll_mat[p, rt, rx]
                        + x_domain                 * C.∂L∂V_quad_values[d,rt,rx] * v_coll_mat[p, rt, rx]
                        + x_domain * timestep(int) * C.∂L∂W_quad_values[d,rt,rx] * w_coll_mat[p, rt, rx])
                end
            end
            for rx in 1:RX
                z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * C.∂u∂P_t₀_quad_values[d,rx,p]) #- C.λ₁_quad_values[d,rx] * C.∂u∂P_t₁_quad_values[d, rx]
            end
            for rt in 1:RT
                z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * C.∂u∂P_x₀_quad_values[d,rt,p] - C.μ₁_quad_values[d,rt] * C.∂u∂P_x₁_quad_values[d,rt,p])
            end
            b[current_idx] = -z # TODO: check the sign
            current_idx += 1
        end
    end

    @assert current_idx == S + 1 "Wrong indexing in residual computation"

    for d in 1:D
        for rx in 1:RX
            b[S + (d - 1) * RX + rx] = x_domain * brx[rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
        end
    end

    for d in 1:D
        for rt in 1:RT
            b[S+ D * RX+(d-1)*RT+rt]= timestep(int) *brt[rt]*(C.ux₀_quad_values[d,rt] - C.boundary_condition_x₀[d,rt])
        end
    end

    for d in 1:D
        for rt in 1:RT
            b[S+ D * RX+ D * RT + (d-1)*RT+rt]= timestep(int) *brt[rt]*(C.ux₁_quad_values[d,rt] - C.boundary_condition_x₁[d,rt])
        end
    end
    # @infiltrate
    @show b
end



function update!(sol_struct, int::PDEIntegrator{<:Galerkin_Bspline_Integrator})
    local D = int.problem.D
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local x = nlsolution(int)
    local S = int.method.basis.S
    local Basis_t = int.method.basis.Basis_t
    local Basis_x = int.method.basis.Basis_x

    x_nodes = collect(xspan[1]:xstep:xspan[2])
    ubasis_x_nodes_values = zeros(S, length(x_nodes))
    vbasis_x_nodes_values = zeros(S, length(x_nodes))
    wbasis_x_nodes_values = zeros(S, length(x_nodes))
    for i in eachindex(x_nodes)
        ubasis_x_nodes_values[:,i] = spline2D_coeff_derivatives((Basis_t, Basis_x),(0.0 ,x_nodes[i])) # bottom
        vbasis_x_nodes_values[:,i] = spline2D_coeff_derivatives_time((Basis_t, Basis_x),(0.0 ,x_nodes[i])) # bottom
        wbasis_x_nodes_values[:,i] = spline2D_coeff_derivatives_space((Basis_t, Basis_x),(0.0 ,x_nodes[i])) # bottom
    end

    for d in 1:D
        for i in eachindex(x_nodes)
            sol_struct.sol.u[sol_struct.current_step][i] = sum(x[1:S] .* ubasis_x_nodes_values[:,i])
            sol_struct.sol.v[sol_struct.current_step][i] = sum(x[1:S] .* vbasis_x_nodes_values[:,i])
            sol_struct.sol.w[sol_struct.current_step][i] = sum(x[1:S] .* wbasis_x_nodes_values[:,i])
        end
    end

    # copy internal variables from cache to solution
    sol_struct.internal.x[sol_struct.current_step] .= cache(int).x 

    sol_struct.t += int.problem.tstep
    # println("In the end of update! function, time = ", sol_struct.t)
end

