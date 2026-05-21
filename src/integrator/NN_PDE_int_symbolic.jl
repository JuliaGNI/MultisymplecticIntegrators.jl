struct NN_PDE_Integrator_Symbolic{MVT,LT,BT<:AbstractPDEBasis,IPMT<:InitialParametersMethod} <: PDEMethod
    basis::BT

    time_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RT::Int # Number of quadrature points in time

    spatial_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix::Matrix{Vector{Float64}} # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights::Matrix{Float64} # Quadrature weights

    t₀_quad::Vector{Vector{Float64}} # quadrature points on t=0 boundary
    t₁_quad::Vector{Vector{Float64}}
    x₀_quad::Vector{Vector{Float64}}
    x₁_quad::Vector{Vector{Float64}}

    t_num_interval::Int
    k_μ_t::Int # order
    μ₀_t::MVT
    μ₁_t::MVT
    Nbasis_μ_t::Int

    x_num_interval::Int
    k_λ_x::Int # order
    λ_x::LT
    Nbasis_λ_x::Int

    mλ_x::Matrix{Float64} # λ_x evaluated at quadrature points
    mμ_t::Matrix{Float64}

    nepochs::Int
    initial_guess_method::IPMT #

    show_status::Bool
    function NN_PDE_Integrator_Symbolic(basis;RT_per_interval::Int = 4,RX_per_interval::Int = 4,
        xspan::Tuple=(0., 1.0),
        nepochs=1000,
        t_num_interval::Int=10, k_μ_t::Int=4, μ::Symbol=:BSplineDirichlet,
        x_num_interval::Int=10, k_λ_x::Int=3, λ::Symbol=:BSplineDirichlet,
        show_status::Bool=false,
        nx::Int = 40,nt::Int= 20,Nw::Int=500, Nb::Int=500,
        initial_guess_method::IPMT=OGA2D(xspan[1], xspan[2],basis.activation_function,nx = nx,nt=nt,Nw=Nw, Nb=Nb), # hyperparameters for OGA2d

        ) where {IPMT,}

        t_quadrature = composite_quadrature(t_num_interval ,RT_per_interval)
        x_quadrature = composite_quadrature(x_num_interval ,RX_per_interval)

        RT = length(t_quadrature.nodes)
        RX = length(x_quadrature.nodes)

        R_list = [RT_per_interval,RX_per_interval]
        grid_matrix, grid_weights = construct_quadrature_grid(R_list,[t_num_interval, x_num_interval])
        # scale grid_matrix
        x0 = xspan[1]
        x_domain = xspan[2] - xspan[1]
        grid_matrix = [collect(grid_matrix[i, j]) for i in axes(grid_matrix, 1), j in axes(grid_matrix, 2)]

        @inbounds for k in eachindex(grid_matrix)
            t, xhat = grid_matrix[k]
            grid_matrix[k][2] = x0 + x_domain * xhat
        end

        # construct quad point at 4 boundary
        t₀_quad = [[0.0, x_quadrature.nodes[i]] for i in 1:RX]
        t₁_quad = [[1.0, x_quadrature.nodes[i]] for i in 1:RX]
        x₀_quad = [[t_quadrature.nodes[i], xspan[1]] for i in 1:RT]
        x₁_quad = [[t_quadrature.nodes[i], xspan[2]] for i in 1:RT]

        # Construct Lagrangian multipliers, defined on [0,1] and need to be scaled carefully when used
        λ_x = Lagrangian_multiplier(λ, x_num_interval, k_λ_x, xspan[1], xspan[2])
        μ₀_t = Lagrangian_multiplier(μ, t_num_interval, k_μ_t, 0.0, 1.0)
        μ₁_t = Lagrangian_multiplier(μ, t_num_interval, k_μ_t, 0.0, 1.0)

        Nbasis_μ_t = length(μ₀_t.b)
        Nbasis_λ_x = length(λ_x.b)

        mλ_x = zeros(Nbasis_λ_x, RX)
        mμ_t = zeros(Nbasis_μ_t, RT)

        for i in 1:Nbasis_λ_x
            mλ_x[i, :] = λ_x.b[i].(xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes)
        end

        for i in 1:Nbasis_μ_t
            mμ_t[i, :] = μ₀_t.b[i].(t_quadrature.nodes)
        end

        new{typeof(μ₀_t),typeof(λ_x),typeof(basis),typeof(initial_guess_method)}(basis,
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            t₀_quad, t₁_quad, x₀_quad, x₁_quad,
            t_num_interval, k_μ_t, μ₀_t, μ₁_t,Nbasis_μ_t,
            x_num_interval, k_λ_x, λ_x,Nbasis_λ_x,
            mλ_x, mμ_t,
            nepochs, initial_guess_method,
            show_status)
    end
end

default_solver(::NN_PDE_Integrator_Symbolic) = NewtonMethod()

struct NN_PDE_IntegratorCache_Symbolic{ST,RT,RX,D,NP,S,Nbasis_μ_t,Nbasis_λ_x,N,M} <: PDEIntegratorCache{ST,D}
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

    ∂u∂P_quad_values::Vector{AbstractArray{ST,3}}
    ∂v∂P_quad_values::Vector{AbstractArray{ST,3}}
    ∂w∂P_quad_values::Vector{AbstractArray{ST,3}}

    λ₀_x_coes::Matrix{ST}
    λ₁_x_coes::Matrix{ST}
    μ₀_t_coes::Matrix{ST}
    μ₁_t_coes::Matrix{ST}

    λ₀_quad_values::Matrix{ST}
    λ₁_quad_values::Matrix{ST}
    μ₀_quad_values::Matrix{ST}
    μ₁_quad_values::Matrix{ST}

    ∂u∂P_t₀_quad_values::Vector{AbstractArray{ST,2}}
    ∂u∂P_t₁_quad_values::Vector{AbstractArray{ST,2}}
    ∂u∂P_x₀_quad_values::Vector{AbstractArray{ST,2}}
    ∂u∂P_x₁_quad_values::Vector{AbstractArray{ST,2}}

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

    ics_ut₀_quad_values::Matrix{ST}
    ics_vt₀_quad_values::Matrix{ST}
    ics_wt₀_quad_values::Matrix{ST}

    bc_ux₀_quad_values::Matrix{ST}
    bc_vx₀_quad_values::Matrix{ST}
    bc_wx₀_quad_values::Matrix{ST}

    bc_ux₁_quad_values::Matrix{ST}
    bc_vx₁_quad_values::Matrix{ST}
    bc_wx₁_quad_values::Matrix{ST}


    init_condition_t₀::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}

    params::Vector{ST}
    flag_done_initial_guess::Vector{ST}

    # fields for OGA2D
    B::Matrix{ST}
    coeffs_full::Vector{ST}
    Wsel::Matrix{ST}
    Bsel::Vector{ST}
    desired::Vector{ST}
    corrs::Vector{ST}
    selected::Vector{Int}
    function NN_PDE_IntegratorCache_Symbolic{ST,RT,RX,D,NP,S,Nbasis_μ_t,Nbasis_λ_x,N,M}() where {ST,RT,RX,D,NP,S,Nbasis_μ_t,Nbasis_λ_x,N,M}
        # x = zeros(ST, NP + 2 * D * RX + 2* D * DX * RT ) # TODO: how to deal with RX being a vector/
        x = zeros(ST, NP + D * Nbasis_λ_x + 2 * D * Nbasis_μ_t) # params, λ₀_x_coes,μ₀_t_coes,μ₁_t_coes
        # TODO:consider when DX is a vector

        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ∂u∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT, RX, D, [NP])
        ∂v∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT, RX, D, [NP])
        ∂w∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT, RX, D, [NP])

        λ₀_x_coes = zeros(ST, D, Nbasis_λ_x)
        λ₁_x_coes = zeros(ST, D, Nbasis_λ_x)
        μ₀_t_coes = zeros(ST, D, Nbasis_μ_t)
        μ₁_t_coes = zeros(ST, D, Nbasis_μ_t)

        λ₀_quad_values = zeros(ST, D, RX)
        λ₁_quad_values = zeros(ST, D, RX)
        μ₀_quad_values = zeros(ST, D, RT)
        μ₁_quad_values = zeros(ST, D, RT)

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

        ics_ut₀_quad_values = zeros(ST, D, RX)
        ics_vt₀_quad_values = zeros(ST, D, RX)
        ics_wt₀_quad_values = zeros(ST, D, RX)

        bc_ux₀_quad_values = zeros(ST, D, RT)
        bc_vx₀_quad_values = zeros(ST, D, RT)
        bc_wx₀_quad_values = zeros(ST, D, RT)

        bc_ux₁_quad_values = zeros(ST, D, RT)
        bc_vx₁_quad_values = zeros(ST, D, RT)
        bc_wx₁_quad_values = zeros(ST, D, RT)

        init_condition_t₀ = zeros(ST, D, RX)

        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)

        params = zeros(ST,4*S) # for fully optimize, store all layers' parameters, for partially optimize, only store the optimized layer's parameters, and regenerate the rest in each iteration
        flag_done_initial_guess = zeros(ST, 1)

        B = zeros(ST, N, S)   # orthonormal basis columns
        coeffs_full = zeros(ST, S)     # coefficients to write into PNN L2
        Wsel = zeros(ST, S, 2)
        Bsel = zeros(ST, S)
        desired = zeros(ST, N)
        corrs = zeros(ST, M)
        selected = zeros(Int, S) # indices of selected atoms in the dictionary

        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂P_quad_values, ∂v∂P_quad_values, ∂w∂P_quad_values,
            λ₀_x_coes, λ₁_x_coes, μ₀_t_coes, μ₁_t_coes,
            λ₀_quad_values, λ₁_quad_values, μ₀_quad_values, μ₁_quad_values,
            ∂u∂P_t₀_quad_values, ∂u∂P_t₁_quad_values, ∂u∂P_x₀_quad_values, ∂u∂P_x₁_quad_values,
            ut₀_quad_values, ut₁_quad_values, vt₀_quad_values, vt₁_quad_values, wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values, vx₀_quad_values, vx₁_quad_values, wx₀_quad_values, wx₁_quad_values,
            ics_ut₀_quad_values, ics_vt₀_quad_values, ics_wt₀_quad_values,
            bc_ux₀_quad_values, bc_vx₀_quad_values, bc_wx₀_quad_values,
            bc_ux₁_quad_values, bc_vx₁_quad_values, bc_wx₁_quad_values,
            init_condition_t₀,
            boundary_condition_x₀, boundary_condition_x₁,
            params, flag_done_initial_guess,
            B, coeffs_full, Wsel, Bsel, desired, corrs, selected)
    end
end

nlsolution(cache::NN_PDE_IntegratorCache_Symbolic) = cache.x

function Cache{ST}(problem::LPDEProblem, int::NN_PDE_Integrator_Symbolic; kwargs...) where {ST}
    NN_PDE_IntegratorCache_Symbolic{ST,int.RT,int.RX,problem.D,int.basis.NP,int.basis.S,int.Nbasis_μ_t,int.Nbasis_λ_x,int.initial_guess_method.N,int.initial_guess_method.M}(; kwargs...)
end

@inline CacheType(ST, problem::LPDEProblem, int::NN_PDE_Integrator_Symbolic) = NN_PDE_IntegratorCache_Symbolic{ST,int.RT,int.RX,problem.D,int.basis.NP,int.basis.S,int.Nbasis_μ_t,int.Nbasis_λ_x,int.initial_guess_method.N,int.initial_guess_method.M}


# zero_vectors!(x::NamedTuple) = foreach(zero_vectors!, values(x))
# zero_vectors!(x::AbstractArray) = fill!(x, zero(eltype(x)))
# zero_vectors!(x) = nothing

function prior_initial_guess!(C, sol, int::PDEIntegrator{<:NN_PDE_Integrator_Symbolic{MVT,LT,BT,IPMT}}) where {MVT,LT,BT,IPMT<:OGA2D}
    local h = timestep(int)
    local S = int.method.basis.S
    local exact_u = int.problem.exact_u
    local u = int.method.basis.u
    local optim_mode = int.method.basis.optim_mode
    local NP = int.method.basis.NP
    local show_status = int.method.show_status

    local quad_nodes = int.method.initial_guess_method.equispaced_quad_nodes
    local quad_weights = int.method.initial_guess_method.quad_weights
    local A_mat = int.method.initial_guess_method.A_mat
    local Φ_raw = int.method.initial_guess_method.Φ_raw
    local N = int.method.initial_guess_method.N
    local M = int.method.initial_guess_method.M
    local tn = sol.t - timestep(int)

    # local desired = C.desired
    # local corrs = C.corrs
    # local Wsel = C.Wsel
    # local Bsel = C.Bsel
    # local B = C.B
    # local coeffs_full = C.coeffs_full

    B = zeros(N, S)   # orthonormal basis columns
    coeffs_full = zeros(S)     # coefficients to write into PNN L2
    Wsel = zeros(S, 2)
    Bsel = zeros(S)
    desired = zeros(N)
    corrs = zeros(M)
    selected = zeros(Int, S) # indices of selected atoms in the dictionary
    zero_vectors!(C.params)

    # Build the desired internal PNN output on all quadrature nodes:
    for i in 1:N
        t = quad_nodes[1, i]
        x = quad_nodes[2, i]
        desired[i] = exact_u(tn + h*t, x) - u(t, x, C.params)[1]
    end

    # Run OGA (orthogonal matching) on Φ_raw to approximate `desired`
    residual = copy(desired)

    for s = 1:S
        # compute correlations with residual (weighted)
        for i in 1:M
            corrs[i] = abs(sum(Φ_raw[i, :] .* (residual .* quad_weights)))
        end
        idx = argmax(corrs)
        selected[s] = idx
        length(Set(selected)) -1 == s ? nothing : @warn "atom repeated at s=$s, idx=$idx"

        # extract raw atom (already normalized) and orthogonalize (Gram-Schmidt)
        φ = copy(Φ_raw[idx, :])

        # append to B
        @views B[:, s] .= φ

        # solve least-squares for coefficients in orthonormal basis
        coeffs = view(B,:,1:s) \ desired         # small system k×1 solved implicitly
        # update residual
        residual = desired - view(B,:,1:s) * coeffs

        # store selection params (note A_mat rows correspond to atoms prior to normalization,
        # yet we normalized Φ_raw; we must store original (w,b) for a neuron consistent with A_mat)
        @views Wsel[s, :] .= A_mat[idx, 1:2]
        Bsel[s] = A_mat[idx, 3]

        coeffs_full[1:s] .= coeffs
        if show_status
            println("s=$s idx=$idx ‖residual‖=$(norm(residual))")
        end
    end

    # for j = 1:S
    #     @views u.params.L1.W[j, :] .= Wsel[j, :]
    #     u.params.L1.b[j] = Bsel[j]
    #     u.params.L2.W[j] = coeffs_full[j]

    #     @views C.params.L1.W[j, :] .= Wsel[j, :]
    #     C.params.L1.b[j] = Bsel[j]
    #     C.params.L2.W[j] = coeffs_full[j]
    # end

    C.params[1:S] .= Wsel[:,1] # L1.W
    C.params[S+1:2*S] .= Wsel[:,2] # L1.W
    C.params[2*S+1:3*S] .= Bsel # L1.b
    C.params[3*S+1:3*S+S] .= coeffs_full # L2.W

    # copy the parameters to the cache
    if optim_mode == :Partially
        C.x[1:NP] = coeffs_full
    elseif optim_mode == :Fully
        C.x[1:2*S] = C.params[1:2*S]
        C.x[2*S+1:3*S] = C.params[2*S+1:3*S]
        C.x[3*S+1:4*S] = C.params[3*S+1:4*S]
    end

    # if show_status
    target_vec = [exact_u(tn + h * quad_nodes[1, i], quad_nodes[2, i]) for i in 1:N]
    approx_vec = [u(quad_nodes[:, i][1], quad_nodes[:, i][2], C.params)[1] for i in 1:N]
    err_vec = abs.(target_vec .- approx_vec)
    println("Max abs error after OGA initial guess: ", maximum(err_vec))
    # println("OGA initial guess completed.")
    # println("Initial guess \n", C.x)
    # # end

end

copy_internal_variables!(C::NN_PDE_IntegratorCache_Symbolic, solstep::SolutionStep) = nothing

function post_initial_guess!(C, sol, int::PDEIntegrator{<:NN_PDE_Integrator_Symbolic}, int_method::NN_PDE_Integrator_Symbolic{MVT,LT,BT,IPMT}) where {MVT<:BSplineDirichlet,LT<:BSplineDirichlet,BT,IPMT}
    local Nbasis_λ_x = int_method.Nbasis_λ_x
    local Nbasis_μ_t = int_method.Nbasis_μ_t
    local D = int.problem.D
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params
    local NP = int_method.basis.NP
    local RX = int.method.RX
    local RT = int.method.RT
    local mλ_x = int_method.mλ_x
    local mμ_t = int_method.mμ_t

    for d in 1:D
        tem_t₁_∂L∂V = zeros(RX)
        tem_x₀_∂L∂W = zeros(RT)
        tem_x₁_∂L∂W = zeros(RT)

        for rx in 1:RX
            tem_t₁_∂L∂V[rx] = lag_sys.∂L∂V[d](C.ut₁_quad_values[d,rx], C.vt₁_quad_values[d,rx], C.wt₁_quad_values[d,rx], params)
        end
        λ_x_tem = mλ_x'\tem_t₁_∂L∂V

        for rx in 1:Nbasis_λ_x
            C.x[NP + (d - 1) * Nbasis_λ_x + rx] = λ_x_tem[rx]
        end

        for rt in 1:RT
            tem_x₀_∂L∂W[rt] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,rt], C.vx₀_quad_values[d,rt], C.wx₀_quad_values[d,rt], params)
            tem_x₁_∂L∂W[rt] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,rt], C.vx₁_quad_values[d,rt], C.wx₁_quad_values[d,rt], params)
        end
        μ₀_t_tem = mμ_t'\tem_x₀_∂L∂W
        μ₁_t_tem = mμ_t'\tem_x₁_∂L∂W

        for rt in 1:Nbasis_μ_t
            C.x[NP + D * Nbasis_λ_x + (d - 1) * Nbasis_μ_t + rt] = μ₀_t_tem[rt]
            C.x[NP + D * Nbasis_λ_x + D * Nbasis_μ_t + (d - 1) * Nbasis_μ_t + rt] = μ₁_t_tem[rt]
        end
    end
    C.flag_done_initial_guess[1] = 1.0

    # @infiltrate
end

function post_initial_guess!(C, sol, int::PDEIntegrator{<:NN_PDE_Integrator_Symbolic}, int_method::NN_PDE_Integrator_Symbolic{MVT,LT,BT,IPMT}) where {MVT<:Lagrange,LT<:Lagrange,BT,IPMT}
    local NP = int_method.basis.NP
    local D = int.problem.D
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params
    local u = [int.method.basis.u]
    local v = [int.method.basis.v]
    local w = [int.method.basis.w]
    local Nbasis_λ_x = int_method.Nbasis_λ_x
    local Nbasis_μ_t = int_method.Nbasis_μ_t
    local xspan = int.problem.xspan
    local λ_x = int_method.λ_x
    local μ₀_t = int_method.μ₀_t
    local show_status = int_method.show_status
    local h = timestep(int)

    ut₀_quad_values_tem = zeros(Nbasis_λ_x)
    ut₁_quad_values_tem = zeros(Nbasis_λ_x)
    vt₀_quad_values_tem = zeros(Nbasis_λ_x)
    vt₁_quad_values_tem = zeros(Nbasis_λ_x)
    wt₀_quad_values_tem = zeros(Nbasis_λ_x)
    wt₁_quad_values_tem = zeros(Nbasis_λ_x)

    ux₀_quad_values_tem = zeros(Nbasis_μ_t)
    ux₁_quad_values_tem = zeros(Nbasis_μ_t)
    vx₀_quad_values_tem = zeros(Nbasis_μ_t)
    vx₁_quad_values_tem = zeros(Nbasis_μ_t)
    wx₀_quad_values_tem = zeros(Nbasis_μ_t)
    wx₁_quad_values_tem = zeros(Nbasis_μ_t)

    for d in 1:D
        for rx in 1:Nbasis_λ_x
            xx = λ_x.x[rx]
            ut₀_quad_values_tem[rx] = (u[d])([0.0, xx], C.params)[1]
            ut₁_quad_values_tem[rx] = (u[d])([1.0, xx], C.params)[1]
            vt₀_quad_values_tem[rx] = (v[d])([0.0, xx], C.params)[1] / h
            vt₁_quad_values_tem[rx] = (v[d])([1.0, xx], C.params)[1] / h
            wt₀_quad_values_tem[rx] = (w[d])([0.0, xx], C.params)[1]
            wt₁_quad_values_tem[rx] = (w[d])([1.0, xx], C.params)[1]
        end

        for rt in 1:Nbasis_μ_t
            tt = μ₀_t.x[rt]
            ux₀_quad_values_tem[rt] = (u[d])([tt, xspan[1]], C.params)[1]
            ux₁_quad_values_tem[rt] = (u[d])([tt, xspan[2]], C.params)[1]
            vx₀_quad_values_tem[rt] = (v[d])([tt, xspan[1]], C.params)[1] / h
            vx₁_quad_values_tem[rt] = (v[d])([tt, xspan[2]], C.params)[1] / h
            wx₀_quad_values_tem[rt] = (w[d])([tt, xspan[1]], C.params)[1]
            wx₁_quad_values_tem[rt] = (w[d])([tt, xspan[2]], C.params)[1]
        end
    end
    for d in 1:D
        for rx in 1:Nbasis_λ_x
            C.x[NP+(d-1)*Nbasis_λ_x+rx] = lag_sys.∂L∂V[d](ut₁_quad_values_tem[rx], vt₁_quad_values_tem[rx], wt₁_quad_values_tem[rx], params)
        end

        for rt in 1:Nbasis_μ_t
            C.x[NP+D*Nbasis_λ_x+(d-1)*Nbasis_μ_t+rt] = lag_sys.∂L∂W[d](ux₀_quad_values_tem[rt], vx₀_quad_values_tem[rt], wx₀_quad_values_tem[rt], params)
            C.x[NP+D*Nbasis_λ_x+D*Nbasis_μ_t+(d-1)*Nbasis_μ_t+rt] = lag_sys.∂L∂W[d](ux₁_quad_values_tem[rt], vx₁_quad_values_tem[rt], wx₁_quad_values_tem[rt], params)
        end
    end
    C.flag_done_initial_guess[1] = 1.0

    if show_status
        local exact_u = int.problem.exact_u
        local exact_v = int.problem.exact_v
        local exact_w = int.problem.exact_w
        ut₀_truth_quad = zeros(Nbasis_λ_x)
        ut₁_truth_quad = zeros(Nbasis_λ_x)
        vt₁_truth_quad = zeros(Nbasis_λ_x)
        vt₀_truth_quad = zeros(Nbasis_λ_x)
        wt₀_truth_quad = zeros(Nbasis_λ_x)
        wt₁_truth_quad = zeros(Nbasis_λ_x)


        ux₀_truth_quad = zeros(Nbasis_μ_t)
        ux₁_truth_quad = zeros(Nbasis_μ_t)
        vx₀_truth_quad = zeros(Nbasis_μ_t)
        vx₁_truth_quad = zeros(Nbasis_μ_t)
        wx₀_truth_quad = zeros(Nbasis_μ_t)
        wx₁_truth_quad = zeros(Nbasis_μ_t)

        for rx in 1:Nbasis_λ_x
            xx = λ_x.x[rx]
            ut₀_truth_quad[rx] = exact_u.(sol.t - timestep(int), xx)
            ut₁_truth_quad[rx] = exact_u.(sol.t, xx)
            vt₀_truth_quad[rx] = exact_v.(sol.t - timestep(int), xx)
            vt₁_truth_quad[rx] = exact_v.(sol.t, xx)
            wt₀_truth_quad[rx] = exact_w.(sol.t - timestep(int), xx)
            wt₁_truth_quad[rx] = exact_w.(sol.t, xx)
        end

        for rt in 1:Nbasis_μ_t
            tt = μ₀_t.x[rt]
            ux₀_truth_quad[rt] = exact_u.(sol.t - timestep(int) + timestep(int) * tt, xspan[1])
            ux₁_truth_quad[rt] = exact_u.(sol.t - timestep(int) + timestep(int) * tt, xspan[2])
            vx₀_truth_quad[rt] = exact_v.(sol.t - timestep(int) + timestep(int) * tt, xspan[1])
            vx₁_truth_quad[rt] = exact_v.(sol.t - timestep(int) + timestep(int) * tt, xspan[2])
            wx₀_truth_quad[rt] = exact_w.(sol.t - timestep(int) + timestep(int) * tt, xspan[1])
            wx₁_truth_quad[rt] = exact_w.(sol.t - timestep(int) + timestep(int) * tt, xspan[2])
        end

        @show maximum(abs.(ut₀_quad_values_tem .- ut₀_truth_quad))
        @show maximum(abs.(vt₀_quad_values_tem .- vt₀_truth_quad))
        @show maximum(abs.(wt₀_quad_values_tem .- wt₀_truth_quad))
        @show maximum(abs.(ut₁_quad_values_tem .- ut₁_truth_quad))
        @show maximum(abs.(vt₁_quad_values_tem .- vt₁_truth_quad))
        @show maximum(abs.(wt₁_quad_values_tem .- wt₁_truth_quad))

        @show maximum(abs.(ux₀_quad_values_tem .- ux₀_truth_quad))
        @show maximum(abs.(vx₀_quad_values_tem .- vx₀_truth_quad))
        @show maximum(abs.(wx₀_quad_values_tem .- wx₀_truth_quad))
        @show maximum(abs.(ux₁_quad_values_tem .- ux₁_truth_quad))
        @show maximum(abs.(vx₁_quad_values_tem .- vx₁_truth_quad))
        @show maximum(abs.(wx₁_quad_values_tem .- wx₁_truth_quad))
    end
    # @infiltrate

end

function components!(x::AbstractVector{ST}, sol, params, int::PDEIntegrator{<:NN_PDE_Integrator_Symbolic}) where {ST}
    local C = cache(int, ST)

    local NP = int.method.basis.NP
    local S = int.method.basis.S
    local RT = int.method.RT
    local RX = int.method.RX
    local D = int.problem.D

    local ∂L∂U = int.problem.lagrangian_system.functions.∂L∂U
    local ∂L∂V = int.problem.lagrangian_system.functions.∂L∂V
    local ∂L∂W = int.problem.lagrangian_system.functions.∂L∂W

    local u = int.method.basis.u
    local v = int.method.basis.v
    local w = int.method.basis.w

    local ∂u∂P = int.method.basis.∂u∂P
    local ∂v∂P = int.method.basis.∂v∂P
    local ∂w∂P = int.method.basis.∂w∂P

    local grid_matrix = int.method.grid_matrix
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local t₀_quad = int.method.t₀_quad
    local t₁_quad = int.method.t₁_quad
    local x₀_quad = int.method.x₀_quad
    local x₁_quad = int.method.x₁_quad

    local lag_params = int.problem.lagrangian_system.params
    local mλ_x = int.method.mλ_x
    local mμ_t = int.method.mμ_t
    local optim_mode = int.method.basis.optim_mode
    local h = timestep(int)
    local Nbasis_λ_x = int.method.Nbasis_λ_x
    local Nbasis_μ_t = int.method.Nbasis_μ_t

    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w
    local show_status = int.method.show_status

    if optim_mode == :Fully
        @views C.params[:] .= x[1:4*S]
    elseif optim_mode == :Partially
        C.params[3*S+1:4*S] .= x[1:S]
    end

    # quad_point = @MArray{2, Float64}(0.0, 0.0)

    # interior values at quadrature points
    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = u(grid_matrix[i, j][1], grid_matrix[i, j][2], C.params)[1]
                C.v_quad_values[d, i, j] = v(grid_matrix[i, j][1], grid_matrix[i, j][2], C.params)[1] / h
                C.w_quad_values[d, i, j] = w(grid_matrix[i, j][1], grid_matrix[i, j][2], C.params)[1]
            end
        end
    end

    if optim_mode == :Fully
        for d in 1:D
            for i in 1:RT
                for j in 1:RX#TODO what if RX is a Vector
                    @views C.∂u∂P_quad_values[d][i, j, :] = ∂u∂P(grid_matrix[i, j][1], grid_matrix[i, j][2], C.params)
                    @views C.∂v∂P_quad_values[d][i, j, :] = ∂v∂P(grid_matrix[i, j][1], grid_matrix[i, j][2], C.params)
                    @views C.∂w∂P_quad_values[d][i, j, :] = ∂w∂P(grid_matrix[i, j][1], grid_matrix[i, j][2], C.params)
                end
            end

            for rx in 1:RX
                @views C.∂u∂P_t₀_quad_values[d][rx, :] = ∂u∂P(t₀_quad[rx][1], t₀_quad[rx][2], C.params)
                @views C.∂u∂P_t₁_quad_values[d][rx, :] = ∂u∂P(t₁_quad[rx][1], t₁_quad[rx][2], C.params)
            end
            for rt in 1:RT
                @views C.∂u∂P_x₀_quad_values[d][rt, :] = ∂u∂P(x₀_quad[rt][1], x₀_quad[rt][2], C.params)
                @views C.∂u∂P_x₁_quad_values[d][rt, :] = ∂u∂P(x₁_quad[rt][1], x₁_quad[rt][2], C.params)
            end
        end
    elseif optim_mode == :Partially
        for d in 1:D
            for i in 1:RT
                for j in 1:RX#TODO what if RX is a Vector
                    @views C.∂u∂P_quad_values[d][i, j, :] = ∂u∂P(grid_matrix[i, j][1], grid_matrix[i, j][2], C.params)
                    @views C.∂v∂P_quad_values[d][i, j, :] = ∂v∂P(grid_matrix[i, j][1], grid_matrix[i, j][2], C.params)
                    @views C.∂w∂P_quad_values[d][i, j, :] = ∂w∂P(grid_matrix[i, j][1], grid_matrix[i, j][2], C.params)
                end
            end
            # q = StaticVector{2, Float64}(0.0, 0.0)
            for rx in 1:RX
                @views C.∂u∂P_t₀_quad_values[d][rx, :] .= ∂u∂P(t₀_quad[rx][1], t₀_quad[rx][2], C.params).L2.W[:]
                @views C.∂u∂P_t₁_quad_values[d][rx, :] .= ∂u∂P(t₁_quad[rx][1], t₁_quad[rx][2], C.params).L2.W[:]
            end
            for rt in 1:RT
                @views C.∂u∂P_x₀_quad_values[d][rt, :] .= ∂u∂P(x₀_quad[rt][1], x₀_quad[rt][2], C.params).L2.W[:]
                @views C.∂u∂P_x₁_quad_values[d][rt, :] .= ∂u∂P(x₁_quad[rt][1], x₁_quad[rt][2], C.params).L2.W[:]
            end
        end
    end

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                current_u = C.u_quad_values[d, i, j]
                current_v = C.v_quad_values[d, i, j]
                current_w = C.w_quad_values[d, i, j]

                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](current_u, current_v, current_w, lag_params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](current_u, current_v, current_w, lag_params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](current_u, current_v, current_w, lag_params)
            end
        end
    end

    for d in 1:D
        for rx in 1:RX
            C.ut₀_quad_values[d, rx] = u(t₀_quad[rx][1], t₀_quad[rx][2], C.params)[1] # bottom
            C.vt₀_quad_values[d, rx] = v(t₀_quad[rx][1], t₀_quad[rx][2], C.params)[1] / h
            C.wt₀_quad_values[d, rx] = w(t₀_quad[rx][1], t₀_quad[rx][2], C.params)[1]
        end
        for rx in 1:RX
            C.ut₁_quad_values[d, rx] = u(t₁_quad[rx][1], t₁_quad[rx][2], C.params)[1] # top
            C.vt₁_quad_values[d, rx] = v(t₁_quad[rx][1], t₁_quad[rx][2], C.params)[1] / h
            C.wt₁_quad_values[d, rx] = w(t₁_quad[rx][1], t₁_quad[rx][2], C.params)[1]
        end

        for rt in 1:RT
            C.ux₀_quad_values[d, rt] = u(x₀_quad[rt][1], x₀_quad[rt][2], C.params)[1]
            C.vx₀_quad_values[d, rt] = v(x₀_quad[rt][1], x₀_quad[rt][2], C.params)[1] / h
            C.wx₀_quad_values[d, rt] = w(x₀_quad[rt][1], x₀_quad[rt][2], C.params)[1]
        end
        for rt in 1:RT
            C.ux₁_quad_values[d, rt] = u(x₁_quad[rt][1], x₁_quad[rt][2], C.params)[1]
            C.vx₁_quad_values[d, rt] = v(x₁_quad[rt][1], x₁_quad[rt][2], C.params)[1] / h
            C.wx₁_quad_values[d, rt] = w(x₁_quad[rt][1], x₁_quad[rt][2], C.params)[1]
        end
    end

    cache(int).flag_done_initial_guess[1] == 0.0 ? post_initial_guess!(cache(int), sol, int, int.method) : nothing
    for d in 1:D
        @views C.λ₁_x_coes[d, :] = x[NP+1:NP+Nbasis_λ_x]
        @views C.μ₀_t_coes[d, :] = x[NP+Nbasis_λ_x+1:NP+Nbasis_λ_x+Nbasis_μ_t]
        @views C.μ₁_t_coes[d, :] = x[NP+Nbasis_λ_x+Nbasis_μ_t+1:NP+Nbasis_λ_x+2*Nbasis_μ_t]
    end

    for d in 1:D
        for rx in 1:RX
            C.λ₀_quad_values[d, rx] = ∂L∂V[d](C.ics_ut₀_quad_values[d, rx], C.ics_vt₀_quad_values[d, rx], C.ics_wt₀_quad_values[d, rx], lag_params)
            @views C.λ₁_quad_values[d, rx] = sum(C.λ₁_x_coes[d, :] .* mλ_x[:, rx])
        end

        for rt in 1:RT
            @views C.μ₀_quad_values[d, rt] = sum(C.μ₀_t_coes[d, :] .* mμ_t[:, rt])
            @views C.μ₁_quad_values[d, rt] = sum(C.μ₁_t_coes[d, :] .* mμ_t[:, rt])
        end
    end

    if show_status
        u_truth_mat = similar(C.u_quad_values)
        v_truth_mat = similar(C.v_quad_values)
        w_truth_mat = similar(C.w_quad_values)

        for d in 1:D
            for i in 1:RT
                for j in 1:RX
                    u_truth_mat[d, i, j] = int.problem.exact_u.(tn + h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                    v_truth_mat[d, i, j] = int.problem.exact_v.(tn + h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                    w_truth_mat[d, i, j] = int.problem.exact_w.(tn + h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                end
            end
        end

        @show maximum(abs.(C.u_quad_values .- u_truth_mat))
        @show maximum(abs.(C.v_quad_values .- v_truth_mat))
        @show maximum(abs.(C.w_quad_values .- w_truth_mat))

        ∂L∂U_truth_mat = similar(C.∂L∂U_quad_values)
        ∂L∂V_truth_mat = similar(C.∂L∂V_quad_values)
        ∂L∂W_truth_mat = similar(C.∂L∂W_quad_values)

        for d in 1:D
            for i in 1:RT
                for j in 1:RX
                    ∂L∂U_truth_mat[d, i, j] = ∂L∂U[d](u_truth_mat[d, i, j], v_truth_mat[d, i, j], w_truth_mat[d, i, j], lag_params)
                    ∂L∂V_truth_mat[d, i, j] = ∂L∂V[d](u_truth_mat[d, i, j], v_truth_mat[d, i, j], w_truth_mat[d, i, j], lag_params)
                    ∂L∂W_truth_mat[d, i, j] = ∂L∂W[d](u_truth_mat[d, i, j], v_truth_mat[d, i, j], w_truth_mat[d, i, j], lag_params)
                end
            end
        end

        @show maximum(abs.(C.∂L∂U_quad_values .- ∂L∂U_truth_mat))
        @show maximum(abs.(C.∂L∂V_quad_values .- ∂L∂V_truth_mat))
        @show maximum(abs.(C.∂L∂W_quad_values .- ∂L∂W_truth_mat))

        # boundary values at quadrature points
        ut₀_quad_values_truth = similar(C.ut₀_quad_values)
        ut₁_quad_values_truth = similar(C.ut₁_quad_values)
        vt₀_quad_values_truth = similar(C.vt₀_quad_values)
        vt₁_quad_values_truth = similar(C.vt₁_quad_values)
        wt₀_quad_values_truth = similar(C.wt₀_quad_values)
        wt₁_quad_values_truth = similar(C.wt₁_quad_values)

        ux₀_quad_values_truth = similar(C.ux₀_quad_values)
        ux₁_quad_values_truth = similar(C.ux₁_quad_values)
        vx₀_quad_values_truth = similar(C.vx₀_quad_values)
        vx₁_quad_values_truth = similar(C.vx₁_quad_values)
        wx₀_quad_values_truth = similar(C.wx₀_quad_values)
        wx₁_quad_values_truth = similar(C.wx₁_quad_values)

        for d in 1:D
            for j in 1:RX
                ut₀_quad_values_truth[d,j] = exact_u.(sol.t - timestep(int), xspan[1] .+ x_domain .* x_quad_nodes[j])
                ut₁_quad_values_truth[d,j] = exact_u.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes[j])
                vt₀_quad_values_truth[d,j] = exact_v.(sol.t - timestep(int), xspan[1] .+ x_domain .* x_quad_nodes[j])
                vt₁_quad_values_truth[d,j] = exact_v.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes[j])
                wt₀_quad_values_truth[d,j] = exact_w.(sol.t - timestep(int), xspan[1] .+ x_domain .* x_quad_nodes[j])
                wt₁_quad_values_truth[d,j] = exact_w.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes[j])
            end

            for i in 1:RT
                ux₀_quad_values_truth[d,i] = exact_u.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[1])
                ux₁_quad_values_truth[d,i] = exact_u.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[2])
                vx₀_quad_values_truth[d,i] = exact_v.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[1])
                vx₁_quad_values_truth[d,i] = exact_v.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[2])
                wx₀_quad_values_truth[d,i] = exact_w.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[1])
                wx₁_quad_values_truth[d,i] = exact_w.(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i], xspan[2])
            end
        end

        @show maximum(abs.(C.ut₀_quad_values .- ut₀_quad_values_truth))
        @show maximum(abs.(C.ut₁_quad_values .- ut₁_quad_values_truth))

        @show maximum(abs.(C.vt₀_quad_values .- vt₀_quad_values_truth))
        @show maximum(abs.(C.vt₁_quad_values .- vt₁_quad_values_truth))

        @show maximum(abs.(C.wt₀_quad_values .- wt₀_quad_values_truth))
        @show maximum(abs.(C.wt₁_quad_values .- wt₁_quad_values_truth))

        @show maximum(abs.(C.ux₀_quad_values .- ux₀_quad_values_truth))
        @show maximum(abs.(C.ux₁_quad_values .- ux₁_quad_values_truth))

        @show maximum(abs.(C.vx₀_quad_values .- vx₀_quad_values_truth))
        @show maximum(abs.(C.vx₁_quad_values .- vx₁_quad_values_truth))

        @show maximum(abs.(C.wx₀_quad_values .- wx₀_quad_values_truth))
        @show maximum(abs.(C.wx₁_quad_values .- wx₁_quad_values_truth))

        @show maximum(abs.(C.λ₁_quad_values .- vt₁_quad_values_truth))
        @show maximum(abs.(0.25 .* C.bc_wx₀_quad_values .+ C.μ₀_quad_values))
        @show maximum(abs.(0.25 .* C.bc_wx₁_quad_values .+ C.μ₁_quad_values))
    end
    # @infiltrate
end

function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{<:NN_PDE_Integrator_Symbolic}) where {ST,}
    local D = int.problem.D
    local RT = int.method.RT
    local RX = int.method.RX
    local NP = int.method.basis.NP
    local quad_b = int.method.grid_weights
    local C = cache(int, ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local brx = int.method.spatial_quadrature.weights
    local brt = int.method.time_quadrature.weights
    local Nbasis_λ_x = int.method.Nbasis_λ_x
    local Nbasis_μ_t = int.method.Nbasis_μ_t
    local mλ_x = int.method.mλ_x
    local mμ_t = int.method.mμ_t
    local show_status = int.method.show_status

    for d in 1:D
        for p in 1:NP
            z = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z += quad_b[rt, rx] *
                         (  x_domain * timestep(int) * C.∂L∂U_quad_values[d, rt, rx] * C.∂u∂P_quad_values[d][rt, rx, p]
                          + x_domain *                 C.∂L∂V_quad_values[d, rt, rx] * C.∂v∂P_quad_values[d][rt, rx, p]
                          + x_domain * timestep(int) * C.∂L∂W_quad_values[d, rt, rx] * C.∂w∂P_quad_values[d][rt, rx, p])
                end
            end
            for rx in 1:RX
                z += x_domain * brx[rx] * (C.λ₀_quad_values[d, rx] * C.∂u∂P_t₀_quad_values[d][rx, p] - C.λ₁_quad_values[d, rx] * C.∂u∂P_t₁_quad_values[d][rx, p])
            end
            for rt in 1:RT
                z += timestep(int) * brt[rt] * (C.μ₀_quad_values[d, rt] * C.∂u∂P_x₀_quad_values[d][rt, p] - C.μ₁_quad_values[d, rt] * C.∂u∂P_x₁_quad_values[d][rt, p])
            end
            b[p] = -z # TODO: check the sign
        end
    end

    for d in 1:D
        for p in 1:Nbasis_λ_x
            z = zero(ST)
            for rx in 1:RX
                z += x_domain * brx[rx] * mλ_x[p, rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
            end
            b[NP+(d-1)*Nbasis_λ_x+p] = -z
        end
    end

    for d in 1:D
        for p in 1:Nbasis_μ_t
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) * brt[rt] * mμ_t[p, rt] * (C.ux₀_quad_values[d, rt] - C.boundary_condition_x₀[d, rt])
            end
            b[NP+D*Nbasis_λ_x+(d-1)*Nbasis_μ_t+p] = -z
        end
    end

    for d in 1:D
        for p in 1:Nbasis_μ_t
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) * brt[rt] * mμ_t[p, rt] * (C.boundary_condition_x₁[d, rt] - C.ux₁_quad_values[d, rt])
            end
            b[NP+D*Nbasis_λ_x+D*Nbasis_μ_t+(d-1)*Nbasis_μ_t+p] = -z
        end
    end
    # @infiltrate
    if show_status
        @show b
    end
end

function update!(sol, int::PDEIntegrator{<:NN_PDE_Integrator_Symbolic})
    local D = int.problem.D
    local u = int.method.basis.u
    local v = int.method.basis.v
    local w = int.method.basis.w
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local C = cache(int)
    local NP = int.method.basis.NP
    local x = nlsolution(int)
    local optim_mode = int.method.basis.optim_mode
    local h = timestep(int)
    local S = int.method.basis.S

    x_nodes = collect(xspan[1]:xstep:xspan[2])

    if optim_mode == :Fully
        @views C.params[:] .= x[1:4*S]
    elseif optim_mode == :Partially
        C.params[3*S+1:4*S] .= x[1:S]
    end

    for i in eachindex(x_nodes)
        sol.u[i] = u(1.0, x_nodes[i], C.params)[1]
        sol.v[i] = v(1.0, x_nodes[i], C.params)[1] / h
        sol.w[i] = w(1.0, x_nodes[i], C.params)[1]
    end

end
