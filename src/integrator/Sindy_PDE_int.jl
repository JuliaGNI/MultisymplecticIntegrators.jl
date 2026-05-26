struct Sindy_PDE_Integrator{MVT,LT,BT<:AbstractPDEBasis} <: PDEMethod
    symbolic_expr_basis::BT
    init_w::Vector

    time_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RT::Int # Number of quadrature points in time

    spatial_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix::Matrix{Vector{Float64}} # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights::Matrix{Float64} # Quadrature weights

    Nbasis_μ_t::Int
    k_μ_t::Int # order
    μ₀_t::MVT
    μ₁_t::MVT

    Nbasis_λ_x::Int
    k_λ_x::Int # order
    λ_x::LT

    mλ_x::Matrix{Float64} # λ_x evaluated at quadrature points
    mμ_t::Matrix{Float64}

    show_status::Bool
    function Sindy_PDE_Integrator(basis,init_w::Vector;RT_per_interval::Int = 4,RX_per_interval::Int = 4,xspan::Tuple = (0.,1.0),
        Nbasis_μ_t::Int = 10,k_μ_t::Int = 4,μ::Symbol = :BSplineDirichlet,
        Nbasis_λ_x::Int = 10,k_λ_x::Int = 4,λ::Symbol = :BSplineDirichlet,
        t_num_interval::Int = 1, x_num_interval::Int = 5,
        show_status = false)

        t_quadrature = composite_quadrature(t_num_interval ,RT_per_interval)
        x_quadrature = composite_quadrature(x_num_interval ,RX_per_interval)

        RT = length(t_quadrature.nodes)
        RX = length(x_quadrature.nodes)

        R_list = [RT_per_interval,RX_per_interval]
        grid_matrix, grid_weights = construct_quadrature_grid(R_list,[t_num_interval, x_num_interval])
        # Construct Lagrangian multipliers, defined on [0,1] and need to be scaled carefully when used
        λ_x = Lagrangian_multiplier(λ,Nbasis_λ_x,k_λ_x,xspan[1],xspan[2])
        μ₀_t = Lagrangian_multiplier(μ,Nbasis_μ_t,k_μ_t,0.0,1.0)
        μ₁_t = Lagrangian_multiplier(μ,Nbasis_μ_t,k_μ_t,0.0,1.0)

        mλ_x = zeros(Nbasis_λ_x, RX)
        mμ_t = zeros(Nbasis_μ_t, RT)

        for i in 1:Nbasis_λ_x
            mλ_x[i,:] = λ_x.b[i].(xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes)
        end

        for i in 1:Nbasis_μ_t
            mμ_t[i,:] = μ₀_t.b[i].(t_quadrature.nodes)
        end

        new{typeof(μ₀_t),typeof(λ_x),typeof(basis)}(basis, init_w,
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            Nbasis_μ_t,k_μ_t, μ₀_t, μ₁_t,
            Nbasis_λ_x, k_λ_x,λ_x,
            mλ_x, mμ_t,show_status)
    end
end

function Base.show(io::IO, method::Sindy_PDE_Integrator)
    print(io, "\n SINDy PDE Integrator with:\n")
    print(io, "   Basis: $(nameof(typeof(method.symbolic_expr_basis))) \n")
    print(io, "   Initial parameter blocks: $(length(method.init_w)) \n")
    print(io, "   Time quadrature points: $(method.RT), space quadrature points: $(method.RX) \n")
    print(io, "   Lagrange multiplier basis functions in time: $(method.Nbasis_μ_t), order: $(method.k_μ_t) \n")
    print(io, "   Lagrange multiplier basis functions in space: $(method.Nbasis_λ_x), order: $(method.k_λ_x) \n")
    print(io, "   Show status: $(method.show_status) \n")
end

default_solver(::Sindy_PDE_Integrator) = NewtonMethod()

struct Sindy_PDE_IntegratorCache{ST,RT,RX,D,NP,Nbasis_μ_t,Nbasis_λ_x} <: PDEIntegratorCache{ST,D}
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

    ics_ut₀_quad_values::Matrix{ST}
    ics_vt₀_quad_values::Matrix{ST}
    ics_wt₀_quad_values::Matrix{ST}

    bc_ux₀_quad_values::Matrix{ST}
    bc_vx₀_quad_values::Matrix{ST}
    bc_wx₀_quad_values::Matrix{ST}

    bc_ux₁_quad_values::Matrix{ST}
    bc_vx₁_quad_values::Matrix{ST}
    bc_wx₁_quad_values::Matrix{ST}

    tem_P::Vector{Vector{ST}} # temporary storage for P values

    init_condition_t₀::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}
    flag_done_initial_guess::Vector{ST}

    function Sindy_PDE_IntegratorCache{ST,RT,RX,D,NP,Nbasis_μ_t,Nbasis_λ_x}(P_sizes) where {ST,RT,RX,D,NP,Nbasis_μ_t,Nbasis_λ_x}
        x = zeros(ST,NP + D * Nbasis_λ_x +2* D * Nbasis_μ_t) # params, λ₀_x_coes,μ₀_t_coes,μ₁_t_coes

        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        ∂u∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,P_sizes)
        ∂v∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,P_sizes)
        ∂w∂P_quad_values = create_interior_quadrature_points_derivative_mat(ST, RT,RX,D,P_sizes)

        λ₀_x_coes = zeros(ST, D, Nbasis_λ_x)
        λ₁_x_coes = zeros(ST, D, Nbasis_λ_x)
        μ₀_t_coes = zeros(ST, D, Nbasis_μ_t)
        μ₁_t_coes = zeros(ST, D, Nbasis_μ_t)

        λ₀_quad_values = zeros(ST, D, RX)
        λ₁_quad_values = zeros(ST, D, RX)
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

        ics_ut₀_quad_values = zeros(ST,D,RX)
        ics_vt₀_quad_values = zeros(ST,D,RX)
        ics_wt₀_quad_values = zeros(ST,D,RX)

        bc_ux₀_quad_values = zeros(ST,D,RT)
        bc_vx₀_quad_values = zeros(ST,D,RT)
        bc_wx₀_quad_values = zeros(ST,D,RT)

        bc_ux₁_quad_values = zeros(ST,D,RT)
        bc_vx₁_quad_values = zeros(ST,D,RT)
        bc_wx₁_quad_values = zeros(ST,D,RT)

        tem_P = create_tem_vector(ST, D, P_sizes)

        init_condition_t₀ = zeros(ST, D, RX)

        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)
        flag_done_initial_guess = zeros(ST, 1)

        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            ∂u∂P_quad_values, ∂v∂P_quad_values, ∂w∂P_quad_values,
            λ₀_x_coes, λ₁_x_coes,μ₀_t_coes,μ₁_t_coes,
            λ₀_quad_values, λ₁_quad_values, μ₀_quad_values, μ₁_quad_values,
            ∂u∂P_t₀_quad_values, ∂u∂P_t₁_quad_values, ∂u∂P_x₀_quad_values, ∂u∂P_x₁_quad_values,
            ut₀_quad_values, ut₁_quad_values,vt₀_quad_values, vt₁_quad_values,wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values,vx₀_quad_values, vx₁_quad_values,wx₀_quad_values, wx₁_quad_values,
            ics_ut₀_quad_values, ics_vt₀_quad_values, ics_wt₀_quad_values,
            bc_ux₀_quad_values, bc_vx₀_quad_values, bc_wx₀_quad_values,
            bc_ux₁_quad_values, bc_vx₁_quad_values, bc_wx₁_quad_values,
            tem_P,
            init_condition_t₀,
            boundary_condition_x₀, boundary_condition_x₁,
            flag_done_initial_guess)
    end
end

nlsolution(cache::Sindy_PDE_IntegratorCache) = cache.x

function Cache{ST}(problem::LPDEProblem, method::Sindy_PDE_Integrator; kwargs...) where {ST}
    Sindy_PDE_IntegratorCache{ST,method.RT,method.RX,problem.D,method.symbolic_expr_basis.NP,method.Nbasis_μ_t,method.Nbasis_λ_x}(method.symbolic_expr_basis.P_sizes; kwargs...)
end

#{ST,RT,RX,D,NP}(P_sizes) where {ST,RT,RX,D,NP}
@inline CacheType(ST, problem::LPDEProblem, method::Sindy_PDE_Integrator) = Sindy_PDE_IntegratorCache{ST,method.RT,method.RX,problem.D,method.symbolic_expr_basis.NP,method.Nbasis_μ_t,method.Nbasis_λ_x}

function prior_initial_guess!(C,sol,int::PDEIntegrator{<:Sindy_PDE_Integrator})
    local P_sizes = int.method.symbolic_expr_basis.P_sizes
    local init_w = int.method.init_w
    local tn = sol.t - timestep(int)

    start_idx = 1
    for (d,P_size) in enumerate(P_sizes)
        if tn == 0.0
            C.x[start_idx:start_idx+P_size-1]= init_w[:]
        else
            C.x[start_idx:start_idx+P_size-1] .= internal(sol).xx[start_idx:start_idx+P_size-1]
            # @infiltrate
        end
        start_idx += P_size
    end
end
copy_internal_variables!(C::Sindy_PDE_IntegratorCache, solstep::SolutionStep) = nothing

function post_initial_guess!(C, sol, int::PDEIntegrator{<:Sindy_PDE_Integrator}, int_method::Sindy_PDE_Integrator{MVT,LT,BT}) where {MVT<:BSplineDirichlet,LT<:BSplineDirichlet,BT}
    local Nbasis_λ_x = int_method.Nbasis_λ_x
    local Nbasis_μ_t = int_method.Nbasis_μ_t
    local D = int.problem.D
    local lag_sys = int.problem.lagrangian_system.functions
    local lag_params = int.problem.lagrangian_system.params
    local NP = int.method.symbolic_expr_basis.NP
    local RX = int.method.RX
    local RT = int.method.RT
    local mλ_x = int_method.mλ_x
    local mμ_t = int_method.mμ_t

    for d in 1:D
        tem_t₁_∂L∂V = zeros(RX)
        tem_x₀_∂L∂W = zeros(RT)
        tem_x₁_∂L∂W = zeros(RT)

        for rx in 1:RX
            tem_t₁_∂L∂V[rx] = lag_sys.∂L∂V[d](C.ut₁_quad_values[d,rx], C.vt₁_quad_values[d,rx], C.wt₁_quad_values[d,rx], lag_params)
        end
        λ_x_tem = mλ_x'\tem_t₁_∂L∂V

        for rx in 1:Nbasis_λ_x
            C.x[NP + (d - 1) * Nbasis_λ_x + rx] = λ_x_tem[rx]
        end

        for rt in 1:RT
            tem_x₀_∂L∂W[rt] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,rt], C.vx₀_quad_values[d,rt], C.wx₀_quad_values[d,rt], lag_params)
            tem_x₁_∂L∂W[rt] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,rt], C.vx₁_quad_values[d,rt], C.wx₁_quad_values[d,rt], lag_params)
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

function post_initial_guess!(C,sol,int::PDEIntegrator{<:Sindy_PDE_Integrator},int_method::Sindy_PDE_Integrator{MVT,LT,BT}) where {MVT<:Lagrange,LT<:Lagrange,BT}
    local NP = int_method.symbolic_expr_basis.NP
    local RT = int_method.RT
    local RX = int_method.RX
    local D = int.problem.D
    local lag_sys = int.problem.lagrangian_system.functions
    local lag_params = int.problem.lagrangian_system.params
    local Nbasis_λ_x = int_method.Nbasis_λ_x
    local Nbasis_μ_t = int_method.Nbasis_μ_t
    local λ_x = int_method.λ_x
    local μ₀_t = int_method.μ₀_t
    local u = int.method.symbolic_expr_basis.u # f = f(Parameters,t,x)
    local v = int.method.symbolic_expr_basis.v
    local w = int.method.symbolic_expr_basis.w
    local xspan = int.problem.xspan
    local show_status = int_method.show_status

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
            ut₀_quad_values_tem[rx] = (u[d])(C.tem_P, sol.t - timestep(int), xx)
            ut₁_quad_values_tem[rx] = (u[d])(C.tem_P, sol.t, xx)
            vt₀_quad_values_tem[rx] = (v[d])(C.tem_P, sol.t - timestep(int), xx)
            vt₁_quad_values_tem[rx] = (v[d])(C.tem_P, sol.t, xx)
            wt₀_quad_values_tem[rx] = (w[d])(C.tem_P, sol.t - timestep(int), xx)
            wt₁_quad_values_tem[rx] = (w[d])(C.tem_P, sol.t, xx)
        end

        for rt in 1:Nbasis_μ_t
            tt = timestep(int) * μ₀_t.x[rt]
            ux₀_quad_values_tem[rt] = (u[d])(C.tem_P, sol.t - timestep(int) + tt, xspan[1])
            ux₁_quad_values_tem[rt] = (u[d])(C.tem_P, sol.t - timestep(int) + tt, xspan[2])
            vx₀_quad_values_tem[rt] = (v[d])(C.tem_P, sol.t - timestep(int) + tt, xspan[1])
            vx₁_quad_values_tem[rt] = (v[d])(C.tem_P, sol.t - timestep(int) + tt, xspan[2])
            wx₀_quad_values_tem[rt] = (w[d])(C.tem_P, sol.t - timestep(int) + tt, xspan[1])
            wx₁_quad_values_tem[rt] = (w[d])(C.tem_P, sol.t - timestep(int) + tt, xspan[2])
        end
    end

    for d in 1:D
        for rx in 1:Nbasis_λ_x
            C.x[NP+(d-1)*Nbasis_λ_x+rx] = lag_sys.∂L∂V[d](ut₁_quad_values_tem[rx], vt₁_quad_values_tem[rx], wt₁_quad_values_tem[rx], lag_params)
        end

        for rt in 1:Nbasis_μ_t
            C.x[NP+D*Nbasis_λ_x+(d-1)*Nbasis_μ_t+rt] = lag_sys.∂L∂W[d](ux₀_quad_values_tem[rt], vx₀_quad_values_tem[rt], wx₀_quad_values_tem[rt], lag_params)
            C.x[NP+D*Nbasis_λ_x+D*Nbasis_μ_t+(d-1)*Nbasis_μ_t+rt] = lag_sys.∂L∂W[d](ux₁_quad_values_tem[rt], vx₁_quad_values_tem[rt], wx₁_quad_values_tem[rt], lag_params)
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
            ux₀_truth_quad[rt] = exact_u.(sol.t - timestep(int) + timestep(int)* tt, xspan[1])
            ux₁_truth_quad[rt] = exact_u.(sol.t - timestep(int) + timestep(int)* tt, xspan[2])
            vx₀_truth_quad[rt] = exact_v.(sol.t - timestep(int) + timestep(int)* tt, xspan[1])
            vx₁_truth_quad[rt] = exact_v.(sol.t - timestep(int) + timestep(int)* tt, xspan[2])
            wx₀_truth_quad[rt] = exact_w.(sol.t - timestep(int) + timestep(int)* tt, xspan[1])
            wx₁_truth_quad[rt] = exact_w.(sol.t - timestep(int) + timestep(int)* tt, xspan[2])
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
end

function components!(x::AbstractVector{ST}, sol, params, int::PDEIntegrator{<:Sindy_PDE_Integrator}) where {ST}
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

    local lag_params = int.problem.lagrangian_system.params
    local mλ_x = int.method.mλ_x
    local mμ_t = int.method.mμ_t

    local Nbasis_λ_x = int.method.Nbasis_λ_x
    local Nbasis_μ_t = int.method.Nbasis_μ_t

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
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], lag_params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], lag_params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], lag_params)
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

    cache(int).flag_done_initial_guess[1] == 0.0 ? post_initial_guess!(cache(int), sol, int, int.method) : nothing

    for d in 1:D
        C.λ₁_x_coes[d, :] = x[NP+1:NP+Nbasis_λ_x]
        C.μ₀_t_coes[d, :] = x[NP+Nbasis_λ_x+1:NP+Nbasis_λ_x+Nbasis_μ_t]
        C.μ₁_t_coes[d, :] = x[NP+Nbasis_λ_x+Nbasis_μ_t+1:NP+Nbasis_λ_x+2*Nbasis_μ_t]
    end

    for d in 1:D
        for rx in 1:RX
            C.λ₀_quad_values[d, rx] = ∂L∂V[d](C.ics_ut₀_quad_values[d, rx], C.ics_vt₀_quad_values[d, rx], C.ics_wt₀_quad_values[d, rx], lag_params)
            C.λ₁_quad_values[d, rx] = sum(C.λ₁_x_coes[d, :] .* mλ_x[:, rx])
        end

        for rt in 1:RT
            C.μ₀_quad_values[d, rt] = sum(C.μ₀_t_coes[d, :] .* mμ_t[:, rt])
            C.μ₁_quad_values[d, rt] = sum(C.μ₁_t_coes[d, :] .* mμ_t[:, rt])
        end
    end

end

function residual!(b::Vector{ST}, sol, params,int::PDEIntegrator{<: Sindy_PDE_Integrator}) where {ST,}
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
    local Nbasis_λ_x = int.method.Nbasis_λ_x
    local Nbasis_μ_t = int.method.Nbasis_μ_t
    local mλ_x = int.method.mλ_x
    local mμ_t = int.method.mμ_t
    local show_status = int.method.show_status

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
                z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * C.∂u∂P_t₀_quad_values[d][rx,p] - C.λ₁_quad_values[d,rx] * C.∂u∂P_t₁_quad_values[d][rx,p])
            end
            for rt in 1:RT
                z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * C.∂u∂P_x₀_quad_values[d][rt,p] - C.μ₁_quad_values[d,rt] * C.∂u∂P_x₁_quad_values[d][rt,p])
            end
            b[current_idx] = z
            current_idx += 1
        end
    end

    @assert current_idx == NP + 1 "Wrong indexing in residual computation"

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

function update!(sol, int::PDEIntegrator{<:Sindy_PDE_Integrator})
    local D = int.problem.D
    local u = int.method.symbolic_expr_basis.u # f = f(Parameters,t,x)
    local v = int.method.symbolic_expr_basis.v
    local w = int.method.symbolic_expr_basis.w
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local C = cache(int)
    local NP = int.method.symbolic_expr_basis.NP

    x_nodes = collect(xspan[1]:xstep:xspan[2])
    tem_P = [nlsolution(int)[1:NP]]

    # println("In update! function, step = ", sol_struct.current_step)
    # println("In update! function, time = ", sol_struct.t)

    for d in 1:D
        for i in eachindex(x_nodes)
            sol.u[i] = (u[d])(tem_P, sol.t, x_nodes[i])
            sol.v[i] = (v[d])(tem_P, sol.t, x_nodes[i])
            sol.w[i] = (w[d])(tem_P, sol.t, x_nodes[i])
        end
    end

end

function internal_variables(method::Sindy_PDE_Integrator, problem::LPDEProblem)
    local D = problem.D
    local RX = method.RX
    local NP = method.symbolic_expr_basis.NP
    local Nbasis_λ_x = method.Nbasis_λ_x
    local Nbasis_μ_t = method.Nbasis_μ_t

    ut₁_quad_values = zeros(D,RX)
    vt₁_quad_values = zeros(D,RX)
    wt₁_quad_values = zeros(D,RX)
    xx = zeros(NP + D * Nbasis_λ_x +2* D * Nbasis_μ_t)

    return (ut₁_quad_values = ut₁_quad_values,
        vt₁_quad_values = vt₁_quad_values,
        wt₁_quad_values = wt₁_quad_values,
        xx = xx
        )
end

function copy_internal_variables!(solstep::SolutionStep,C::Sindy_PDE_IntegratorCache)
    # copy internal variables from cache to internal,
    haskey(internal(solstep), :ut₁_quad_values) && copyto!(internal(solstep).ut₁_quad_values,C.ut₁_quad_values)
    haskey(internal(solstep), :vt₁_quad_values) && copyto!(internal(solstep).vt₁_quad_values,C.vt₁_quad_values)
    haskey(internal(solstep), :wt₁_quad_values) && copyto!(internal(solstep).wt₁_quad_values,C.wt₁_quad_values)
    haskey(internal(solstep), :xx) && copyto!(internal(solstep).xx,C.x)
end
