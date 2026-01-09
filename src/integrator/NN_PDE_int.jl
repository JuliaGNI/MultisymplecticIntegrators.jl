struct NN_PDE_Integrator{MVT,LT,BT<:AbstractPDEBasis,IPMT<:InitialParametersMethod} <: PDEMethod
    basis::BT

    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights

    Nbasis_μ_t
    k_μ_t::Int # order
    μ₀_t::MVT
    μ₁_t::MVT

    Nbasis_λ_x
    k_λ_x::Int # order 
    λ_x::LT

    mλ_x # λ_x evaluated at quadrature points
    mμ_t

    nepochs::Int
    initial_guess_method::IPMT # :LSGD or :GroundTruth

    Nw                 # angular directions
    Nb

    show_status::Bool
    function NN_PDE_Integrator(basis; RT::Int=6, RX::Int=8, xspan::Tuple=(0., 1.0),
        nepochs=1000, initial_guess_method::IPMT=OGA2D(), Nw::Int=500, Nb::Int=500,
        Nbasis_μ_t::Int=10, k_μ_t::Int=4, μ::Symbol=:BSplineDirichlet,
        Nbasis_λ_x::Int=10, k_λ_x::Int=3, λ::Symbol=:BSplineDirichlet,
        show_status::Bool=false) where {IPMT,}

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

        # Construct Lagrangian multipliers, defined on [0,1] and need to be scaled carefully when used
        λ_x = Lagrangian_multiplier(λ, Nbasis_λ_x, k_λ_x, xspan[1], xspan[2])
        μ₀_t = Lagrangian_multiplier(μ, Nbasis_μ_t, k_μ_t, 0.0, 1.0)
        μ₁_t = Lagrangian_multiplier(μ, Nbasis_μ_t, k_μ_t, 0.0, 1.0)

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
            Nbasis_μ_t, k_μ_t, μ₀_t, μ₁_t,
            Nbasis_λ_x, k_λ_x, λ_x,
            mλ_x, mμ_t,
            nepochs, initial_guess_method,
            Nw, Nb,show_status)
    end
end

default_solver(::NN_PDE_Integrator) = NewtonMethod()

struct NN_PDE_IntegratorCache{ST,RT,RX,D,NP,S,Nbasis_μ_t,Nbasis_λ_x} <: PDEIntegratorCache{ST,D}
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


    init_condition_t₀::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}

    sol_params
    flag_done_initial_guess::Vector{ST}
    function NN_PDE_IntegratorCache{ST,RT,RX,D,NP,S,Nbasis_μ_t,Nbasis_λ_x}() where {ST,RT,RX,D,NP,S,Nbasis_μ_t,Nbasis_λ_x}
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

        sol_params = (L1=(W=zeros(ST, S, 2), b=zeros(ST, S)), L2=(W=zeros(ST, 1, S),))
        flag_done_initial_guess = zeros(ST, 1)

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
            sol_params, flag_done_initial_guess)
    end
end

nlsolution(cache::NN_PDE_IntegratorCache) = cache.x

function Cache{ST}(problem::LPDEProblem, int::NN_PDE_Integrator; kwargs...) where {ST}
    NN_PDE_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.NP,int.basis.S,int.Nbasis_μ_t,int.Nbasis_λ_x}(; kwargs...)
end

@inline CacheType(ST, problem::LPDEProblem, int::NN_PDE_Integrator) = NN_PDE_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.NP,int.basis.S,int.Nbasis_μ_t,int.Nbasis_λ_x}
@inline function Base.getindex(c::NN_PDE_IntegratorCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function prior_initial_guess!(C, sol, int::PDEIntegrator{<:NN_PDE_Integrator{MVT,LT,BT,IPMT}}) where {MVT,LT,BT,IPMT<:OGA2D}
    local h = timestep(int)
    local Nw = int.method.Nw
    local Nb = int.method.Nb
    local activation = int.method.basis.activation_function
    local S = int.method.basis.S
    local a, b = int.problem.xspan[1], int.problem.xspan[2]
    local exact_u = int.problem.exact_u
    local u = int.method.basis.u
    local optim_mode = int.method.basis.optim_mode
    local NP = int.method.basis.NP
    local show_status = int.method.show_status
    # Equidistant Quadrature / sampling grid
    nx = 40
    nt = 20
    xs = range(a, b, length=nx)
    ts = range(0.0, 1.0, length=nt)

    # build list of sample coords as 2×N matrix (t; x)
    coords = [(t, x) for t in ts, x in xs]   # nt × nx array of tuples
    N = length(coords)
    quad_nodes = zeros(2, N)
    for i in 1:N
        quad_nodes[1, i] = coords[i][1]
        quad_nodes[2, i] = coords[i][2]
    end

    # simple uniform quadrature weights (you can switch to Simpson)
    quad_weights = fill(1.0 / N, N)
    thetas = range(-π, π, length=Nw + 1)
    dirs = [[cos(θ), sin(θ)] for θ in thetas]  # length Nw+1

    biases = range(-π, π, length=Nb + 1)       # larger bias range works well for sinusoids

    # make dictionary rows (M × 3)
    Arows = Float64[]
    for w in dirs, b in biases
        append!(Arows, [w[1], w[2], b])
    end
    A_mat = reshape(Arows, 3, :)'   # M × 3
    M = size(A_mat, 1)

    # build augmented coordinates (for bias): 3 × N
    Xaug = vcat(quad_nodes, ones(1, N))

    # precompute dictionary activations (M×N)
    Φ_raw = activation.(A_mat * Xaug)   # M × N
    # This performs up to `max_iter` outer iterations to account for boundary terms depending on PNN
    selected = Int[]
    B = Matrix{Float64}(undef, N, 0)   # orthonormal basis columns
    coeffs_full = zeros(S)             # coefficients to write into PNN L2
    Wsel = zeros(S, 2)
    Bsel = zeros(S)

    # Build the desired internal PNN output on all quadrature nodes:
    # desired = target + T1NN - T2NN - C1 + C2  (evaluated with current PNN.params)
    desired = zeros(N)
    for i in 1:N
        t = quad_nodes[1, i]
        x = quad_nodes[2, i]
        desired[i] = exact_u(h * t, x) - u([t, x], C.sol_params)[1]
    end

    # Run OGA (orthogonal matching) on Φ_raw to approximate `desired`
    residual = copy(desired)

    for s = 1:S
        # compute correlations with residual (weighted)
        corrs = zeros(M)
        for i in 1:M
            corrs[i] = abs(sum(Φ_raw[i, :] .* (residual .* quad_weights)))
        end
        idx = argmax(corrs)
        push!(selected, idx)
        length(Set(selected)) == s ? nothing : @warn "atom repeated at s=$s, idx=$idx"

        # extract raw atom (already normalized) and orthogonalize (Gram-Schmidt)
        φ = copy(Φ_raw[idx, :])

        # append to B
        B = hcat(B, φ)

        # solve least-squares for coefficients in orthonormal basis
        coeffs = B \ desired         # small system k×1 solved implicitly
        # update residual
        residual = desired - B * coeffs

        # store selection params (note A_mat rows correspond to atoms prior to normalization,
        # yet we normalized Φ_raw; we must store original (w,b) for a neuron consistent with A_mat)
        Wsel[s, :] .= A_mat[idx, 1:2]
        Bsel[s] = A_mat[idx, 3]

        coeffs_full[1:s] .= coeffs
        if show_status
            println("s=$s idx=$idx ‖residual‖=$(norm(residual))")
        end
    end

    for j = 1:S
        u.params.L1.W[j, :] .= Wsel[j, :]
        u.params.L1.b[j] = Bsel[j]
        u.params.L2.W[j] = coeffs_full[j]

        C.sol_params.L1.W[j, :] .= Wsel[j, :]
        C.sol_params.L1.b[j] = Bsel[j]
        C.sol_params.L2.W[j] = coeffs_full[j]
    end


    # copy the parameters to the cache
    if optim_mode == :Partially
        C.x[1:NP] = C.sol_params.L2.W[:]
    elseif optim_mode == :Fully
        C.x[1:2*S] = reshape(C.sol_params.L1.W, :, 1)
        C.x[2*S+1:3*S] = C.sol_params.L1.b[:]
        C.x[3*S+1:4*S] = C.sol_params.L2.W[:]
        @assert 4 * S == NP
    end

    if show_status
        target_vec = [exact_u(h * quad_nodes[1, i], quad_nodes[2, i]) for i in 1:N]
        approx_vec = [u(quad_nodes[:, i], C.sol_params)[1] for i in 1:N]
        err_vec = abs.(target_vec .- approx_vec)
        println("Max abs error after OGA initial guess: ", maximum(err_vec))
        println("OGA initial guess completed.")
        println("Initial guess \n", C.x)
    end
    

end

copy_internal_variables!(C::NN_PDE_IntegratorCache, solstep::SolutionStep) = nothing

function post_initial_guess!(C, sol, int::PDEIntegrator{<:NN_PDE_Integrator}, int_method::NN_PDE_Integrator{MVT,LT,BT,IPMT}) where {MVT<:BSplineDirichlet,LT<:BSplineDirichlet,BT,IPMT}
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
            C.x[NP + D * Nbasis_λ_x + + D * Nbasis_μ_t + (d - 1) * Nbasis_μ_t + rt] = μ₁_t_tem[rt]
        end
    end
    C.flag_done_initial_guess[1] = 1.0

    # @infiltrate
end

function post_initial_guess!(C, sol, int::PDEIntegrator{<:NN_PDE_Integrator}, int_method::NN_PDE_Integrator{MVT,LT,BT,IPMT}) where {MVT<:Lagrange,LT<:Lagrange,BT,IPMT}
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
            ut₀_quad_values_tem[rx] = (u[d])([0.0, xx], C.sol_params)[1]
            ut₁_quad_values_tem[rx] = (u[d])([1.0, xx], C.sol_params)[1]
            vt₀_quad_values_tem[rx] = (v[d])([0.0, xx], C.sol_params)[1] / h
            vt₁_quad_values_tem[rx] = (v[d])([1.0, xx], C.sol_params)[1] / h
            wt₀_quad_values_tem[rx] = (w[d])([0.0, xx], C.sol_params)[1]
            wt₁_quad_values_tem[rx] = (w[d])([1.0, xx], C.sol_params)[1]
        end

        for rt in 1:Nbasis_μ_t
            tt = μ₀_t.x[rt]
            ux₀_quad_values_tem[rt] = (u[d])([tt, xspan[1]], C.sol_params)[1]
            ux₁_quad_values_tem[rt] = (u[d])([tt, xspan[2]], C.sol_params)[1]
            vx₀_quad_values_tem[rt] = (v[d])([tt, xspan[1]], C.sol_params)[1] / h
            vx₁_quad_values_tem[rt] = (v[d])([tt, xspan[2]], C.sol_params)[1] / h
            wx₀_quad_values_tem[rt] = (w[d])([tt, xspan[1]], C.sol_params)[1]
            wx₁_quad_values_tem[rt] = (w[d])([tt, xspan[2]], C.sol_params)[1]
        end
    end
    for d in 1:D
        for rx in 1:Nbasis_λ_x
            C.x[NP+(d-1)*Nbasis_λ_x+rx] = lag_sys.∂L∂V[d](ut₁_quad_values_tem[rx], vt₁_quad_values_tem[rx], wt₁_quad_values_tem[rx], params)
        end

        for rt in 1:Nbasis_μ_t
            C.x[NP+D*Nbasis_λ_x+(d-1)*Nbasis_μ_t+rt] = lag_sys.∂L∂W[d](ux₀_quad_values_tem[rt], vx₀_quad_values_tem[rt], wx₀_quad_values_tem[rt], params)
            C.x[NP+D*Nbasis_λ_x++D*Nbasis_μ_t+(d-1)*Nbasis_μ_t+rt] = lag_sys.∂L∂W[d](ux₁_quad_values_tem[rt], vx₁_quad_values_tem[rt], wx₁_quad_values_tem[rt], params)
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



function components!(x::AbstractVector{ST}, sol, params, int::PDEIntegrator{<:NN_PDE_Integrator}) where {ST}
    local C = cache(int, ST)

    local NP = int.method.basis.NP
    local S = int.method.basis.S
    local RT = int.method.RT
    local RX = int.method.RX
    local D = int.problem.D

    local ∂L∂U = int.problem.lagrangian_system.functions.∂L∂U
    local ∂L∂V = int.problem.lagrangian_system.functions.∂L∂V
    local ∂L∂W = int.problem.lagrangian_system.functions.∂L∂W

    local u = [int.method.basis.u]
    local v = [int.method.basis.v]
    local w = [int.method.basis.w]

    local ∂u∂P = int.method.basis.∂u∂P
    local ∂v∂P = int.method.basis.∂v∂P
    local ∂w∂P = int.method.basis.∂w∂P

    local grid_matrix = int.method.grid_matrix
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]

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

    last_layer = keys(u[1].params)[end]

    #copy part of x into the network parameter
    if optim_mode == :Fully
        C.sol_params.L1.W[:,1] = x[1:S]
        C.sol_params.L1.W[:,2] = x[S+1:2*S]
        C.sol_params.L1.b[:] = x[2*S+1:3*S]
        C.sol_params.L2.W[:] = x[3*S+1:4*S]
    elseif optim_mode == :Partially
        for (name, layer) in zip(keys(u[1].params), values(u[1].params))
            if hasfield(typeof(layer), :b)
                C.sol_params[name].W[:] = layer.W[:]
                C.sol_params[name].b[:] = layer.b[:]
            else
                # For layers without bias (e.g., output), just regenerate W
                C.sol_params[name].W[:] = layer.W[:]
            end
        end
        C.sol_params[last_layer].W[:] = x[1:NP]
    end

    # interior values at quadrature points
    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = (u[d])([grid_matrix[i, j][1], xspan[1] + x_domain * grid_matrix[i, j][2]], C.sol_params)[1]
                C.v_quad_values[d, i, j] = (v[d])([grid_matrix[i, j][1], xspan[1] + x_domain * grid_matrix[i, j][2]], C.sol_params)[1] / h
                C.w_quad_values[d, i, j] = (w[d])([grid_matrix[i, j][1], xspan[1] + x_domain * grid_matrix[i, j][2]], C.sol_params)[1]
            end
        end
    end

    if optim_mode == :Fully
        for d in 1:D
            for i in 1:RT
                for j in 1:RX#TODO what if RX is a Vector
                    C.∂u∂P_quad_values[d][i, j, :] = flatten_params(∂u∂P([grid_matrix[i, j][1], xspan[1] + x_domain * grid_matrix[i, j][2]], NeuralNetworkParameters(C.sol_params)))
                    C.∂v∂P_quad_values[d][i, j, :] = flatten_params(∂v∂P([grid_matrix[i, j][1], xspan[1] + x_domain * grid_matrix[i, j][2]], NeuralNetworkParameters(C.sol_params)))
                    C.∂w∂P_quad_values[d][i, j, :] = flatten_params(∂w∂P([grid_matrix[i, j][1], xspan[1] + x_domain * grid_matrix[i, j][2]], NeuralNetworkParameters(C.sol_params)))
                end
            end

            for rx in 1:RX
                C.∂u∂P_t₀_quad_values[d][rx, :] = flatten_params(∂u∂P([0.0, xspan[1] + x_domain * x_quad_nodes[rx]], NeuralNetworkParameters(C.sol_params)))
                C.∂u∂P_t₁_quad_values[d][rx, :] = flatten_params(∂u∂P([1.0, xspan[1] + x_domain * x_quad_nodes[rx]], NeuralNetworkParameters(C.sol_params)))
            end
            for rt in 1:RT
                C.∂u∂P_x₀_quad_values[d][rt, :] = flatten_params(∂u∂P([t_quad_nodes[rt], xspan[1]], NeuralNetworkParameters(C.sol_params)))
                C.∂u∂P_x₁_quad_values[d][rt, :] = flatten_params(∂u∂P([t_quad_nodes[rt], xspan[2]], NeuralNetworkParameters(C.sol_params)))
            end
        end
    elseif optim_mode == :Partially
        for d in 1:D
            for i in 1:RT
                for j in 1:RX#TODO what if RX is a Vector
                    C.∂u∂P_quad_values[d][i, j, :] = ∂u∂P([grid_matrix[i, j][1], xspan[1] + x_domain * grid_matrix[i, j][2]], NeuralNetworkParameters(C.sol_params))[last_layer].W[:]
                    C.∂v∂P_quad_values[d][i, j, :] = ∂v∂P([grid_matrix[i, j][1], xspan[1] + x_domain * grid_matrix[i, j][2]], NeuralNetworkParameters(C.sol_params))[last_layer].W[:]
                    C.∂w∂P_quad_values[d][i, j, :] = ∂w∂P([grid_matrix[i, j][1], xspan[1] + x_domain * grid_matrix[i, j][2]], NeuralNetworkParameters(C.sol_params))[last_layer].W[:]
                end
            end

            for rx in 1:RX
                C.∂u∂P_t₀_quad_values[d][rx, :] = ∂u∂P([0.0, xspan[1] + x_domain * x_quad_nodes[rx]], NeuralNetworkParameters(C.sol_params))[last_layer].W[:]
                C.∂u∂P_t₁_quad_values[d][rx, :] = ∂u∂P([1.0, xspan[1] + x_domain * x_quad_nodes[rx]], NeuralNetworkParameters(C.sol_params))[last_layer].W[:]
            end
            for rt in 1:RT
                C.∂u∂P_x₀_quad_values[d][rt, :] = ∂u∂P([t_quad_nodes[rt], xspan[1]], NeuralNetworkParameters(C.sol_params))[last_layer].W[:]
                C.∂u∂P_x₁_quad_values[d][rt, :] = ∂u∂P([t_quad_nodes[rt], xspan[2]], NeuralNetworkParameters(C.sol_params))[last_layer].W[:]
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

    for d in 1:D
        for j in 1:RX
            C.ut₀_quad_values[d, j] = u[d]([0.0, xspan[1] + x_domain * x_quad_nodes[j]], C.sol_params)[1] # bottom 
            C.ut₁_quad_values[d, j] = u[d]([1.0, xspan[1] + x_domain * x_quad_nodes[j]], C.sol_params)[1] # top
            C.vt₀_quad_values[d, j] = v[d]([0.0, xspan[1] + x_domain * x_quad_nodes[j]], C.sol_params)[1] / h
            C.vt₁_quad_values[d, j] = v[d]([1.0, xspan[1] + x_domain * x_quad_nodes[j]], C.sol_params)[1] / h
            C.wt₀_quad_values[d, j] = w[d]([0.0, xspan[1] + x_domain * x_quad_nodes[j]], C.sol_params)[1]
            C.wt₁_quad_values[d, j] = w[d]([1.0, xspan[1] + x_domain * x_quad_nodes[j]], C.sol_params)[1]
        end

        for i in 1:RT
            C.ux₀_quad_values[d, i] = u[d]([t_quad_nodes[i], xspan[1]], C.sol_params)[1]
            C.ux₁_quad_values[d, i] = u[d]([t_quad_nodes[i], xspan[2]], C.sol_params)[1]
            C.vx₀_quad_values[d, i] = v[d]([t_quad_nodes[i], xspan[1]], C.sol_params)[1] / h
            C.vx₁_quad_values[d, i] = v[d]([t_quad_nodes[i], xspan[2]], C.sol_params)[1] / h
            C.wx₀_quad_values[d, i] = w[d]([t_quad_nodes[i], xspan[1]], C.sol_params)[1]
            C.wx₁_quad_values[d, i] = w[d]([t_quad_nodes[i], xspan[2]], C.sol_params)[1]
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

    if show_status
        u_truth_mat = similar(C.u_quad_values)
        v_truth_mat = similar(C.v_quad_values)
        w_truth_mat = similar(C.w_quad_values)

        for d in 1:D
            for i in 1:RT
                for j in 1:RX
                    u_truth_mat[d, i, j] = int.problem.exact_u.(h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                    v_truth_mat[d, i, j] = int.problem.exact_v.(h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                    w_truth_mat[d, i, j] = int.problem.exact_w.(h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
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

function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{<:NN_PDE_Integrator}) where {ST,}
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

function update!(sol, int::PDEIntegrator{<:NN_PDE_Integrator})
    local D = int.problem.D
    local u = [int.method.basis.u]
    local v = [int.method.basis.v]
    local w = [int.method.basis.w]
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local C = cache(int)
    local NP = int.method.basis.NP
    local x = nlsolution(int)
    local optim_mode = int.method.basis.optim_mode
    last_layer = keys(u[1].params)[end]

    x_nodes = collect(xspan[1]:xstep:xspan[2])
    if optim_mode == :Fully
        tem_params = NeuralNetworkParameters(reconstruct_params(x[1:NP], u[1].params))
        for (name, layer) in zip(keys(tem_params), values(tem_params))
            if hasfield(typeof(layer), :b)
                C.sol_params[name].W[:] = layer.W[:]
                C.sol_params[name].b[:] = layer.b[:]
            else
                # For layers without bias (e.g., output), just regenerate W
                C.sol_params[name].W[:] = layer.W[:]
            end
        end
    elseif optim_mode == :Partially
        for (name, layer) in zip(keys(u[1].params), values(u[1].params))
            if hasfield(typeof(layer), :b)
                C.sol_params[name].W[:] = layer.W[:]
                C.sol_params[name].b[:] = layer.b[:]
            else
                # For layers without bias (e.g., output), just regenerate W
                C.sol_params[name].W[:] = layer.W[:]
            end
        end
        C.sol_params[last_layer].W[:] = x[1:NP]
    end

    for d in 1:D
        for i in eachindex(x_nodes)
            sol.u[i] = u[d]([1.0, x_nodes[i]], C.sol_params)[1]
            sol.v[i] = v[d]([1.0, x_nodes[i]], C.sol_params)[1]
            sol.w[i] = w[d]([1.0, x_nodes[i]], C.sol_params)[1]
        end
    end
end

