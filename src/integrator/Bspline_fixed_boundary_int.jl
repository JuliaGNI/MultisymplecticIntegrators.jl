struct Constraint_Bspline_Integrator{T,BT<:AbstractPDEBasis} <: PDEMethod
    basis::BT

    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions
    
    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights
    
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

    NP::Int # number of internal basis functions
    function Constraint_Bspline_Integrator(basis; RT::Int = 16,RX::Int = 32,xspan::Tuple = (0.,1.0),tstep::T = 1.0,) where {T}

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
                xx = xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes[rx]
                tt = t_quadrature.nodes[rt]
                # xx = x_quadrature.nodes[rx]
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
            xx = xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes[rx]
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

        Nbasis_x = basis.Nbasis_x
        Nbasis_t = basis.Nbasis_t
        NP = (Nbasis_x-2) * (Nbasis_t-1) # number of internal basis functions

        new{T,typeof(basis)}(basis, 
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            u_collocation_matrix, v_collocation_matrix, w_collocation_matrix,
            ut₀_basis_quad_values,ut₁_basis_quad_values,
            vt₀_basis_quad_values,vt₁_basis_quad_values,
            wt₀_basis_quad_values,wt₁_basis_quad_values,
            ux₀_basis_quad_values,ux₁_basis_quad_values,
            vx₀_basis_quad_values,vx₁_basis_quad_values,
            wx₀_basis_quad_values,wx₁_basis_quad_values,
            NP
            )
    end 
end

default_solver(::Constraint_Bspline_Integrator) = NewtonMethod()

struct Galerkin_Bspline_IntegratorCache{ST,RT,RX,D,NP} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    DX = dimension of input of u,v,w, i.e. number of spatial dimensions
    """
    x::Vector{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

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
    function Galerkin_Bspline_IntegratorCache{ST,RT,RX,D,NP}() where {ST,RT,RX,D,NP}
        x = zeros(ST,NP)
        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

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
            ∂u∂P_t₀_quad_values, ∂u∂P_t₁_quad_values, ∂u∂P_x₀_quad_values, ∂u∂P_x₁_quad_values,
            ut₀_quad_values, ut₁_quad_values,vt₀_quad_values, vt₁_quad_values,wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values,vx₀_quad_values, vx₁_quad_values,wx₀_quad_values, wx₁_quad_values,
            init_condition_t₀, 
            boundary_condition_x₀, boundary_condition_x₁,
            )
    end
end

nlsolution(cache::Galerkin_Bspline_IntegratorCache) = cache.x

function Cache{ST}(problem::LPDEProblem, int::Constraint_Bspline_Integrator; kwargs...) where {ST}
    Galerkin_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.NP}(; kwargs...)
end

#{ST,RT,RX,D,NP}(NP) where {ST,RT,RX,D,NP}
@inline CacheType(ST, problem::LPDEProblem, int::Constraint_Bspline_Integrator) = Galerkin_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.NP}
@inline function Base.getindex(c::Galerkin_Bspline_IntegratorCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function initialize_bcs_ics!(sol,int::PDEIntegrator{<:Constraint_Bspline_Integrator})
    local C = cache(int)
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local D = int.problem.D
    local RT = int.method.RT
    local ic_fun = int.problem.ics_function
    local tn = sol.t - timestep(int)
    local bc_fun = int.problem.bcs_function
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    for d in 1:D
        # println("update initial condition, current time = ", sol.t, "the initial condition is at time = ", sol.t - timestep(int))

        if tn == 0.0
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

copy_internal_variables!(C::Galerkin_Bspline_IntegratorCache,solstep::SolutionStep) = nothing
copy_internal_variables!(solstep::SolutionStep,C::Galerkin_Bspline_IntegratorCache) = nothing


function prior_initial_guess!(C,sol,int::PDEIntegrator{<:Constraint_Bspline_Integrator})
    local exact_u = int.problem.exact_u
    local xs = int.method.basis.xs
    local ts = int.method.basis.ts # t \in [0, 1.0]

    # local ts = int.method.time_quadrature.nodes
    # local xs = int.method.spatial_quadrature.nodes
    # xs = a .+ (b - a) .* xs # scale to physical domain
    local Nbasis_x = int.method.basis.Nbasis_x
    local Nbasis_t = int.method.basis.Nbasis_t
    local Cx = int.method.basis.Collocation_x
    local Ct = int.method.basis.Collocation_t
    local Bx = int.method.basis.Basis_x
    local Bt = int.method.basis.Basis_t
    local S = int.method.basis.S
    local h = timestep(int) 
    local a = int.problem.xspan[1]
    local b = int.problem.xspan[2]
    local grid_matrix = int.method.grid_matrix
    local RT = int.method.RT
    local RX = int.method.RX

    fdata = exact_u.(h .* ts, xs') # fdata[i,j] = exact_u(h .* ts[i], xs[j])

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

    bspline_approx = similar(fdata)
    # Verification: evaluate 2D spline at data points
    for m ∈ eachindex(ts), n ∈ eachindex(xs)
        t = ts[m]
        x = xs[n]
        bspline_approx[m, n] = eval_spline2D(coefs, (Bt, Bx), (t, x))
    end

    # println("Error Matrix between exact solution and B-spline approximation at grid points:")
    # @show bspline_approx .- fdata
    println("Max error in initial guess: ", maximum(abs.(fdata .- bspline_approx)))

    bspline_quad_approx = zeros(RT, RX)
    f_quad_value = zeros(RT, RX)
    for ti in 1:RT
        for xi in 1:RX
            tt = grid_matrix[ti,xi][1]
            xx = a + (b - a) * grid_matrix[ti,xi][2]
            bspline_quad_approx[ti,xi] = eval_spline2D(coefs, (Bt, Bx), (tt, xx))
            f_quad_value[ti,xi] = exact_u(h * tt, xx)
        end
    end

    # println("Error Matrix between exact solution and B-spline approximation at quadrature points:")
    # @show bspline_quad_approx .- f_quad_value
    println("Max error in initial guess at quadrature points: ", maximum(abs.(f_quad_value .- bspline_quad_approx)))

    C.x[1:S] = reshape(coefs, :, 1)
    # @infiltrate
end

post_initial_guess!(internal_coes, C,sol_struct,int::PDEIntegrator{<:Constraint_Bspline_Integrator},int_method::Constraint_Bspline_Integrator) = nothing
post_initial_guess!(internal_coes, C,sol_struct,int::PDEIntegrator{<:Constraint_Bspline_Integrator},int_method::Constraint_Bspline_Integrator) = nothing
    

function components!(x::AbstractVector{ST}, sol, params, int::PDEIntegrator{<:Constraint_Bspline_Integrator}) where {ST}
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

    u_truth_mat = similar(C.u_quad_values)
    v_truth_mat = similar(C.v_quad_values)
    w_truth_mat = similar(C.w_quad_values)
    # interior values at quadrature points
    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = sum(x[1:S] .* int.method.u_collocation_mat[:, i, j])
                C.v_quad_values[d, i, j] = sum(x[1:S] .* int.method.v_collocation_mat[:, i, j]) / h 
                C.w_quad_values[d, i, j] = sum(x[1:S] .* int.method.w_collocation_mat[:, i, j])

                u_truth_mat[d, i, j] = int.problem.exact_u.(h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                v_truth_mat[d, i, j] = int.problem.exact_v.(h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                w_truth_mat[d, i, j] = int.problem.exact_w.(h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
            end
        end
    end

    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w
    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local h = timestep(int)
    
    @show C.u_quad_values[1,1,:] .- exact_u.(h * t_quad_nodes[1], xspan[1] .+ x_domain .* x_quad_nodes)
    @show C.v_quad_values[1,1,:] .- exact_v.(h * t_quad_nodes[1], xspan[1] .+ x_domain .* x_quad_nodes)
    @show C.w_quad_values[1,1,:] .- exact_w.(h * t_quad_nodes[1], xspan[1] .+ x_domain .* x_quad_nodes)

    @show maximum(abs.(C.u_quad_values .- u_truth_mat))
    @show maximum(abs.(C.v_quad_values .- v_truth_mat))
    @show maximum(abs.(C.w_quad_values .- w_truth_mat))

    ∂L∂U_truth_mat = similar(C.∂L∂U_quad_values)
    ∂L∂V_truth_mat = similar(C.∂L∂V_quad_values)
    ∂L∂W_truth_mat = similar(C.∂L∂W_quad_values)

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.∂L∂U_quad_values[d, i, j] = ∂L∂U[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], lag_params)
                C.∂L∂V_quad_values[d, i, j] = ∂L∂V[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], lag_params)
                C.∂L∂W_quad_values[d, i, j] = ∂L∂W[d](C.u_quad_values[d, i, j], C.v_quad_values[d, i, j], C.w_quad_values[d, i, j], lag_params)

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
    for d in 1:D
        for j in 1:RX
            C.ut₀_quad_values[d,j] = sum(x[1:S] .* int.method.ut₀_basis_quad_values[:,j])
            C.ut₁_quad_values[d,j] = sum(x[1:S] .* int.method.ut₁_basis_quad_values[:,j])

            C.vt₀_quad_values[d,j] = sum(x[1:S] .* int.method.vt₀_basis_quad_values[:,j]) / h
            C.vt₁_quad_values[d,j] = sum(x[1:S] .* int.method.vt₁_basis_quad_values[:,j]) / h

            C.wt₀_quad_values[d,j] = sum(x[1:S] .* int.method.wt₀_basis_quad_values[:,j])
            C.wt₁_quad_values[d,j] = sum(x[1:S] .* int.method.wt₁_basis_quad_values[:,j])
        end

        for i in 1:RT
            C.ux₀_quad_values[d,i] = sum(x[1:S] .* int.method.ux₀_basis_quad_values[:,i])
            C.ux₁_quad_values[d,i] = sum(x[1:S] .* int.method.ux₁_basis_quad_values[:,i])
            C.vx₀_quad_values[d,i] = sum(x[1:S] .* int.method.vx₀_basis_quad_values[:,i])/ h
            C.vx₁_quad_values[d,i] = sum(x[1:S] .* int.method.vx₁_basis_quad_values[:,i])/ h
            C.wx₀_quad_values[d,i] = sum(x[1:S] .* int.method.wx₀_basis_quad_values[:,i]) 
            C.wx₁_quad_values[d,i] = sum(x[1:S] .* int.method.wx₁_basis_quad_values[:,i]) 
        end

    end

    (x == cache(int).x && eltype(x) == eltype(cache(int).x)) ? post_initial_guess!(x[1:S], cache(int),sol,int,int.method) : nothing

    for d in 1:D
        C.λ₀_x_coes[d,:] = x[S+1:S+k_λ_x]
        C.μ₀_t_coes[d,:] = x[S+k_λ_x+1:S+k_λ_x+k_μ_t]
        C.μ₁_t_coes[d,:] = x[S+k_λ_x+k_μ_t+1:S+k_λ_x+2*k_μ_t]
    end

    for d in 1:D
        for rx in 1:RX
            C.λ₀_quad_values[d,rx] = sum([C.λ₀_x_coes[d,i] * mλ₀_x[i,rx]  for i in 1:k_λ_x])
        end

        for rt in 1:RT
            C.μ₀_quad_values[d,rt] = sum([C.μ₀_t_coes[d,i]*mμ_t[i,rt] for i in 1:k_μ_t]) 
            C.μ₁_quad_values[d,rt] = sum([C.μ₁_t_coes[d,i]*mμ_t[i,rt] for i in 1:k_μ_t])
        end
    end
    @infiltrate
end

function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{<:Constraint_Bspline_Integrator})
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
    local k_λ_x = int.method.k_λ_x
    local k_μ_t = int.method.k_μ_t
    local mλ₀_x = int.method.mλ₀_x
    local mμ_t = int.method.mμ_t

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
                z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * C.ut₀_basis_quad_values[p,rx]) #- C.λ₁_quad_values[d,rx] * C.∂u∂P_t₁_quad_values[d, rx]
            end
            for rt in 1:RT
                z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * C.ux₀_basis_quad_values[p,rt] - C.μ₁_quad_values[d,rt] * C.ux₁_basis_quad_values[p,rt])
            end
            b[current_idx] = -z # TODO: check the sign
            current_idx += 1
        end
    end

    @assert current_idx == S + 1 "Wrong indexing in residual computation"

    for d in 1:D
        for p in 1:k_λ_x
            z = zero(ST)
            for rx in 1:RX
                z += x_domain * brx[rx] * mλ₀_x[p,rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
            end
            b[S + (d - 1) * k_λ_x + p] = - z
        end
    end

    for d in 1:D
        for p in 1:k_μ_t
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) *brt[rt] * mμ_t[p,rt] *(C.ux₀_quad_values[d,rt] - C.boundary_condition_x₀[d,rt])
            end
            b[S + D * k_λ_x + (d - 1) * k_μ_t + p] = - z
        end
    end

    for d in 1:D
        for p in 1:k_μ_t
            z = zero(ST)
            for rt in 1:RT
                z += timestep(int) *brt[rt] * mμ_t[p,rt] *(C.boundary_condition_x₁[d,rt] - C.ux₁_quad_values[d,rt])
            end
            b[S + D * k_λ_x + D * k_μ_t + (d - 1) * k_μ_t + p] = - z
        end
    end
    @infiltrate
    @show b
end

function update!(sol, int::PDEIntegrator{<:Constraint_Bspline_Integrator})
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
            sol.u[i] = sum(x[1:S] .* ubasis_x_nodes_values[:,i])
            sol.v[i] = sum(x[1:S] .* vbasis_x_nodes_values[:,i])
            sol.w[i] = sum(x[1:S] .* wbasis_x_nodes_values[:,i])
        end
    end

    # copy internal variables from cache to solution
    # sol.internal.x[] .= cache(int).x 
    # println("In the end of update! function, time = ", sol_struct.t)
end

