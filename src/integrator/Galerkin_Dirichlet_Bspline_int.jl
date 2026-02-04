struct Galerkin_Dirichlet_Bspline_Integrator{BT<:AbstractPDEBasis} <: PDEMethod
    basis::BT
    
    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions
    
    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights

    Nbasis_λ_x

    mλ_x # λ_x evaluated at quadrature points

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

    show_status
    function Galerkin_Dirichlet_Bspline_Integrator(basis; RT_per_interval::Int = 4,RX_per_interval::Int = 4,xspan::Tuple = (0.,1.0), 
        show_status = false)

        # The quadrature nodes in [0.0,1.0]
        t_num_interval = length(basis.ts) - 1
        x_num_interval = length(basis.xs) - 1
        t_quadrature = composite_quadrature(t_num_interval ,RT_per_interval)
        x_quadrature = composite_quadrature(x_num_interval ,RX_per_interval)

        R_list = [RT_per_interval,RX_per_interval]  
        grid_matrix, grid_weights = construct_quadrature_grid(R_list,[t_num_interval, x_num_interval])
        
        S = basis.S
        @show S
        RT = length(t_quadrature.nodes)
        RX = length(x_quadrature.nodes)
        
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

        
        mλ_x = zeros(basis.Nbasis_t, RX)
        for i in 1:basis.Nbasis_t
            mλ_x[i,:] = basis.Basis_t[i].(xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes,BSplineKit.Derivative(1))#
        end 
        Nbasis_λ_x = basis.Nbasis_t

        new{typeof(basis)}(basis, 
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            Nbasis_λ_x,
            mλ_x,
            u_collocation_matrix, v_collocation_matrix, w_collocation_matrix,
            ut₀_basis_quad_values,ut₁_basis_quad_values,
            vt₀_basis_quad_values,vt₁_basis_quad_values,
            wt₀_basis_quad_values,wt₁_basis_quad_values,
            ux₀_basis_quad_values,ux₁_basis_quad_values,
            vx₀_basis_quad_values,vx₁_basis_quad_values,
            wx₀_basis_quad_values,wx₁_basis_quad_values,
            show_status
            )
    end 
end

default_solver(::Galerkin_Dirichlet_Bspline_Integrator) = NewtonMethod()

struct Galerkin_Dirichlet_Bspline_IntegratorCache{ST,RT,RX,D,S,Nbasis_λ_x} <: PDEIntegratorCache{ST,D}
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
    λ₁_x_coes::Matrix{ST}

    λ₀_quad_values::Matrix{ST} 
    λ₁_quad_values::Matrix{ST}

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
    flag_done_initial_guess::Vector{ST}

    function Galerkin_Dirichlet_Bspline_IntegratorCache{ST,RT,RX,D,S,Nbasis_λ_x}() where {ST,RT,RX,D,S,Nbasis_λ_x}
        # x = zeros(ST, NP + 2 * D * RX + 2* D * DX * RT ) # TODO: how to deal with RX being a vector/
        x = zeros(ST,S + D * Nbasis_λ_x) # params, λ_x_coes,μ₀_t_coes,μ₁_t_coes
        # TODO:consider when DX is a vector
        # x = zeros(ST,S)
        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        λ₀_x_coes = zeros(ST, D, Nbasis_λ_x)
        λ₁_x_coes = zeros(ST, D, Nbasis_λ_x)

        λ₀_quad_values = zeros(ST, D, RX) 
        λ₁_quad_values = zeros(ST, D, RX)

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

        init_condition_t₀ = zeros(ST, D, RX)
        boundary_condition_x₀ = zeros(ST, D, RT)
        boundary_condition_x₁ = zeros(ST, D, RT)
        flag_done_initial_guess = zeros(ST, 1)
        
        new(x,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
            λ₀_x_coes, λ₁_x_coes,
            λ₀_quad_values,λ₁_quad_values,
            ut₀_quad_values, ut₁_quad_values,vt₀_quad_values, vt₁_quad_values,wt₀_quad_values, wt₁_quad_values,
            ux₀_quad_values, ux₁_quad_values,vx₀_quad_values, vx₁_quad_values,wx₀_quad_values, wx₁_quad_values,
            ics_ut₀_quad_values, ics_vt₀_quad_values, ics_wt₀_quad_values,
            bc_ux₀_quad_values, bc_vx₀_quad_values, bc_wx₀_quad_values,
            bc_ux₁_quad_values, bc_vx₁_quad_values, bc_wx₁_quad_values,
            init_condition_t₀, 
            boundary_condition_x₀, boundary_condition_x₁,
            flag_done_initial_guess
            )
    end
end

nlsolution(cache::Galerkin_Dirichlet_Bspline_IntegratorCache) = cache.x

function Cache{ST}(problem::LPDEProblem, int::Galerkin_Dirichlet_Bspline_Integrator; kwargs...) where {ST}
    Galerkin_Dirichlet_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.S,int.Nbasis_λ_x}(; kwargs...)
end

#{ST,RT,RX,D,NP}(NP) where {ST,RT,RX,D,NP}
@inline CacheType(ST, problem::LPDEProblem, int::Galerkin_Dirichlet_Bspline_Integrator) = Galerkin_Dirichlet_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.S,int.Nbasis_λ_x}
@inline function Base.getindex(c::Galerkin_Dirichlet_Bspline_IntegratorCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

copy_internal_variables!(C::Galerkin_Dirichlet_Bspline_IntegratorCache,solstep::SolutionStep) = nothing

function prior_initial_guess!(C,sol,int::PDEIntegrator{<:Galerkin_Dirichlet_Bspline_Integrator})
    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w

    local x_collocation_points = int.method.basis.collocation_points_x
    local t_collocation_points = int.method.basis.collocation_points_t
    local Cx = int.method.basis.collocation_matrix_x
    local Ct = int.method.basis.collocation_matrix_t

    local Bx = int.method.basis.Basis_x
    local Bt = int.method.basis.Basis_t
    local S = int.method.basis.S
    local h = timestep(int) 
    local a = int.problem.xspan[1]
    local b = int.problem.xspan[2]
    local grid_matrix = int.method.grid_matrix
    local RT = int.method.RT
    local RX = int.method.RX
    local show_status = int.method.show_status

    udata = exact_u.(h .* t_collocation_points, x_collocation_points') # fdata[i,j] = exact_u(h .* ts[i], xs[j])
    
    # 2D B-spline coefficients (output)
    coefs = similar(udata)

    # Solve linear systems
    for j ∈ eachindex(x_collocation_points)
        @views ldiv!(coefs[:, j], Ct, udata[:, j])
    end
    for i ∈ eachindex(t_collocation_points)
        @views ldiv!(Cx, coefs[i, :])
    end
    C.x[1:S] = reshape(coefs, :, 1)

    if show_status
        vdata = exact_v.(h .* t_collocation_points, x_collocation_points')
        wdata = exact_w.(h .* t_collocation_points, x_collocation_points')

        u_approx = similar(udata)
        v_approx = similar(vdata)
        w_approx = similar(wdata)
        # Verification: evaluate 2D spline at data points
        for m ∈ eachindex(t_collocation_points), n ∈ eachindex(x_collocation_points)
            t = t_collocation_points[m]
            x = x_collocation_points[n]
            u_approx[m, n] = eval_spline2D(coefs, (Bt, Bx), (t, x))
            v_approx[m, n] = eval_spline2D_dt(coefs, (Bt, Bx), (t, x)) / h
            w_approx[m, n] = eval_spline2D_dx(coefs, (Bt, Bx), (t, x))
        end

        u_quad_approx = zeros(RT, RX)
        v_quad_approx = zeros(RT, RX)
        w_quad_approx = zeros(RT, RX)
        for ti in 1:RT
            for xi in 1:RX
                tt = grid_matrix[ti,xi][1]
                xx = a + (b - a) * grid_matrix[ti,xi][2]

                u_quad_approx[ti,xi] = eval_spline2D(coefs, (Bt, Bx), (tt, xx))
                v_quad_approx[ti,xi] = eval_spline2D_dt(coefs, (Bt, Bx), (tt, xx)) / h
                w_quad_approx[ti,xi] = eval_spline2D_dx(coefs, (Bt, Bx), (tt, xx))
            end
        end

        u_quad_value = zeros(RT, RX)
        v_quad_value = zeros(RT, RX)
        w_quad_value = zeros(RT, RX)
        for ti in 1:RT
            for xi in 1:RX
                tt = grid_matrix[ti,xi][1]
                xx = a + (b - a) * grid_matrix[ti,xi][2]

                u_quad_value[ti,xi] = exact_u(h * tt, xx)
                v_quad_value[ti,xi] = exact_v(h * tt, xx)
                w_quad_value[ti,xi] = exact_w(h * tt, xx)
            end
        end

        println("Error Matrix between exact solution and B-spline approximation at grid points:")
        # @show u_approx .- fdata
        println("Max error in initial guess for u: ", maximum(abs.(udata .- u_approx)))
        println("Max error in initial guess for v: ", maximum(abs.(vdata .- v_approx))) 
        println("Max error in initial guess for w: ", maximum(abs.(wdata .- w_approx)))

        println("Error Matrix between exact solution and B-spline approximation at quadrature points:")
        println("Max error in initial guess at quadrature points for u: ", maximum(abs.(u_quad_value .- u_quad_approx)))
        println("Max error in initial guess at quadrature points for v: ", maximum(abs.(v_quad_value .- v_quad_approx)))
        println("Max error in initial guess at quadrature points for w: ", maximum(abs.(w_quad_value .- w_quad_approx)))
    end

    # @infiltrate
end

function post_initial_guess!(internal_coes, C,sol_struct,int::PDEIntegrator{<:Galerkin_Dirichlet_Bspline_Integrator},int_method::Galerkin_Dirichlet_Bspline_Integrator{BT}) where {BT}
    local RX = int_method.RX
    local D = int.problem.D 
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params
    local Nbasis_λ_x = int_method.Nbasis_λ_x
    local mλ_x = int_method.mλ_x
    local S = int_method.basis.S

    for d in 1:D
        tem_t₁_∂L∂V = zeros(RX)
        for rx in 1:RX
            tem_t₁_∂L∂V[rx] = lag_sys.∂L∂V[d](C.ut₁_quad_values[d,rx], C.vt₁_quad_values[d,rx], C.wt₁_quad_values[d,rx], params)
        end
        λ_x_tem = mλ_x'\tem_t₁_∂L∂V

        for rx in 1:Nbasis_λ_x
            C.x[S + (d - 1) * Nbasis_λ_x + rx] = λ_x_tem[rx]
        end
    end

    C.flag_done_initial_guess[1] = 1.0
end


function components!(x::AbstractVector{ST}, sol, params, int::PDEIntegrator{<:Galerkin_Dirichlet_Bspline_Integrator}) where {ST}
    local C = cache(int,ST)
    local h = timestep(int)
    local RT = int.method.RT
    local RX = int.method.RX
    local D = int.problem.D
    local S = int.method.basis.S

    local ∂L∂U = int.problem.lagrangian_system.functions.∂L∂U
    local ∂L∂V = int.problem.lagrangian_system.functions.∂L∂V
    local ∂L∂W = int.problem.lagrangian_system.functions.∂L∂W

    local x_quad_nodes = int.method.spatial_quadrature.nodes
    local t_quad_nodes = int.method.time_quadrature.nodes
    local xspan = int.problem.xspan
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]

    local lag_params = int.problem.lagrangian_system.params
    local Nbasis_λ_x = int.method.Nbasis_λ_x
    local mλ_x = int.method.mλ_x
    
    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w
    local show_status = int.method.show_status

    # interior values at quadrature points
    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = sum(x[1:S] .* int.method.u_collocation_mat[:, i, j])
                C.v_quad_values[d, i, j] = sum(x[1:S] .* int.method.v_collocation_mat[:, i, j]) / h 
                C.w_quad_values[d, i, j] = sum(x[1:S] .* int.method.w_collocation_mat[:, i, j])
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

    # initial guess for the coefficients of Lagrangian multipliers
    cache(int).flag_done_initial_guess[1] == 0.0 ? post_initial_guess!(x[1:S], cache(int),sol,int,int.method) : nothing
    
    for d in 1:D
        C.λ₁_x_coes[d,:] = x[S+1:S+Nbasis_λ_x]
    end

    for d in 1:D
        for rx in 1:RX
            C.λ₀_quad_values[d,rx] = ∂L∂V[d](C.ics_ut₀_quad_values[d,rx], C.ics_vt₀_quad_values[d,rx],C.ics_wt₀_quad_values[d,rx], lag_params)
            C.λ₁_quad_values[d,rx] = sum(C.λ₁_x_coes[d,:] .* mλ_x[:,rx])
        end
    end

    if show_status
        u_truth_mat = similar(C.u_quad_values)
        v_truth_mat = similar(C.v_quad_values)
        w_truth_mat = similar(C.w_quad_values)
        # interior values at quadrature points
        for d in 1:D
            for i in 1:RT
                for j in 1:RX
                    u_truth_mat[d, i, j] = exact_u.(h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                    v_truth_mat[d, i, j] = exact_v.(h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
                    w_truth_mat[d, i, j] = exact_w.(h * t_quad_nodes[i], xspan[1] .+ x_domain .* x_quad_nodes[j])
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

        @show maximum(abs.(C.λ₁_quad_values[1,:] .- exact_v.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes)))
    end
end


function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{<: Galerkin_Dirichlet_Bspline_Integrator}) where {ST}
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
    local Nbasis_λ_x = int.method.Nbasis_λ_x
    local mλ_x = int.method.mλ_x
    local ut₀_basis_quad_values = int.method.ut₀_basis_quad_values
    local ut₁_basis_quad_values = int.method.ut₁_basis_quad_values
    local ux₀_basis_quad_values = int.method.ux₀_basis_quad_values
    local ux₁_basis_quad_values = int.method.ux₁_basis_quad_values
    local show_status = int.method.show_status

    for d in 1:D 
        for p in 1:S
            z_in = zero(ST)
            for rt in 1:RT
                for rx in 1:RX
                    z_in +=  quad_b[rt,rx] * 
                        ( x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx] * u_coll_mat[p, rt, rx]
                        + x_domain                 * C.∂L∂V_quad_values[d,rt,rx] * v_coll_mat[p, rt, rx]
                        + x_domain * timestep(int) * C.∂L∂W_quad_values[d,rt,rx] * w_coll_mat[p, rt, rx])
                end
            end
            # println(z_in)
            z_bd = zero(ST)
            for rx in 1:RX
                z_bd += x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * ut₀_basis_quad_values[p,rx] - C.λ₁_quad_values[d,rx] * ut₁_basis_quad_values[p,rx])
            end
            # println(z_bd)
            b[p] = -(z_in + z_bd)
        end
    end

    for d in 1:D
        for p in 1:Nbasis_λ_x
            z = zero(ST)
            for rx in 1:RX
                z += x_domain * brx[rx] * mλ_x[p,rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
            end
            b[S + (d - 1) * Nbasis_λ_x + p] = - z
        end
    end

    if show_status
        @show b
    end
end

function update!(sol, int::PDEIntegrator{<:Galerkin_Dirichlet_Bspline_Integrator})
    local D = int.problem.D
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local x = nlsolution(int)
    local S = int.method.basis.S
    local Basis_t = int.method.basis.Basis_t
    local Basis_x = int.method.basis.Basis_x
    local h = timestep(int)
    local Nbasis_x = int.method.basis.Nbasis_x
    local Nbasis_t = int.method.basis.Nbasis_t
    
    x_nodes = collect(xspan[1]:xstep:xspan[2])
    coefs = reshape(x[1:S], Nbasis_t, Nbasis_x)

    for d in 1:D
        for i in eachindex(x_nodes)
            sol.u[i] = eval_spline2D(coefs, (Basis_t, Basis_x), (1.0, x_nodes[i]))
            sol.v[i] = eval_spline2D_dt(coefs, (Basis_t, Basis_x), (1.0, x_nodes[i])) /h
            sol.w[i] = eval_spline2D_dx(coefs, (Basis_t, Basis_x), (1.0, x_nodes[i]))
        end
    end
    
end

