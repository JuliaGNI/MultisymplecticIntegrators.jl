struct Galerkin_Bspline_Integrator{T,MVT,LT,BT<:AbstractPDEBasis} <: PDEMethod
    basis::BT
    
    time_quadrature
    RT::Int # Number of quadrature points in time

    spatial_quadrature
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions
    
    grid_matrix # Quadrature grid points: [(t1,x1), (t1,x2), ]
    grid_weights # Quadrature weights
    
    k_μ_t::Int
    μ₀_t::MVT
    μ₁_t::MVT

    k_λ_x::Int
    λ_x::LT

    mλ_x # λ_x evaluated at quadrature points
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

    function Galerkin_Bspline_Integrator(basis; RT::Int = 8,RX::Int = 8,xspan::Tuple = (0.,1.0),tstep::T = 1.0, k_μ_t::Int = 4,k_λ_x::Int = 4,
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


        # Construct Lagrangian multipliers, defined on [0,1] and need to be scaled carefully when used
        λ_x = Lagrangian_multiplier(λ,k_λ_x,xspan[1],xspan[2])
        μ₀_t = Lagrangian_multiplier(μ,k_μ_t,0.0,1.0)
        μ₁_t = Lagrangian_multiplier(μ,k_μ_t,0.0,1.0)

        mλ_x = zeros(k_λ_x, RX)
        mμ_t = zeros(k_μ_t, RT)

        for i in 1:k_λ_x
            mλ_x[i,:] = λ_x.b[i].(xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes)
        end
    
        for i in 1:k_μ_t
            mμ_t[i,:] = μ₀_t.b[i].(t_quadrature.nodes)
        end

        new{T,typeof(μ₀_t),typeof(λ_x),typeof(basis)}(basis, 
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            k_μ_t, μ₀_t, μ₁_t,
            k_λ_x,λ_x, #λ₁_x,
            mλ_x, mμ_t,
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

default_solver(::Galerkin_Bspline_Integrator) = NewtonMethod()

struct Galerkin_Bspline_IntegratorCache{ST,RT,RX,D,S,k_μ_t,k_λ_x} <: PDEIntegratorCache{ST,D}
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
    μ₀_t_coes::Matrix{ST}
    μ₁_t_coes::Matrix{ST}

    λ₀_quad_values::Matrix{ST} 
    λ₁_quad_values::Matrix{ST}
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

    function Galerkin_Bspline_IntegratorCache{ST,RT,RX,D,S,k_μ_t,k_λ_x}() where {ST,RT,RX,D,S,k_μ_t,k_λ_x}
        # x = zeros(ST, NP + 2 * D * RX + 2* D * DX * RT ) # TODO: how to deal with RX being a vector/
        x = zeros(ST,S + D * k_λ_x + 2 * D * k_μ_t) # params, λ_x_coes,μ₀_t_coes,μ₁_t_coes
        # TODO:consider when DX is a vector
        # x = zeros(ST,S)
        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

        λ₀_x_coes = zeros(ST, D, k_λ_x)
        λ₁_x_coes = zeros(ST, D, k_λ_x)
        μ₀_t_coes = zeros(ST, D, k_μ_t)
        μ₁_t_coes = zeros(ST, D, k_μ_t)

        λ₀_quad_values = zeros(ST, D, RX) 
        λ₁_quad_values = zeros(ST, D, RX)
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
            λ₀_x_coes, λ₁_x_coes,μ₀_t_coes, μ₁_t_coes,
            λ₀_quad_values,λ₁_quad_values, μ₀_quad_values, μ₁_quad_values,
            ∂u∂P_t₀_quad_values, ∂u∂P_t₁_quad_values, ∂u∂P_x₀_quad_values, ∂u∂P_x₁_quad_values,
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

nlsolution(cache::Galerkin_Bspline_IntegratorCache) = cache.x

function Cache{ST}(problem::LPDEProblem, int::Galerkin_Bspline_Integrator; kwargs...) where {ST}
    Galerkin_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.S,int.k_μ_t,int.k_λ_x}(; kwargs...)
end

#{ST,RT,RX,D,NP}(NP) where {ST,RT,RX,D,NP}
@inline CacheType(ST, problem::LPDEProblem, int::Galerkin_Bspline_Integrator) = Galerkin_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.S,int.k_μ_t,int.k_λ_x}
@inline function Base.getindex(c::Galerkin_Bspline_IntegratorCache, ST::DataType)
    key = hash(Threads.threadid(), hash(ST))
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache{ST}(c.problem, c.method)
    end::CacheType(ST, c.problem, c.method)
end

function initialize_bcs_ics!(sol,int::PDEIntegrator{<:Galerkin_Bspline_Integrator})
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

            C.ics_ut₀_quad_values[d,:] .= ic_fun(xspan[1] .+ x_domain .* x_quad_nodes).u
            C.ics_vt₀_quad_values[d,:] .= ic_fun(xspan[1] .+ x_domain .* x_quad_nodes).v
            C.ics_wt₀_quad_values[d,:] .= ic_fun(xspan[1] .+ x_domain .* x_quad_nodes).w
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

            C.bc_ux₀_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.u
            C.bc_vx₀_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.v
            C.bc_wx₀_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₀.w

            C.bc_ux₁_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.u
            C.bc_vx₁_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.v
            C.bc_wx₁_quad_values[d,i] = bc_fun(sol.t - timestep(int) + timestep(int)* t_quad_nodes[i],xspan).bc₁.w
        end
        # println("left boundary condition = " , C.boundary_condition_x₀[d,:])
    end

end

copy_internal_variables!(C::Galerkin_Bspline_IntegratorCache,solstep::SolutionStep) = nothing
copy_internal_variables!(solstep::SolutionStep,C::Galerkin_Bspline_IntegratorCache) = nothing


function prior_initial_guess!(C,sol,int::PDEIntegrator{<:Galerkin_Bspline_Integrator})
    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w

    local xs = int.method.basis.xs
    local ts = int.method.basis.ts # t \in [0, 1.0]

    # local ts = int.method.time_quadrature.nodes
    # local xs = int.method.spatial_quadrature.nodes
    # xs = a .+ (b - a) .* xs # scale to physical domain

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

    udata = exact_u.(h .* ts, xs') # fdata[i,j] = exact_u(h .* ts[i], xs[j])
    vdata = exact_v.(h .* ts, xs')
    wdata = exact_w.(h .* ts, xs')

    # 2D B-spline coefficients (output)
    coefs = similar(udata)

    # Solve linear systems
    for j ∈ eachindex(xs)
        @views ldiv!(coefs[:, j], Ct, udata[:, j])
    end
    for i ∈ eachindex(ts)
        @views ldiv!(Cx, coefs[i, :])
    end

    u_approx = similar(udata)
    v_approx = similar(vdata)
    w_approx = similar(wdata)
    # Verification: evaluate 2D spline at data points
    for m ∈ eachindex(ts), n ∈ eachindex(xs)
        t = ts[m]
        x = xs[n]
        u_approx[m, n] = eval_spline2D(coefs, (Bt, Bx), (t, x))
        v_approx[m, n] = eval_spline2D_dt(coefs, (Bt, Bx), (t, x)) / h
        w_approx[m, n] = eval_spline2D_dx(coefs, (Bt, Bx), (t, x))
    end

    # println("Error Matrix between exact solution and B-spline approximation at grid points:")
    # @show u_approx .- fdata
    println("Max error in initial guess for u: ", maximum(abs.(udata .- u_approx)))
    println("Max error in initial guess for v: ", maximum(abs.(vdata .- v_approx))) 
    println("Max error in initial guess for w: ", maximum(abs.(wdata .- w_approx)))

    u_quad_value = zeros(RT, RX)
    v_quad_value = zeros(RT, RX)
    w_quad_value = zeros(RT, RX)

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

            u_quad_value[ti,xi] = exact_u(h * tt, xx)
            v_quad_value[ti,xi] = exact_v(h * tt, xx)
            w_quad_value[ti,xi] = exact_w(h * tt, xx)
        end
    end

    # println("Error Matrix between exact solution and B-spline approximation at quadrature points:")
    println("Max error in initial guess at quadrature points for u: ", maximum(abs.(u_quad_value .- u_quad_approx)))
    println("Max error in initial guess at quadrature points for v: ", maximum(abs.(v_quad_value .- v_quad_approx)))
    println("Max error in initial guess at quadrature points for w: ", maximum(abs.(w_quad_value .- w_quad_approx)))

    C.x[1:S] = reshape(coefs, :, 1)
    # @infiltrate
end

function post_initial_guess!(internal_coes, C,sol_struct,int::PDEIntegrator{<:Galerkin_Bspline_Integrator},int_method::Galerkin_Bspline_Integrator{T,MVT,LT,BT}) where {T,MVT<:BSplineDirichlet{T},LT<:BSplineDirichlet{T},BT}
    local RT = int_method.RT
    local RX = int_method.RX
    local D = int.problem.D 
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params
    local xspan = int.problem.xspan
    local x_quad_nodes = int_method.spatial_quadrature.nodes
    local k_λ_x = int_method.k_λ_x
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
        λ_x = interpolate(tem_x, tem_t₀_∂L∂V, BSplineOrder(k_λ_x))
        for i in 1:RX
            C.x[S + (d - 1) * RX + i] = λ_x.spline.coefs[i]
        end

        tem_x₀_∂L∂W = zeros(RT)
        for i in 1:RT
            tem_x₀_∂L∂W[i] = lag_sys.∂L∂W[d](C.ux₀_quad_values[d,i], C.vx₀_quad_values[d,i], C.wx₀_quad_values[d,i], params)
        end

        tem_t = sol_struct.t .- timestep(int) .+ timestep(int) .* t_quad_nodes
        μ₀_t = interpolate(tem_t, tem_x₀_∂L∂W, BSplineOrder(int.k_μ_t))
        for i in 1:RT
            C.x[S + D * RX + (d - 1) * RT + i] = μ₀_t.spline.coefs[i]
        end

        tem_x₁_∂L∂W = zeros(RT)
        for i in 1:RT
            tem_x₁_∂L∂W[i] = lag_sys.∂L∂W[d](C.ux₁_quad_values[d,i], C.vx₁_quad_values[d,i], C.wx₁_quad_values[d,i], params)
        end

        μ₁_t = interpolate(tem_t, tem_x₁_∂L∂W, BSplineOrder(int.k_μ_t))
        for i in 1:RT
            C.x[S+ D * RX + D * RT + (d-1)*RT+i] = μ₁_t.spline.coefs[i]
        end

    end
end

function post_initial_guess!(internal_coes, C,sol_struct,int::PDEIntegrator{<:Galerkin_Bspline_Integrator},int_method::Galerkin_Bspline_Integrator{T,MVT,LT,BT}) where {T,MVT<:Lagrange,LT<:Lagrange,BT}
    local S = int_method.basis.S
    local λ_x = int_method.λ_x
    local k_λ_x = int_method.k_λ_x
    local h = timestep(int)
    local μ₀_t = int_method.μ₀_t
    local k_μ_t = int_method.k_μ_t

    local xspan = int.problem.xspan
    local D = int.problem.D 
    local lag_sys = int.problem.lagrangian_system.functions
    local params = int.problem.lagrangian_system.params

    local basis = int_method.basis
    local Nbasis_x = int_method.basis.Nbasis_x
    local Nbasis_t = int_method.basis.Nbasis_t

    ut₀_quad_values_tem = zeros(k_λ_x)
    ut₁_quad_values_tem = zeros(k_λ_x)
    vt₀_quad_values_tem = zeros(k_λ_x)
    vt₁_quad_values_tem = zeros(k_λ_x)
    wt₀_quad_values_tem = zeros(k_λ_x)
    wt₁_quad_values_tem = zeros(k_λ_x)

    ux₀_quad_values_tem = zeros(k_μ_t)
    ux₁_quad_values_tem = zeros(k_μ_t)
    vx₀_quad_values_tem = zeros(k_μ_t)
    vx₁_quad_values_tem = zeros(k_μ_t)
    wx₀_quad_values_tem = zeros(k_μ_t)
    wx₁_quad_values_tem = zeros(k_μ_t)

    coefs = reshape(internal_coes, Nbasis_t, Nbasis_x)

    for rx in 1:k_λ_x
        xx = λ_x.x[rx]
        ut₀_quad_values_tem[rx] = eval_spline2D(coefs, (basis.Basis_t, basis.Basis_x), (0.0, xx))
        ut₁_quad_values_tem[rx] = eval_spline2D(coefs, (basis.Basis_t, basis.Basis_x), (1.0, xx))
        vt₀_quad_values_tem[rx] = eval_spline2D_dt(coefs, (basis.Basis_t, basis.Basis_x), (0.0, xx)) /h 
        vt₁_quad_values_tem[rx] = eval_spline2D_dt(coefs, (basis.Basis_t, basis.Basis_x), (1.0, xx)) /h
        wt₀_quad_values_tem[rx] = eval_spline2D_dx(coefs, (basis.Basis_t, basis.Basis_x), (0.0, xx))
        wt₁_quad_values_tem[rx] = eval_spline2D_dx(coefs, (basis.Basis_t, basis.Basis_x), (1.0, xx))
    end

    for rt in 1:k_μ_t
        tt = μ₀_t.x[rt]
        ux₀_quad_values_tem[rt] = eval_spline2D(coefs, (basis.Basis_t, basis.Basis_x), (tt, xspan[1]))
        ux₁_quad_values_tem[rt] = eval_spline2D(coefs, (basis.Basis_t, basis.Basis_x), (tt, xspan[2]))
        vx₀_quad_values_tem[rt] = eval_spline2D_dt(coefs, (basis.Basis_t, basis.Basis_x), (tt, xspan[1]))/h
        vx₁_quad_values_tem[rt] = eval_spline2D_dt(coefs, (basis.Basis_t, basis.Basis_x), (tt, xspan[2]))/h
        wx₀_quad_values_tem[rt] = eval_spline2D_dx(coefs, (basis.Basis_t, basis.Basis_x), (tt, xspan[1]))
        wx₁_quad_values_tem[rt] = eval_spline2D_dx(coefs, (basis.Basis_t, basis.Basis_x), (tt, xspan[2]))
    end

    for d in 1:D
        for rx in 1:k_λ_x
            C.x[S + (d - 1) * k_λ_x + rx] = lag_sys.∂L∂V[d](ut₁_quad_values_tem[rx], vt₁_quad_values_tem[rx], wt₁_quad_values_tem[rx], params)
            # C.x[S + D * k_λ_x + (d - 1) * k_λ_x + rx] = lag_sys.∂L∂V[d](ut₁_quad_values_tem[rx], vt₁_quad_values_tem[rx], wt₁_quad_values_tem[rx], params)
        end

        for rt in 1:k_μ_t
            C.x[S + D * k_λ_x + (d - 1) * k_μ_t + rt] = lag_sys.∂L∂W[d](ux₀_quad_values_tem[rt], vx₀_quad_values_tem[rt], wx₀_quad_values_tem[rt], params)
            C.x[S + D * k_λ_x + + D * k_μ_t + (d - 1) * k_μ_t + rt] = lag_sys.∂L∂W[d](ux₁_quad_values_tem[rt], vx₁_quad_values_tem[rt], wx₁_quad_values_tem[rt], params)
        end
    end

    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w
    ut₀_truth_quad = zeros(k_λ_x)
    ut₁_truth_quad = zeros(k_λ_x)
    vt₁_truth_quad = zeros(k_λ_x)
    vt₀_truth_quad = zeros(k_λ_x)
    wt₀_truth_quad = zeros(k_λ_x)
    wt₁_truth_quad = zeros(k_λ_x)
    

    ux₀_truth_quad = zeros(k_μ_t)
    ux₁_truth_quad = zeros(k_μ_t)
    vx₀_truth_quad = zeros(k_μ_t)
    vx₁_truth_quad = zeros(k_μ_t)
    wx₀_truth_quad = zeros(k_μ_t)
    wx₁_truth_quad = zeros(k_μ_t)

    for rx in 1:k_λ_x
        xx = λ_x.x[rx]
        ut₀_truth_quad[rx] = exact_u.(sol_struct.t - timestep(int), xx)
        ut₁_truth_quad[rx] = exact_u.(sol_struct.t, xx)
        vt₀_truth_quad[rx] = exact_v.(sol_struct.t - timestep(int), xx)
        vt₁_truth_quad[rx] = exact_v.(sol_struct.t, xx)
        wt₀_truth_quad[rx] = exact_w.(sol_struct.t - timestep(int), xx)
        wt₁_truth_quad[rx] = exact_w.(sol_struct.t, xx)
    end

    for rt in 1:k_μ_t
        tt = μ₀_t.x[rt]
        ux₀_truth_quad[rt] = exact_u.(sol_struct.t - timestep(int) + timestep(int)* tt, xspan[1])
        ux₁_truth_quad[rt] = exact_u.(sol_struct.t - timestep(int) + timestep(int)* tt, xspan[2])
        vx₀_truth_quad[rt] = exact_v.(sol_struct.t - timestep(int) + timestep(int)* tt, xspan[1])
        vx₁_truth_quad[rt] = exact_v.(sol_struct.t - timestep(int) + timestep(int)* tt, xspan[2])
        wx₀_truth_quad[rt] = exact_w.(sol_struct.t - timestep(int) + timestep(int)* tt, xspan[1])
        wx₁_truth_quad[rt] = exact_w.(sol_struct.t - timestep(int) + timestep(int)* tt, xspan[2])
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

    C.flag_done_initial_guess[1] = 1.0
    # @infiltrate
end

function components!(x::AbstractVector{ST}, sol, params, int::PDEIntegrator{<:Galerkin_Bspline_Integrator}) where {ST}
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
    local k_λ_x = int.method.k_λ_x
    local k_μ_t = int.method.k_μ_t
    local mλ_x = int.method.mλ_x
    local mμ_t = int.method.mμ_t
    

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

    # C.∂L∂U_quad_values .= ∂L∂U_truth_mat
    # C.∂L∂V_quad_values .= ∂L∂V_truth_mat
    # C.∂L∂W_quad_values .= ∂L∂W_truth_mat


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

    # boundary values at quadrature points
    for d in 1:D
        for j in 1:RX
            C.ut₀_quad_values[d,j] = sum(x[1:S] .* int.method.ut₀_basis_quad_values[:,j])
            C.ut₁_quad_values[d,j] = sum(x[1:S] .* int.method.ut₁_basis_quad_values[:,j])

            C.vt₀_quad_values[d,j] = sum(x[1:S] .* int.method.vt₀_basis_quad_values[:,j]) / h
            C.vt₁_quad_values[d,j] = sum(x[1:S] .* int.method.vt₁_basis_quad_values[:,j]) / h

            C.wt₀_quad_values[d,j] = sum(x[1:S] .* int.method.wt₀_basis_quad_values[:,j])
            C.wt₁_quad_values[d,j] = sum(x[1:S] .* int.method.wt₁_basis_quad_values[:,j])

            ut₀_quad_values_truth[d,j] = exact_u.(sol.t - timestep(int), xspan[1] .+ x_domain .* x_quad_nodes[j])
            ut₁_quad_values_truth[d,j] = exact_u.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes[j])

            vt₀_quad_values_truth[d,j] = exact_v.(sol.t - timestep(int), xspan[1] .+ x_domain .* x_quad_nodes[j])
            vt₁_quad_values_truth[d,j] = exact_v.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes[j])

            wt₀_quad_values_truth[d,j] = exact_w.(sol.t - timestep(int), xspan[1] .+ x_domain .* x_quad_nodes[j])
            wt₁_quad_values_truth[d,j] = exact_w.(sol.t, xspan[1] .+ x_domain .* x_quad_nodes[j])
        end

        for i in 1:RT
            C.ux₀_quad_values[d,i] = sum(x[1:S] .* int.method.ux₀_basis_quad_values[:,i])
            C.ux₁_quad_values[d,i] = sum(x[1:S] .* int.method.ux₁_basis_quad_values[:,i])
            C.vx₀_quad_values[d,i] = sum(x[1:S] .* int.method.vx₀_basis_quad_values[:,i])/ h
            C.vx₁_quad_values[d,i] = sum(x[1:S] .* int.method.vx₁_basis_quad_values[:,i])/ h
            C.wx₀_quad_values[d,i] = sum(x[1:S] .* int.method.wx₀_basis_quad_values[:,i]) 
            C.wx₁_quad_values[d,i] = sum(x[1:S] .* int.method.wx₁_basis_quad_values[:,i]) 

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


    cache(int).flag_done_initial_guess[1] == 0.0 ? post_initial_guess!(x[1:S], cache(int),sol,int,int.method) : nothing

    for d in 1:D
        C.λ₁_x_coes[d,:] = x[S+1:S+k_λ_x]
        C.μ₀_t_coes[d,:] = x[S+k_λ_x+1:S+k_λ_x+k_μ_t]
        C.μ₁_t_coes[d,:] = x[S+k_λ_x+k_μ_t+1:S+k_λ_x+2*k_μ_t]
    end

    for d in 1:D
        for rx in 1:RX
            C.λ₀_quad_values[d,rx] = ∂L∂V[d](C.ics_ut₀_quad_values[d,rx], C.ics_vt₀_quad_values[d,rx],C.ics_wt₀_quad_values[d,rx], lag_params)
            C.λ₁_quad_values[d,rx] = sum(C.λ₁_x_coes[d,:] .* mλ_x[:,rx])
        end

        for rt in 1:RT
            C.μ₀_quad_values[d,rt] = sum(C.μ₀_t_coes[d,:] .* mμ_t[:,rt]) 
            C.μ₁_quad_values[d,rt] = sum(C.μ₁_t_coes[d,:] .* mμ_t[:,rt])
        end
    end
    @show maximum(abs.(C.λ₁_quad_values[1,:] .- exact_v.(h, xspan[1] .+ x_domain .* x_quad_nodes)))
    @show maximum(abs.(0.25 .* C.bc_wx₀_quad_values .+ C.μ₀_quad_values))
    @show maximum(abs.(0.25 .* C.bc_wx₁_quad_values .+ C.μ₁_quad_values))

    # @show x 
    # @infiltrate
end

# function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{IT}) where {ST, T, MVT <: BSplineDirichlet,LT <: BSplineDirichlet,BT,IT <: Galerkin_Bspline_Integrator{T, MVT, LT, BT},}
#     local D = int.problem.D 
#     local RT = int.method.RT
#     local RX = int.method.RX
#     local quad_b = int.method.grid_weights
#     local C = cache(int,ST)
#     local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
#     local brx = int.method.spatial_quadrature.weights
#     local brt = int.method.time_quadrature.weights
#     local mλ_x = int.method.mλ_x
#     local mμ_t = int.method.mμ_t
#     local S = int.method.basis.S
#     local u_coll_mat = int.method.u_collocation_mat
#     local v_coll_mat = int.method.v_collocation_mat
#     local w_coll_mat = int.method.w_collocation_mat
#     local ut₀_basis_quad_values = int.method.ut₀_basis_quad_values
#     local ut₁_basis_quad_values = int.method.ut₁_basis_quad_values
#     local ux₀_basis_quad_values = int.method.ux₀_basis_quad_values
#     local ux₁_basis_quad_values = int.method.ux₁_basis_quad_values

#     current_idx = 1
#     for d in 1:D 
#         for p in 1:S
#             z = zero(ST)
#             for rt in 1:RT
#                 for rx in 1:RX
#                     z +=  quad_b[rt,rx] * 
#                         ( x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx] * u_coll_mat[p, rt, rx]
#                         + x_domain                 * C.∂L∂V_quad_values[d,rt,rx] * v_coll_mat[p, rt, rx]
#                         + x_domain * timestep(int) * C.∂L∂W_quad_values[d,rt,rx] * w_coll_mat[p, rt, rx])
#                 end
#             end
#             for rx in 1:RX
#                 z+= x_domain * brx[rx] * (C.λ₀_quad_values[d,rx] * ut₀_basis_quad_values[p,rx] - C.λ₁_quad_values[d,rx] * ut₁_basis_quad_values[p,rx])
#             end
#             for rt in 1:RT
#                 z+= timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * ux₀_basis_quad_values[p,rt] - C.μ₁_quad_values[d,rt] * ux₁_basis_quad_values[p,rt])
#             end
#             b[current_idx] = z 
#             current_idx += 1
#         end
#     end
#     # @infiltrate 
#     @assert current_idx == S + 1 "Wrong indexing in residual computation"

#     for d in 1:D
#         for i in 1:k_λ_x
#             z = zero(ST)
#             for rx in 1:RX
#                 z += x_domain * brx[rx] * mλ_x[i,rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
#             end
#             b[S + (d - 1) * RX + i] = z
#         end
#     end

#     for d in 1:D
#         for i in 1:k_μ_t
#             z = zero(ST)
#             for rt in 1:RT
#                 z += timestep(int) *brt[rt] * mμ_t[i,rt] *(C.ux₀_quad_values[d,rt] - C.boundary_condition_x₀[d,rt])
#             end
#             b[S + D * RX + (d - 1) * RT + i] = z
#         end
#     end

#     for d in 1:D
#         for i in 1:k_μ_t
#             z = zero(ST)
#             for rt in 1:RT
#                 z += timestep(int) *brt[rt] * mμ_t[i,rt] *(C.boundary_condition_x₁[d,rt] - C.ux₁_quad_values[d,rt])
#             end
#             b[S + D * RX + D * RT + (d - 1) * RT + i] = z
#         end
#     end
#     # @infiltrate
# end

function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{IT}) where {ST, T, MVT <: Lagrange,LT <: Lagrange,BT,IT <: Galerkin_Bspline_Integrator{T, MVT, LT, BT}}
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
    local mλ_x = int.method.mλ_x
    local mμ_t = int.method.mμ_t
    local ut₀_basis_quad_values = int.method.ut₀_basis_quad_values
    local ut₁_basis_quad_values = int.method.ut₁_basis_quad_values
    local ux₀_basis_quad_values = int.method.ux₀_basis_quad_values
    local ux₁_basis_quad_values = int.method.ux₁_basis_quad_values

    current_idx = 1
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
            for rt in 1:RT
                z_bd += timestep(int)* brt[rt] * (C.μ₀_quad_values[d,rt] * ux₀_basis_quad_values[p,rt] - C.μ₁_quad_values[d,rt] * ux₁_basis_quad_values[p,rt])
            end
            # println(z_bd)
            b[current_idx] = -(z_in + z_bd) # TODO: check the sign
            current_idx += 1
        end
    end

    @assert current_idx == S + 1 "Wrong indexing in residual computation"

    for d in 1:D
        for p in 1:k_λ_x
            z = zero(ST)
            for rx in 1:RX
                z += x_domain * brx[rx] * mλ_x[p,rx] * (C.ut₀_quad_values[d, rx] - C.init_condition_t₀[d, rx])
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
    # @infiltrate
    @show b
end

function update!(sol, int::PDEIntegrator{<:Galerkin_Bspline_Integrator})
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

    # copy internal variables from cache to solution
    # sol.internal.x[] .= cache(int).x 
    # println("In the end of update! function, time = ", sol_struct.t)
end

