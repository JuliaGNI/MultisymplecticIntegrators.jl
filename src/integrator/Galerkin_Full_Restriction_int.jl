struct Galerkin_Full_Restriction_Bspline_Integrator{BT<:AbstractPDEBasis} <: PDEMethod
    basis::BT
    N_t_interval::Int
    N_x_interval::Int

    time_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RT::Int # Number of quadrature points in time

    spatial_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RX::Int # Number of quadrature points in spatial dimension, for simplicity, set the same for all dimensions

    grid_matrix::Matrix{Vector{Float64}} # Quadrature grid points: [[t1,x1], [t1,x2], ]
    grid_weights::Matrix{Float64} # Quadrature weights

    u_collocation_indices::Array{Int,3}
    u_collocation_values::Array{Float64,3}
    v_collocation_indices::Array{Int,3}
    v_collocation_values::Array{Float64,3}
    w_collocation_indices::Array{Int,3}
    w_collocation_values::Array{Float64,3}

    ut₀_basis_quad_indices::Matrix{Int}
    ut₀_basis_quad_values::Matrix{Float64}
    ut₁_basis_quad_indices::Matrix{Int}
    ut₁_basis_quad_values::Matrix{Float64}
    vt₀_basis_quad_indices::Matrix{Int}
    vt₀_basis_quad_values::Matrix{Float64}
    vt₁_basis_quad_indices::Matrix{Int}
    vt₁_basis_quad_values::Matrix{Float64}
    wt₀_basis_quad_indices::Matrix{Int}
    wt₀_basis_quad_values::Matrix{Float64}
    wt₁_basis_quad_indices::Matrix{Int}
    wt₁_basis_quad_values::Matrix{Float64}

    ux₀_basis_quad_indices::Matrix{Int}
    ux₀_basis_quad_values::Matrix{Float64}
    ux₁_basis_quad_indices::Matrix{Int}
    ux₁_basis_quad_values::Matrix{Float64}
    vx₀_basis_quad_indices::Matrix{Int}
    vx₀_basis_quad_values::Matrix{Float64}
    vx₁_basis_quad_indices::Matrix{Int}
    vx₁_basis_quad_values::Matrix{Float64}
    wx₀_basis_quad_indices::Matrix{Int}
    wx₀_basis_quad_values::Matrix{Float64}
    wx₁_basis_quad_indices::Matrix{Int}
    wx₁_basis_quad_values::Matrix{Float64}

    show_status::Bool
    function Galerkin_Full_Restriction_Bspline_Integrator(basis; RT_per_interval::Int = 4,RX_per_interval::Int = 4,xspan::Tuple = (0.,1.0),
        show_status = false)

        # The quadrature nodes in [0.0,1.0]
        N_t_interval = length(unique(basis.ts)) - 1
        N_x_interval = length(unique(basis.xs)) - 1
        t_quadrature = composite_quadrature(N_t_interval ,RT_per_interval)
        x_quadrature = composite_quadrature(N_x_interval ,RX_per_interval)

        R_list = [RT_per_interval,RX_per_interval]
        grid_matrix, grid_weights = construct_quadrature_grid(R_list,[N_t_interval, N_x_interval])

        x0 = xspan[1]
        x_domain = xspan[2] - xspan[1]
        grid_matrix = [collect(grid_matrix[i, j]) for i in axes(grid_matrix, 1), j in axes(grid_matrix, 2)]

        @inbounds for k in eachindex(grid_matrix)
            t, xhat = grid_matrix[k]
            grid_matrix[k][2] = x0 + x_domain * xhat
        end

        S = basis.S
        RT = length(t_quadrature.nodes)
        RX = length(x_quadrature.nodes)

        active_size = BSplineKit.order(basis.Basis_t) * BSplineKit.order(basis.Basis_x)
        function fill_active!(indices_array, values_array, active_indices, active_values, q...)
            for n in eachindex(active_indices)
                indices_array[n, q...] = active_indices[n]
                values_array[n, q...] = active_values[n]
            end
        end

        u_collocation_indices = zeros(Int, active_size, RT, RX)
        u_collocation_values = zeros(active_size, RT, RX)
        v_collocation_indices = zeros(Int, active_size, RT, RX)
        v_collocation_values = zeros(active_size, RT, RX)
        w_collocation_indices = zeros(Int, active_size, RT, RX)
        w_collocation_values = zeros(active_size, RT, RX)
        for rt in 1:RT
            for rx in 1:RX
                active_indices, active_values = spline2D_active_coeff_derivatives((basis.Basis_t, basis.Basis_x), grid_matrix[rt,rx])
                fill_active!(u_collocation_indices, u_collocation_values, active_indices, active_values, rt, rx)
                active_indices, active_values = spline2D_active_coeff_derivatives_time((basis.Basis_t, basis.Basis_x), grid_matrix[rt,rx])
                fill_active!(v_collocation_indices, v_collocation_values, active_indices, active_values, rt, rx)
                active_indices, active_values = spline2D_active_coeff_derivatives_space((basis.Basis_t, basis.Basis_x), grid_matrix[rt,rx])
                fill_active!(w_collocation_indices, w_collocation_values, active_indices, active_values, rt, rx)
            end
        end

        ut₀_basis_quad_indices = zeros(Int, active_size,RX)
        ut₀_basis_quad_values = zeros(active_size,RX)
        ut₁_basis_quad_indices = zeros(Int, active_size,RX)
        ut₁_basis_quad_values = zeros(active_size,RX)
        vt₀_basis_quad_indices = zeros(Int, active_size,RX)
        vt₀_basis_quad_values = zeros(active_size,RX)
        vt₁_basis_quad_indices = zeros(Int, active_size,RX)
        vt₁_basis_quad_values = zeros(active_size,RX)
        wt₀_basis_quad_indices = zeros(Int, active_size,RX)
        wt₀_basis_quad_values = zeros(active_size,RX)
        wt₁_basis_quad_indices = zeros(Int, active_size,RX)
        wt₁_basis_quad_values = zeros(active_size,RX)

        ux₀_basis_quad_indices = zeros(Int, active_size,RT)
        ux₀_basis_quad_values = zeros(active_size,RT)
        ux₁_basis_quad_indices = zeros(Int, active_size,RT)
        ux₁_basis_quad_values = zeros(active_size,RT)
        vx₀_basis_quad_indices = zeros(Int, active_size,RT)
        vx₀_basis_quad_values = zeros(active_size,RT)
        vx₁_basis_quad_indices = zeros(Int, active_size,RT)
        vx₁_basis_quad_values = zeros(active_size,RT)
        wx₀_basis_quad_indices = zeros(Int, active_size,RT)
        wx₀_basis_quad_values = zeros(active_size,RT)
        wx₁_basis_quad_indices = zeros(Int, active_size,RT)
        wx₁_basis_quad_values = zeros(active_size,RT)

        for rx in 1:RX
            xx = xspan[1] .+ (xspan[2] - xspan[1]) .* x_quadrature.nodes[rx]
            active_indices, active_values = spline2D_active_coeff_derivatives((basis.Basis_t, basis.Basis_x),(0.0 ,xx))
            fill_active!(ut₀_basis_quad_indices, ut₀_basis_quad_values, active_indices, active_values, rx)
            active_indices, active_values = spline2D_active_coeff_derivatives((basis.Basis_t, basis.Basis_x),(1.0 ,xx))
            fill_active!(ut₁_basis_quad_indices, ut₁_basis_quad_values, active_indices, active_values, rx)
            active_indices, active_values = spline2D_active_coeff_derivatives_time((basis.Basis_t, basis.Basis_x),(0.0 ,xx))
            fill_active!(vt₀_basis_quad_indices, vt₀_basis_quad_values, active_indices, active_values, rx)
            active_indices, active_values = spline2D_active_coeff_derivatives_time((basis.Basis_t, basis.Basis_x),(1.0 ,xx))
            fill_active!(vt₁_basis_quad_indices, vt₁_basis_quad_values, active_indices, active_values, rx)
            active_indices, active_values = spline2D_active_coeff_derivatives_space((basis.Basis_t, basis.Basis_x),(0.0 ,xx))
            fill_active!(wt₀_basis_quad_indices, wt₀_basis_quad_values, active_indices, active_values, rx)
            active_indices, active_values = spline2D_active_coeff_derivatives_space((basis.Basis_t, basis.Basis_x),(1.0 ,xx))
            fill_active!(wt₁_basis_quad_indices, wt₁_basis_quad_values, active_indices, active_values, rx)
        end

        for i in 1:RT
            tt = t_quadrature.nodes[i]
            active_indices, active_values = spline2D_active_coeff_derivatives((basis.Basis_t, basis.Basis_x),(tt,xspan[1]))
            fill_active!(ux₀_basis_quad_indices, ux₀_basis_quad_values, active_indices, active_values, i)
            active_indices, active_values = spline2D_active_coeff_derivatives((basis.Basis_t, basis.Basis_x),(tt,xspan[2]))
            fill_active!(ux₁_basis_quad_indices, ux₁_basis_quad_values, active_indices, active_values, i)
            active_indices, active_values = spline2D_active_coeff_derivatives_time((basis.Basis_t, basis.Basis_x),(tt,xspan[1]))
            fill_active!(vx₀_basis_quad_indices, vx₀_basis_quad_values, active_indices, active_values, i)
            active_indices, active_values = spline2D_active_coeff_derivatives_time((basis.Basis_t, basis.Basis_x),(tt,xspan[2]))
            fill_active!(vx₁_basis_quad_indices, vx₁_basis_quad_values, active_indices, active_values, i)
            active_indices, active_values = spline2D_active_coeff_derivatives_space((basis.Basis_t, basis.Basis_x),(tt,xspan[1]))
            fill_active!(wx₀_basis_quad_indices, wx₀_basis_quad_values, active_indices, active_values, i)
            active_indices, active_values = spline2D_active_coeff_derivatives_space((basis.Basis_t, basis.Basis_x),(tt,xspan[2]))
            fill_active!(wx₁_basis_quad_indices, wx₁_basis_quad_values, active_indices, active_values, i)
        end


        new{typeof(basis)}(basis,N_t_interval,N_x_interval,
            t_quadrature, RT,
            x_quadrature, RX,
            grid_matrix, grid_weights,
            u_collocation_indices, u_collocation_values,
            v_collocation_indices, v_collocation_values,
            w_collocation_indices, w_collocation_values,
            ut₀_basis_quad_indices, ut₀_basis_quad_values,
            ut₁_basis_quad_indices, ut₁_basis_quad_values,
            vt₀_basis_quad_indices, vt₀_basis_quad_values,
            vt₁_basis_quad_indices, vt₁_basis_quad_values,
            wt₀_basis_quad_indices, wt₀_basis_quad_values,
            wt₁_basis_quad_indices, wt₁_basis_quad_values,
            ux₀_basis_quad_indices, ux₀_basis_quad_values,
            ux₁_basis_quad_indices, ux₁_basis_quad_values,
            vx₀_basis_quad_indices, vx₀_basis_quad_values,
            vx₁_basis_quad_indices, vx₁_basis_quad_values,
            wx₀_basis_quad_indices, wx₀_basis_quad_values,
            wx₁_basis_quad_indices, wx₁_basis_quad_values,
            show_status
            )
    end
end


function Base.show(io::IO, method::Galerkin_Full_Restriction_Bspline_Integrator)
    print(io, "\n Spline Integrator without Multipliers, Only for Zero Dirichlet Boundary Condition with:\n")
    print(io, "   Basis Order in each dimension k:$(method.basis.k) \n")
    print(io, "   Nbasis_x: $(basis.Nbasis_x), Nbasis_t: $(basis.Nbasis_t), total DOFs:$(basis.S) \n")
    print(io, "   scaled Breaks_x: $(basis.xs) \n")
    print(io, "   scaled Breaks_t: $(basis.ts) \n")

end

default_solver(::Galerkin_Full_Restriction_Bspline_Integrator) = NewtonMethod()

struct Galerkin_Full_Restriction_Bspline_IntegratorCache{ST,RT,RX,D,S,Nx,Nt} <: PDEIntegratorCache{ST,D}
    """
    RT = number of quadrature points in time
    RX = number of quadrature points in space
    D = dimension of output of u,v,w, i.e. scaler value function (D = 1) or vector function
    DX = dimension of input of u,v,w, i.e. number of spatial dimensions
    NP = number of parameters in the expression
    """
    x::Vector{ST}
    known_dofs::Vector{ST}

    p₀_quad_values::Matrix{ST}

    u_quad_values::Array{ST}
    v_quad_values::Array{ST}
    w_quad_values::Array{ST}

    ∂L∂U_quad_values::Array{ST}
    ∂L∂V_quad_values::Array{ST}
    ∂L∂W_quad_values::Array{ST}

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

    function Galerkin_Full_Restriction_Bspline_IntegratorCache{ST,RT,RX,D,S,Nx,Nt}() where {ST,RT,RX,D,S,Nx,Nt}
        # x = zeros(ST, NP + 2 * D * RX + 2* D * DX * RT ) # TODO: how to deal with RX being a vector/
        x = zeros(ST,Nx*(Nt-1)) # params, λ_x_coes,μ₀_t_coes,μ₁_t_coes
        known_dofs = zeros(ST,Nx)

        p₀_quad_values = zeros(ST, D, RX)

        # TODO:consider when DX is a vector
        # x = zeros(ST,S)
        u_quad_values = zeros(ST, D, RT, RX)
        v_quad_values = zeros(ST, D, RT, RX)
        w_quad_values = zeros(ST, D, RT, RX)

        ∂L∂U_quad_values = zeros(ST, D, RT, RX)
        ∂L∂V_quad_values = zeros(ST, D, RT, RX)
        ∂L∂W_quad_values = zeros(ST, D, RT, RX)

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
            known_dofs,p₀_quad_values,
            u_quad_values, v_quad_values, w_quad_values,
            ∂L∂U_quad_values, ∂L∂V_quad_values, ∂L∂W_quad_values,
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

nlsolution(cache::Galerkin_Full_Restriction_Bspline_IntegratorCache) = cache.x

function Cache{ST}(problem::LPDEProblem, int::Galerkin_Full_Restriction_Bspline_Integrator; kwargs...) where {ST}
    Galerkin_Full_Restriction_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.S,int.basis.Nbasis_x,int.basis.Nbasis_t}(; kwargs...)
end

#{ST,RT,RX,D,NP}(NP) where {ST,RT,RX,D,NP}
@inline CacheType(ST, problem::LPDEProblem, int::Galerkin_Full_Restriction_Bspline_Integrator) = Galerkin_Full_Restriction_Bspline_IntegratorCache{ST,int.RT,int.RX,problem.D,int.basis.S,int.basis.Nbasis_x,int.basis.Nbasis_t}

function copy_internal_variables!(C::Galerkin_Full_Restriction_Bspline_IntegratorCache, solstep::SolutionStep)
    carried_internal = internal(solstep)
    haskey(carried_internal, :ut₁_quad_values) && copyto!(C.ics_ut₀_quad_values, carried_internal.ut₁_quad_values)
    haskey(carried_internal, :vt₁_quad_values) && copyto!(C.ics_vt₀_quad_values, carried_internal.vt₁_quad_values)
    haskey(carried_internal, :wt₁_quad_values) && copyto!(C.ics_wt₀_quad_values, carried_internal.wt₁_quad_values)
    haskey(carried_internal, :known_dofs) && copyto!(C.known_dofs, carried_internal.known_dofs)
    haskey(carried_internal, :p₀_quad_values) && copyto!(C.p₀_quad_values, carried_internal.p₀_quad_values)
    return nothing
end

function prior_initial_guess!(C,sol,int::PDEIntegrator{<:Galerkin_Full_Restriction_Bspline_Integrator})
    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w

    local x_collocation_points = int.method.basis.collocation_points_x
    local t_collocation_points = int.method.basis.collocation_points_t
    local Cx = int.method.basis.collocation_matrix_x
    local Ct = int.method.basis.collocation_matrix_t

    local Bx = int.method.basis.Basis_x
    local Bt = int.method.basis.Basis_t
    local Nx = int.method.basis.Nbasis_x
    local Nt = int.method.basis.Nbasis_t

    local S = int.method.basis.S
    local h = timestep(int)
    local a = int.problem.xspan[1]
    local b = int.problem.xspan[2]
    local grid_matrix = int.method.grid_matrix
    local RT = int.method.RT
    local RX = int.method.RX
    local show_status = int.method.show_status
    local tn = sol.t - timestep(int)

    udata = exact_u.(tn .+ h .* t_collocation_points, x_collocation_points') # fdata[i,j] = exact_u(h .* ts[i], xs[j])

    # 2D B-spline coefficients (output)
    coefs = similar(udata)

    # Solve linear systems
    for j ∈ eachindex(x_collocation_points)
        @views ldiv!(coefs[:, j], Ct, udata[:, j])
    end
    for i ∈ eachindex(t_collocation_points)
        @views ldiv!(Cx, coefs[i, :])
    end

    if iszero(tn)
        C.known_dofs[:] = coefs[1,:]
    else
        C.known_dofs[:] = internal(sol).known_dofs
    end
    C.x[:] = vec(coefs[2:end, :])

    if show_status
        vdata = exact_v.(tn .+ h .* t_collocation_points, x_collocation_points')
        wdata = exact_w.(tn .+ h .* t_collocation_points, x_collocation_points')

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

                u_quad_value[ti,xi] = exact_u(tn + h * tt, xx)
                v_quad_value[ti,xi] = exact_v(tn + h * tt, xx)
                w_quad_value[ti,xi] = exact_w(tn + h * tt, xx)
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

post_initial_guess!(internal_coes, C,sol_struct,int::PDEIntegrator{<:Galerkin_Full_Restriction_Bspline_Integrator},int_method::Galerkin_Full_Restriction_Bspline_Integrator{BT}) where {BT} = nothing


function components!(x::AbstractVector{ST}, sol, params, int::PDEIntegrator{<:Galerkin_Full_Restriction_Bspline_Integrator}) where {ST}
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

    local exact_u = int.problem.exact_u
    local exact_v = int.problem.exact_v
    local exact_w = int.problem.exact_w
    local show_status = int.method.show_status
    local tn = sol.t - h

    local Nt = int.method.basis.Nbasis_t
    local Nx = int.method.basis.Nbasis_x

    # interior values at quadrature points
    full_mat = zeros(ST,Nt,Nx)
    full_mat[1,:] = cache(int).known_dofs
    full_mat[2:end, :] .= reshape(x, Nt-1, Nx)
    full_coefs = vec(full_mat)

    function active_dot(indices, values, q...)
        z = zero(ST)
        for n in axes(indices, 1)
            p = indices[n, q...]
            iszero(p) && break
            z += full_coefs[p] * values[n, q...]
        end
        return z
    end

    for d in 1:D
        for i in 1:RT
            for j in 1:RX
                C.u_quad_values[d, i, j] = active_dot(
                    int.method.u_collocation_indices, int.method.u_collocation_values, i, j)
                C.v_quad_values[d, i, j] = active_dot(
                    int.method.v_collocation_indices, int.method.v_collocation_values, i, j) / h
                C.w_quad_values[d, i, j] = active_dot(
                    int.method.w_collocation_indices, int.method.w_collocation_values, i, j)
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
            C.ut₀_quad_values[d,j] = active_dot(int.method.ut₀_basis_quad_indices, int.method.ut₀_basis_quad_values, j)
            C.ut₁_quad_values[d,j] = active_dot(int.method.ut₁_basis_quad_indices, int.method.ut₁_basis_quad_values, j)
            C.vt₀_quad_values[d,j] = active_dot(int.method.vt₀_basis_quad_indices, int.method.vt₀_basis_quad_values, j) / h
            C.vt₁_quad_values[d,j] = active_dot(int.method.vt₁_basis_quad_indices, int.method.vt₁_basis_quad_values, j) / h
            C.wt₀_quad_values[d,j] = active_dot(int.method.wt₀_basis_quad_indices, int.method.wt₀_basis_quad_values, j)
            C.wt₁_quad_values[d,j] = active_dot(int.method.wt₁_basis_quad_indices, int.method.wt₁_basis_quad_values, j)
        end

        for i in 1:RT
            C.ux₀_quad_values[d,i] = active_dot(int.method.ux₀_basis_quad_indices, int.method.ux₀_basis_quad_values, i)
            C.ux₁_quad_values[d,i] = active_dot(int.method.ux₁_basis_quad_indices, int.method.ux₁_basis_quad_values, i)
            C.vx₀_quad_values[d,i] = active_dot(int.method.vx₀_basis_quad_indices, int.method.vx₀_basis_quad_values, i) / h
            C.vx₁_quad_values[d,i] = active_dot(int.method.vx₁_basis_quad_indices, int.method.vx₁_basis_quad_values, i) / h
            C.wx₀_quad_values[d,i] = active_dot(int.method.wx₀_basis_quad_indices, int.method.wx₀_basis_quad_values, i)
            C.wx₁_quad_values[d,i] = active_dot(int.method.wx₁_basis_quad_indices, int.method.wx₁_basis_quad_values, i)
        end
    end

    # initial guess for the coefficients of Lagrangian multipliers
    # cache(int).flag_done_initial_guess[1] == 0.0 ? post_initial_guess!(x, cache(int),sol,int,int.method) : nothing


    for d in 1:D
        for rx in 1:RX
            C.p₀_quad_values[d,rx] = ∂L∂V[d](C.ics_ut₀_quad_values[d,rx], C.ics_vt₀_quad_values[d,rx],C.ics_wt₀_quad_values[d,rx], lag_params)
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
                    u_truth_mat[d, i, j] = exact_u(tn + h * t_quad_nodes[i], xspan[1] + x_domain * x_quad_nodes[j])
                    v_truth_mat[d, i, j] = exact_v(tn + h * t_quad_nodes[i], xspan[1] + x_domain * x_quad_nodes[j])
                    w_truth_mat[d, i, j] = exact_w(tn + h * t_quad_nodes[i], xspan[1] + x_domain * x_quad_nodes[j])
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
                xq = xspan[1] + x_domain * x_quad_nodes[j]
                ut₀_quad_values_truth[d,j] = exact_u(tn, xq)
                ut₁_quad_values_truth[d,j] = exact_u(sol.t, xq)
                vt₀_quad_values_truth[d,j] = exact_v(tn, xq)
                vt₁_quad_values_truth[d,j] = exact_v(sol.t, xq)
                wt₀_quad_values_truth[d,j] = exact_w(tn, xq)
                wt₁_quad_values_truth[d,j] = exact_w(sol.t, xq)
            end

            for i in 1:RT
                tq = tn + h * t_quad_nodes[i]
                ux₀_quad_values_truth[d,i] = exact_u(tq, xspan[1])
                ux₁_quad_values_truth[d,i] = exact_u(tq, xspan[2])
                vx₀_quad_values_truth[d,i] = exact_v(tq, xspan[1])
                vx₁_quad_values_truth[d,i] = exact_v(tq, xspan[2])
                wx₀_quad_values_truth[d,i] = exact_w(tq, xspan[1])
                wx₁_quad_values_truth[d,i] = exact_w(tq, xspan[2])
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

        p₀_bottom_momentum = similar(C.p₀_quad_values)
        p₀_exact_momentum = similar(C.p₀_quad_values)
        for d in 1:D
            for rx in 1:RX
                xq = xspan[1] + x_domain * x_quad_nodes[rx]
                p₀_bottom_momentum[d, rx] = ∂L∂V[d](
                    C.ics_ut₀_quad_values[d, rx],
                    C.ics_vt₀_quad_values[d, rx],
                    C.ics_wt₀_quad_values[d, rx],
                    lag_params,
                )
                p₀_exact_momentum[d, rx] = exact_v(tn, xq)
            end
        end

        @show maximum(abs.(C.p₀_quad_values .- p₀_bottom_momentum))
        @show maximum(abs.(C.p₀_quad_values .- p₀_exact_momentum))
        @show maximum(abs.(C.ut₀_quad_values .- C.init_condition_t₀))
        @infiltrate
    end
end


function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{<: Galerkin_Full_Restriction_Bspline_Integrator}) where {ST}
    local D = int.problem.D
    local RT = int.method.RT
    local RX = int.method.RX

    local quad_b = int.method.grid_weights
    local C = cache(int,ST)
    local x_domain = int.problem.xspan[2] - int.problem.xspan[1]
    local brx = int.method.spatial_quadrature.weights
    local brt = int.method.time_quadrature.weights
    local u_collocation_indices = int.method.u_collocation_indices
    local u_collocation_values = int.method.u_collocation_values
    local v_collocation_indices = int.method.v_collocation_indices
    local v_collocation_values = int.method.v_collocation_values
    local w_collocation_indices = int.method.w_collocation_indices
    local w_collocation_values = int.method.w_collocation_values
    local S = int.method.basis.S
    local ut₀_basis_quad_indices = int.method.ut₀_basis_quad_indices
    local ut₀_basis_quad_values = int.method.ut₀_basis_quad_values
    local show_status = int.method.show_status
    local Nx = int.method.basis.Nbasis_x
    local Nt = int.method.basis.Nbasis_t

    fill!(b, zero(ST))

    for d in 1:D
        for rt in 1:RT
            for rx in 1:RX
                zU = quad_b[rt,rx] * x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx]
                zV = quad_b[rt,rx] * x_domain * C.∂L∂V_quad_values[d,rt,rx]
                zW = quad_b[rt,rx] * x_domain * timestep(int) * C.∂L∂W_quad_values[d,rt,rx]
                for n in axes(u_collocation_indices, 1)
                    p = u_collocation_indices[n, rt, rx]
                    iszero(p) && break
                    i = p - (div(p - 1, Nt) * Nt)
                    i > Nt - 1 && continue
                    b_idx = div(p - 1, Nt) * (Nt - 1) + i
                    b[b_idx] += zU * u_collocation_values[n, rt, rx]
                end
                for n in axes(v_collocation_indices, 1)
                    p = v_collocation_indices[n, rt, rx]
                    iszero(p) && break
                    i = p - (div(p - 1, Nt) * Nt)
                    i > Nt - 1 && continue
                    b_idx = div(p - 1, Nt) * (Nt - 1) + i
                    b[b_idx] += zV * v_collocation_values[n, rt, rx]
                end
                for n in axes(w_collocation_indices, 1)
                    p = w_collocation_indices[n, rt, rx]
                    iszero(p) && break
                    i = p - (div(p - 1, Nt) * Nt)
                    i > Nt - 1 && continue
                    b_idx = div(p - 1, Nt) * (Nt - 1) + i
                    b[b_idx] += zW * w_collocation_values[n, rt, rx]
                end
            end
        end

        for rx in 1:RX
            z_p0 = x_domain * brx[rx] * C.p₀_quad_values[d, rx]
            for n in axes(ut₀_basis_quad_indices, 1)
                p = ut₀_basis_quad_indices[n, rx]
                iszero(p) && break
                i = p - (div(p - 1, Nt) * Nt)
                i == 1 || continue
                b_idx = div(p - 1, Nt) * (Nt - 1) + i
                b[b_idx] += z_p0 * ut₀_basis_quad_values[n, rx]
            end
        end
    end


    # for d in 1:D
    #     for p in Nx+1 : (Nt-1)*Nx
    #         z_in = zero(ST)
    #         for rt in 1:RT
    #             for rx in 1:RX
    #                 z_in +=  quad_b[rt,rx] *
    #                     ( x_domain * timestep(int) * C.∂L∂U_quad_values[d,rt,rx] * u_coll_mat[p, rt, rx]
    #                     + x_domain                 * C.∂L∂V_quad_values[d,rt,rx] * v_coll_mat[p, rt, rx]
    #                     + x_domain * timestep(int) * C.∂L∂W_quad_values[d,rt,rx] * w_coll_mat[p, rt, rx])
    #             end
    #         end
    #         # println(z_bd)
    #         b[p] = z_in
    #     end
    # end

    if show_status
        @show b
    end
end

function update!(sol, int::PDEIntegrator{<:Galerkin_Full_Restriction_Bspline_Integrator})
    local D = int.problem.D
    local xspan = int.problem.xspan
    local xstep = int.problem.xstep
    local x = nlsolution(int)
    local S = int.method.basis.S
    local Basis_t = int.method.basis.Basis_t
    local Basis_x = int.method.basis.Basis_x
    local h = timestep(int)
    local Nx = int.method.basis.Nbasis_x
    local Nt = int.method.basis.Nbasis_t
    local RX = int.method.RX
    local lag_params = int.problem.lagrangian_system.params

    x_nodes = collect(xspan[1]:xstep:xspan[2])
    full_mat = zeros(Nt, Nx)
    full_mat[1, :] = cache(int).known_dofs
    full_mat[2:end, :] .= reshape(x, Nt-1, Nx)

    for d in 1:D
        for i in eachindex(x_nodes)
            sol.u[i] = eval_spline2D(full_mat, (Basis_t, Basis_x), (1.0, x_nodes[i]))
            sol.v[i] = eval_spline2D_dt(full_mat, (Basis_t, Basis_x), (1.0, x_nodes[i])) /h
            sol.w[i] = eval_spline2D_dx(full_mat, (Basis_t, Basis_x), (1.0, x_nodes[i]))
        end
    end

    # cache(int).known_dofs[:] = full_mat[end,:]
    # for d in 1:D
    #     for rx in 1:RX
    #         cache(int).p₀_quad_values = ∂L∂V[d](u_end, v_end, w_end, lag_params)
    #     end
    # end
end

function internal_variables(method::Galerkin_Full_Restriction_Bspline_Integrator, problem::LPDEProblem)
    local D = problem.D
    local RX = method.RX
    local Nx = method.basis.Nbasis_x
    ut₁_quad_values = zeros(D,RX)
    vt₁_quad_values = zeros(D,RX)
    wt₁_quad_values = zeros(D,RX)

    known_dofs = zeros(Nx)
    p₀_quad_values = zeros(D, RX)

    return (ut₁_quad_values = ut₁_quad_values,
        vt₁_quad_values = vt₁_quad_values,
        wt₁_quad_values = wt₁_quad_values,
        known_dofs = known_dofs,
        p₀_quad_values = p₀_quad_values
        )
end

function copy_internal_variables!(solstep::SolutionStep, C::Galerkin_Full_Restriction_Bspline_IntegratorCache)
    target_internal = internal(solstep)
    haskey(target_internal, :ut₁_quad_values) && copyto!(target_internal.ut₁_quad_values, C.ut₁_quad_values)
    haskey(target_internal, :vt₁_quad_values) && copyto!(target_internal.vt₁_quad_values, C.vt₁_quad_values)
    haskey(target_internal, :wt₁_quad_values) && copyto!(target_internal.wt₁_quad_values, C.wt₁_quad_values)
    haskey(target_internal, :p₀_quad_values) && copyto!(target_internal.p₀_quad_values, C.p₀_quad_values)

    if haskey(target_internal, :known_dofs)
        Nx = length(C.known_dofs)
        Nt = length(C.x) ÷ Nx + 1
        full_mat = Matrix{eltype(C.x)}(undef, Nt, Nx)
        full_mat[1, :] .= C.known_dofs
        full_mat[2:end, :] .= reshape(C.x, Nt - 1, Nx)
        copyto!(target_internal.known_dofs, view(full_mat, Nt, :))
    end
    return nothing
end

# function copy_internal_variables!(C::Galerkin_Full_Restriction_Bspline_IntegratorCache,solstep::SolutionStep)
#     # # copy internal variables from internal to cache, e.g. after the first iteration, we can update the initial guess for the Lagrangian multipliers at t = 0 based on the current solution
#     # haskey(C, :ut₁_quad_values) && copyto!(C.ut₁_quad_values, internal(solstep).ut₁_quad_values)
#     # haskey(C, :vt₁_quad_values) && copyto!(C.vt₁_quad_values, internal(solstep).vt₁_quad_values)
#     # haskey(C, :wt₁_quad_values) && copyto!(C.wt₁_quad_values, internal(solstep).wt₁_quad_values)
#     # haskey(C, :known_dofs)      && copyto!(C.known_dofs, internal(solstep).known_dofs)
#     # haskey(C, :p₀_quad_values)  && copyto!(C.p₀_quad_values, internal(solstep).p₀_quad_values)
#     nothing
# end
