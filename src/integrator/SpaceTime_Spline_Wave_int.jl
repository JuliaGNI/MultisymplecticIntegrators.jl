struct SpaceTime_Spline_Wave_Integrator{BT <: AbstractPDEBasis} <: PDEMethod
    basis::BT
    time_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RT::Int
    spatial_quadrature::NamedTuple{
        (:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RX::Int
    grid_matrix::Matrix{Vector{Float64}}
    grid_weights::Matrix{Float64}
    action_matrix::Matrix{Float64}
    system_matrix::Matrix{Float64}
    system_factor::LU{Float64, Matrix{Float64}, Vector{Int64}}
    unknown_indices::Vector{Int}
    equation_indices::Vector{Int}
    bottom_indices::Vector{Int}
    top_indices::Vector{Int}
    eval_matrix::Matrix{Float64}
    eval_derivative_matrix::Matrix{Float64}
    c::Float64
    show_status::Bool

    function SpaceTime_Spline_Wave_Integrator(
            basis::Dirichlet_BSpline2D,
            problem::LPDEProblem;
            c = problem.params.c,
            quadrature_order::Int = basis.k + 2,
            show_status::Bool = false)
        @assert problem.D == 1 "SpaceTime_Spline_Wave_Integrator currently supports scalar 1+1 wave problems."
        @assert hasproperty(problem.params, :c) "The wave speed parameter `c` is required."

        h = problem.timestep
        Nt = basis.Nbasis_t
        Nx = basis.Nbasis_x
        xspan = problem.xspan
        x_domain = xspan[2] - xspan[1]

        time_quadrature = composite_quadrature(length(unique(basis.ts)) - 1, quadrature_order)
        spatial_ref_quadrature = composite_quadrature(length(unique(basis.xs)) - 1, quadrature_order)
        spatial_nodes = xspan[1] .+ x_domain .* spatial_ref_quadrature.nodes
        spatial_weights = x_domain .* spatial_ref_quadrature.weights
        spatial_quadrature = (
            nodes = spatial_ref_quadrature.nodes, weights = spatial_ref_quadrature.weights)
        RT = length(time_quadrature.nodes)
        RX = length(spatial_quadrature.nodes)
        grid_matrix = [Float64[τ, x] for τ in time_quadrature.nodes, x in spatial_nodes]
        grid_weights = [time_quadrature.weights[i] * spatial_quadrature.weights[j]
                        for i in eachindex(time_quadrature.nodes),
        j in eachindex(spatial_quadrature.nodes)]

        Mt = _spline_gram_matrix(basis.Basis_t, time_quadrature.nodes, time_quadrature.weights)
        Kt = _spline_gram_matrix(basis.Basis_t, time_quadrature.nodes,
            time_quadrature.weights; derivative = true)
        Mx = _spline_gram_matrix(basis.Basis_x, spatial_nodes, spatial_weights)
        Kx = _spline_gram_matrix(basis.Basis_x, spatial_nodes, spatial_weights; derivative = true)

        action_matrix = kron(Mx, Kt / h) - (h * c^2) * kron(Kx, Mt)

        bottom_indices = [_st_index(1, b, Nt) for b in 1:Nx]
        top_indices = [_st_index(Nt, b, Nt) for b in 1:Nx]
        unknown_indices = [_st_index(a, b, Nt) for b in 1:Nx for a in 2:Nt]
        equation_indices = [_st_index(a, b, Nt) for b in 1:Nx for a in 1:(Nt - 1)]
        system_matrix = action_matrix[equation_indices, unknown_indices]
        system_factor = lu(system_matrix)

        x_nodes = collect(problem.xspan[1]:problem.xstep:problem.xspan[2])
        eval_matrix = _spline_value_matrix(basis.Basis_x, x_nodes)
        eval_derivative_matrix = _spline_value_matrix(basis.Basis_x, x_nodes; derivative = true)

        new{typeof(basis)}(
            basis,
            time_quadrature, RT,
            spatial_quadrature, RX,
            grid_matrix, grid_weights,
            action_matrix, system_matrix, system_factor,
            unknown_indices, equation_indices, bottom_indices, top_indices,
            eval_matrix, eval_derivative_matrix,
            Float64(c), show_status)
    end
end

_st_index(a::Integer, b::Integer, Nt::Integer) = a + (b - 1) * Nt

function _spline_value_matrix(B, nodes; derivative::Bool = false)
    values = zeros(length(nodes), length(B))
    for (q, x) in enumerate(nodes)
        i, bx = derivative ? B(x, BSplineKit.Derivative(1)) : B(x)
        for δ in eachindex(bx)
            j = i - δ + 1
            if 1 ≤ j ≤ length(B)
                values[q, j] = bx[δ]
            end
        end
    end
    return values
end

function _spline_gram_matrix(B, nodes, weights; derivative::Bool = false)
    V = _spline_value_matrix(B, nodes; derivative = derivative)
    return V' * Diagonal(weights) * V
end

function Base.show(io::IO, method::SpaceTime_Spline_Wave_Integrator)
    print(io, "\n Space-Time Spline Wave Integrator with:\n")
    print(io, "   Basis order k: $(method.basis.k) \n")
    print(io, "   Nbasis_x: $(method.basis.Nbasis_x), Nbasis_t: $(method.basis.Nbasis_t) \n")
    print(io, "   Unknowns per slab: $(length(method.unknown_indices)) \n")
    print(io, "   Quadrature points: RT=$(method.RT), RX=$(method.RX) \n")
    print(io, "   Wave speed c: $(method.c) \n")
    print(io, "   Show status: $(method.show_status) \n")
end

default_solver(::SpaceTime_Spline_Wave_Integrator) = NewtonMethod()
default_options(::SpaceTime_Spline_Wave_Integrator) = (max_iterations = 1, verbosity = 0)

struct SpaceTime_Spline_Wave_IntegratorCache{ST, RT, RX, D, Nx, Nt, NFree} <:
       PDEIntegratorCache{ST, D}
    x::Vector{ST}
    full_coeffs::Vector{ST}
    bottom_dofs::Vector{ST}
    top_dofs::Vector{ST}
    p₀_dofs::Matrix{ST}
    p₁_dofs::Matrix{ST}
    u_quad_values::Array{ST, 3}
    v_quad_values::Array{ST, 3}
    w_quad_values::Array{ST, 3}
    ut₁_quad_values::Matrix{ST}
    vt₁_quad_values::Matrix{ST}
    wt₁_quad_values::Matrix{ST}
    init_condition_t₀::Matrix{ST}
    ics_ut₀_quad_values::Matrix{ST}
    ics_vt₀_quad_values::Matrix{ST}
    ics_wt₀_quad_values::Matrix{ST}
    boundary_condition_x₀::Matrix{ST}
    boundary_condition_x₁::Matrix{ST}
    bc_ux₀_quad_values::Matrix{ST}
    bc_vx₀_quad_values::Matrix{ST}
    bc_wx₀_quad_values::Matrix{ST}
    bc_ux₁_quad_values::Matrix{ST}
    bc_vx₁_quad_values::Matrix{ST}
    bc_wx₁_quad_values::Matrix{ST}
    flag_done_initial_guess::Vector{ST}

    function SpaceTime_Spline_Wave_IntegratorCache{
            ST, RT, RX, D, Nx, Nt, NFree}() where {ST, RT, RX, D, Nx, Nt, NFree}
        new(
            zeros(ST, NFree),
            zeros(ST, Nx * Nt),
            zeros(ST, Nx),
            zeros(ST, Nx),
            zeros(ST, D, Nx),
            zeros(ST, D, Nx),
            zeros(ST, D, RT, RX),
            zeros(ST, D, RT, RX),
            zeros(ST, D, RT, RX),
            zeros(ST, D, RX),
            zeros(ST, D, RX),
            zeros(ST, D, RX),
            zeros(ST, D, RX),
            zeros(ST, D, RX),
            zeros(ST, D, RX),
            zeros(ST, D, RX),
            zeros(ST, D, RT),
            zeros(ST, D, RT),
            zeros(ST, D, RT),
            zeros(ST, D, RT),
            zeros(ST, D, RT),
            zeros(ST, D, RT),
            zeros(ST, D, RT),
            zeros(ST, D, RT),
            zeros(ST, 1)
        )
    end
end

nlsolution(cache::SpaceTime_Spline_Wave_IntegratorCache) = cache.x

function Cache{ST}(problem::LPDEProblem, int::SpaceTime_Spline_Wave_Integrator; kwargs...) where {ST}
    SpaceTime_Spline_Wave_IntegratorCache{ST, int.RT, int.RX, problem.D,
        int.basis.Nbasis_x, int.basis.Nbasis_t, length(int.unknown_indices)}(;
        kwargs...)
end

@inline CacheType(ST, problem::LPDEProblem,
    int::SpaceTime_Spline_Wave_Integrator) = SpaceTime_Spline_Wave_IntegratorCache{
    ST, int.RT, int.RX, problem.D,
    int.basis.Nbasis_x, int.basis.Nbasis_t, length(int.unknown_indices)}

function internal_variables(method::SpaceTime_Spline_Wave_Integrator, problem::LPDEProblem)
    D = problem.D
    RX = method.RX
    Nx = method.basis.Nbasis_x
    return (
        ut₁_quad_values = zeros(D, RX),
        vt₁_quad_values = zeros(D, RX),
        wt₁_quad_values = zeros(D, RX),
        bottom_dofs = zeros(Nx),
        p₁_dofs = zeros(D, Nx)
    )
end

function copy_internal_variables!(C::SpaceTime_Spline_Wave_IntegratorCache, solstep::SolutionStep)
    carried_internal = internal(solstep)
    haskey(carried_internal, :bottom_dofs) &&
        copyto!(C.bottom_dofs, carried_internal.bottom_dofs)
    haskey(carried_internal, :p₁_dofs) && copyto!(C.p₀_dofs, carried_internal.p₁_dofs)
    return nothing
end

function copy_internal_variables!(solstep::SolutionStep, C::SpaceTime_Spline_Wave_IntegratorCache)
    target_internal = internal(solstep)
    haskey(target_internal, :ut₁_quad_values) &&
        copyto!(target_internal.ut₁_quad_values, C.ut₁_quad_values)
    haskey(target_internal, :vt₁_quad_values) &&
        copyto!(target_internal.vt₁_quad_values, C.vt₁_quad_values)
    haskey(target_internal, :wt₁_quad_values) &&
        copyto!(target_internal.wt₁_quad_values, C.wt₁_quad_values)
    haskey(target_internal, :bottom_dofs) &&
        copyto!(target_internal.bottom_dofs, C.top_dofs)
    haskey(target_internal, :p₁_dofs) && copyto!(target_internal.p₁_dofs, C.p₁_dofs)
    return nothing
end

function prior_initial_guess!(C, sol, int::PDEIntegrator{<:SpaceTime_Spline_Wave_Integrator})
    nothing
end
function components!(x::AbstractVector{ST}, sol, params,
        int::PDEIntegrator{<:SpaceTime_Spline_Wave_Integrator}) where {ST}
    nothing
end

function _project_to_spatial_spline!(dofs, values, int::PDEIntegrator{<:SpaceTime_Spline_Wave_Integrator})
    tmp = copy(values)
    ldiv!(dofs, int.method.basis.collocation_matrix_x, tmp)
    return nothing
end

function _project_momentum!(p_dofs, velocity_values, int::PDEIntegrator{<:SpaceTime_Spline_Wave_Integrator})
    Bx = int.method.basis.Basis_x
    nodes = int.method.grid_matrix[1, :]
    weights = int.problem.xspan[2] - int.problem.xspan[1]
    fill!(p_dofs, zero(eltype(p_dofs)))
    for rx in 1:int.method.RX
        x = nodes[rx][2]
        i, bx = Bx(x)
        for δ in eachindex(bx)
            j = i - δ + 1
            if 1 ≤ j ≤ length(Bx)
                p_dofs[j] += weights * int.method.spatial_quadrature.weights[rx] *
                             velocity_values[rx] * bx[δ]
            end
        end
    end
    return nothing
end

function _initialize_bottom_data!(C, sol, int::PDEIntegrator{<:SpaceTime_Spline_Wave_Integrator})
    tn = sol.t - timestep(int)
    if iszero(tn)
        x_collocation = int.method.basis.collocation_points_x
        u0 = int.problem.ics_function(x_collocation).u
        _project_to_spatial_spline!(C.bottom_dofs, u0, int)

        v0 = [int.problem.ics_function(int.method.grid_matrix[1, rx][2]).v
              for rx in 1:int.method.RX]
        _project_momentum!(view(C.p₀_dofs, 1, :), v0, int)
    end
    return nothing
end

function _solve_spacetime_spline_slab!(C, sol, int::PDEIntegrator{<:SpaceTime_Spline_Wave_Integrator})
    _initialize_bottom_data!(C, sol, int)
    A = int.method.action_matrix
    q_known = zeros(eltype(C.x), length(C.full_coeffs))
    q_known[int.method.bottom_indices] .= C.bottom_dofs

    rhs = -(A[int.method.equation_indices, int.method.bottom_indices] * C.bottom_dofs)
    Nt = int.method.basis.Nbasis_t
    for b in 1:int.method.basis.Nbasis_x
        rhs[(b - 1) * (Nt - 1) + 1] -= C.p₀_dofs[1, b]
    end
    C.x .= int.method.system_factor \ rhs

    C.full_coeffs .= q_known
    C.full_coeffs[int.method.unknown_indices] .= C.x
    C.top_dofs .= C.full_coeffs[int.method.top_indices]
    C.p₁_dofs[1, :] .= (A[int.method.top_indices, :] * C.full_coeffs)
    return nothing
end

function _fill_spacetime_spline_values!(C, sol, int::PDEIntegrator{<:SpaceTime_Spline_Wave_Integrator})
    h = timestep(int)
    Nt = int.method.basis.Nbasis_t
    Nx = int.method.basis.Nbasis_x
    coeffs = reshape(C.full_coeffs, Nt, Nx)
    Bt = int.method.basis.Basis_t
    Bx = int.method.basis.Basis_x

    for rt in 1:int.method.RT, rx in 1:int.method.RX

        τ, x = int.method.grid_matrix[rt, rx]
        C.u_quad_values[1, rt, rx] = eval_spline2D(coeffs, (Bt, Bx), (τ, x))
        C.v_quad_values[1, rt, rx] = eval_spline2D_dt(coeffs, (Bt, Bx), (τ, x)) / h
        C.w_quad_values[1, rt, rx] = eval_spline2D_dx(coeffs, (Bt, Bx), (τ, x))
    end

    for rx in 1:int.method.RX
        x = int.method.grid_matrix[1, rx][2]
        C.ut₁_quad_values[1, rx] = eval_spline2D(coeffs, (Bt, Bx), (1.0, x))
        C.vt₁_quad_values[1, rx] = eval_spline2D_dt(coeffs, (Bt, Bx), (1.0, x)) / h
        C.wt₁_quad_values[1, rx] = eval_spline2D_dx(coeffs, (Bt, Bx), (1.0, x))
    end
    return nothing
end

function integrate_step!(sol, history, params, int::PDEIntegrator{<:SpaceTime_Spline_Wave_Integrator})
    C = cache(int)
    _solve_spacetime_spline_slab!(C, sol, int)
    _fill_spacetime_spline_values!(C, sol, int)
    update!(sol, int)
    int.method.show_status && println("Space-time spline max |u| = ", maximum(abs.(sol.u)))
    return nothing
end

function update!(sol, int::PDEIntegrator{<:SpaceTime_Spline_Wave_Integrator})
    C = cache(int)
    h = timestep(int)
    Nt = int.method.basis.Nbasis_t
    Nx = int.method.basis.Nbasis_x
    coeffs = reshape(C.full_coeffs, Nt, Nx)
    top = @view coeffs[Nt, :]

    sol.u .= int.method.eval_matrix * top
    sol.w .= int.method.eval_derivative_matrix * top

    coeffs_dt_top = zeros(eltype(C.x), Nx)
    _, bt_dt = int.method.basis.Basis_t(1.0, BSplineKit.Derivative(1))
    it_dt = length(int.method.basis.Basis_t)
    for b in 1:Nx
        for δ in eachindex(bt_dt)
            a = it_dt - δ + 1
            if 1 ≤ a ≤ Nt
                coeffs_dt_top[b] += coeffs[a, b] * bt_dt[δ]
            end
        end
    end
    sol.v .= (int.method.eval_matrix * coeffs_dt_top) ./ h
    return nothing
end
