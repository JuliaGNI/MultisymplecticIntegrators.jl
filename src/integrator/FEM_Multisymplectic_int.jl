struct FEM_Multisymplectic_Integrator <: PDEMethod
    time_quadrature::NamedTuple{(:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RT::Int
    spatial_quadrature::NamedTuple{
        (:nodes, :weights), Tuple{Vector{Float64}, Vector{Float64}}}
    RX::Int
    grid_matrix::Matrix{Vector{Float64}}
    grid_weights::Matrix{Float64}
    x_nodes::Vector{Float64}
    mass_matrix::Matrix{Float64}
    laplace_matrix::Matrix{Float64}
    lhs_matrix::Matrix{Float64}
    lhs_factor::LU{Float64, Matrix{Float64}, Vector{Int64}}
    interior::Vector{Int}
    boundary::Vector{Int}
    c::Float64
    startup::Symbol
    show_status::Bool

    function FEM_Multisymplectic_Integrator(problem::LPDEProblem;
            c = problem.params.c,
            startup::Symbol = :taylor,
            show_status::Bool = false)
        @assert problem.D == 1 "FEM_Multisymplectic_Integrator currently supports scalar 1+1 PDEs."
        @assert hasproperty(problem.params, :c) "The wave speed parameter `c` is required."
        @assert startup in (:taylor, :exact) "startup must be :taylor or :exact."

        x_nodes = collect(problem.xspan[1]:problem.xstep:problem.xspan[2])
        N = length(x_nodes)
        @assert N ≥ 3 "At least three spatial nodes are required."
        dx = problem.xstep
        h = problem.timestep
        cfl = abs(c) * h / dx
        @assert cfl ≤ 1 + sqrt(eps(Float64)) "Bilinear FEM multisymplectic wave integrator requires abs(c) * timestep / xstep ≤ 1; got $(cfl)."
        interior = collect(2:(N - 1))
        boundary = [1, N]

        mass_matrix = zeros(N, N)
        laplace_matrix = zeros(N, N)
        for i in interior
            mass_matrix[i, i - 1] = 1 / 6
            mass_matrix[i, i] = 4 / 6
            mass_matrix[i, i + 1] = 1 / 6
            laplace_matrix[i, i - 1] = 1
            laplace_matrix[i, i] = -2
            laplace_matrix[i, i + 1] = 1
        end

        lhs_full = mass_matrix / h^2 - c^2 * laplace_matrix / (6 * dx^2)
        lhs_matrix = lhs_full[interior, interior]
        lhs_factor = lu(lhs_matrix)

        nodes = (x_nodes .- problem.xspan[1]) ./ (problem.xspan[2] - problem.xspan[1])
        weights = fill(1 / (N - 1), N)
        weights[begin] *= 0.5
        weights[end] *= 0.5
        time_quadrature = (nodes = [1.0], weights = [1.0])
        spatial_quadrature = (nodes = collect(nodes), weights = weights)
        grid_matrix = [Float64[t, x] for t in time_quadrature.nodes, x in x_nodes]
        grid_weights = reshape(copy(weights), 1, N)

        new(time_quadrature, 1,
            spatial_quadrature, N,
            grid_matrix, grid_weights,
            x_nodes,
            mass_matrix, laplace_matrix, lhs_matrix, lhs_factor,
            interior, boundary, Float64(c), startup, show_status)
    end
end

function Base.show(io::IO, method::FEM_Multisymplectic_Integrator)
    print(io, "\n FEM Multisymplectic Integrator with:\n")
    print(io, "   Element: bilinear Lagrange on rectangular space-time cells \n")
    print(io, "   Spatial nodes: $(length(method.x_nodes)), interior unknowns: $(length(method.interior)) \n")
    print(io, "   Wave speed c: $(method.c) \n")
    print(io, "   Startup: $(method.startup) \n")
    print(io, "   Show status: $(method.show_status) \n")
end

default_solver(::FEM_Multisymplectic_Integrator) = NewtonMethod()
default_options(::FEM_Multisymplectic_Integrator) = (max_iterations = 1, verbosity = 0)

struct FEM_Multisymplectic_IntegratorCache{ST, D, RX} <: PDEIntegratorCache{ST, D}
    x::Vector{ST}
    previous_u::Vector{ST}
    current_u::Vector{ST}
    next_u::Vector{ST}
    startup_done::Vector{Bool}
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

    function FEM_Multisymplectic_IntegratorCache{ST, D, RX}() where {ST, D, RX}
        x = zeros(ST, RX - 2)
        previous_u = zeros(ST, RX)
        current_u = zeros(ST, RX)
        next_u = zeros(ST, RX)
        startup_done = [false]
        u_quad_values = zeros(ST, D, 1, RX)
        v_quad_values = zeros(ST, D, 1, RX)
        w_quad_values = zeros(ST, D, 1, RX)
        ut₁_quad_values = zeros(ST, D, RX)
        vt₁_quad_values = zeros(ST, D, RX)
        wt₁_quad_values = zeros(ST, D, RX)
        init_condition_t₀ = zeros(ST, D, RX)
        ics_ut₀_quad_values = zeros(ST, D, RX)
        ics_vt₀_quad_values = zeros(ST, D, RX)
        ics_wt₀_quad_values = zeros(ST, D, RX)
        boundary_condition_x₀ = zeros(ST, D, 1)
        boundary_condition_x₁ = zeros(ST, D, 1)
        bc_ux₀_quad_values = zeros(ST, D, 1)
        bc_vx₀_quad_values = zeros(ST, D, 1)
        bc_wx₀_quad_values = zeros(ST, D, 1)
        bc_ux₁_quad_values = zeros(ST, D, 1)
        bc_vx₁_quad_values = zeros(ST, D, 1)
        bc_wx₁_quad_values = zeros(ST, D, 1)
        flag_done_initial_guess = zeros(ST, 1)

        new(x, previous_u, current_u, next_u, startup_done,
            u_quad_values, v_quad_values, w_quad_values,
            ut₁_quad_values, vt₁_quad_values, wt₁_quad_values,
            init_condition_t₀,
            ics_ut₀_quad_values, ics_vt₀_quad_values, ics_wt₀_quad_values,
            boundary_condition_x₀, boundary_condition_x₁,
            bc_ux₀_quad_values, bc_vx₀_quad_values, bc_wx₀_quad_values,
            bc_ux₁_quad_values, bc_vx₁_quad_values, bc_wx₁_quad_values,
            flag_done_initial_guess)
    end
end

nlsolution(cache::FEM_Multisymplectic_IntegratorCache) = cache.x

function Cache{ST}(problem::LPDEProblem, int::FEM_Multisymplectic_Integrator; kwargs...) where {ST}
    FEM_Multisymplectic_IntegratorCache{ST, problem.D, int.RX}(; kwargs...)
end

@inline CacheType(ST, problem::LPDEProblem,
    int::FEM_Multisymplectic_Integrator) = FEM_Multisymplectic_IntegratorCache{
    ST, problem.D, int.RX}

function copy_internal_variables!(C::FEM_Multisymplectic_IntegratorCache, solstep::SolutionStep)
    nothing
end
prior_initial_guess!(C, sol, int::PDEIntegrator{<:FEM_Multisymplectic_Integrator}) = nothing

function _fem_boundary_values(int::PDEIntegrator{<:FEM_Multisymplectic_Integrator}, t)
    bc = int.problem.bcs_function(t, int.problem.xspan)
    return (left = bc.bc₀.u, right = bc.bc₁.u)
end

function _fem_start_previous!(C, sol, int::PDEIntegrator{<:FEM_Multisymplectic_Integrator})
    h = timestep(int)
    t0 = sol.t - h
    x_nodes = int.method.x_nodes
    if int.method.startup == :exact
        C.previous_u .= int.problem.exact_u.(t0 - h, x_nodes)
        return nothing
    end

    C.previous_u .= sol.u .- h .* sol.v
    acceleration = int.method.c^2 .* (int.method.laplace_matrix * sol.u) ./
                   int.problem.xstep^2
    C.previous_u[int.method.interior] .+= 0.5 * h^2 .* acceleration[int.method.interior]
    bc = _fem_boundary_values(int, t0 - h)
    C.previous_u[begin] = bc.left
    C.previous_u[end] = bc.right
    return nothing
end

function _fem_step!(C::FEM_Multisymplectic_IntegratorCache, sol,
        int::PDEIntegrator{<:FEM_Multisymplectic_Integrator})
    h = timestep(int)
    dx = int.problem.xstep
    I = int.method.interior
    B = int.method.boundary
    M = int.method.mass_matrix
    A = int.method.laplace_matrix
    c2 = int.method.c^2

    C.current_u .= sol.u
    if !C.startup_done[1]
        _fem_start_previous!(C, sol, int)
        C.startup_done[1] = true
    end

    t_next = sol.t
    bc_next = _fem_boundary_values(int, t_next)
    C.next_u[begin] = bc_next.left
    C.next_u[end] = bc_next.right

    lhs_full = M / h^2 - c2 * A / (6 * dx^2)
    rhs_full = 2 .* (M * C.current_u) ./ h^2 .- (M * C.previous_u) ./ h^2 .+
               c2 .* (A * (C.previous_u .+ 4 .* C.current_u)) ./ (6 * dx^2)
    rhs = rhs_full[I] .- lhs_full[I, B] * C.next_u[B]

    C.x .= int.method.lhs_factor \ rhs
    C.next_u[I] .= C.x
    int.method.show_status &&
        println("FEM multisymplectic max |u_next| = ", maximum(abs.(C.next_u)))
    return nothing
end

function _fem_fill_derivatives!(
        C::FEM_Multisymplectic_IntegratorCache, previous_u, current_u,
        int::PDEIntegrator{<:FEM_Multisymplectic_Integrator})
    h = timestep(int)
    dx = int.problem.xstep
    N = length(int.method.x_nodes)

    C.u_quad_values[1, 1, :] .= C.next_u
    C.v_quad_values[1, 1, :] .= (3 .* C.next_u .- 4 .* current_u .+ previous_u) ./ (2 * h)
    for i in 2:(N - 1)
        C.w_quad_values[1, 1, i] = (C.next_u[i + 1] - C.next_u[i - 1]) / (2 * dx)
    end
    C.w_quad_values[1, 1, 1] = (C.next_u[2] - C.next_u[1]) / dx
    C.w_quad_values[1, 1, N] = (C.next_u[N] - C.next_u[N - 1]) / dx

    C.ut₁_quad_values[1, :] .= C.next_u
    C.vt₁_quad_values[1, :] .= C.v_quad_values[1, 1, :]
    C.wt₁_quad_values[1, :] .= C.w_quad_values[1, 1, :]
    return nothing
end

function integrate_step!(sol, history, params, int::PDEIntegrator{<:FEM_Multisymplectic_Integrator})
    C = cache(int)
    old_u = copy(sol.u)
    _fem_step!(C, sol, int)
    old_previous_u = copy(C.previous_u)
    sol.u .= C.next_u
    _fem_fill_derivatives!(C, old_previous_u, old_u, int)
    sol.v .= C.v_quad_values[1, 1, :]
    sol.w .= C.w_quad_values[1, 1, :]
    C.previous_u .= old_u
    return nothing
end

function components!(x::AbstractVector{ST}, sol, params,
        int::PDEIntegrator{<:FEM_Multisymplectic_Integrator}) where {ST}
    return nothing
end

function residual!(b::Vector{ST}, sol, params, int::PDEIntegrator{<:FEM_Multisymplectic_Integrator}) where {ST}
    C = cache(int, ST)
    h = timestep(int)
    dx = int.problem.xstep
    I = int.method.interior
    M = int.method.mass_matrix
    A = int.method.laplace_matrix
    c2 = int.method.c^2
    full_next = copy(C.next_u)
    full_next[I] .= C.x
    r = (M * (full_next .- 2 .* C.current_u .+ C.previous_u)) ./ h^2 .-
        c2 .* (A * (C.previous_u .+ 4 .* C.current_u .+ full_next)) ./ (6 * dx^2)
    b .= r[I]
    return nothing
end

function update!(sol, int::PDEIntegrator{<:FEM_Multisymplectic_Integrator})
    C = cache(int)
    sol.u .= C.next_u
    sol.v .= C.v_quad_values[1, 1, :]
    sol.w .= C.w_quad_values[1, 1, :]
    return nothing
end
