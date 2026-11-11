"""
    BoxScheme() <: SpaceTimeMethod

The Preissmann box scheme: the baseline every other method of the package is compared against,
advanced by [`integrate`](@ref) like any other method.

It discretises the first-order form

```math
K z_t + L z_x = \\nabla S(z), \\qquad z = (u, v, w),
```

of a density `λ(½|v|² − ½c²|w|² − V(u))` with `λ > 0` and `V` a function of `u` alone, in which
`v = ∂ₜu` and the third state is `w = −c² ∂ₓu`. The constant matrices are

```math
K = \\begin{pmatrix} 0 & -1 & 0 \\\\ 1 & 0 & 0 \\\\ 0 & 0 & 0 \\end{pmatrix}, \\qquad
L = \\begin{pmatrix} 0 & 0 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 0 & 0 \\end{pmatrix},
```

both skew-symmetric, and `S(z) = V(u) + ½v² − ½w²/c²`. The scheme evaluates every row of that form
at the centre of a box of the grid — the mean of its four corners for the value, the box's own
differences for the derivatives — which is second order in the time step and in the space step, and
which satisfies the discrete multisymplectic conservation law `δₜω + δₓκ = 0` of the form, with
`ω = UᵀKV` and `κ = UᵀLV` of two solutions of the scheme's linearisation, to round-off per box.

# The class of densities

The example problems whose density is of that shape — `Wave`, `VectorWave` and `SineGordon` — are
integrated; a density outside the class, such as `LinearTransport`'s doubled field, raises an
`ArgumentError` when the integrator is built. The class is decided on the partial derivatives the
density supplies, never on the problem's type nor on its component count: `SineGordon`'s density is
in the class up to the constant factor `c` that its `v` term carries, and `LinearTransport` has two
components and no such shape.

# The resolution, and the state

The resolution is the problem's own: one box per `timestep` and `xstep`, and the state lives at the
nodes of the output grid, `(u, v, w)` per component, so that the trace [`top_trace`](@ref) gives and
the solution are on the grid the other methods write. The third state is `w = −c² ∂ₓu`, and the
trace carries the physical slope `∂ₓu`.

The system of one step is square: `u` at the interior nodes, `v` at every node but the first and `w`
at every node are the unknowns, and the three rows of every box are the step's equations. `u` at
both end nodes and `v` at the first node are the problem's boundary data; `v` needs one of them
because the box rows see `v` only through sums over the box, so that an alternating mode of `v` is
invisible to them and one datum fixes it. The step's initial guess is the level it starts from,
continued, and the ecosystem's Newton solver — the one every method of the package uses —
determines the step, so the solve's residuals and its convergence are the run's own record.

# The energy

The energy that [`energy_error`](@ref) reports for a box scheme is the density's Hamiltonian at the
nodes of the output grid, integrated by that grid's trapezoidal rule, which is
[`box_quadrature_weights`](@ref).
"""
struct BoxScheme <: SpaceTimeMethod end

default_solver(::BoxScheme) = Newton()

"""
    box_grid(problem) -> NamedTuple

The lengths one run of the box scheme works with: the number of nodes of the output grid, the
number of components of the field, the number of boxes between the nodes and the size of one step's
system.
"""
function box_grid(problem::LagrangianPDEProblem)
    nx = length(xgrid(problem))
    D = components(problem)
    return (; nx, D, boxes = nx - 1, unknowns = 3 * D * (nx - 1))
end

# The index of one unknown in the solver's vector, component-major: the interior nodes of `u`, then
# `v` from the second node on, then `w` at every node.
@inline function box_u_index(d::Integer, j::Integer, nx::Integer)
    return (d - 1) * (3nx - 3) + (j - 1)
end

@inline function box_v_index(d::Integer, j::Integer, nx::Integer)
    return (d - 1) * (3nx - 3) + (nx - 2) + (j - 1)
end

@inline function box_w_index(d::Integer, j::Integer, nx::Integer)
    return (d - 1) * (3nx - 3) + (2nx - 3) + j
end

"""
    box_quadrature_weights(problem)

The weights of the output grid's trapezoidal rule, which is the rule the box scheme's energy is
integrated by: the trace lives at the nodes, so a node's weight is half the distance to each
neighbour, and an end node gets half of its one-sided gap.
"""
function box_quadrature_weights(problem::LagrangianPDEProblem{T}) where {T}
    nodes = xgrid(problem)
    nx = length(nodes)
    weights = Vector{T}(undef, nx)
    for j in 1:nx
        left = j == 1 ? zero(T) : nodes[j] - nodes[j - 1]
        right = j == nx ? zero(T) : nodes[j + 1] - nodes[j]
        weights[j] = (left + right) / 2
    end
    return weights
end

# The partial derivative of the density with respect to `u`, component `d`, at one point. A density
# of one component at a time is applied to each component of the point; a density that couples them
# is handed the `D` values at once, as the slab's own energy quadrature does.
@inline function potential_derivative(
        density::LagrangianDensity, u, v, w, d::Integer, params)
    return derivative_u(density, u[d], v[d], w[d], params)
end

@inline function potential_derivative(density, u, v, w, d::Integer, params)
    return derivatives(density, u, v, w, params).u[d]
end

"""
    check_box_density(problem) -> (λ, c²)

The coefficients `λ` and `c²` of the density's first-order form, or an `ArgumentError` naming the
part of the class the density misses.

The class is `λ(½|v|² − ½c²|w|² − V(u))` with `λ > 0`, `c² > 0` and `V` a function of `u` alone, and
it is decided by probing the partial derivatives the density itself supplies: `∂L/∂v` must be `λ v`
with one positive `λ` for every component, `∂L/∂w` must be `−λc² w` with the same `λ` and one
positive `c²`, and `∂L/∂u` must move with `u` alone. A probe compares values the density computes,
so a density whose text resembles the class and whose partials are not in it — the doubled field of a
transport equation, whose `∂L/∂u` reads the other component — is refused.
"""
function check_box_density(problem::LagrangianPDEProblem{T}) where {T}
    D = components(problem)
    density = problem.density
    params = problem.params
    tol(scale) = 64 * eps(T) * max(one(T), abs(scale))
    grid(scale) = ones(T, 1, D) .* scale
    partials(u, v, w) = point_derivatives(
        density, grid(u), grid(v), grid(w), 1, params, Val(D))
    # ∂L/∂v = λ v, with one positive λ for every component
    probe = partials(one(T), one(T), zero(T))
    λ = probe.v
    first(λ) > zero(T) ||
        throw(ArgumentError("the box scheme integrates a density λ(½|v|² − ½c²|w|² − V(u)) with " *
                            "λ > 0; this density's ∂L/∂v is $(first(λ)) at v = 1"))
    all(value -> abs(value - first(λ)) ≤ tol(first(λ)), λ) ||
        throw(ArgumentError("the box scheme integrates one λ for every component of the field; " *
                            "this density's ∂L/∂v is $λ at v = 1"))
    doubled = partials(one(T), T(2), zero(T))
    all(value -> abs(value - 2 * first(λ)) ≤ tol(4 * first(λ)), doubled.v) ||
        throw(ArgumentError("the box scheme integrates a density whose ∂L/∂v is λ v, linear in " *
                            "v; this density's ∂L/∂v is $(doubled.v) at v = 2 and $λ at v = 1"))
    moved = partials(T(2), one(T), T(2))
    all(value -> abs(value - first(λ)) ≤ tol(first(λ)), moved.v) ||
        throw(ArgumentError("the box scheme integrates a density whose ∂L/∂v does not move with " *
                            "u or w; this density's ∂L/∂v is $(moved.v) at (u, v, w) = (2, 1, 2) " *
                            "and $λ at (u, v, w) = (1, 1, 0)"))
    # ∂L/∂w = −λ c² w, with one positive λ c² for every component
    slope = partials(one(T), zero(T), one(T)).w
    all(value -> abs(value - first(slope)) ≤ tol(first(slope)), slope) ||
        throw(ArgumentError("the box scheme integrates one λc² for every component of the " *
                            "field; this density's ∂L/∂w is $slope at w = 1"))
    first(slope) < zero(T) ||
        throw(ArgumentError("the box scheme integrates a density λ(½|v|² − ½c²|w|² − V(u)) with " *
                            "c² > 0; this density's ∂L/∂w is $slope at w = 1"))
    steeper = partials(one(T), zero(T), T(2)).w
    all(value -> abs(value - 2 * first(slope)) ≤ tol(4 * first(slope)), steeper) ||
        throw(ArgumentError("the box scheme integrates a density whose ∂L/∂w is −λc² w, linear " *
                            "in w; this density's ∂L/∂w is $steeper at w = 2 and $slope at w = 1"))
    held = partials(T(2), one(T), one(T)).w
    all(value -> abs(value - first(slope)) ≤ tol(first(slope)), held) ||
        throw(ArgumentError("the box scheme integrates a density whose ∂L/∂w does not move with " *
                            "u or v; this density's ∂L/∂w is $held at (u, v, w) = (2, 1, 1) and " *
                            "$slope at (u, v, w) = (1, 0, 1)"))
    # ∂L/∂u moves with u alone
    still = partials(one(T), zero(T), zero(T)).u
    others = partials(one(T), one(T), one(T)).u
    all(d -> abs(others[d] - still[d]) ≤ tol(abs(still[d]) + first(λ)), 1:D) ||
        throw(ArgumentError("the box scheme integrates a density whose ∂L/∂u moves with u " *
                            "alone; this density's ∂L/∂u is $others at (u, v, w) = (1, 1, 1) and " *
                            "$still at (u, v, w) = (1, 0, 0)"))
    c² = -first(slope) / first(λ)
    c² > zero(T) ||
        throw(ArgumentError("the box scheme integrates a density λ(½|v|² − ½c²|w|² − V(u)) with " *
                            "c² > 0; this density's −(∂L/∂w)/(∂L/∂v) is $c²"))
    return first(λ), c²
end

# The boundary data of the level being solved: `u` at both end nodes and `v` at the first, the three
# of them per component.
function boundary_data!(c, sol, problem::LagrangianPDEProblem)
    side = problem.bcs_function(first(sol.t), problem.params)
    for d in eachindex(c.left_u)
        c.left_u[d] = side.left.u[d]
        c.right_u[d] = side.right.u[d]
        c.left_v[d] = side.left.v[d]
    end
    return nothing
end

"""
    BoxSchemeCache{ST}

The cache of the box scheme: the level the last step ended at, as `u`, `v` and `w` at the nodes of
the output grid and component-major; the boundary data of the level being solved; the new level the
step assembles; the solver's vector, and the record of the solve of every step.

`λ` and `c²` are the coefficients of the density's first-order form, read once when the cache is
built by [`check_box_density`](@ref), which also refuses a density outside the class.
"""
mutable struct BoxSchemeCache{ST} <: IntegratorCache{ST}
    u::Vector{ST}
    v::Vector{ST}
    w::Vector{ST}
    new_u::Vector{ST}
    new_v::Vector{ST}
    new_w::Vector{ST}
    left_u::Vector{ST}
    right_u::Vector{ST}
    left_v::Vector{ST}
    x::Vector{ST}
    iterations::Vector{Int}
    residuals::Vector{ST}
    initial_residuals::Vector{ST}
    converged::Vector{Bool}
    steps::Int
    λ::ST
    c²::ST
end

function Cache{ST}(problem::LagrangianPDEProblem, method::BoxScheme; kwargs...) where {ST}
    λ, c² = check_box_density(problem)
    nx = length(xgrid(problem))
    D = components(problem)
    cells = D * nx
    return BoxSchemeCache{ST}(
        zeros(ST, cells), zeros(ST, cells), zeros(ST, cells),
        zeros(ST, cells), zeros(ST, cells), zeros(ST, cells),
        zeros(ST, D), zeros(ST, D), zeros(ST, D),
        zeros(ST, 3 * D * (nx - 1)),
        Int[], ST[], ST[], Bool[], 0, ST(λ), ST(c²))
end

CacheType(ST, problem::LagrangianPDEProblem, ::BoxScheme) = BoxSchemeCache{ST}

nlsolution(c::BoxSchemeCache) = c.x

function Base.show(io::IO, cache::BoxSchemeCache{ST}) where {ST}
    print(io, "BoxSchemeCache{$ST} with $(length(cache.x)) unknowns")
end

"""
    reset!(cache::BoxSchemeCache)

Clear what a run leaves behind: the step counter and the record of the steps' Newton solves. The
level a run starts from is the problem's initial data, which the first step reads from the solution
step, so a cache that still counted the previous run's steps would start the new one at the old
level.
"""
function reset!(c::BoxSchemeCache)
    c.steps = 0
    empty!(c.iterations)
    empty!(c.residuals)
    empty!(c.initial_residuals)
    empty!(c.converged)
    return nothing
end

"""
    solve_stats(cache::BoxSchemeCache)

The records of the steps' Newton solves, as the named tuple
`(; iterations, residuals, initial_residuals, converged)`, one entry per step taken since the last
[`reset!`](@ref).
"""
function solve_stats(c::BoxSchemeCache)
    return (; iterations = c.iterations, residuals = c.residuals,
        initial_residuals = c.initial_residuals, converged = c.converged)
end

"""
    top_trace(cache::BoxSchemeCache)

The trace `(u, v, w)` at the top of the last step, at the nodes of the output grid and
component-major: the solved level's field, its velocity and its slope.
"""
top_trace(c::BoxSchemeCache) = (u = vec(c.u), v = vec(c.v), w = vec(c.w))

# The new level's field at a node: the solved unknown in the interior, and the boundary datum at an
# end node.
@inline function new_u(
        x::AbstractVector, c::BoxSchemeCache, d::Integer, j::Integer, nx::Integer)
    j == 1 && return c.left_u[d]
    j == nx && return c.right_u[d]
    return x[box_u_index(d, j, nx)]
end

@inline function new_v(
        x::AbstractVector, c::BoxSchemeCache, d::Integer, j::Integer, nx::Integer)
    j == 1 && return c.left_v[d]
    return x[box_v_index(d, j, nx)]
end

@inline function new_w(
        x::AbstractVector, c::BoxSchemeCache, d::Integer, j::Integer, nx::Integer)
    return x[box_w_index(d, j, nx)]
end

# The level being solved, written into the cache's own arrays once per residual evaluation.
function assemble_new!(c::BoxSchemeCache, x::AbstractVector, problem::LagrangianPDEProblem)
    nx = box_grid(problem).nx
    D = components(problem)
    for d in 1:D, j in 1:nx

        i = (d - 1) * nx + j
        c.new_u[i] = new_u(x, c, d, j, nx)
        c.new_v[i] = new_v(x, c, d, j, nx)
        c.new_w[i] = new_w(x, c, d, j, nx)
    end
    return nothing
end

"""
    box_residual!(b, x, cache, problem)

The residual of one step, in the solver's layout: for every component and every box, the three rows
of the first-order form at the centre of the box,

```math
-\\delta_t v - \\delta_x w = V'(u), \\qquad \\delta_t u = v, \\qquad \\delta_x u = w,
```

where `δₜ` and `δₓ` are the box's own differences and the value on a right-hand side is the mean of
the box's four corners. The rows are ordered component-major, the three rows of one box together.
"""
function box_residual!(b::AbstractVector, x::AbstractVector, c::BoxSchemeCache,
        problem::LagrangianPDEProblem)
    return box_residual!(b, x, c, problem, Val(components(problem)))
end

function box_residual!(b::AbstractVector, x::AbstractVector, c::BoxSchemeCache,
        problem::LagrangianPDEProblem, ::Val{D}) where {D}
    nx = box_grid(problem).nx
    h = timestep(problem)
    dx = xstep(problem)
    λ = c.λ
    c² = c.c²
    density = problem.density
    params = problem.params
    rows = 3 * (nx - 1)
    assemble_new!(c, x, problem)
    for d in 1:D
        offset = (d - 1) * rows
        for j in 1:(nx - 1)
            i0 = (d - 1) * nx + j
            i1 = i0 + 1
            # the four corners of the box, the level being solved second
            ua = c.u[i0]
            ub = c.new_u[i0]
            uc = c.u[i1]
            ud = c.new_u[i1]
            va = c.v[i0]
            vb = c.new_v[i0]
            vc = c.v[i1]
            vd = c.new_v[i1]
            wa = c.w[i0]
            wb = c.new_w[i0]
            wc = c.w[i1]
            wd = c.new_w[i1]
            potential = box_centre_potential(density, c, j, d, nx, params, Val(D))
            b[offset + 3(j - 1) + 1] = ((vb + vd) - (va + vc)) / (2h) -
                                       c² * ((wc + wd) - (wa + wb)) / (2dx) -
                                       potential / λ
            b[offset + 3(j - 1) + 2] = ((ub + ud) - (ua + uc)) / (2h) -
                                       (va + vb + vc + vd) / 4
            b[offset + 3(j - 1) + 3] = ((uc + ud) - (ua + ub)) / (2dx) -
                                       (wa + wb + wc + wd) / 4
        end
    end
    return nothing
end

# `∂L/∂u` of component `d` at the centre of box `j`: the mean of the box's four corners of the `D`
# components, which is the point the density is read at.
function box_centre_potential(density, c::BoxSchemeCache, j::Integer, d::Integer,
        nx::Integer, params, ::Val{D}) where {D}
    return potential_derivative(density,
        ntuple(e -> box_corner_mean(c.u, c.new_u, e, j, nx), Val(D)),
        ntuple(e -> box_corner_mean(c.v, c.new_v, e, j, nx), Val(D)),
        ntuple(e -> box_corner_mean(c.w, c.new_w, e, j, nx), Val(D)), d, params)
end

@inline function box_corner_mean(old::AbstractVector, new::AbstractVector, e::Integer,
        j::Integer, nx::Integer)
    k = (e - 1) * nx + j
    m = k + 1
    return (old[k] + new[k] + old[m] + new[m]) / 4
end

"""
    residual!(b, x, sol, params, int)

The nonlinear residual of one step of a box scheme, in the solver's own layout.

The level the step starts from, and the boundary data of the level it solves, live in the cache of
the problem's element type; they are copied into the cache the solver works in, as the slab
integrator copies its coupling's data, so that an evaluation at the solver's element type reads
neither a stale level nor an allocation.
"""
function residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:BoxScheme, <:LagrangianPDEProblem}) where {ST}
    data = cache(int)
    c = cache(int, ST)
    copyto!(c.u, data.u)
    copyto!(c.v, data.v)
    copyto!(c.w, data.w)
    copyto!(c.left_u, data.left_u)
    copyto!(c.right_u, data.right_u)
    copyto!(c.left_v, data.left_v)
    assemble_new!(c, x, problem(int))
    box_residual!(b, x, c, problem(int))
    return nothing
end

"""
    initial_guess!(sol, history, params, int)

The initial guess of one step: the level it starts from, continued.

A box scheme's step is a Newton solve, and the previous level is the closest thing the method has to
an answer for the new one: the two are one time step apart, so the guess differs from the solution
by the size of a step's change.
"""
function initial_guess!(sol, history, params,
        int::GeometricIntegrator{<:BoxScheme, <:LagrangianPDEProblem})
    c = cache(int)
    p = problem(int)
    if c.steps == 0
        copyto!(c.u, sol.u)
        copyto!(c.v, sol.v)
        copyto!(c.w, sol.w)
    end
    nx = box_grid(p).nx
    D = components(p)
    for d in 1:D, j in 1:nx

        i = (d - 1) * nx + j
        (j > 1 && j < nx) && (c.x[box_u_index(d, j, nx)] = c.u[i])
        j > 1 && (c.x[box_v_index(d, j, nx)] = c.v[i])
        c.x[box_w_index(d, j, nx)] = c.w[i]
    end
    return nothing
end

"""
    integrate_step!(sol, history, params, int)

One box step: the boundary data of the new level is read from the problem, the ecosystem's Newton
solver determines the step's unknowns, the cache records the solve, and [`update!`](@ref) writes the
solved level into the solution and the cache.

The convergence of the step is recorded under `residual ≤ f_abstol + f_reltol · r₀`, with the
tolerances of the method's own solver options, exactly as the slab integrator records it.
"""
function integrate_step!(sol, history, params,
        int::GeometricIntegrator{<:BoxScheme, <:LagrangianPDEProblem})
    c = cache(int)
    p = problem(int)
    c.steps == 0 && read_initial_level!(c, sol)
    boundary_data!(c, sol, p)
    status = solve_with_status!(nlsolution(int), solver(int), solverstate(int),
        (sol, params, int))
    check_solver_status(status, int)
    push!(c.iterations, status.iterations)
    push!(c.residuals, convert(eltype(c.residuals), status.rfₐ))
    push!(c.initial_residuals,
        convert(eltype(c.initial_residuals), initial_residual(solverstate(int))))
    push!(c.converged,
        step_converged(last(c.residuals), last(c.initial_residuals), config(solver(int))))
    update!(sol, params, nlsolution(int), int)
    return nothing
end

# The first step of a run starts at the problem's initial data, which the solution step holds.
function read_initial_level!(c::BoxSchemeCache, sol)
    copyto!(c.u, sol.u)
    copyto!(c.v, sol.v)
    copyto!(c.w, sol.w)
    return nothing
end

"""
    update!(sol, params, x, int)

Write the solved level into the solution step: the field, its velocity and its slope, at the nodes of
the output grid and component-major, which is also the trace [`top_trace`](@ref) gives.
"""
function update!(sol, params, x::AbstractVector{ST},
        int::GeometricIntegrator{<:BoxScheme, <:LagrangianPDEProblem}) where {ST}
    c = cache(int, ST)
    p = problem(int)
    assemble_new!(c, x, p)
    copyto!(sol.u, c.new_u)
    copyto!(sol.v, c.new_v)
    copyto!(sol.w, c.new_w)
    copyto!(c.u, c.new_u)
    copyto!(c.v, c.new_v)
    copyto!(c.w, c.new_w)
    c.steps += 1
    return nothing
end
