"""
    SpaceTimeGalerkin{A, C, G} <: SpaceTimeAnsatz

The space-time Galerkin slab integrator: a tensor B-spline ansatz `A`, a coupling `C` of
consecutive slabs and an initial guess `G`.

One step of the integrator solves one slab. The unknown is the coefficient array of the ansatz,
`(Nt, Nx)` per component; the bottom row of the time direction is a datum that the coupling
supplies — the initial condition at the first step and the previous slab's top row after it — and
the remaining `Nt - 1` rows are determined by the interior weak Euler–Lagrange form of the slab,
tested against the basis functions that vanish at the top of the slab. The `v` term of that form
leaves one boundary term at the bottom of the slab, the momentum flux, which the coupling supplies
as well; the top of the slab is free, so its flux is not imposed but measured, and handed on as the
next slab's bottom flux.

The unknowns are the `Nt - 1` free rows of each space function and each component, so a step is a
nonlinear system of `(Nt - 1) * Nx * D` equations, which the ecosystem's Newton solver solves. The
iterate count and the residual of every step are recorded in the cache, and the trace `(u, v, w)` at
the top of the slab, at the nodes of the space quadrature, is kept for the diagnostics.

```julia
method = SpaceTimeGalerkin(
    TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0)),
    CoefficientRestriction();
    initial_guess = Continuation())

result = integrate(problem, method)
```
"""
struct SpaceTimeGalerkin{A <: SpaceTimeAnsatz, C <: SlabCoupling, G <: SlabInitialGuess} <:
       SpaceTimeAnsatz
    ansatz::A
    coupling::C
    initial_guess::G
end

function SpaceTimeGalerkin(ansatz::SpaceTimeAnsatz,
        coupling::SlabCoupling = CoefficientRestriction();
        initial_guess::SlabInitialGuess = Continuation())
    return SpaceTimeGalerkin{typeof(ansatz), typeof(coupling), typeof(initial_guess)}(
        ansatz, coupling, initial_guess)
end

"""The ansatz of the method."""
ansatz(method::SpaceTimeGalerkin) = method.ansatz

"""The coupling of the method."""
coupling(method::SpaceTimeGalerkin) = method.coupling

"""The initial guess of the method."""
initial_guess(method::SpaceTimeGalerkin) = method.initial_guess

default_solver(::SpaceTimeGalerkin) = Newton()

"""
    check_method(problem, method)

Reject a method that the problem cannot support, before any step runs.

The refusals are `ArgumentError`s: a target that reads the exact solution of a problem that has
none, a fit that needs an ansatz linear in its coefficients on one that is not, an ansatz of another
element type than the problem's, and an ansatz on another space interval than the problem's. The
last two name the two values they compared, and the element type is compared first, so a pair that
mismatches in both hears about the type.
"""
function check_method(problem::LagrangianPDEProblem{PT}, method::SpaceTimeGalerkin) where {PT}
    guess = initial_guess(method)
    if guess isa FitTo
        if needs_exact_solution(guess.target) && problem.exact_u === nothing
            throw(ArgumentError("FitTo(ExactSolution()) needs a problem with an exact solution; " *
                                "this problem has exact_u = nothing"))
        end
        if fit_method(guess, ansatz(method)) isa Projection &&
           !is_linear_in_coefficients(ansatz(method))
            throw(ArgumentError("Projection() needs an ansatz that is linear in its " *
                                "coefficients; $(nameof(typeof(ansatz(method)))) is not"))
        end
    end
    a = ansatz(method)
    a isa TensorBSplineAnsatz || return nothing
    S = eltype(a.xspan)
    S === PT || throw(ArgumentError(
        "the ansatz element type $S does not match the problem's $PT"))
    xspan(a) == xspan(problem) || throw(ArgumentError(
        "the ansatz xspan $(xspan(a)) does not match the problem's $(xspan(problem))"))
    return nothing
end

"""
    SlabEvaluation{T}

The evaluation tables of one slab: the space basis at the output grid and at the nodes of the space
quadrature, and the time basis at the top of the slab, with its first derivative.
"""
struct SlabEvaluation{T}
    output::Matrix{T}
    output_slope::Matrix{T}
    trace::Matrix{T}
    trace_slope::Matrix{T}
    top_slope::Vector{T}
end

function SlabEvaluation(ansatz::TensorBSplineAnsatz{T}, grid, nodes) where {T}
    tb = time_basis(ansatz)
    xb = space_basis(ansatz)
    nx = nbasis(xb)
    nt = nbasis(tb)
    return SlabEvaluation{T}(
        [evaluate(xb, j, x) for x in grid, j in 1:nx],
        [evaluate(xb, j, x, 1) for x in grid, j in 1:nx],
        [evaluate(xb, j, x) for x in nodes, j in 1:nx],
        [evaluate(xb, j, x, 1) for x in nodes, j in 1:nx],
        [evaluate(tb, a, one(T), 1) for a in 1:nt])
end

"""
    SpaceTimeGalerkinCache{ST, T}

The cache of the slab integrator.

`x` is the nonlinear solver's vector, the free coefficients of the slab, laid out component-major
with the time index fastest: the entry of row `a ∈ 2:Nt`, space function `j` and component `d` is
`a - 1 + (j - 1) * (Nt - 1) + (d - 1) * (Nt - 1) * Nx`.

The data of a step are `bottom` and `flux`, the coupling's datum and its momentum load, and the
results of the last step are `top`, the top row of the coefficient array, `residual`, the last weak
form, and the trace at the top of the slab. `iterations`, `residuals` and `initial_residuals`
record the Newton solve of every step: its iterate count, its final residual `‖F(x)‖₂` and its
initial residual `r₀ = ‖F(x₀)‖₂`. `steps` counts the steps taken, which is what tells the coupling
and the guess whether the slab is the run's first.
"""
mutable struct SpaceTimeGalerkinCache{ST, T} <: IntegratorCache{ST}
    x::Vector{ST}
    coeffs::Array{ST, 3}
    residual::Array{ST, 3}
    u::Array{ST, 3}
    v::Array{ST, 3}
    w::Array{ST, 3}
    du::Array{ST, 3}
    dv::Array{ST, 3}
    dw::Array{ST, 3}
    bottom::Matrix{ST}
    flux::Matrix{ST}
    top::Matrix{ST}
    trace_u::Matrix{ST}
    trace_v::Matrix{ST}
    trace_w::Matrix{ST}
    iterations::Vector{Int}
    residuals::Vector{ST}
    initial_residuals::Vector{ST}
    steps::Int
    tabulation::SlabTabulation{T}
    evaluation::SlabEvaluation{T}
end

function Cache{ST}(problem::LagrangianPDEProblem,
        method::SpaceTimeGalerkin; kwargs...) where {ST}
    check_method(problem, method)
    a = ansatz(method)
    a isa TensorBSplineAnsatz || throw(ArgumentError(
        "SpaceTimeGalerkin needs a TensorBSplineAnsatz, got $(nameof(typeof(a)))"))
    T = eltype(a.xspan)
    tabulation = SlabTabulation(a)
    evaluation = SlabEvaluation(a, xgrid(problem), space_quadrature_nodes(a))
    nt = ntime_functions(a)
    nx = nspace_functions(a)
    D = components(problem)
    free = total_free_coefficients(method, problem)
    qt = nquadrature(tabulation.time)
    qx = nquadrature(tabulation.space)
    nq = length(space_quadrature_nodes(a))
    coeffs = zeros(ST, nt, nx, D)
    return SpaceTimeGalerkinCache{ST, T}(
        zeros(ST, free),
        coeffs,
        similar(coeffs),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, nx, D),
        zeros(ST, nx, D),
        zeros(ST, nx, D),
        zeros(ST, nq, D),
        zeros(ST, nq, D),
        zeros(ST, nq, D),
        Int[],
        ST[],
        ST[],
        0,
        tabulation,
        evaluation)
end

@inline CacheType(ST, problem::LagrangianPDEProblem{T},
    method::SpaceTimeGalerkin{<:TensorBSplineAnsatz{T}}) where {T} = SpaceTimeGalerkinCache{
    ST, T}

nlsolution(cache::SpaceTimeGalerkinCache) = cache.x

function Base.show(io::IO, cache::SpaceTimeGalerkinCache{ST, T}) where {ST, T}
    print(io, "SpaceTimeGalerkinCache{$ST, $T} with $(length(cache.x)) free coefficients")
end

"""
    reset!(cache::SpaceTimeGalerkinCache)

Clear the state a run leaves in the cache: the step counter, which is what tells the coupling and
the guess whether the slab is the run's first, and the per-step records of the Newton solve.

A run starts at the problem's initial data, so a cache that still counted the previous run's steps
would hand that run's last top row to the new first slab as its bottom datum.
"""
function reset!(c::SpaceTimeGalerkinCache)
    c.steps = 0
    empty!(c.iterations)
    empty!(c.residuals)
    empty!(c.initial_residuals)
    return nothing
end

"""Scatter the solver's vector into the coefficient array, with the coupling's bottom row."""
function scatter!(coeffs::Array{ST, 3}, x::AbstractVector{ST}, bottom::AbstractMatrix{ST},
        nt::Integer) where {ST}
    nx, D = size(bottom)
    a_free = nt - 1
    for d in 1:D
        for j in 1:nx
            coeffs[1, j, d] = bottom[j, d]
            for a in 2:nt
                coeffs[a, j, d] = x[(a - 1) + (j - 1) * a_free + (d - 1) * a_free * nx]
            end
        end
    end
    return nothing
end

"""The coefficient index of the free row `a ∈ 2:Nt`, space function `j`, component `d`."""
@inline function free_index(a::Integer, j::Integer, d::Integer, nt::Integer, nx::Integer)
    return (a - 1) + (j - 1) * (nt - 1) + (d - 1) * (nt - 1) * nx
end

"""
    slab_weak_form!(cache, problem, method)

The interior weak form of the slab the cache's coefficient array describes, into `cache.residual`.
"""
function slab_weak_form!(c::SpaceTimeGalerkinCache{ST}, problem::LagrangianPDEProblem,
        method::SpaceTimeGalerkin) where {ST}
    slab_values!(c.u, c.v, c.w, c.coeffs, c.tabulation, timestep(problem))
    density_derivatives!(c.du, c.dv, c.dw, problem.density, c.u, c.v, c.w,
        parameters(problem))
    interior_weak_form!(c.residual, c.du, c.dv, c.dw, c.tabulation, timestep(problem))
    return nothing
end

"""
    residual!(b, x, sol, params, int)

The nonlinear residual of one slab step, in the solver's own layout: `(Nt - 1) * Nx * D` entries,
the free rows of the coefficient array ordered as the solver's vector is.
"""
function residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:SpaceTimeGalerkin, <:LagrangianPDEProblem}) where {ST}
    data = cache(int)
    c = cache(int, ST)
    copy_datum!(c, data)
    nt = ntime_functions(ansatz(method(int)))
    nx, D = size(c.bottom)
    scatter!(c.coeffs, x, c.bottom, nt)
    slab_weak_form!(c, problem(int), method(int))
    for d in 1:D, j in 1:nx, a in 1:(nt - 1)
        b[a + (j - 1) * (nt - 1) + (d - 1) * (nt - 1) * nx] = c.residual[a, j, d] +
                                                              (a == 1 ?
                                                               flux_load(coupling(method(int)), c, j, d) :
                                                               zero(ST))
    end
    return nothing
end

"""Copy the coupling's datum of the step from the data cache into a cache of another element type."""
function copy_datum!(c::SpaceTimeGalerkinCache, data::SpaceTimeGalerkinCache)
    c.bottom .= data.bottom
    c.flux .= data.flux
    return nothing
end

"""
    update!(sol, params, x, int)

Write the solution of the slab into `sol`, keep its trace, and hand its top row and top flux to the
next step.
"""
function update!(sol, params, x::AbstractVector{ST},
        int::GeometricIntegrator{<:SpaceTimeGalerkin, <:LagrangianPDEProblem}) where {ST}
    c = cache(int, ST)
    m = method(int)
    a = ansatz(m)
    p = problem(int)
    nt = ntime_functions(a)
    nx, D = size(c.bottom)
    scatter!(c.coeffs, x, c.bottom, nt)
    c.top .= view(c.coeffs, nt, :, :)
    h = timestep(p)
    assign!(sol.u, sol.v, sol.w, c, a, nx, D, h)
    slab_weak_form!(c, p, m)
    c.flux .= view(c.residual, nt, :, :)
    c.steps += 1
    return nothing
end

"""
    assign!(u, v, w, cache, ansatz, nx, D, h)

The solution and its two derivatives at the top of the slab, on the output grid, and beside them the
trace `(u, v, w)` at the nodes of the space quadrature, in one pass over the top row.

The two tables are the space basis evaluated at two sets of points, so the row is walked once and
the value of each function is written to both.
"""
function assign!(u::AbstractVector{ST}, v::AbstractVector{ST}, w::AbstractVector{ST},
        c::SpaceTimeGalerkinCache{ST}, a::TensorBSplineAnsatz, nx::Integer, D::Integer,
        h) where {ST}
    e = c.evaluation
    nt = ntime_functions(a)
    n = size(e.output, 1)
    m = size(e.trace, 1)
    hh = convert(ST, h)
    fill!(u, zero(ST))
    fill!(v, zero(ST))
    fill!(w, zero(ST))
    fill!(c.trace_u, zero(ST))
    fill!(c.trace_v, zero(ST))
    fill!(c.trace_w, zero(ST))
    for d in 1:D
        for j in 1:nx
            # the time derivative at the top of the slab, per space function: `∂ₜu` is `∂_{t̂}u / h`
            tderiv = zero(ST)
            for k in 1:nt
                tderiv += c.coeffs[k, j, d] * e.top_slope[k]
            end
            tderiv /= hh
            for i in 1:n
                u[(d - 1) * n + i] += c.top[j, d] * e.output[i, j]
                w[(d - 1) * n + i] += c.top[j, d] * e.output_slope[i, j]
                v[(d - 1) * n + i] += tderiv * e.output[i, j]
            end
            for i in 1:m
                c.trace_u[i, d] += c.top[j, d] * e.trace[i, j]
                c.trace_w[i, d] += c.top[j, d] * e.trace_slope[i, j]
                c.trace_v[i, d] += tderiv * e.trace[i, j]
            end
        end
    end
    return nothing
end

"""
    top_trace(cache::SpaceTimeGalerkinCache)

The trace `(u, v, w)` at the top of the last slab, at the nodes of the space quadrature,
component-major.
"""
function top_trace(c::SpaceTimeGalerkinCache{ST}) where {ST}
    return (u = vec(c.trace_u), v = vec(c.trace_v), w = vec(c.trace_w))
end

"""
    slab_bottom!(cache, coupling, problem, ansatz, sol)

Fix the datum of the slab's bottom trace and its momentum load, in `cache.bottom` and `cache.flux`.

A coupling implements this for the two cases it distinguishes: the first slab of a run, where the
datum is the problem's initial condition, and every later slab, where it is the trace the previous
slab carried out. The cache counts its steps, so `cache.steps` says which case this is.
"""
function slab_bottom!(cache, coupling::SlabCoupling, problem, ansatz, sol)
    throw(ArgumentError("$(nameof(typeof(coupling))) does not implement slab_bottom!"))
end

"""
    integrate_step!(sol, history, params, int)

One slab: the coupling fixes the bottom datum and its momentum load, the initial guess fills the
solver's vector, the ecosystem's Newton solver solves the slab, and the update writes the solution,
the trace and the data the next slab starts from.
"""
function integrate_step!(sol, history, params,
        int::GeometricIntegrator{<:SpaceTimeGalerkin, <:LagrangianPDEProblem})
    c = cache(int)
    m = method(int)
    slab_bottom!(c, coupling(m), problem(int), ansatz(m), sol)
    initial_guess!(sol, history, params, int)
    status = solve_with_status!(nlsolution(int), solver(int), solverstate(int),
        (sol, params, int))
    check_solver_status(status, int)
    push!(c.iterations, status.iterations)
    push!(c.residuals, convert(eltype(c.residuals), status.rfₐ))
    push!(c.initial_residuals,
        convert(eltype(c.initial_residuals), initial_residual(solverstate(int))))
    update!(sol, params, nlsolution(int), int)
    return nothing
end

"""
    total_free_coefficients(method, problem)

The size of one slab's nonlinear system: one equation per free coefficient.
"""
function total_free_coefficients(method::SpaceTimeGalerkin, problem::LagrangianPDEProblem)
    a = ansatz(method)
    return (ntime_functions(a) - 1) * nspace_functions(a) * components(problem)
end
