"""
    SpaceTimeGalerkin{A, C, G} <: SpaceTimeMethod

The space-time Galerkin slab integrator: a tensor B-spline ansatz `A`, a coupling `C` of
consecutive slabs and an initial guess `G`.

The ansatz is a [`SpaceTimeAnsatz`](@ref), the coupling a [`SlabCoupling`](@ref) and the guess a
[`SlabInitialGuess`](@ref) — [`Continuation`](@ref) or [`FitTo`](@ref). A value of another role,
a method in the place of the ansatz or a target in the place of the guess, is refused when the
method is built.

One step of the integrator solves one slab. The unknown is the coefficient array of the ansatz,
`(Nt, Nx)` per component; the bottom row of the time direction is a datum that the coupling
supplies — the initial condition at the first step and the previous slab's top row after it — and
the remaining `Nt - 1` rows are determined by the interior weak Euler–Lagrange form of the slab,
tested against the basis functions that vanish at the top of the slab. The `v` term of that form
leaves one boundary term at the bottom of the slab, the momentum flux, which the coupling supplies
as well; the top of the slab is free, so its flux is not imposed but measured, and handed on as the
next slab's bottom flux.

The unknown count is the coupling's: the free rows of each space function and each component, plus
the coupling's own unknowns behind them. A [`CoefficientRestriction`](@ref) fixes the bottom row, so
it leaves `(Nt - 1) * Nx * D` free coefficients; a weak-multiplier coupling imposes the initial
condition with multipliers of its own, so all `Nt` rows are free and the multipliers follow them. A
step is a nonlinear system of [`total_free_coefficients`](@ref)`(method, problem)` equations, which
the ecosystem's Newton solver solves. The iterate count and the residual of every step are recorded
in the cache, and the trace `(u, v, w)` at the top of the slab, at the nodes of the space
quadrature, is kept for the diagnostics.

```julia
method = SpaceTimeGalerkin(
    TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0)),
    CoefficientRestriction();
    initial_guess = Continuation())

result = integrate(problem, method)
```
"""
struct SpaceTimeGalerkin{A <: SpaceTimeAnsatz, C <: SlabCoupling, G <: SlabInitialGuess} <:
       SpaceTimeMethod
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

function SpaceTimeGalerkin(method::SpaceTimeMethod, args...; kwargs...)
    throw(ArgumentError("SpaceTimeGalerkin needs a SpaceTimeAnsatz, got the method " *
                        "$(nameof(typeof(method)))"))
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
    check_coupling(problem, coupling(method), a)
    return nothing
end

"""
    internal_variables(method::SpaceTimeGalerkin, problem)

The coupling's own state of a step, which the ecosystem's `copy_internal_variables!` fills after
every step.
"""
function GeometricIntegratorsBase.internal_variables(method::SpaceTimeGalerkin,
        problem::LagrangianPDEProblem)
    return coupling_internal(coupling(method), ansatz(method), problem)
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
    CouplingData

The coupling's block of the slab cache.

A coupling fixes how the free coefficients of a slab are laid out, adds terms to the residual, and
may carry unknowns of its own between the coefficients and behind them. The block holds what it
keeps between slabs — its datum, its unknowns, the tables its residual term evaluates and the
scratch of that evaluation. It is built once, when the cache is built, and its type is a parameter
of the cache, so that a coupling that keeps nothing pays for nothing.
"""
abstract type CouplingData end

"""The block of a coupling that keeps no data of its own."""
struct NoCouplingData <: CouplingData end

"""
    coupling_data(coupling, ansatz, problem, ST)

The coupling's block of the slab cache, built once per integrator.
"""
function coupling_data(::SlabCoupling, ::SpaceTimeAnsatz, ::LagrangianPDEProblem, ::Type{ST}) where {ST}
    return NoCouplingData()
end

"""
    coupling_data_type(coupling, ansatz, problem, ST)

The type of [`coupling_data`](@ref)`(coupling, ansatz, problem, ST)`, computed without building the
block, because the ecosystem asserts the cache's type on every lookup.
"""
function coupling_data_type(::SlabCoupling, ::SpaceTimeAnsatz, ::LagrangianPDEProblem,
        ::Type{ST}) where {ST}
    return NoCouplingData
end

"""
    free_rows(coupling, ansatz)

The number of free time rows of the slab: the rows of the coefficient array that the solver
determines, the time index running fastest over the coefficient part of the solver's vector.

A coupling that fixes the bottom row of the coefficient array leaves `Nt - 1` free rows; one that
imposes the initial condition weakly leaves all `Nt`.
"""
free_rows(coupling::SlabCoupling, ansatz::SpaceTimeAnsatz) = ntime_functions(ansatz) - 1

"""The number of free time rows, from the block the cache holds."""
free_rows(::NoCouplingData, ansatz::SpaceTimeAnsatz) = ntime_functions(ansatz) - 1

"""
    coupling_unknowns(coupling, ansatz, problem)

The number of the coupling's extra unknowns: the entries of the solver's vector that follow the
coefficients.
"""
coupling_unknowns(::SlabCoupling, ::SpaceTimeAnsatz, ::LagrangianPDEProblem) = 0

"""
    coupling_term!(b, x, coupling, data, cache, ansatz, problem)

The coupling's own part of the residual: its terms on the free rows the interior weak form already
carries, and its own equations after them. The coupling is the third argument, so a coupling's term
is reached whether or not it keeps a block of the cache, and the fourth argument is that block.
"""
coupling_term!(b, x, ::SlabCoupling, ::CouplingData, cache, ansatz, problem) = nothing

"""
    coupling_guess!(data, cache, ansatz, problem, sol)

The coupling's initial guess of its unknowns, once per slab.
"""
coupling_guess!(::NoCouplingData, cache, ansatz, problem, sol) = nothing

"""
    coupling_update!(data, x, cache, ansatz, problem)

Read the coupling's unknowns out of the solver's vector once a step has been solved.
"""
coupling_update!(::NoCouplingData, x, cache, ansatz, problem) = nothing

"""
    copy_coupling_data!(cache_data, data)

Copy the coupling's datums from the cache of the problem's element type into the cache the solver
works in.
"""
copy_coupling_data!(::CouplingData, ::CouplingData) = nothing

"""
    coupling_internal(coupling, ansatz, problem)

The coupling's own state of a step: the entries of the solution step's `internal` named tuple, which
the ecosystem's `copy_internal_variables!` fills after every step. A coupling that carries nothing
between steps declares no entry.
"""
coupling_internal(::SlabCoupling, ::SpaceTimeAnsatz, ::LagrangianPDEProblem) = NamedTuple()

"""
    check_coupling(problem, coupling, ansatz)

Reject a coupling that the ansatz cannot support, when the integrator is built.
"""
check_coupling(::LagrangianPDEProblem, ::SlabCoupling, ::SpaceTimeAnsatz) = nothing

"""
    SpaceTimeGalerkinCache{ST, T, CD}

The cache of the slab integrator.

`x` is the nonlinear solver's vector: the free coefficients of the slab, laid out component-major
with the time index fastest — the entry of free row `a ∈ 1:Nf`, space function `j` and component `d`
is `a + (j - 1) * Nf + (d - 1) * Nf * Nx`, with `Nf` the coupling's free-row count — and then the
coupling's own unknowns, in the layout of the coupling's block `coupling`, whose type is `CD`.

The data of a step are `bottom` and `flux`, the coupling's datum and its momentum load, and the
results of the last step are `top`, the top row of the coefficient array, `residual`, the last weak
form, and the trace at the top of the slab. `iterations`, `residuals` and `initial_residuals`
record the Newton solve of every step: its iterate count, its final residual `‖F(x)‖₂` and its
initial residual `r₀ = ‖F(x₀)‖₂`. `steps` counts the steps taken, which is what tells the coupling
and the guess whether the slab is the run's first.

The buffers of a [`FitTo`](@ref) guess keep a step's guess from allocating: `sample`, of size
`(Qt, Qx, D)`, holds the target at the quadrature points; `fit`, of size `(Nt, Nx, D)`, holds the
fitted coefficients; `initial_sample`, of size `(Qx, D)`, holds the problem's initial condition at
the nodes of the space quadrature, evaluated once; and `taylor` holds the Taylor coefficients of
the top time cell that [`PreviousSlab`](@ref) expands. `weighted`, `load` and `fibre` are the
scratch of the [`Projection`](@ref) fit: the quadrature-weighted sample, its load in the time
basis, and one space fibre of the coefficients.
"""
mutable struct SpaceTimeGalerkinCache{ST, T, CD <: CouplingData} <: IntegratorCache{ST}
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
    sample::Array{ST, 3}
    fit::Array{ST, 3}
    initial_sample::Matrix{ST}
    taylor::Array{ST, 3}
    weighted::Matrix{ST}
    load::Matrix{ST}
    fibre::Vector{ST}
    coupling::CD
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
    initial_sample = initial_condition_sample(ST, problem, a, qx)
    return SpaceTimeGalerkinCache{
        ST, T, coupling_data_type(coupling(method), a, problem, ST)}(
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
        evaluation,
        zeros(ST, qt, qx, D),
        similar(coeffs),
        initial_sample,
        zeros(ST, nx, D, degree(time_basis(a)) + 1),
        zeros(ST, qt, qx),
        zeros(ST, nt, qx),
        zeros(ST, nx),
        coupling_data(coupling(method), a, problem, ST))
end

# The problem's initial condition at the nodes of the space quadrature, per component.
function initial_condition_sample(::Type{ST}, problem::LagrangianPDEProblem,
        ansatz::TensorBSplineAnsatz, qx::Integer) where {ST}
    xq = collect(space_quadrature_nodes(ansatz))
    values = problem.ics_function(xq, parameters(problem))
    D = components(problem)
    return ST[convert(ST, values.u[(d - 1) * qx + r]) for r in 1:qx, d in 1:D]
end

@inline CacheType(ST, problem::LagrangianPDEProblem{T},
    method::SpaceTimeGalerkin{<:TensorBSplineAnsatz{T}}) where {T} = SpaceTimeGalerkinCache{
    ST, T, coupling_data_type(coupling(method), ansatz(method), problem, ST)}

"""
    cache(int::GeometricIntegrator, ST)
    cache(int::GeometricIntegrator)

The slab cache at element type `ST`, and at the problem's own element type.

The ecosystem's accessor takes the element type as a value, which the compiler cannot specialise on,
so the type assertion of the cache dictionary rebuilds the cache's type at every lookup. These two
methods take it as a type parameter, which lets that assertion fold away, so that one slab's residual
evaluation allocates nothing.
"""
@inline function cache(
        int::GeometricIntegrator{<:SpaceTimeGalerkin, <:LagrangianPDEProblem}, ::Type{ST}) where {ST}
    return int.caches[ST]
end

@inline function cache(
        int::GeometricIntegrator{<:SpaceTimeGalerkin, <:LagrangianPDEProblem{T}}) where {T}
    return cache(int, T)
end

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
        nt::Integer, nf::Integer) where {ST}
    nx, D = size(bottom)
    m = nt - nf
    for d in 1:D
        for j in 1:nx
            m == 0 || (coeffs[1, j, d] = bottom[j, d])
            for a in (m + 1):nt
                coeffs[a, j, d] = x[free_index(a - m, j, d, nf, nx)]
            end
        end
    end
    return nothing
end

"""The solver index of the free coefficient of free row `a`, space function `j`, component `d`."""
@inline function free_index(a::Integer, j::Integer, d::Integer, nf::Integer, nx::Integer)
    return a + (j - 1) * nf + (d - 1) * nf * nx
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

The nonlinear residual of one slab step, in the solver's own layout: one equation per free
coefficient, ordered as the solver's vector is, and after them the coupling's own equations.

The equations of the free coefficients are the interior weak form tested against the basis
functions that the slab's free rows carry, and the coupling adds its own terms to them.
"""
function residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params,
        int::GeometricIntegrator{<:SpaceTimeGalerkin, <:LagrangianPDEProblem}) where {ST}
    data = cache(int)
    c = cache(int, ST)
    copy_datum!(c, data)
    a = ansatz(method(int))
    nt = ntime_functions(a)
    nx, D = size(c.bottom)
    nf = free_rows(c.coupling, a)
    scatter!(c.coeffs, x, c.bottom, nt, nf)
    slab_weak_form!(c, problem(int), method(int))
    for d in 1:D, j in 1:nx, k in 1:nf
        b[free_index(k, j, d, nf, nx)] = c.residual[k, j, d]
    end
    coupling_term!(b, x, coupling(method(int)), c.coupling, c, a, problem(int))
    return nothing
end

"""Copy the coupling's datum of the step from the data cache into a cache of another element type."""
function copy_datum!(c::SpaceTimeGalerkinCache, data::SpaceTimeGalerkinCache)
    c.bottom .= data.bottom
    c.flux .= data.flux
    copy_coupling_data!(c.coupling, data.coupling)
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
    nf = free_rows(c.coupling, a)
    scatter!(c.coeffs, x, c.bottom, nt, nf)
    c.top .= view(c.coeffs, nt, :, :)
    h = timestep(p)
    assign!(sol.u, sol.v, sol.w, c, a, nx, D, h)
    slab_weak_form!(c, p, m)
    c.flux .= view(c.residual, nt, :, :)
    coupling_update!(c.coupling, x, c, a, p)
    c.steps += 1
    return nothing
end

"""
    trace!(trace_u, trace_v, trace_w, cache, ansatz, h)

The trace `(u, v, w)` of the slab's top row at the nodes of the space quadrature, component-major,
into three matrices with a row per node of the space rule.

`∂ₜu` is the time derivative of the normalised coordinate divided by the slab's `h`, which is the
factor the interior weak form carries as well.
"""
function trace!(trace_u::AbstractMatrix{ST}, trace_v::AbstractMatrix{ST},
        trace_w::AbstractMatrix{ST}, c::SpaceTimeGalerkinCache{ST}, a::TensorBSplineAnsatz,
        h) where {ST}
    e = c.evaluation
    nt = ntime_functions(a)
    nx, D = size(c.bottom)
    hh = convert(ST, h)
    fill!(trace_u, zero(ST))
    fill!(trace_v, zero(ST))
    fill!(trace_w, zero(ST))
    for d in 1:D
        for j in 1:nx
            # the time derivative at the top of the slab, per space function: `∂ₜu` is `∂_{t̂}u / h`
            tderiv = zero(ST)
            for k in 1:nt
                tderiv += c.coeffs[k, j, d] * e.top_slope[k]
            end
            tderiv /= hh
            for i in axes(e.trace, 1)
                trace_u[i, d] += c.coeffs[nt, j, d] * e.trace[i, j]
                trace_w[i, d] += c.coeffs[nt, j, d] * e.trace_slope[i, j]
                trace_v[i, d] += tderiv * e.trace[i, j]
            end
        end
    end
    return nothing
end

"""
    assign!(u, v, w, cache, ansatz, nx, D, h)

The solution and its two derivatives at the top of the slab, on the output grid, and beside them the
trace `(u, v, w)` at the nodes of the space quadrature, which [`trace!`](@ref) fills from the same
top row.
"""
function assign!(u::AbstractVector{ST}, v::AbstractVector{ST}, w::AbstractVector{ST},
        c::SpaceTimeGalerkinCache{ST}, a::TensorBSplineAnsatz, nx::Integer, D::Integer,
        h) where {ST}
    e = c.evaluation
    nt = ntime_functions(a)
    n = size(e.output, 1)
    hh = convert(ST, h)
    fill!(u, zero(ST))
    fill!(v, zero(ST))
    fill!(w, zero(ST))
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
        end
    end
    trace!(c.trace_u, c.trace_v, c.trace_w, c, a, h)
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

The size of one slab's nonlinear system: one equation per free coefficient, and one per unknown the
coupling adds after them.
"""
function total_free_coefficients(method::SpaceTimeGalerkin, problem::LagrangianPDEProblem)
    a = ansatz(method)
    c = coupling(method)
    return free_rows(c, a) * nspace_functions(a) * components(problem) +
           coupling_unknowns(c, a, problem)
end
