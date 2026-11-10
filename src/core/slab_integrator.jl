"""
    SpaceTimeGalerkin{A, C, G} <: SpaceTimeMethod

The space-time Galerkin slab integrator: an ansatz `A`, a coupling `C` of consecutive slabs and an
initial guess `G`.

The ansatz is a [`SpaceTimeAnsatz`](@ref), the coupling a [`SlabCoupling`](@ref) and the guess a
[`SlabInitialGuess`](@ref) — [`Continuation`](@ref) or [`FitTo`](@ref). A value of another role,
a method in the place of the ansatz or a target in the place of the guess, is refused when the
method is built.

One step of the integrator solves one slab. The unknown of a [`TensorBSplineAnsatz`](@ref) is its
coefficient array, `(Nt, Nx)` per component, the unknown of a [`SymbolicAnsatz`](@ref) the
parameter vector of its expression, and the unknown of a [`NetworkAnsatz`](@ref) the free part of
its parameter vector. For the coefficient array, a coupling that fixes the bottom
row of the time direction supplies it as a datum — the initial condition at the first step and the
previous slab's top row after it — and the remaining `Nt - 1` rows are determined by the interior
weak Euler–Lagrange form of the slab, tested against the basis functions that vanish at the top of
the slab. The `v` term of that form leaves one boundary term at the bottom of the slab, the
momentum flux, which the coupling supplies as well; the top of the slab is free, so its flux is not
imposed but measured, and handed on as the next slab's bottom flux.

The unknown count is the coupling's: the free rows of each space function and each component, plus
the coupling's own unknowns behind them. A [`CoefficientRestriction`](@ref) fixes the bottom row, so
it leaves `(Nt - 1) * Nx * D` free coefficients; a weak-multiplier coupling imposes the initial
condition with multipliers of its own, so all `Nt` rows are free and the multipliers follow them.
A symbolic ansatz has no rows, and neither has a network: their parameters are their free
unknowns, and [`WeakInitialAndBoundaryConditions`](@ref), the one coupling either takes, puts its
multipliers after them. A step is a nonlinear system of
[`total_free_coefficients`](@ref)`(method, problem)` equations, which the ecosystem's Newton solver
solves, with a pivoted QR as its linear solve for a symbolic or a network ansatz. With
`linear_solver = true`, for an ansatz whose step is affine in its unknowns and a linear density, one
linear solve replaces Newton's iterations: the step's equations are the slab's weak-form rows — a
network's interior weak form beside the coupling's weak initial and boundary rows — with the
Jacobian of the whole system, and a solution is read from zero and then refined once with the same
factorisation, two solves in all and no iterate. The factorisation is the solver's own linear
method — the `linear_solver_method` of its configuration, which is
`SimpleSolvers.LapackPivotedQR` for a network — taken at the run's first step and again at each step
whose guess has moved the parameters the step holds fixed, because the Jacobian depends on them: for
a `:last_layer` network those are the hidden parameters before its free output weights, of which the
cache keeps a copy and compares it exactly. The Jacobian of such a step is a saddle point whose
constraint rows reach only the free output weights of a `:last_layer` network, so the pivoted QR of a
network solves it by its minimum-norm step, where a factorisation without rank detection raises on a
zero pivot. In `Float32` that step leaves a residual of order `1e-2` of the initial one — at most
`2.4e-2` over three steps of `LinearTransport` at `S = 20` and `D = 2`, where `Float64` leaves
`1.5e-14` — so the solver's default tolerances record such a step as not converged. The iterate
count, the residual, the initial residual and the convergence of every step are recorded in the
cache, and the trace `(u, v, w)` at the top of the slab, at the nodes of the space quadrature, is
kept for the diagnostics.

The roles of the ansatz, the coupling and the guess, and the functions through which the core
reaches each of them, are on the page *The slab seam* of the manual.

```julia
method = SpaceTimeGalerkin(
    TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0)),
    CoefficientRestriction();
    initial_guess = Continuation(),
    linear_solver = false)

result = integrate(problem, method)
```
"""
struct SpaceTimeGalerkin{A <: SpaceTimeAnsatz, C <: SlabCoupling, G <: SlabInitialGuess} <:
       SpaceTimeMethod
    ansatz::A
    coupling::C
    initial_guess::G
    linear_solver::Bool
end

function SpaceTimeGalerkin(ansatz::A,
        coupling::C,
        initial_guess::G; linear_solver::Bool = false) where {
        A <: SpaceTimeAnsatz, C <: SlabCoupling, G <: SlabInitialGuess}
    return SpaceTimeGalerkin{A, C, G}(ansatz, coupling, initial_guess, linear_solver)
end

function SpaceTimeGalerkin(ansatz::SpaceTimeAnsatz,
        coupling::SlabCoupling = CoefficientRestriction();
        initial_guess::SlabInitialGuess = Continuation(),
        linear_solver::Bool = false)
    return SpaceTimeGalerkin{typeof(ansatz), typeof(coupling), typeof(initial_guess)}(
        ansatz, coupling, initial_guess, linear_solver)
end

function SpaceTimeGalerkin(method::SpaceTimeMethod, args...; kwargs...)
    throw(ArgumentError("SpaceTimeGalerkin needs a SpaceTimeAnsatz, got the method " *
                        "$(nameof(typeof(method)))"))
end

"""The ansatz of the method."""
ansatz(method::SpaceTimeGalerkin) = method.ansatz

"""The coupling of the method, or the coupling that a coupling's block of the cache carries."""
coupling(method::SpaceTimeGalerkin) = method.coupling

"""The initial guess of the method."""
initial_guess(method::SpaceTimeGalerkin) = method.initial_guess

default_solver(::SpaceTimeGalerkin) = Newton()

"""
    check_method(problem, method)

Reject a method that the problem cannot support, before any step runs.

The refusals are `ArgumentError`s: a target that reads the exact solution of a problem that has
none, a fit that needs an ansatz linear in its coefficients on one that is not, a direct linear
solve (`linear_solver = true`) on an ansatz whose step is not affine in its unknowns or a density
that is not linear, and what the ansatz's [`check_ansatz`](@ref) and the coupling's
[`check_coupling`](@ref) refuse.

The direct linear solve is offered to an ansatz whose step is affine in its unknowns, which
[`has_affine_step`](@ref) reports.
"""
function check_method(problem::LagrangianPDEProblem, method::SpaceTimeGalerkin)
    a = ansatz(method)
    if method.linear_solver
        has_affine_step(a) ||
            throw(ArgumentError("linear_solver = true needs an ansatz whose step is affine in its " *
                                "unknowns; $(nameof(typeof(a))) is not"))
        is_linear_density(problem.density) ||
            throw(ArgumentError("linear_solver = true needs a linear Lagrangian density; " *
                                "the problem's density is not linear"))
    end
    guess = initial_guess(method)
    if guess isa FitTo
        if needs_exact_solution(guess.target) && problem.solutions === nothing
            throw(ArgumentError("FitTo(ExactSolution()) needs a problem with an exact solution; " *
                                "this problem has solutions = nothing"))
        end
        if fit_method(guess, ansatz(method)) isa Projection &&
           !is_linear_in_coefficients(ansatz(method))
            throw(ArgumentError("Projection() needs an ansatz that is linear in its " *
                                "coefficients; $(nameof(typeof(ansatz(method)))) is not"))
        end
    end
    check_ansatz(problem, a)
    check_coupling(problem, coupling(method), a)
    return nothing
end

"""
    check_ansatz(problem, ansatz)

Reject an ansatz that the problem cannot support, when the integrator is built. An ansatz that
declares no requirement is accepted.
"""
check_ansatz(::LagrangianPDEProblem, ::SpaceTimeAnsatz) = nothing

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
    CouplingData

The coupling's block of the slab cache.

A coupling fixes how the free coefficients of a slab are laid out, adds terms to the residual, may
transform the field, and may carry unknowns of its own between the coefficients and behind them.
The block carries the coupling itself, in a field `coupling`, and what the coupling keeps between
slabs — its datum, its unknowns, the tables its residual term evaluates and the scratch of that
evaluation. It is built once, when the cache is built, and its type is a parameter of the cache.
Every function of the coupling that runs within a step dispatches on the block.
"""
abstract type CouplingData end

coupling(block::CouplingData) = block.coupling

"""
    NoCouplingData{C}

The block of a coupling `C` that keeps no data of its own: it carries the coupling alone, and has
nothing to guess, to update or to copy.
"""
struct NoCouplingData{C <: SlabCoupling} <: CouplingData
    coupling::C
end

"""
    coupling_data(coupling, ansatz, problem, ST)

The coupling's block of the slab cache, built once per integrator. A coupling that keeps no data
of its own gets a [`NoCouplingData`](@ref).
"""
function coupling_data(coupling::SlabCoupling, ::SpaceTimeAnsatz, ::LagrangianPDEProblem,
        ::Type{ST}) where {ST}
    return NoCouplingData(coupling)
end

"""
    coupling_data_type(coupling, ansatz, problem, ST)

The type of [`coupling_data`](@ref)`(coupling, ansatz, problem, ST)`, computed without building the
block, because the ecosystem asserts the cache's type on every lookup.
"""
function coupling_data_type(
        coupling::SlabCoupling, ::SpaceTimeAnsatz, ::LagrangianPDEProblem,
        ::Type{ST}) where {ST}
    return NoCouplingData{typeof(coupling)}
end

"""
    free_rows(coupling, ansatz)

The number of free time rows of the slab: the rows of the coefficient array that the solver
determines, the time index running fastest over the coefficient part of the solver's vector.

Every coupling declares its own: one that fixes the bottom row of the coefficient array leaves
`Nt - 1` free rows, and one that imposes the initial condition weakly leaves all `Nt`.
"""
function free_rows end

"""
    coupling_unknowns(coupling, ansatz, problem)

The number of the coupling's extra unknowns: the entries of the solver's vector that follow the
coefficients.
"""
coupling_unknowns(::SlabCoupling, ::SpaceTimeAnsatz, ::LagrangianPDEProblem) = 0

"""
    coupling_term!(b, x, data, cache, ansatz, problem)

The coupling's own part of the residual, from its block `data`: its terms on the free rows the
interior weak form already carries, and its own equations after them.

There is no default: a coupling whose residual has no term of its own declares its no-op.
"""
function coupling_term! end

"""
    coupling_guess!(data, cache, ansatz, problem, sol)

The coupling's initial guess of its unknowns, once per slab.
"""
coupling_guess!(::NoCouplingData, cache, ansatz, problem, sol) = nothing

"""
    coupling_update!(data, x, cache, ansatz, problem)

Read the coupling's state out of the solved slab: its unknowns in the solver's vector, and what it
hands on to the next slab.
"""
coupling_update!(::NoCouplingData, x, cache, ansatz, problem) = nothing

"""
    copy_coupling_data!(cache_data, data)

Copy the coupling's datums from the cache of the problem's element type into the cache the solver
works in.
"""
copy_coupling_data!(::NoCouplingData, ::NoCouplingData) = nothing

"""
    copy_ansatz_data!(cache_data, data)

Copy what the ansatz block keeps outside the solver's vector — the parameters it holds fixed — from
the cache of the problem's element type into the cache the solver works in, where the solver
evaluates at another element type.

An ansatz whose every unknown is a row of the solver's vector keeps nothing here, and the default
does nothing; a network in `:last_layer` mode frees its output weights alone and leaves its hidden
layer in the block, which the block of another element type does not inherit.
"""
copy_ansatz_data!(cache_data, data) = nothing

"""
    has_affine_step(ansatz)

Whether the ansatz's step is affine in its unknowns, which the direct linear solve
(`linear_solver = true`) needs. An ansatz linear in its coefficients has one; a
[`NetworkAnsatz`](@ref) has one in `:last_layer` mode alone, whose free output weights enter the
step affinely while its hidden layer is data, and not in `:all` mode, where Newton determines every
layer.
"""
has_affine_step(a) = is_linear_in_coefficients(a)

"""
    held_parameters(block)

The parameters the block's step holds fixed: the ones it reads and never frees, on which the step's
Jacobian therefore depends. An ansatz whose every unknown is a row of the solver's vector holds
none, and the default is the empty range.
"""
held_parameters(block) = 1:0

"""
    bottom_datum(data)

The bottom row of the coefficient array that the coupling fixes, of size `(Nx, D)`, or `nothing`
for a coupling whose free rows are all `Nt` rows.
"""
bottom_datum(::NoCouplingData) = nothing

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
    slab_bottom!(cache, data, problem, ansatz, sol)

Fix the coupling's datum of the slab, in its block `data`.

A coupling implements this for the two cases it distinguishes: the first slab of a run, where the
datum comes from the problem's initial condition, and every later slab, where it is what the
previous slab carried out. The cache counts its steps, so `cache.steps` says which case this is.
"""
function slab_bottom! end

"""
    SpaceTimeGalerkinCache{ST, AD, CD}

The cache of the slab integrator.

`x` is the nonlinear solver's vector: the free coefficients of the slab, laid out component-major
with the time index fastest — the entry of free row `a ∈ 1:Nf`, space function `j` and component `d`
is `a + (j - 1) * Nf + (d - 1) * Nf * Nx`, with `Nf` the coupling's free-row count — and then the
coupling's own unknowns, in the layout of the coupling's block.

`u`, `v` and `w` hold the field and its derivatives at the points of the slab's quadrature, and
`du`, `dv` and `dw` the density's derivatives there; `trace_u`, `trace_v` and `trace_w` hold the
trace at the top of the last slab. `iterations`, `residuals`, `initial_residuals` and `converged`
record the Newton solve of every step: its iterate count, its final residual `‖F(x)‖₂`, its initial
residual `r₀ = ‖F(x₀)‖₂` and whether the step converged. The direct linear solve records its two
solves, the first and the refinement step, as two iterates. `steps` counts the steps taken, which is
what tells the coupling and the guess whether the slab is the run's first.

`ansatz` is the ansatz's block, of type `AD`, which holds what the ansatz keeps of a slab, and
`coupling` the coupling's block, of type `CD`.
"""
mutable struct SpaceTimeGalerkinCache{ST, AD, CD <: CouplingData} <: IntegratorCache{ST}
    x::Vector{ST}
    u::Array{ST, 3}
    v::Array{ST, 3}
    w::Array{ST, 3}
    du::Array{ST, 3}
    dv::Array{ST, 3}
    dw::Array{ST, 3}
    trace_u::Matrix{ST}
    trace_v::Matrix{ST}
    trace_w::Matrix{ST}
    iterations::Vector{Int}
    residuals::Vector{ST}
    initial_residuals::Vector{ST}
    converged::Vector{Bool}
    steps::Int
    factorisations::Int
    linear_held::Vector{ST}
    linear_rhs::Vector{ST}
    linear_zero::Vector{ST}
    ansatz::AD
    coupling::CD
end

function Cache{ST}(problem::LagrangianPDEProblem,
        method::SpaceTimeGalerkin; kwargs...) where {ST}
    check_method(problem, method)
    a = ansatz(method)
    slab = ansatz_data(a, problem, ST)
    tnodes, xnodes = quadrature_points(slab)
    qt = length(tnodes)
    qx = length(xnodes)
    D = components(problem)
    nfree = total_free_coefficients(method, problem)
    nlin = method.linear_solver ? nfree : 0
    nheld = method.linear_solver ? length(held_parameters(slab)) : 0
    return SpaceTimeGalerkinCache{ST, typeof(slab),
        coupling_data_type(coupling(method), a, problem, ST)}(
        zeros(ST, nfree),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, qt, qx, D),
        zeros(ST, qx, D),
        zeros(ST, qx, D),
        zeros(ST, qx, D),
        Int[],
        ST[],
        ST[],
        Bool[],
        0,
        0,
        zeros(ST, nheld),
        zeros(ST, nlin),
        zeros(ST, nlin),
        slab,
        coupling_data(coupling(method), a, problem, ST))
end

@inline function CacheType(ST, problem::LagrangianPDEProblem, method::SpaceTimeGalerkin)
    return SpaceTimeGalerkinCache{ST, ansatz_data_type(ansatz(method), problem, ST),
        coupling_data_type(coupling(method), ansatz(method), problem, ST)}
end

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

function Base.show(io::IO, cache::SpaceTimeGalerkinCache{ST}) where {ST}
    print(io, "SpaceTimeGalerkinCache{$ST} with $(length(cache.x)) free coefficients")
end

"""
    reset!(cache::SpaceTimeGalerkinCache)

Clear the state a run leaves in the cache: the step counter, which is what tells the coupling and
the guess whether the slab is the run's first, the counter of linear-solver factorisations, at zero
of which the next step factorises again, and the per-step records of the Newton solve.

A run starts at the problem's initial data, so a cache that still counted the previous run's steps
would hand that run's last top row to the new first slab as its bottom datum.
"""
function reset!(c::SpaceTimeGalerkinCache)
    c.steps = 0
    c.factorisations = 0
    empty!(c.iterations)
    empty!(c.residuals)
    empty!(c.initial_residuals)
    empty!(c.converged)
    return nothing
end

"""
    factorisations(cache::SpaceTimeGalerkinCache)
    factorisations(int::GeometricIntegrator)

The number of Jacobian factorisations computed by the direct linear solver since the last `reset!`
of the cache, which is one per run whose guess leaves the parameters the step holds fixed unchanged,
and one more at every step whose guess has moved them.
"""
factorisations(c::SpaceTimeGalerkinCache) = c.factorisations
function factorisations(int::GeometricIntegrator{
        <:SpaceTimeGalerkin, <:LagrangianPDEProblem})
    return factorisations(cache(int))
end

"""
    solve_stats(cache::SpaceTimeGalerkinCache)

The records of the Newton solves of the steps taken, as the named tuple
`(; iterations, residuals, initial_residuals, converged)`, one entry per step.
"""
function solve_stats(c::SpaceTimeGalerkinCache)
    return (; iterations = c.iterations, residuals = c.residuals,
        initial_residuals = c.initial_residuals, converged = c.converged)
end

"""The solver index of the free coefficient of free row `a`, space function `j`, component `d`."""
@inline function free_index(a::Integer, j::Integer, d::Integer, nf::Integer, nx::Integer)
    return a + (j - 1) * nf + (d - 1) * nf * nx
end

# The field at the top of the slab, `t̂ = 1`, at the points of `evaluation`, with the coupling's
# transform: `u`, `v` and `w` are indexed `[i, d]`, a row per point.
function top_field!(u::AbstractMatrix, v::AbstractMatrix, w::AbstractMatrix,
        c::SpaceTimeGalerkinCache{ST}, evaluation, problem::LagrangianPDEProblem) where {ST}
    h = timestep(problem)
    top_values!(u, v, w, c.ansatz, evaluation, h)
    n, D = size(u)
    transform_field!(reshape(u, 1, n, D), reshape(v, 1, n, D), reshape(w, 1, n, D),
        c.coupling, c.ansatz, (one(ST),), evaluation_points(evaluation), h, problem)
    return nothing
end

"""
    transform_field!(u, v, w, c.coupling, c.ansatz, tnodes, xnodes, h, problem)

The coupling's transform of the field, applied to the ansatz's field `u` and its derivatives
`v = ∂ₜu` and `w = ∂ₓu` at the points `(t̂[q], x[r])` of the slab, as `u[q, r, d]`, in place. `t̂` is
the normalised time of the slab, so the physical-time derivative of a term `ℓ(t̂, x)` is
`∂_{t̂}ℓ / h`.

The core applies it wherever it evaluates the field: in the residual, at the nodes of the slab's
quadrature, and at the top of the slab, in the trace and in the solution. It hands the coupling's
block and the ansatz's block, so a transform that is not a function of the point alone — one that
projects the field, for instance — reads the parameters the step is solving for from the ansatz's
block. The default leaves the field as the ansatz gives it; a coupling that lifts the field adds
its lift here.
"""
function transform_field!(u, v, w, ::CouplingData, block, t̂, x, h, problem)
    return nothing
end

"""
    slab_residual!(cache, x, problem, nf)

The interior weak form of the slab that the solver's vector `x` describes, in the ansatz's block:
the ansatz's field at the nodes of the slab's quadrature, with the coupling's fixed bottom row and
its transform of the field, the density's derivatives there, and their weak form.
"""
function slab_residual!(
        c::SpaceTimeGalerkinCache, x::AbstractVector, problem::LagrangianPDEProblem,
        nf::Integer)
    h = timestep(problem)
    slab_field!(c.u, c.v, c.w, c.ansatz, x, bottom_datum(c.coupling), nf, h)
    tnodes, xnodes = quadrature_points(c.ansatz)
    transform_field!(c.u, c.v, c.w, c.coupling, c.ansatz, tnodes, xnodes, h, problem)
    density_derivatives!(c.du, c.dv, c.dw, problem.density, c.u, c.v, c.w,
        parameters(problem))
    slab_weak_form!(c.ansatz, c.du, c.dv, c.dw, h)
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
    copy_coupling_data!(c.coupling, data.coupling)
    copy_ansatz_data!(c.ansatz, data.ansatz)
    a = ansatz(method(int))
    p = problem(int)
    nf = free_rows(coupling(c.coupling), a)
    slab_residual!(c, x, p, nf)
    free_residual!(b, c.ansatz, nf)
    coupling_term!(b, x, c.coupling, c, a, p)
    return nothing
end

"""
    update!(sol, params, x, int)

Write the solution of the slab into `sol`, keep its trace, and let the coupling read what it hands
on to the next step.
"""
function update!(sol, params, x::AbstractVector{ST},
        int::GeometricIntegrator{<:SpaceTimeGalerkin, <:LagrangianPDEProblem}) where {ST}
    c = cache(int, ST)
    a = ansatz(method(int))
    p = problem(int)
    slab_residual!(c, x, p, free_rows(coupling(c.coupling), a))
    keep_top!(c.ansatz)
    assign!(sol.u, sol.v, sol.w, c, p)
    coupling_update!(c.coupling, x, c, a, p)
    c.steps += 1
    return nothing
end

"""
    trace!(cache, problem)

The trace `(u, v, w)` at the top of the slab at the nodes of the space quadrature, component-major,
into the cache's three trace matrices, with a row per node of the space rule.
"""
function trace!(c::SpaceTimeGalerkinCache, problem::LagrangianPDEProblem)
    top_field!(c.trace_u, c.trace_v, c.trace_w, c, trace_evaluation(c.ansatz), problem)
    return nothing
end

"""
    assign!(u, v, w, cache, problem)

The solution and its two derivatives at the top of the slab, on the output grid, component-major,
and beside them the trace `(u, v, w)` at the nodes of the space quadrature, which [`trace!`](@ref)
fills. Both come from one evaluation of the top of the slab, at two sets of points.
"""
function assign!(u::AbstractVector, v::AbstractVector, w::AbstractVector,
        c::SpaceTimeGalerkinCache, problem::LagrangianPDEProblem)
    D = size(c.trace_u, 2)
    n = length(u) ÷ D
    top_field!(reshape(u, n, D), reshape(v, n, D), reshape(w, n, D), c,
        output_evaluation(c.ansatz), problem)
    trace!(c, problem)
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
    step_converged(residual, initial_residual, options)

Whether a step's Newton solve converged: `residual ≤ f_abstol + f_reltol · initial_residual`, with
the tolerances of the solver's `options`. The relative term counts only for a finite
`initial_residual`, as in SimpleSolvers' own residual test, so the bound is finite and a residual
of `Inf` or `NaN` never converges.
"""
function step_converged(residual, initial_residual, options)
    relative = isfinite(initial_residual) ? options.f_reltol * initial_residual :
               zero(residual)
    return residual ≤ options.f_abstol + relative
end

"""
    integrate_step!(sol, history, params, int)

One slab: the coupling fixes its datum, the initial guess fills the solver's vector, the
ecosystem's Newton solver (or, when `method.linear_solver` is `true`, one solve with the cached
factorisation of the solver's own linear method, from zero, and one step of iterative refinement
with it) solves the slab, the cache records the solve, and the update writes the solution, the trace
and the data the next slab starts from.
"""
function integrate_step!(sol, history, params,
        int::GeometricIntegrator{<:SpaceTimeGalerkin, <:LagrangianPDEProblem})
    c = cache(int)
    m = method(int)
    slab_bottom!(c, c.coupling, problem(int), ansatz(m), sol)
    initial_guess!(sol, history, params, int)
    if m.linear_solver
        x = nlsolution(int)
        s = solver(int)
        residual!(c.linear_rhs, x, sol, params, int)
        r₀ = norm(c.linear_rhs)
        lsolver = linearsolver(s)
        held = held_parameters(c.ansatz)
        if c.factorisations == 0 || held != c.linear_held
            jacobian!(s, c.linear_zero, (sol, params, int))
            factorize!(lsolver, jacobianmatrix(s))
            c.factorisations += 1
            copyto!(c.linear_held, held)
        end
        residual!(c.linear_rhs, c.linear_zero, sol, params, int)
        ldiv!(c.linear_rhs, lsolver, c.linear_rhs)
        x .= .-c.linear_rhs
        # one step of iterative refinement with the same factorisation
        residual!(c.linear_rhs, x, sol, params, int)
        ldiv!(c.linear_rhs, lsolver, c.linear_rhs)
        x .-= c.linear_rhs
        residual!(c.linear_rhs, x, sol, params, int)
        rfₐ = norm(c.linear_rhs)
        push!(c.iterations, 2)
        push!(c.residuals, convert(eltype(c.residuals), rfₐ))
        push!(c.initial_residuals, convert(eltype(c.initial_residuals), r₀))
        push!(c.converged,
            step_converged(last(c.residuals), last(c.initial_residuals), config(s)))
        update!(sol, params, x, int)
        return nothing
    end
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

"""
    nunknowns(ansatz, coupling, problem)

The number of the ansatz's own unknowns of one slab: the entries of the solver's vector that precede
the coupling's.

A tensor B-spline ansatz has one unknown per free row of its coefficient array, per space function
and per component, so its count is

```julia
nunknowns(ansatz::TensorBSplineAnsatz, coupling, problem) =
    free_rows(coupling, ansatz) * nspace_functions(ansatz) * components(problem)
```

with the coupling that leaves the free rows and the problem that fixes the component count in scope.
An ansatz that keeps its unknowns as a parameter vector declares its own count instead; a
[`NetworkAnsatz`](@ref) has one per parameter its mode frees.
"""
function nunknowns(ansatz::TensorBSplineAnsatz, coupling::SlabCoupling,
        problem::LagrangianPDEProblem)
    return free_rows(coupling, ansatz) * nspace_functions(ansatz) * components(problem)
end

"""
    slab_unknowns(ansatz, coupling, problem)

The size of one slab's nonlinear system: one equation per unknown of the ansatz, and one per unknown
the coupling adds after them.
"""
function slab_unknowns(ansatz, coupling, problem)
    nunknowns(ansatz, coupling, problem) + coupling_unknowns(coupling, ansatz, problem)
end

"""
    total_free_coefficients(method, problem)

The size of one slab's nonlinear system: one equation per unknown of the ansatz, and one per unknown
the coupling adds after them.
"""
function total_free_coefficients(method::SpaceTimeGalerkin, problem::LagrangianPDEProblem)
    return slab_unknowns(ansatz(method), coupling(method), problem)
end
