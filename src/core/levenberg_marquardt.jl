"""
    LevenbergMarquardt(; variable_projection = false) <: NonlinearSolverMethod

The Levenberg–Marquardt nonlinear solver of a fixed-boundary network slab.

The step is damped by a shift on the singular values of the Jacobian,

```math
\\delta(\\mu) = -V \\operatorname{diag}\\bigl(\\sigma_i / (\\sigma_i^2 + \\mu)\\bigr) U^\\top r,
```

from one decomposition of the Jacobian per accepted iterate, with the identity as the damping
matrix. `μ` starts at `1e-6`, is divided by 3 after an accepted step down to `1e-15`, and
multiplied by 4 after a rejected one. A trial point whose residual is not finite is rejected, and a
guess whose residual is not finite ends the step at once with no iteration. The solve stops on the
rule of `step_converged` with the integrator's own options, at `max_iterations` (300 by default),
or once `μ` reaches `1e12`. Its iteration count is the number of trial steps, accepted or rejected,
and its residual is `‖F‖` of the whole system, so the slab's record path is the one every family
uses.

With `variable_projection = true` the bubble's output weights are eliminated: an inner Newton solve
brings their own residuals to the floor by the minimum-norm step of a truncated decomposition, and
the outer solve is Levenberg–Marquardt on the Schur complement `J_oo − J_oa J_aa⁺ J_ao`. The
residuals the cache records stay those of the whole system, so that a joint row and a projected row
of the benchmark matrix compare one quantity.

A fixed-boundary slab's residual has a floor above the machine epsilon of its type under either
variant, and the projected solve reaches a lower one than the joint solve; a row's convergence is
recorded and not asserted, which is why both are offered.

```julia
integrate(problem, SpaceTimeGalerkin(FixedBoundaryNetwork(xspan = (0.0, 1.0)), FixedBoundary());
    solver = LevenbergMarquardt(variable_projection = true))
```
"""
struct LevenbergMarquardt <: NonlinearSolverMethod
    variable_projection::Bool
end

function LevenbergMarquardt(; variable_projection = false)
    return LevenbergMarquardt(Bool(variable_projection))
end

"""The damping `μ` a solve starts from."""
const LM_MU_START = 1e-6

"""The smallest damping an accepted step divides down to."""
const LM_MU_MIN = 1e-15

"""The damping at which a solve gives up: every singular value is damped away there."""
const LM_MU_MAX = 1e12

"""The iteration cap of a Levenberg–Marquardt solve when the integrator's options name none."""
const LM_MAX_ITERATIONS = 300

"""The iteration cap of the inner Newton solve of the projected variant."""
const LM_INNER_ITERATIONS = 50

"""The relative singular value below which a decomposition's rank stops."""
const LM_TRUNCATION = 1e-12

"""
    LevenbergMarquardtSolver{ST, XT} <: AbstractSolver

The solver object of one integrator: the options the integrator was handed, a zero vector of the
solution's type and length for the state, the flag of the projected variant, and the positions in
the solver's vector of the unknowns that the projection eliminates, or `nothing` for a method that
names none.
"""
struct LevenbergMarquardtSolver{ST, XT <: AbstractVector{ST}} <:
       GeometricBase.AbstractSolver
    config::Options{ST}
    zero::XT
    variable_projection::Bool
    projected::Union{Nothing, Vector{Int}}
end

config(s::LevenbergMarquardtSolver) = s.config

"""
    LevenbergMarquardtState{ST} <: AbstractSolverState

What one solve keeps between its iterations: the iteration count, the iterate, and the initial
residual `‖F(x₀)‖` that the convergence rule is relative to.
"""
mutable struct LevenbergMarquardtState{ST, XT <: AbstractVector{ST}} <:
               GeometricBase.AbstractSolverState
    iterations::Int
    x::XT
    r₀::ST
end

function LevenbergMarquardtState(s::LevenbergMarquardtSolver{ST}) where {ST}
    return LevenbergMarquardtState{ST, typeof(s.zero)}(0, copy(s.zero), ST(NaN))
end

SimpleSolvers.SolverState(s::LevenbergMarquardtSolver) = LevenbergMarquardtState(s)

"""The initial residual `‖F(x₀)‖` of the solve, which the convergence rule is relative to."""
initial_residual(state::LevenbergMarquardtState) = state.r₀

"""
    LevenbergMarquardtStatus{ST}

The outcome of one Levenberg–Marquardt solve: its number of trial steps, its final residual, its
initial residual, and whether it converged.
"""
struct LevenbergMarquardtStatus{ST}
    iterations::Int
    rfₐ::ST
    rf₀::ST
    converged::Bool
end

isconverged(status::LevenbergMarquardtStatus) = status.converged

function Base.show(io::IO, status::LevenbergMarquardtStatus)
    print(io, "LevenbergMarquardtStatus(",
        status.converged ? "converged" : "not converged", ", iterations = ",
        status.iterations, ", residual = ", status.rfₐ, ", initial residual = ", status.rf₀,
        ")")
end

"""
    lm_step!(δ, F, r, μ, truncation = 0)

Write `δ(μ) = −V diag(σᵢ/(σᵢ² + μ)) Uᵀ r` into `δ`, from the decomposition `F` of a Jacobian.

A singular value at or below `truncation` times the largest one contributes nothing, which is the
minimum-norm step of a rank-deficient system; with `μ > 0` every singular value is damped, so the
damping alone already removes the null directions.
"""
function lm_step!(δ::AbstractVector{T}, F, r::AbstractVector{T},
        μ, truncation = zero(T)) where {T}
    σ = F.S
    cut = truncation * (isempty(σ) ? zero(T) : first(σ))
    fill!(δ, zero(T))
    for k in eachindex(σ)
        σk = σ[k]
        σk ≤ cut && continue
        scale = σk / (σk^2 + μ)
        c = scale * dot(view(F.U, :, k), r)
        for i in eachindex(δ)
            δ[i] -= F.V[i, k] * c
        end
    end
    return δ
end

"""
    jacobian_of(f!, x, n)

The Jacobian, at `x`, of the `n`-residual `f!(b, x)`. The residual buffer is the routine's own, so
that a caller's residual is not overwritten by the differentiation.
"""
function jacobian_of(f!, x::AbstractVector{T}, n::Integer) where {T}
    return jacobian(
        z -> (b = Vector{eltype(z)}(undef, n); f!(b, z); b), x)
end

"""
    least_squares!(x, f!, n; iterations, μ₀, μ_min)

Bring `x` to a least-squares minimum of `‖f(x)‖` — `f!` writes the `n` residuals of `x` — by
Levenberg–Marquardt with the step of [`lm_step!`](@ref), and return `x`.

One routine serves both fits of the family, the datum's and the level's, so that one damping
schedule and one acceptance rule hold for both; the slab's solve is [`lm_solve!`](@ref).
"""
function least_squares!(x::AbstractVector{T}, f!, n::Integer;
        iterations::Integer = 400, μ₀ = LM_MU_START, μ_min = 1e-12) where {T}
    r = zeros(T, n)
    r̂ = zeros(T, n)
    x̂ = similar(x)
    δ = similar(x)
    f!(r, x)
    μ = T(μ₀)
    for _ in 1:iterations
        lm_step!(δ, svd(jacobian_of(f!, x, n)), r, μ)
        x̂ .= x .+ δ
        f!(r̂, x̂)
        if all(isfinite, r̂) && norm(r̂) < norm(r)
            copyto!(x, x̂)
            copyto!(r, r̂)
            μ = max(μ / 3, T(μ_min))
        else
            μ *= 4
        end
    end
    return x
end

"""
    lm_solve!(x, f!, n; max_iterations, μ₀, f_abstol, f_reltol, projected)

Solve `f(x) = 0` for the `n`-residual `f!` by Levenberg–Marquardt, from `x` and in place, and return
`(; iterations, residual, initial_residual, converged)`.

`projected`, where given, holds the positions of the unknowns that the projected variant eliminates
by an inner Newton solve; the outer step is then taken on the Schur complement of that block.
"""
function lm_solve!(x::AbstractVector{T}, f!, n::Integer;
        max_iterations::Integer = LM_MAX_ITERATIONS, μ₀ = LM_MU_START,
        f_abstol = zero(T), f_reltol = sqrt(eps(T)),
        projected::Union{Nothing, Vector{Int}} = nothing) where {T}
    options = (f_abstol = f_abstol, f_reltol = f_reltol)
    r = zeros(T, n)
    r̂ = zeros(T, n)
    x̂ = similar(x)
    δ = similar(x)
    δo = similar(x, projected === nothing ? 0 : n - length(projected))
    f!(r, x)
    r₀ = norm(r)
    iterations = 0
    all(isfinite, r) ||
        return (; iterations, residual = r₀, initial_residual = r₀, converged = false)
    converged = step_converged(r₀, r₀, options)
    μ = T(μ₀)
    while !converged && iterations < max_iterations && μ < T(LM_MU_MAX)
        J = jacobian_of(f!, x, n)
        if projected === nothing || isempty(projected)
            F = svd(J)
            reduced, reduced_residual = nothing, r
        else
            F = nothing
            reduced, reduced_residual = project!(x, f!, r, J, projected)
        end
        accepted = false
        while !accepted && iterations < max_iterations && μ < T(LM_MU_MAX)
            iterations += 1
            if F === nothing
                lm_step!(δo, svd(reduced), reduced_residual, μ)
                lm_widen!(x̂, x, δo, projected)
            else
                lm_step!(δ, F, r, μ)
                x̂ .= x .+ δ
            end
            f!(r̂, x̂)
            if all(isfinite, r̂) && norm(r̂) < norm(r)
                copyto!(x, x̂)
                copyto!(r, r̂)
                μ = max(μ / 3, T(LM_MU_MIN))
                accepted = true
            else
                μ *= 4
            end
            converged = step_converged(norm(r), r₀, options)
        end
    end
    return (; iterations, residual = norm(r), initial_residual = r₀, converged)
end

"""
    lm_widen!(x̂, x, δ, projected)

Write the outer step `δ` of the solved block onto the iterate `x` into `x̂`; the eliminated block
stays where the inner solve put it.
"""
function lm_widen!(x̂::AbstractVector, x::AbstractVector, δ::AbstractVector,
        projected::Vector{Int})
    copyto!(x̂, x)
    k = 0
    for i in eachindex(x)
        i in projected && continue
        k += 1
        x̂[i] += δ[k]
    end
    return x̂
end

"""The positions `1:n` without the eliminated block."""
function lm_solved_indices(n::Integer, projected::Vector{Int})
    return [i for i in 1:n if !(i in projected)]
end

"""
    project!(x, f!, r, J, projected)

Bring the eliminated block of `x` to a root of its own equations by Newton's method with the
minimum-norm step of a truncated decomposition, and return the Schur complement
`J_oo − J_oa J_aa⁺ J_ao` and the residual of the solved block.

`J` stays the one the outer iterate was given for the whole loop, so the inner step is Newton's
method of a frozen Jacobian; it is recomputed once, after the loop, for the Schur complement.

The inner solve stops when a step no longer lowers the eliminated block's residual, when it starts
there, or after [`LM_INNER_ITERATIONS`](@ref) steps.
"""
function project!(x::AbstractVector{T}, f!, r::AbstractVector{T}, J::AbstractMatrix{T},
        projected::Vector{Int}) where {T}
    solved = lm_solved_indices(length(x), projected)
    δa = zeros(T, length(projected))
    r̂ = similar(r)
    x̂ = similar(x)
    m = norm(view(r, projected))
    for _ in 1:LM_INNER_ITERATIONS
        Jaa = J[projected, projected]
        lm_step!(δa, svd(Jaa), view(r, projected), zero(T), T(LM_TRUNCATION))
        copyto!(x̂, x)
        for (k, i) in enumerate(projected)
            x̂[i] += δa[k]
        end
        f!(r̂, x̂)
        all(isfinite, r̂) || break
        m̂ = norm(view(r̂, projected))
        m̂ < m || break
        copyto!(x, x̂)
        copyto!(r, r̂)
        m = m̂
        iszero(m) && break
    end
    J = jacobian_of(f!, x, length(r))
    Jaa = J[projected, projected]
    A = svd(Jaa)
    cut = T(LM_TRUNCATION) * (isempty(A.S) ? zero(T) : first(A.S))
    keep = [k for k in eachindex(A.S) if A.S[k] > cut]
    Jao = J[projected, solved]
    JoaV = J[solved, projected] * view(A.V, :, keep)
    reduced = J[solved, solved]
    UtJoa = transpose(view(A.U, :, keep)) * Jao
    for k in eachindex(keep), j in eachindex(solved), i in eachindex(solved)
        reduced[i, j] -= JoaV[i, k] * UtJoa[k, j] / A.S[keep[k]]
    end
    return reduced, view(r, solved)
end

"""
    solve_with_status!(x, solver, state, params)

Solve one slab by Levenberg–Marquardt and return its [`LevenbergMarquardtStatus`](@ref). `params`
is the tuple `(sol, params, int)` that the record path hands to `residual!`.
"""
function solve_with_status!(x::AbstractVector{T}, s::LevenbergMarquardtSolver{T},
        state::LevenbergMarquardtState{T}, params) where {T}
    f!(b, z) = residual!(b, z, params)
    options = config(s)
    result = lm_solve!(x, f!, length(x);
        max_iterations = options.max_iterations, f_abstol = options.f_abstol,
        f_reltol = options.f_reltol,
        projected = s.variable_projection ? s.projected : nothing)
    state.iterations = result.iterations
    state.r₀ = convert(T, result.initial_residual)
    copyto!(state.x, x)
    return LevenbergMarquardtStatus{T}(result.iterations, convert(T, result.residual),
        convert(T, result.initial_residual), result.converged)
end

"""
    variable_projection_indices(method)

The positions, in the solver's vector, of the unknowns that the projected variant eliminates. A
method that keeps no block of its own answers `nothing`.
"""
variable_projection_indices(::GeometricMethod) = nothing

"""
    initsolver(solvermethod::LevenbergMarquardt, method::GeometricMethod, caches; kwargs...)

Build the Levenberg–Marquardt solver of one integrator from the options the integrator was handed,
with the family's defaults underneath them. A method that names no block for the projected variant
refuses it rather than silently solving the whole system.
"""
function initsolver(solvermethod::LevenbergMarquardt, method::GeometricMethod,
        caches::CacheDict; kwargs...)
    x = nlsolution(caches)
    ismissing(x) && throw(ArgumentError(string(
        nameof(typeof(method)), " does not support problems of type ",
        nameof(typeof(equation(caches.problem))), ".")))
    T = eltype(x)
    projected = variable_projection_indices(method)
    if solvermethod.variable_projection && projected === nothing
        throw(ArgumentError("variable_projection = true needs a method whose ansatz names the " *
                            "block it eliminates; $(nameof(typeof(method))) names none"))
    end
    options = merge(default_options(method, caches.problem), values(kwargs))
    return LevenbergMarquardtSolver{T, typeof(x)}(Options(T; options...), x,
        solvermethod.variable_projection, projected)
end
