"""
    Continuation() <: SlabInitialGuess

The default initial guess: the previous slab's solution, carried into the new slab.

For the tensor B-spline ansatz under a [`CoefficientRestriction`](@ref) that is the previous slab's
top coefficient row, repeated in every row: each row of the new slab starts where the previous slab
ended. At the first slab there is nothing to carry, and the guess falls back to
`FitTo(InitialCondition())` with the ansatz's default fit method.
"""
struct Continuation <: SlabInitialGuess end

"""
    InitialCondition() <: SlabTarget

The problem's initial condition `u₀(x)`, constant in time over the slab.
"""
struct InitialCondition <: SlabTarget end

"""
    PreviousSlab() <: SlabTarget

The previous slab's solution, extrapolated in time beyond the top of that slab.

For the B-spline ansatz the extrapolation is the polynomial piece of the slab's top time cell,
expanded about its top and evaluated at the displacement `t̂` beyond it: a spline is a polynomial on
each cell, so that piece extends the solution without inventing anything. At the first slab, where
there is no previous solution, the target falls back to `InitialCondition()`.
"""
struct PreviousSlab <: SlabTarget end

"""
    ExactSolution() <: SlabTarget

The problem's exact solution `u(t, x)` at the physical time of the slab.

A problem without an exact solution cannot supply it, and `FitTo(ExactSolution())` on one throws an
`ArgumentError` at construction.
"""
struct ExactSolution <: SlabTarget end

"""
    TargetFunction(f) <: SlabTarget

A user-defined target: `f(t, x, params)` in physical time.

The result is converted to the problem's element type, so a `Float32` problem started from a function
that returns `Float64` keeps `Float32` coefficients. A `D`-component problem may return a number,
which every component takes, or a vector of the `D` values.
"""
struct TargetFunction{F} <: SlabTarget
    f::F
end

"""
    FitTo(target; method = nothing) <: SlabInitialGuess

Fit the ansatz to `target` on the slab, with the fit method `method`.

A target is evaluated on the slab — see [`InitialCondition`](@ref), [`PreviousSlab`](@ref),
[`ExactSolution`](@ref) and [`TargetFunction`](@ref) — and the fit brings the ansatz to it. A
`method` that is named is used as it is; an unnamed one resolves through
[`default_fit`](@ref)`(ansatz)`, so that an ansatz with a fit method of its own gets it.

`target` is a [`SlabTarget`](@ref) and `method` a [`SlabFitMethod`](@ref); a value of another role,
such as `FitTo(Continuation())`, has no method and is refused at construction.

A user-defined target is a subtype of `SlabTarget` with one method of

```julia
evaluate!(sample, target, cache, ansatz, problem, sol)
```

filling `sample`, of size `(Qt, Qx, D)`, with the target's value at the quadrature points of the
slab. The fit that follows takes `sample` from the cache, so a target that allocates nothing keeps
the guess free of allocations.
"""
struct FitTo{T <: SlabTarget, M <: Union{Nothing, SlabFitMethod}} <: SlabInitialGuess
    target::T
    method::M
end

function FitTo(target::SlabTarget; method::Union{Nothing, SlabFitMethod} = nothing)
    FitTo(target, method)
end

"""
    fit_method(fit::FitTo, ansatz)

The fit method a [`FitTo`](@ref) uses: the one it names, or the ansatz's own
[`default_fit`](@ref) where the caller named none.
"""
function fit_method(fit::FitTo, ansatz::SpaceTimeAnsatz)
    fit.method === nothing ? default_fit(ansatz) : fit.method
end

"""Whether a target needs the problem's exact solution."""
needs_exact_solution(::SlabTarget) = false
needs_exact_solution(::ExactSolution) = true

"""
    initial_guess!(sol, history, params, int)

Fill the nonlinear solver's vector of the slab from the method's initial guess, and then the
coupling's own unknowns from the guess the coupling defines for them.
"""
function initial_guess!(sol, history, params,
        int::GeometricIntegrator{<:SpaceTimeGalerkin, <:LagrangianPDEProblem})
    c = cache(int)
    m = method(int)
    a = ansatz(m)
    apply_guess!(c, initial_guess(m), a, problem(int), sol)
    coupling_guess!(c.coupling, c, a, problem(int), sol)
    return nothing
end

function apply_guess!(c::SpaceTimeGalerkinCache, guess::Continuation,
        ansatz::TensorBSplineAnsatz, problem::LagrangianPDEProblem, sol)
    if c.steps == 0
        apply_guess!(c, FitTo(InitialCondition()), ansatz, problem, sol)
    else
        nf = free_rows(c.coupling, ansatz)
        nx = nspace_functions(ansatz)
        for d in axes(c.top, 2), j in 1:nx, a in 1:nf
            c.x[free_index(a, j, d, nf, nx)] = c.top[j, d]
        end
    end
    return nothing
end

function apply_guess!(c::SpaceTimeGalerkinCache, guess::FitTo,
        ansatz::TensorBSplineAnsatz, problem::LagrangianPDEProblem, sol)
    D = size(c.bottom, 2)
    evaluate!(c.sample, guess.target, c, ansatz, problem, sol)
    fit!(c.fit, fit_method(guess, ansatz), ansatz, c.sample, c)
    nt = ntime_functions(ansatz)
    nx = nspace_functions(ansatz)
    nf = free_rows(c.coupling, ansatz)
    for d in 1:D, j in 1:nx, a in 1:nf
        c.x[free_index(a, j, d, nf, nx)] = c.fit[a + nt - nf, j, d]
    end
    return nothing
end

"""
    evaluate!(sample, target, cache, ansatz, problem, sol)

Fill `sample`, of size `(Qt, Qx, D)`, with the value of `target` at the quadrature points of the
slab, of which `sol` fixes the physical time.
"""
function evaluate!(sample::AbstractArray{ST, 3}, ::InitialCondition,
        c::SpaceTimeGalerkinCache, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem, sol) where {ST}
    for d in axes(sample, 3), r in axes(sample, 2), q in axes(sample, 1)
        sample[q, r, d] = c.initial_sample[r, d]
    end
    return nothing
end

function evaluate!(sample::AbstractArray{ST, 3}, ::PreviousSlab,
        c::SpaceTimeGalerkinCache, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem, sol) where {ST}
    if c.steps == 0
        return evaluate!(sample, InitialCondition(), c, ansatz, problem, sol)
    end
    tb = time_basis(ansatz)
    nt = ntime_functions(ansatz)
    nx = nspace_functions(ansatz)
    D = size(c.bottom, 2)
    order = degree(tb)
    T = eltype(ansatz.xspan)
    # the Taylor coefficients of the top time cell at t̂ = 1, one per space function and component
    deriv = c.taylor
    for k in 0:order, d in 1:D, j in 1:nx
        coefficient = zero(ST)
        for a in 1:nt
            coefficient += c.coeffs[a, j, d] * evaluate(tb, a, one(T), k)
        end
        deriv[j, d, k + 1] = coefficient
    end
    t = c.tabulation.time
    x = c.tabulation.space
    for q in 1:nquadrature(t)
        # the displacement beyond the top of the previous slab, about which `deriv` expands
        δ = t.node[q]
        for r in 1:nquadrature(x)
            for d in 1:D
                value = zero(ST)
                for l in axes(x.index, 2)
                    j = x.index[r, l]
                    j == 0 && continue
                    poly = zero(ST)
                    factor = one(ST)
                    for k in 0:order
                        poly += deriv[j, d, k + 1] * factor
                        factor *= δ / (k + 1)
                    end
                    value += poly * x.value[r, l]
                end
                sample[q, r, d] = value
            end
        end
    end
    return nothing
end

"""
    target_values!(sample, f, tabulation, ansatz, problem, sol)

The value of a pointwise target `f(t, x, params)` on the slab, per component, converted to the
problem's element type. A `f` that returns a number gives every component that number; one that
returns a vector gives component `d` its `d`-th entry.
"""
function target_values!(sample::AbstractArray{ST, 3}, f, tabulation::SlabTabulation,
        ansatz::TensorBSplineAnsatz, problem::LagrangianPDEProblem, sol) where {ST}
    t = tabulation.time
    x = tabulation.space
    t0 = sol.t - timestep(problem)
    h = timestep(problem)
    for q in 1:nquadrature(t), r in 1:nquadrature(x)

        time = t0 + h * t.node[q]
        value = f(time, x.node[r], parameters(problem))
        for d in axes(sample, 3)
            sample[q, r, d] = convert(ST, value isa Number ? value : value[d])
        end
    end
    return nothing
end

function evaluate!(sample::AbstractArray{ST, 3}, target::TargetFunction,
        c::SpaceTimeGalerkinCache, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem, sol) where {ST}
    return target_values!(sample, target.f, c.tabulation, ansatz, problem, sol)
end

function evaluate!(sample::AbstractArray{ST, 3}, ::ExactSolution,
        c::SpaceTimeGalerkinCache, ansatz::TensorBSplineAnsatz,
        problem::LagrangianPDEProblem, sol) where {ST}
    return target_values!(sample, problem.exact_u, c.tabulation, ansatz, problem, sol)
end
