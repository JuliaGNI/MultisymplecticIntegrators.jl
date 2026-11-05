"""
    top_trace(cache)

The top trace `(u, v, w)` of the last step, at the nodes of the space quadrature.

A method whose cache keeps no trace returns `nothing`, which is the default. A method that keeps one
extends this function, and the trace is stored beside the solution by [`integrate`](@ref).
"""
top_trace(::Any) = nothing

"""
    solve_stats(cache)

The records of the nonlinear solves of the steps taken since the last `reset!` of the cache, as the
named tuple `(; iterations, residuals, initial_residuals, converged)`, one entry per step.

[`integrate`](@ref) reads it after every step. A method whose cache keeps no record returns
`nothing`, which is the default. A method that keeps one extends this function for its cache.
"""
solve_stats(::Any) = nothing

# The two values of the keyword `on_failure` of `integrate`.
function check_on_failure(on_failure)
    on_failure in (:warn, :error) ||
        throw(ArgumentError("on_failure must be :warn or :error, got $(repr(on_failure))"))
    return nothing
end

"""
    integrate(problem::LagrangianPDEProblem, method::SpaceTimeMethod; on_failure = :warn, kwargs...)
    integrate(integrator::GeometricIntegrator; on_failure = :warn)

Integrate `problem` with `method` and return the named tuple `(; solution, traces, stats)`.

`solution` is the `GeometricSolution` on the output grid, and `traces` holds one entry per stored
step: the top trace `(u, v, w)` of that step at the nodes of the space quadrature, as the method's
[`top_trace`](@ref) gives it, or `nothing` where the method keeps none. The trace is needed at the
quadrature nodes rather than the interior ones because the energy diagnostics integrate it there, at
the accuracy of the Gauss rule.

`stats` is the named tuple `(; iterations, residuals, initial_residuals, converged)` of the
nonlinear solves, with one entry per stored trace, as the method's [`solve_stats`](@ref) records
them, or `nothing` where the method keeps no record; `on_failure` then has nothing to act on. A
step has converged when its residual is at most `f_abstol + f_reltol · r₀`, with `r₀` its
initial residual and the tolerances of the integrator's own solver options; the term in `r₀`
counts only for a finite `r₀`, so a residual that is not finite never converges. A step that has
not converged warns with `on_failure = :warn`, the default, and throws an `ErrorException` with
`on_failure = :error`; both name the step. Any other value of `on_failure` throws an
`ArgumentError` before the first step.

The steps are driven one at a time, because the trace of a step exists only while its cache does.
Every call starts a new run: the integrator's cache is reset first, so its first slab takes the
problem's initial data and the steps it records belong to this run alone.
As in the ecosystem's `integrate!`, a step whose state holds a NaN ends the run with a warning;
`traces` then ends at that step, and the solution holds the NaN there. A step whose nonlinear solve
raises a `NonlinearSolverException` ends the run with a warning too; `traces` then ends at the step
before it, and the solution is valid up to that step only. The keyword arguments other than
`on_failure` go to the integrator's constructor, and to nothing else.
"""
function integrate(problem::LagrangianPDEProblem, method::SpaceTimeMethod;
        on_failure = :warn, kwargs...)
    return integrate(GeometricIntegrator(problem, method; kwargs...); on_failure)
end

function integrate(::LagrangianPDEProblem, ansatz::SpaceTimeAnsatz; kwargs...)
    throw(ArgumentError("$(nameof(typeof(ansatz))) is an ansatz, not a method; " *
                        "integrate with SpaceTimeGalerkin(ansatz)"))
end

function integrate(
        integrator::GeometricIntegrator{
            <:SpaceTimeMethod, <:LagrangianPDEProblem};
        on_failure = :warn)
    check_on_failure(on_failure)
    c = cache(integrator)
    reset!(c)
    solution = GeometricSolution(problem(integrator))
    steps = ntime(solution)
    sample = top_trace(c)
    traces = sizehint!(Vector{typeof(sample)}(), steps)
    solstep = solutionstep(integrator, solution[0])
    return integrate_steps!(solution, solstep, traces, integrator, c, on_failure)
end

function integrate_steps!(solution, solstep, traces, integrator, c, on_failure)
    steps = ntime(solution)
    for n in 1:steps
        reset!(solstep, timesteps(solution)[n])
        try
            integrate!(solstep, integrator)
        catch e
            e isa NonlinearSolverException || rethrow()
            @warn "The nonlinear solver failed at timestep n=$(n): $(e.msg). " *
                  "The solution is valid up to n=$(n - 1); later entries were never computed."
            break
        end
        copy!(solution, current(solstep), n)
        push!(traces, deepcopy(top_trace(c)))
        record = solve_stats(c)
        if record !== nothing && !record.converged[n]
            message = "The nonlinear solve did not converge at timestep n=$(n): residual " *
                      "$(record.residuals[n]) after $(record.iterations[n]) iterations, from " *
                      "an initial residual of $(record.initial_residuals[n])."
            on_failure === :error && error(message)
            @warn message
        end
        if isnan(current(solstep))
            @warn "The state holds a NaN at timestep n=$(n); the integration stops there."
            break
        end
    end
    record = solve_stats(c)
    stats = record === nothing ? nothing : map(entries -> entries[1:length(traces)], record)
    return (; solution, traces, stats)
end
