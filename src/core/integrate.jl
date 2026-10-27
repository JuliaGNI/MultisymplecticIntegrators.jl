"""
    top_trace(cache)

The top trace `(u, v, w)` of the last step, at the nodes of the space quadrature.

A method whose cache keeps no trace returns `nothing`, which is the default. A method that keeps one
extends this function, and the trace is stored beside the solution by [`integrate`](@ref).
"""
top_trace(::Any) = nothing

"""
    integrate(problem::LagrangianPDEProblem, method::SpaceTimeAnsatz; kwargs...)
    integrate(integrator::GeometricIntegrator)

Integrate `problem` with `method` and return the named tuple `(; solution, traces)`.

`solution` is the `GeometricSolution` on the output grid, and `traces` holds one entry per stored
step: the top trace `(u, v, w)` of that step at the nodes of the space quadrature, as the method's
[`top_trace`](@ref) gives it, or `nothing` where the method keeps none. The trace is needed at the
quadrature nodes rather than the interior ones because the energy diagnostics integrate it there, at
the accuracy of the Gauss rule.

The steps are driven one at a time, because the trace of a step exists only while its cache does.
Every call starts a new run: the integrator's cache is reset first, so its first slab takes the
problem's initial data and the steps it records belong to this run alone.
As in the ecosystem's `integrate!`, a step whose state holds a NaN ends the run with a warning;
`traces` then ends at that step, and the solution holds the NaN there. The keyword arguments go to
the integrator's constructor, and to nothing else.
"""
function integrate(problem::LagrangianPDEProblem, method::SpaceTimeAnsatz; kwargs...)
    return integrate(GeometricIntegrator(problem, method; kwargs...))
end

function integrate(integrator::GeometricIntegrator{
        <:SpaceTimeAnsatz, <:LagrangianPDEProblem})
    reset!(cache(integrator))
    solution = GeometricSolution(problem(integrator))
    steps = ntime(solution)
    sample = top_trace(cache(integrator))
    traces = sizehint!(Vector{typeof(sample)}(), steps)
    solstep = solutionstep(integrator, solution[0])
    for n in 1:steps
        reset!(solstep, timesteps(solution)[n])
        integrate!(solstep, integrator)
        copy!(solution, current(solstep), n)
        push!(traces, deepcopy(top_trace(cache(integrator))))
        if isnan(current(solstep))
            @warn "The state holds a NaN at timestep n=$(n); the integration stops there."
            break
        end
    end
    return (; solution, traces)
end
