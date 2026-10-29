# One step's residual, timed: what the slab costs and where it goes.
#
#   julia --project=scripts scripts/profile_residual.jl
#   julia --project=scripts scripts/profile_residual.jl --steps 0.05,0.025
#
# The residual is the nonlinear system of one slab — the interior weak form, plus the coupling's own
# equations behind the free coefficients. Its size, its cost and its allocations are what decide
# whether a run is worth its time, so they are measured rather than reasoned about.

include(joinpath(@__DIR__, "archives.jl"))
include(joinpath(@__DIR__, "experiments.jl"))
include(joinpath(@__DIR__, "spline_model.jl"))

using GeometricBase: parameters
using GeometricIntegratorsBase: GeometricIntegrator, cache, current, integrate!, reset!,
                                residual!, solutionstep
using GeometricSolutions: GeometricSolution

function time_call(label, f)
    f()
    GC.gc()
    seconds = @elapsed f()
    @printf("%-32s %9.4f s\n", label, seconds)
    return seconds
end

function profile(entry)
    problem = spline_problem(entry)
    method = spline_method(entry)
    integrator = GeometricIntegrator(problem, method)
    step = solutionstep(integrator, GeometricSolution(problem)[0])
    reset!(step, problem.timespan[1] + problem.timestep)
    # One step, so that the cache holds the bottom datum and the momentum flux the residual reads.
    integrate!(step, integrator)

    c = cache(integrator)
    x = c.x
    b = similar(x)
    banner("$(length(x)) unknowns, ansatz $(entry.ncells), h = $(entry.timestep)")
    time_call("residual!",
        () -> residual!(b, x, current(step), parameters(step), integrator))
    report("bytes per residual evaluation",
        @allocated residual!(b, x, current(step), parameters(step), integrator))
    time_call("one step, residual and solve", () -> integrate!(step, integrator))
    return 0
end

function main(args)
    _, options = parse_arguments(args, ("--steps",))
    template = smallest_entry(SPLINE)
    steps = option_steps(options, (template.timestep,))
    for timestep in steps
        profile(merge(template, (timestep = timestep,)))
    end
    return 0
end

exit(main(ARGS))
