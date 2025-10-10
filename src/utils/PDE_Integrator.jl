struct PDEIntegrator{
    MT<:PDEMethod,
    PT<:PDEProblem,
    CT<:CacheDict{PT,MT},
    ST <: Union{NonlinearSolver,SolverMethod},
    IT <: Extrapolation} <: AbstractPDEIntegrator 
    
    problem::PT
    method::MT
    caches::CT
    solver::ST
    iguess::IT
end

function PDEIntegrator(problem::PDEProblem,
    integratormethod::PDEMethod,
    solvermethod::NewtonMethod,
    iguess::Extrapolation;
    options = default_options(),
    method = initmethod(integratormethod, problem),
    caches = CacheDict(problem, method),
    solver = initsolver(solvermethod, options, method, caches)
)
    PDEIntegrator(problem,integratormethod, caches, solver, iguess)
end

function PDEIntegrator(
    problem::PDEProblem,
    method::PDEMethod;
    solver = default_solver(method),
    initialguess = default_iguess(method), 
    kwargs...
)
    PDEIntegrator(problem, method, solver, initialguess; kwargs...)
end

initmethod(method::PDEMethod, problem::PDEProblem) = method
problem(integrator::PDEIntegrator) = integrator.problem
caches(int::PDEIntegrator) = int.caches
cache(int::PDEIntegrator, DT) = caches(int)[DT]
cache(int::PDEIntegrator) = cache(int, datatype(problem(int)))
nlsolution(int::PDEIntegrator) = nlsolution(cache(int))
solver(int::PDEIntegrator) = int.solver
timestep(int::PDEIntegrator) = timestep(problem(int))


function integrate(problem::PDEProblem, method::PDEMethod; kwargs...)
    integrator = PDEIntegrator(problem, method; kwargs...)
    integrate(integrator)
end

function integrate(integrator::AbstractPDEIntegrator)
    sol = LPDE_solution(problem(integrator),internal = internal_variables(integrator,problem(integrator)))
    integrate!(sol, integrator)
end

function integrate!(sol::AbstractPDESolution, int::AbstractPDEIntegrator)
    integrate!(sol, int, 1, ntime(sol))
    return sol
end

function integrate!(sol::AbstractPDESolution, int::AbstractPDEIntegrator, n₁::Int, n₂::Int)
    # check time steps range for consistency
    @assert n₁ ≥ 1
    @assert n₂ ≥ n₁
    @assert n₂ ≤ ntime(sol)

    # copy initial condition from solution to solutionstep and initialize
    # solstep = solutionstep(int, sol[n₁-1])ß
    # loop over time steps
    for sol.current_step in n₁:n₂
        println("start integrating step = ", sol.current_step, "current time = ", sol.t)
        
        prior_initial_guess!(cache(int),sol,int)
        initialize_bcs_ics!(sol,int)

        # integrate one step and copy solution from cache to solution
        integrate_step!(sol, int)
        println("finish integrating step  = ", sol.current_step, "current time = ", sol.t)

    end

    return sol
end

# function integrate!(sol::AbstractPDESolution, int::AbstractPDEIntegrator, n::Int)
#     integrate_step!(sol, int)
#     return sol
# end

function integrate_step!(sol::AbstractPDESolution, int::AbstractPDEIntegrator)
    # call nonlinear solver
    solve!(cache(int).x, (b,x) -> residual!(b, x, sol, int), solver(int))

    # print solver status
    # println(status(solver))

    # check if solution contains NaNs or error bounds are violated
    # println(meets_stopping_criteria(status(solver)))

    # compute final update
    update!(sol, int)
end

function residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol::AbstractPDESolution, int::AbstractPDEIntegrator) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    # compute stages of implicit Runge-Kutta methods from nonlinear solver solution x
    components!(x, sol, int)

    # compute right-hand side b of nonlinear solver
    residual!(b, sol, int)
end

function internal_variables(int,problem::PDEProblem)
    local x = cache(int).x
    ntime = Int((problem.tspan[2] - problem.tspan[1]) / problem.tstep)
    xx = (x, ntuple( _ -> zeros(size(x)...), ntime)...)
    return (x = xx,)
end