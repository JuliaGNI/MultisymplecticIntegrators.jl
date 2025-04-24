struct PDEIntegrator{PT,#::PDEMethod,
    MT,#::PDEProblem,
    CT,#::PDEIntegratorCache,
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
    PDEIntegrator(integratormethod, problem, caches, solver, iguess)
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
problem(integrator::AbstractPDEIntegrator) = integrator.problem

function integrate(problem::PDEProblem, method::PDEMethod; kwargs...)
    integrator = PDEIntegrator(problem, method; kwargs...)
    integrate(integrator)
end

function integrate(integrator::AbstractPDEIntegrator)
    sol = LPDE_solution(problem(integrator))
    integrate!(sol, integrator)
end

function integrate!(sol::GeometricPDESolution, int::AbstractPDEIntegrator)
    integrate!(sol, int, 1, ntime(sol))
    return sol
end

function integrate!(sol::GeometricPDESolution, int::AbstractPDEIntegrator, n₁::Int, n₂::Int)
    # check time steps range for consistency
    @assert n₁ ≥ 1
    @assert n₂ ≥ n₁
    @assert n₂ ≤ ntime(sol)

    # copy initial condition from solution to solutionstep and initialize
    # solstep = solutionstep(int, sol[n₁-1])
ß
    # loop over time steps
    for sol.current_step in n₁:n₂
        # integrate one step and copy solution from cache to solution
        integrate!(sol, int, sol.current_step)
        sol.current_step += 1
        sol.current_time += int.problem.tstep
    end

    return sol
end

function integrate!(sol::GeometricPDESolution, int::AbstractPDEIntegrator, current_step::Int)

    initial_guess!(cache(int),sol,int,current_step)

    integrate_step!(sol, int)

    return sol
end



function integrate_step!(sol::GeometricPDESolution, int::AbstractPDEIntegrator)
    # call nonlinear solver
    solve!(cache(int).x, (b,x) -> residual!(b, x, sol, int), solver(int))

    # print solver status
    # println(status(solver))

    # check if solution contains NaNs or error bounds are violated
    # println(meets_stopping_criteria(status(solver)))

    # compute final update
    # update!(sol, nlsolution(int), int)
end

function residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol::GeometricPDESolution, int::AbstractPDEIntegrator) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    # compute stages of implicit Runge-Kutta methods from nonlinear solver solution x
    components!(x, sol, int)

    # compute right-hand side b of nonlinear solver
    residual!(b, sol, int)
end
