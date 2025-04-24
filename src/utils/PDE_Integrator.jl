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
    solution = LPDE_solution(problem(integrator))
    integrate!(solution, integrator)
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
    solstep = solutionstep(int, sol[n₁-1])

    # loop over time steps
    for n in n₁:n₂
        # integrate one step and copy solution from cache to solution
        sol[n] = integrate!(solstep, int)

        # try
        #     sol[n] = integrate!(int)
        # catch ex
        #     tstr = " in time step " * string(n)
        #
        #     if m₁ ≠ m₂
        #         tstr *= " for initial condition " * string(m)
        #     end
        #
        #     tstr *= "."
        #
        #     if isa(ex, DomainError)
        #         @warn("Domain error" * tstr)
        #     elseif isa(ex, ErrorException)
        #         @warn("Simulation exited early" * tstr)
        #         @warn(ex.msg)
        #     else
        #         @warn(string(typeof(ex)) * tstr)
        #         throw(ex)
        #     end
        # end
    end

    return sol
end