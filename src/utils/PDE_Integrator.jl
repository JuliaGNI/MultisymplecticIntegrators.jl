struct PDEIntegrator{
    MT<:PDEMethod,
    PT<:LPDEProblem,
    CT<:CacheDict{PT,MT},
    ST <: Union{NonlinearSolver,SolverMethod}} <: AbstractPDEIntegrator 
    
    problem::PT
    method::MT
    caches::CT
    solver::ST
end

function PDEIntegrator(problem::LPDEProblem,
    integratormethod::PDEMethod,
    solvermethod::NewtonMethod;
    method = initmethod(integratormethod, problem),
    caches = CacheDict(problem, method),
    options...
)
    solver = initsolver(solvermethod, method, caches; (length(options) == 0 ? default_options(integratormethod) : options)...)
    PDEIntegrator(problem,integratormethod, caches, solver)
end

function PDEIntegrator(
    problem::LPDEProblem,
    method::PDEMethod;
    solver = default_solver(method),
    kwargs...
)
    PDEIntegrator(problem, method, solver; kwargs...)
end

initmethod(method::PDEMethod, problem::LPDEProblem) = method
problem(integrator::PDEIntegrator) = integrator.problem
caches(int::PDEIntegrator) = int.caches
cache(int::PDEIntegrator, DT) = caches(int)[DT]
cache(int::PDEIntegrator) = cache(int, datatype(problem(int)))
nlsolution(int::PDEIntegrator) = nlsolution(cache(int))
solver(int::PDEIntegrator) = int.solver
timestep(int::PDEIntegrator) = timestep(problem(int))
method(int::PDEIntegrator) = int.method
_state(a::Vector{TT}) where {TT} = zeros(TT,length(a))
_vectorfield(a::Vector{TT}) where TT = missing
nlsolution(int::PDEIntegratorCache) = cache(int).x

function integrate(problem::LPDEProblem, method::PDEMethod; kwargs...)
    integrator = PDEIntegrator(problem, method; kwargs...)
    integrate(integrator)
end

function integrate(integrator::AbstractPDEIntegrator)
    sol = GeometricSolution(problem(integrator)) #,internal = internal_variables(integrator,problem(integrator)
    integrate!(sol, integrator)
end

function integrate!(sol::GeometricSolution, int::AbstractPDEIntegrator)
    integrate!(sol, int, 1, ntime(sol))
    return sol
end

function integrate!(sol::GeometricSolution, int::AbstractPDEIntegrator, n₁::Int, n₂::Int;kwargs...)
    # check time steps range for consistency
    @assert n₁ ≥ 1
    @assert n₂ ≥ n₁
    @assert n₂ ≤ ntime(sol)

    # copy initial condition from solution to solutionstep and initialize
    solstep = solutionstep(int, sol[n₁-1])
    # loop over time steps
    for n in n₁:n₂
        sol[n] = integrate!(solstep, int)


    end

    return sol
end

function integrate!(solstep::SolutionStep, int::AbstractPDEIntegrator)
    reset!(solstep, timestep(int))

    prior_initial_guess!(cache(int),solstep,int)

    initialize_bcs_ics!(solstep,int)

    copy_internal_variables!(cache(int),solstep)
    # integrate one step and copy solution from cache to solution
    integrate_step!(current(solstep), history(solstep), parameters(solstep), int)

    copy_internal_variables!(solstep,cache(int))

    return solstep
end


function integrate_step!(sol, history, params, int::AbstractPDEIntegrator)
    # call nonlinear solver
    solve!(nlsolution(int), solver(int), (sol,params,int))

    # print solver status
    # println(status(solver))

    # check if solution contains NaNs or error bounds are violated
    # println(meets_stopping_criteria(status(solver)))

    # compute final update
    update!(sol, int)
end

function residual!(b::AbstractVector{ST}, x::AbstractVector{ST}, sol, params, int::AbstractPDEIntegrator) where {ST}
    # check that x and b are compatible
    @assert axes(x) == axes(b)

    components!(x, sol, params, int)

    residual!(b, sol, params, int)
end

function internal_variables(int,problem::LPDEProblem)
    local x = cache(int).x
    ntime = Int((problem.tspan[2] - problem.tspan[1]) / problem.tstep)
    xx = (x, ntuple( _ -> zeros(size(x)...), ntime)...)
    return (x = xx,)
end