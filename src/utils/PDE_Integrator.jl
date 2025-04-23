struct PDEIntegrator
    {MT::PDEMethod,
    PT::PDEProblem,
    CT::PDEIntegratorCache,
    ST <: Union{NonlinearSolver,SolverMethod},
    } <: AbstractIntegrator
    
    method::MT
    problem::PT
    caches::CT
    solver::ST
end

function 