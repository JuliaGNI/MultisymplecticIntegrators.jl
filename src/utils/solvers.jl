# Basically the same with GeometricIntegrator/src/integrators/solvers.jl

default_options() = Options(
    min_iterations = 1,
    x_abstol = 8eps(),
    f_abstol = 8eps(),
)

# create nonlinear solver
function initsolver(::NewtonMethod, config::Options, ::PDEMethod, caches::CacheDict; kwargs...)
    x = zero(nlsolution(caches))
    y = zero(nlsolution(caches))
    NewtonSolver(x, y; linesearch = Backtracking(), config = config, kwargs...)
end