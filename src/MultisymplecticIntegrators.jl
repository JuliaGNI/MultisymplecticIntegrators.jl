module MultisymplecticIntegrators

using GeometricBase
using GeometricEquations
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra
using QuadratureRules
using Symbolics

import QuadratureRules: nodes, weights, nnodes
import GeometricBase: datatype, timetype, arrtype, equtype, timespan, timestep, parameters,
                      periodicity, ntime, initialtime, finaltime, initialstate, timesteps
import GeometricEquations: initial_conditions, GeometricProblem, GeometricEquation,
                           compute_vectorfields!
import GeometricIntegratorsBase: GeometricIntegrator, Cache, CacheType, IntegratorCache,
                                 solutionstep, reset!, integrate!, integrate_step!, cache,
                                 current, problem, method, _extrapolate!

# abstract types
include("methods.jl")
export SpaceTimeAnsatz, SlabCoupling, SlabInitialGuess

# problems
include("problems/lagrangian_density.jl")
export LagrangianDensity, derivative_u, derivative_v, derivative_w, derivatives

include("problems/lagrangian_problem.jl")
export LagrangianEquation, LagrangianPDEProblem, xspan, xstep, xgrid, components

include("problems/wave.jl")
export Wave, VectorWave, VectorDensity

# quadrature
include("quadrature/spacetime_quadrature.jl")
export SpaceTimeQuadrature, boundary_points, nodes, weights, nnodes

# the integrator shell
include("core/integrate.jl")
export integrate, top_trace

end
