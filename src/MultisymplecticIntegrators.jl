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
                      periodicity, ntime, initialstate, timesteps, integrate
import GeometricEquations: initial_conditions, GeometricProblem, GeometricEquation,
                           compute_vectorfields!, problem, equation
import GeometricIntegratorsBase: GeometricIntegrator, solutionstep, reset!, integrate!,
                                 cache,
                                 current

# abstract types
include("methods.jl")
export SpaceTimeAnsatz, SlabCoupling, SlabInitialGuess

# problems
include("problems/lagrangian_density.jl")
export LagrangianDensity, derivative_u, derivative_v, derivative_w, derivatives

include("problems/lagrangian_problem.jl")
export LagrangianEquation, LagrangianPDEProblem, xspan, xstep, xgrid, components,
       field_initial_data, field_boundary_data, field_exact, convert_parameters

include("problems/wave.jl")
export Wave, VectorWave

include("problems/sine_gordon.jl")
export SineGordon

include("problems/linear_transport.jl")
export LinearTransport

# quadrature
include("quadrature/spacetime_quadrature.jl")
export SpaceTimeQuadrature, boundary_points, nodes, weights, nnodes

# the integrator shell
include("core/integrate.jl")
export integrate, top_trace

end
