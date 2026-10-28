module MultisymplecticIntegrators

using GeometricBase
using GeometricEquations
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: Diagonal, I, mul!
using QuadratureRules
using SimpleSplines
using SimpleSolvers
using SparseArrays
using Symbolics

import QuadratureRules: nodes, weights, nnodes
import SimpleSplines: SplineQuadrature, TensorProductBasis, TensorProductQuadrature,
                      BSplineBasis, UniformMesh, evaluate, basis, bases,
                      basis_values, degree, nbasis, ncells, quadrature_nodes,
                      quadrature_weights, l2_projection!, nodes as spline_nodes
import SimpleSolvers: initial_residual, method, solve_with_status!, NonlinearSolverException
import GeometricBase: datatype, timetype, arrtype, equtype, timespan, timestep, parameters,
                      periodicity, ntime, initialstate, timesteps, integrate, update!
import GeometricEquations: initial_conditions, GeometricProblem, GeometricEquation,
                           compute_vectorfields!, problem, equation
import GeometricIntegratorsBase: GeometricIntegrator, solutionstep, reset!, integrate!,
                                 cache, current, default_extrapolation, NoExtrapolation
import GeometricIntegratorsBase: Cache, CacheType, IntegratorCache, check_solver_status,
                                 nlsolution, default_solver, initial_guess!,
                                 integrate_step!, residual!, solver,
                                 solverstate, copy_internal_variables!, internal

# abstract types
include("methods.jl")
export SpaceTimeAnsatz, SpaceTimeMethod, SlabCoupling, SlabInitialGuess, SlabTarget,
       SlabFitMethod

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

# the ansatz
include("ansatz/tensor_bspline.jl")
export TensorBSplineAnsatz, degree, ncells, output_grid, quadrature,
       is_linear_in_coefficients,
       space_quadrature_nodes

# the slab core
include("core/weak_form.jl")
include("core/slab_integrator.jl")
export SpaceTimeGalerkin, SpaceTimeGalerkinCache, ansatz, coupling, total_free_coefficients

# the couplings
include("couplings/coefficient_restriction.jl")
export CoefficientRestriction

include("couplings/weak_multipliers.jl")
export WeakInitialCondition, WeakInitialAndBoundaryConditions

# the initial guess
include("initial_guess/fit_methods.jl")
include("initial_guess/guess.jl")
export Continuation, FitTo, Projection, InitialCondition, PreviousSlab
export ExactSolution, TargetFunction
export default_fit

# the integrator shell
include("core/integrate.jl")
export integrate, top_trace

end
