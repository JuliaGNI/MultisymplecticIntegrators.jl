module MultisymplecticIntegrators

using GeometricBase
using GeometricEquations
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: Diagonal, I, UpperTriangular, axpy!, dot, ldiv!,
                     mul!, norm, rdiv!, svd
using SimpleSplines
using SimpleSolvers
import ForwardDiff: gradient, jacobian
using SparseArrays
using Random: AbstractRNG, Xoshiro
using Symbolics

import SimpleSplines: SplineQuadrature, TensorProductBasis, TensorProductQuadrature,
                      BSplineBasis, UniformMesh, evaluate, basis, bases,
                      basis_values, degree, nbasis, ncells, quadrature_nodes,
                      quadrature_weights, l2_projection!, nodes as spline_nodes
import SimpleSolvers: initial_residual, method, solve_with_status!, NonlinearSolverException
import SimpleSolvers: config, jacobian!, jacobianmatrix, linearsolver, isconverged
import AbstractNeuralNetworks: Chain, Dense, NeuralNetwork, layers, params
import NeuralNetworkParameters: flatten, unflatten, unflatten!
import SymbolicNeuralNetworks: SymbolicNeuralNetwork, Jacobian, derivative,
                               build_flat_function, flat_parameter_gradient
import GeometricBase: datatype, timetype, arrtype, equtype, timespan, timestep, parameters,
                      periodicity, ntime, initialstate, timesteps, integrate, update!,
                      nodes
import GeometricEquations: initial_conditions, GeometricProblem, GeometricEquation,
                           compute_vectorfields!, problem, equation
import GeometricIntegratorsBase: GeometricIntegrator, solutionstep, reset!, integrate!,
                                 cache, current, default_extrapolation, NoExtrapolation
import GeometricIntegratorsBase: Cache, CacheType, IntegratorCache, check_solver_status,
                                 nlsolution, default_solver, default_options,
                                 initial_guess!,
                                 integrate_step!, residual!, solver, initsolver,
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
       field_initial_data, field_side_data, field_exact, convert_parameters

include("problems/wave.jl")
export Wave, VectorWave

include("problems/sine_gordon.jl")
export SineGordon

include("problems/linear_transport.jl")
export LinearTransport

# the weak form of a slab on its tensor quadrature
include("core/weak_form.jl")

# the ansatz
include("ansatz/tensor_bspline.jl")
export TensorBSplineAnsatz, degree, ncells, quadrature,
       is_linear_in_coefficients,
       space_quadrature_nodes

# the slab core
include("core/slab_integrator.jl")
export SpaceTimeGalerkin, SpaceTimeGalerkinCache, ansatz, coupling, total_free_coefficients

# the nonlinear solver of the fixed-boundary family, which plugs into the record path
include("core/levenberg_marquardt.jl")
export LevenbergMarquardt

include("ansatz/symbolic.jl")
export SymbolicAnsatz

# the initial guess: the roles first, then the fits a network uses and the ansatz they seed
include("initial_guess/fit_methods.jl")
include("initial_guess/guess.jl")
include("initial_guess/orthogonal_greedy.jl")
include("initial_guess/trained_hidden_layer.jl")
export TrainedHiddenLayer, PINNTraining

# The fixed-boundary coupling and the network block refer to each other: the block holds the
# coupling's fit workspace, and the coupling's methods dispatch on the block. This file is therefore
# included twice — its first pass declares the two blocks and the fits, which the network's block
# needs, and its second pass, after the network, declares the methods that dispatch on it.
include("couplings/fixed_boundary.jl")

# the network ansatz, which the coupling's carrier and the guess hooks extend
include("ansatz/network.jl")
export NetworkAnsatz, trace_basis, OrthogonalGreedy

include("couplings/fixed_boundary.jl")
export FixedBoundaryNetwork, FixedBoundary, BubbleGreedy, Predictor, bubble_atoms,
       fixed_boundary_f_reltol

# the couplings
include("couplings/coefficient_restriction.jl")
export CoefficientRestriction

include("couplings/weak_multipliers.jl")
export WeakInitialCondition, WeakInitialAndBoundaryConditions

include("couplings/hard_constraint_lifting.jl")
export HardConstraintLifting

export Continuation, FitTo, Projection, InitialCondition, PreviousSlab
export ExactSolution, TargetFunction
export default_fit

# the integrator shell
include("core/integrate.jl")
export integrate, top_trace, solve_stats

# the diagnostics
include("diagnostics.jl")
export energy_density, energy_error, solution_data

# the plotting API, implemented in the `MultisymplecticIntegratorsPlots` extension
include("plots.jl")

end
