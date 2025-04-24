using GeometricIntegrators:GeometricMethod
abstract type PDEMethod <: GeometricMethod end

using GeometricIntegrators:AbstractProblem
abstract type PDEProblem <: AbstractProblem end

using GeometricIntegrators:IntegratorCache
abstract type PDEIntegratorCache{DT,D} <: IntegratorCache{DT,D} end

using GeometricIntegrators:AbstractIntegrator
abstract type AbstractPDEIntegrator end


abstract type GeometricPDESolution end


# Solvers 
using SimpleSolvers:NewtonMethod, Options, NonlinearSolver