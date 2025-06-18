using GeometricIntegrators:GeometricMethod
abstract type PDEMethod <: GeometricMethod end

using GeometricIntegrators:AbstractProblem
abstract type PDEProblem <: AbstractProblem end

using GeometricIntegrators:IntegratorCache
abstract type PDEIntegratorCache{DT,D} <: IntegratorCache{DT,D} end

using GeometricIntegrators:AbstractIntegrator
abstract type AbstractPDEIntegrator <: AbstractIntegrator end

# using GeometricSolutions:AbstractSolution
abstract type AbstractPDESolution end #<: AbstractSolution

# using CompactBasisFunctions: Basis
abstract type AbstractPDEBasis end #<: Basis 

abstract type InitialParametersMethod end
struct LSGD <: InitialParametersMethod end
struct GroundTruth <: InitialParametersMethod end