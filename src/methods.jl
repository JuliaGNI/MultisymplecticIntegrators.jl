using GeometricIntegratorsBase:GeometricMethod
abstract type PDEMethod <: GeometricMethod end

# using GeometricBase:AbstractProblem
# abstract type PDEProblem <: AbstractProblem end


using GeometricEquations: GeometricEquation
using GeometricBase:OptionalInvariants,OptionalParameters,OptionalPeriodicity


using GeometricIntegratorsBase:IntegratorCache
abstract type PDEIntegratorCache{DT,D} <: IntegratorCache{DT,D} end

using GeometricBase:AbstractIntegrator
abstract type AbstractPDEIntegrator <: AbstractIntegrator end

# using GeometricBase:AbstractSolution
# abstract type AbstractPDESolution <: AbstractSolution end

# using CompactBasisFunctions: Basis
abstract type AbstractPDEBasis end #<: Basis 

abstract type InitialParametersMethod end
struct LSGD <: InitialParametersMethod end
struct OGA2D <: InitialParametersMethod end
struct ELM <: InitialParametersMethod end
struct PINN <: InitialParametersMethod end
struct TrialOGA2D <: InitialParametersMethod end
