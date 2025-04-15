using GeometricIntegrators:GeometricMethod
abstract type PDEMethod <: GeometricMethod end

using GeometricIntegrators:IntegratorCache
abstract type PDEIntegratorCache{DT,D} <: IntegratorCache{DT,D} end

using GeometricEquations:GeometricProblem, GeometricEquation

abstract type PartialDifferentialEquation{invType,parType,perType} <: GeometricEquation{invType,parType,perType} end