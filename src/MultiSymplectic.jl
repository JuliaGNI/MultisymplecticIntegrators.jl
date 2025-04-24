module MultiSymplectic

    using GeometricIntegrators
    using Symbolics
    using CompactBasisFunctions
    using QuadratureRules
    using Parameters: @unpack
    using LinearAlgebra
    using IterTools
    
    using SimpleSolvers:NewtonMethod, Options, NonlinearSolver

    # basis
    include("basis/Sindy_PDE_basis.jl")
    export SindyPDEBasis    

    # integrators
    include("integrator/Sindy_PDE_int.jl")
    export Sindy_PDE_Integrator

    #problems
    include("problem/sine_Gordon.jl")
    export SineGordon

    # abstract types
    include("methods.jl")
    export PDEMethod, PDEProblem, PDEIntegratorCache, AbstractPDEIntegrator, GeometricPDESolution
end