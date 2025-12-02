module MultiSymplectic

    using GeometricIntegrators
    import GeometricIntegrators.Integrators: default_solver, default_iguess,default_options,initsolver,datatype
    import GeometricIntegrators.Integrators: CacheDict, Cache,cache, caches,CacheType,solver
    using Symbolics
    using AbstractNeuralNetworks
    using SymbolicNeuralNetworks
    using GeometricMachineLearning    
    using CompactBasisFunctions
    using QuadratureRules
    using Parameters: @unpack
    using LinearAlgebra
    using IterTools:product
    using SimpleSolvers:NewtonMethod, Options, NonlinearSolver,Newton,solve!
    using Random
    using BSplineKit
    import BSplineKit.SplineInterpolations:make_knots
    using Zygote
    using Statistics
    using NonlinearSolve
    using ForwardDiff

    # abstract types
    include("methods.jl")
    export PDEMethod, PDEProblem, PDEIntegratorCache, AbstractPDEIntegrator, GeometricPDESolution
    export InitialParametersMethod, LSGD, ELM, PINN, TrialOGA2D, OGA2D
    # utils
    include("utils/common.jl")
    export LPDE_variables,symbolize,substitute_parameters,Lagrangian_multiplier

    include("utils/Lagrangian_PDE_problem.jl")
    export LPDEProblem

    include("utils/Lagrangian_PDE_solution.jl")
    export LPDE_solution
    
    include("utils/Lagrangian_PDE_system.jl")
    export LPDESystem

    include("utils/PDE_Integrator.jl")
    export PDEIntegrator

    # basis
    include("basis/Sindy_PDE_basis.jl")
    export SindyPDEBasis    

    include("basis/BSplineBasis.jl")
    export BSplineDirichlet

    include("basis/Network_PDE_Basis.jl")
    export NetworkPDEBasis

    include("basis/NN_Basis.jl")
    export NN_Basis

    include("basis/TrialSolution_Basis.jl")
    export Trial_Solution_Basis

    # integrators
    include("integrator/Sindy_PDE_int.jl")
    export Sindy_PDE_Integrator

    include("integrator/NN_PDE_int.jl")
    export NN_PDE_Integrator

    include("integrator/ELM_PDE_int.jl")
    export ELM_PDE_int
    
    include("integrator/TrialNN_PDE_int.jl")
    export TrialNN_PDE_int
    #problems
    include("problem/sine_Gordon.jl")
    export SineGordon

    include("problem/wave.jl")
    export Wave

    include("problem/linear_transport.jl")
    export LinearTransport
end