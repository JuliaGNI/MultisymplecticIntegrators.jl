module MultiSymplectic

    # using GeometricIntegrators
    using GeometricBase
    using GeometricIntegratorsBase
    import GeometricIntegratorsBase: default_solver,default_options,initsolver,CacheDict,Cache,cache,CacheType,solutionstep,reset!
    import GeometricIntegratorsBase: problem,method,parameters,SolverMethod,history, solver,residual!,copy_internal_variables!,internal
    import GeometricIntegratorsBase: _state, _vectorfield, compute_vectorfields!,_extrapolate!,internal_variables,nlsolution
    import GeometricBase: datatype,timetype,ntime
    import GeometricBase: initialtime, finaltime, timespan, timestep,periodicity, NullPeriodicity
    import GeometricEquations:GeometricProblem,initial_conditions
    using GeometricSolutions:GeometricSolution

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
    using ForwardDiff
    using Infiltrator

    # abstract types
    include("methods.jl")
    export PDEMethod, PDEIntegratorCache, AbstractPDEIntegrator, GeometricPDESolution
    export InitialParametersMethod, LSGD, ELM, PINN, TrialOGA2D, OGA2D

    include("utils/Lagrangian_LPDE_problem.jl")
    export LPDEProblem

    # include("utils/Lagrangian_PDE_solution.jl")
    # export LPDE_solution
    
    include("utils/Lagrangian_LPDE_system.jl")
    export LPDESystem

    include("utils/PDE_Integrator.jl")
    export PDEIntegrator
    
    # utils
    include("utils/common.jl")
    export LPDE_variables,symbolize,substitute_parameters,Lagrangian_multiplier
    export initialize_bcs_ics!,vector_hessian
    # basis
    include("basis/Sindy_PDE_basis.jl")
    export SindyPDEBasis    

    include("basis/BSplineBasis.jl")
    export BSplineDirichlet, BSpline2D

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

    include("integrator/Galerkin_Bspline_int.jl")
    export Galerkin_Bspline_Integrator
    
    #problems
    include("problem/sine_Gordon.jl")
    export SineGordon

    include("problem/wave.jl")
    export Wave

    include("problem/linear_transport.jl")
    export LinearTransport
end