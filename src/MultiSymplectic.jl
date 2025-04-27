module MultiSymplectic

    using GeometricIntegrators
    import GeometricIntegrators.Integrators: default_solver, default_iguess,default_options,initsolver,datatype
    import GeometricIntegrators.Integrators: CacheDict, Cache,cache, caches,CacheType,solver
    using Symbolics
    using CompactBasisFunctions
    using QuadratureRules
    using Parameters: @unpack
    using LinearAlgebra
    import IterTools:product
    using SimpleSolvers:NewtonMethod, Options, NonlinearSolver,Newton,solve!

    # abstract types
    include("methods.jl")
    export PDEMethod, PDEProblem, PDEIntegratorCache, AbstractPDEIntegrator, GeometricPDESolution

    # utils
    include("utils/common.jl")
    export LPDE_variables,symbolize,substitute_parameters

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

    # integrators
    include("integrator/Sindy_PDE_int.jl")
    export Sindy_PDE_Integrator

    #problems
    include("problem/sine_Gordon.jl")
    export SineGordon


end