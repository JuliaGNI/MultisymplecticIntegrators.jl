module MultiSymplectic

    using Symbolics
    using CompactBasisFunctions
    using QuadratureRules
    using Parameters: @unpack
    using LinearAlgebra

    include("symbolics_expr_basis.jl")
    export SindyPDEBasis


end