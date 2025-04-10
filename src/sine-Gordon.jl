"""
    Nonlinear Sine-Gordon equation
    u_xx - u_tt = - sin(u) = d(V(u))/du = d(1+cos(u)) / du
"""

using Parameters: @unpack
using EulerLagrange
using LinearAlgebra
using Symbolics
using CompactBasisFunctions
using QuadratureRules

include("./Lagrangian_common.jl")
module SineGordon
    using Parameters: @unpack
    using EulerLagrange
    using LinearAlgebra
    using Symbolics

    const D = 1
    const DX = 1

    const tstep = 0.01
    const tspan = (0.0, 10.0)
    const xspan = (0.0, 10.0)

    const default_parameters = (
        c=1.0,
    )

    function lagrangian(t, x, u, v, w, params)
        @unpack c = params
        1 / 2 * (c * v[1]^2 - w[1,1]^2) + (1 + cos(u[1]))
    end

    function hamiltonian(t, x, u, v, w, params)
        @unpack c = params
        1 / 2 * (c * v[1]^2 + w[1]^2) - (1 + cos(u[1]))
    end

    function lpdeproblem(u₀=u₀, v₀=v₀, w₀=w₀; tspan=tspan, tstep=tstep, xspan=xspan, xstep=xstep, parameters=default_parameters)
        @assert tstep < xstep "tstep must be less than xstep for stability"
        @assert tspan[1] < tspan[2] "tspan must be increasing"
        @assert xspan[1] < xspan[2] "xspan must be increasing"

        t,x,U,V,W = lagrangianPDE_variables(1, 1) # U,V,W does not have t,x dependence 
        sparams = symbolize(parameters)
        lag_sys = LagrangianPDESystem(lagrangian(t,x,U,V,W, sparams), t,x,U,V,W, sparams)
        #TODO LPDEProblem(lag_sys, tspan, tstep, xspan=xspan, xstep=xstep, u₀, v₀, w₀; v̄=θ̇, parameters=parameters)
    end

    export lagrangian, hamiltonian
end


for i in 1:2
    for j in 1:5
        res[i,j] = i + j
    end
end

res = [2*res[i] for i in eachindex(res)]