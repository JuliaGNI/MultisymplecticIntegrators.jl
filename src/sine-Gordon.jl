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

        t, x, u, v, w = lagrangianPDE_variables(1, 1)
        sparams = symbolize(parameters)
        lag_sys = LagrangianPDESystem(lagrangian(t, x, u, v, w, sparams), t, x, u, v, w, sparams)
        LPDEProblem(lag_sys, tspan, tstep, xspan=xspan, xstep=xstep, u₀, v₀, w₀; v̄=θ̇, parameters=parameters)
    end

    export lagrangian, hamiltonian
end


# function test_lagrangian(t, x, u, v, w, params)
#     @unpack c = params
#     1 / 2 * (c * v[1]^2-  5 * w[1]^2 - w[2]^2) + (1 + cos(u[1]+u[2]))
# end

# sum(hcat([expand_derivatives(dx.(∂L∂w)) for dx in Dx]...),dims=2)[:,1]



@variables W[1:6]
u = W[1] * sin(W[2] * t[1] + W[3]) * cos(W[4] * x[1] + W[5]) + W[6]

t,x,u,v,w = lagrangianPDE_variables(1, 1)


Dt = Differential(t)
Dx = Differential(x)

v_expr = expand_derivatives(Dt(u_expr))
w_expr = expand_derivatives(Dx(u_expr))

u_func = eval(build_function(u_expr, W, t, x))
# u_func(rand(6), rand(1)[1], rand(1)[1])
v_func = eval(build_function(v_expr, W, t, x))
w_func = eval(build_function(w_expr, W, t, x))

Du = Differential(u)
Dv = Differential(v)
Dw = Differential(w)


sparams = symbolize(default_parameters)
Ls = SineGordon.lagrangian(t, x, u, v, w, sparams)

expand_derivatives(Du(Ls))

lag_sys = LagrangianPDESystem(Ls, t, x, u, v, w, sparams)
typeof(lag_sys.equations.∂L∂u) 
∂L∂u = eval(build_function(lag_sys.equations.∂L∂u[1],u))
∂L∂u(0.1)


QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))


μ₁_t = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))
μ₂_t = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))

λ₁_x = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(8)))
λ₂_x = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QuadratureRules.GaussLegendreQuadrature(8)))

# Cache

RT = 4
RX = 8
∂L∂u_mat = zeros(ST,RX,RT)
∂L∂v_mat = zeros(ST,RX,RT)
∂L∂w_mat = zeros(ST,RX,RT)

NΘ = length(W)
D = 1
∂u∂θ_mat = zeros(ST,D,NΘ)
∂v∂θ_mat = zeros(ST,D,NΘ)
∂w∂θ_mat = zeros(ST,D,NΘ) 

μ₁_t_coes = zeros(ST,RT)
μ₂_t_coes = zeros(ST,RT)
λ₁_x_coes = zeros(ST,RX)
λ₂_x_coes = zeros(ST,RX) 

# components

