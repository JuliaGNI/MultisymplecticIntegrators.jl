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

        t,x,U,V,W = lagrangianPDE_variables(1, 1) # U,V,W does not have t,x dependence 
        sparams = symbolize(parameters)
        lag_sys = LagrangianPDESystem(lagrangian(t,x,U,V,W, sparams), t,x,U,V,W, sparams)
        LPDEProblem(lag_sys, tspan, tstep, xspan=xspan, xstep=xstep, u₀, v₀, w₀; v̄=θ̇, parameters=parameters)
    end

    export lagrangian, hamiltonian
end


# function test_lagrangian(t, x, u, v, w, params)
#     @unpack c = params
#     1 / 2 * (c * v[1]^2-  5 * w[1]^2 - w[2]^2) + (1 + cos(u[1]+u[2]))
# end

# sum(hcat([expand_derivatives(dx.(∂L∂w)) for dx in Dx]...),dims=2)[:,1]

RT = 4
RX = 8

include("./common.jl")
dimensions = [RX,RT]  # Number of quadrature points in each dimension
grid_matrix, grid_weights = construct_quadrature_grid(dimensions)

@variables P[1:6]
t,x,U,V,W = lagrangianPDE_variables(1, 1) # U,V,W does not have t,x dependence 

u_expr = P[1] * sin(P[2] * t[1] + P[3]) * cos(P[4] * x[1] + P[5]) + P[6]

Symbolics.derivative(u_expr,P[1])

∂u_expr∂P = [Symbolics.derivative(u_expr,P[i]) for i in eachindex(P)]

Dt = Differential(t)
Dx = Differential(x)

v_expr = expand_derivatives(Dt(u_expr))
w_expr = [Symbolics.derivative(u_expr, x[i]) for i in eachindex(x)]

∂v_expr∂P = [Symbolics.derivative(v_expr,P[i]) for i in eachindex(P)]
∂w_expr∂P = [Symbolics.derivative(w_expr[i],P[j]) for i in eachindex(w_expr) for j in eachindex(P)]

∂u∂P = [Symbolics.eval(build_function(∂u_expr∂P[i], P, t, x)) for i in eachindex(∂u_expr∂P)]
∂v∂P = [Symbolics.eval(build_function(∂v_expr∂P[i], P, t, x)) for i in eachindex(∂v_expr∂P)]
∂w∂P = [Symbolics.eval(build_function(∂w_expr∂P[i], P, t, x)) for i in eachindex(∂w_expr∂P)]
# ∂w∂P[1](rand(6), rand(1)[1], rand(1)[1])



u = eval(build_function(u_expr, P, t, x))
v = eval(build_function(v_expr, P, t, x))
w = [eval(build_function(w_expr[i], P, t, x)) for i in eachindex(w_expr)]
# u(rand(6), rand(1)[1], rand(1)[1])

DU = Differential(U)
DV = Differential(V)
DW = Differential(W)



sparams = symbolize(SineGordon.default_parameters)
Ls = SineGordon.lagrangian(t, x, U, V, W, sparams)

∂L∂U_expr = [Symbolics.derivative(Ls, U[i]) for i in eachindex(U)]
∂L∂V_expr = [Symbolics.derivative(Ls, V[i]) for i in eachindex(V)]
∂L∂W_expr = [Symbolics.derivative(Ls, W[i]) for i in eachindex(W)]

∂L∂U = [substitute_parameters(Symbolics.build_function(∂L∂U_expr[i], U, V, W,sparams...; nanmath = false),sparams) for i in eachindex(∂L∂U_expr)]
∂L∂V = [substitute_parameters(Symbolics.build_function(∂L∂V_expr[i], U, V, W,sparams...; nanmath = false),sparams) for i in eachindex(∂L∂V_expr)]
∂L∂W = [substitute_parameters(Symbolics.build_function(∂L∂W_expr[i], U, V, W,sparams...; nanmath = false),sparams) for i in eachindex(∂L∂W_expr)]

∂L∂U = [Symbolics.eval(∂L∂U[i]) for i in eachindex(∂L∂U)]
∂L∂V = [Symbolics.eval(∂L∂V[i]) for i in eachindex(∂L∂V)]
∂L∂W = [Symbolics.eval(∂L∂W[i]) for i in eachindex(∂L∂W)]

u_quad_values = zeros(Float64,RX,RT)
v_quad_values = zeros(Float64,RX,RT)
w_quad_values = zeros(Float64,RX,RT)

for i in 1:RX
    for j in 1:RT
        u_quad_values[i,j] = u(rand(6),grid_matrix[i,j][2], grid_matrix[i,j][1])
        v_quad_values[i,j] = v(rand(6),grid_matrix[i,j][2], grid_matrix[i,j][1])
        w_quad_values[i,j] = w[1](rand(6),grid_matrix[i,j][2], grid_matrix[i,j][1])
    end
end


∂L∂U_quad_values = zeros(Float64,RX,RT)
∂L∂V_quad_values = zeros(Float64,RX,RT)
∂L∂W_quad_values = zeros(Float64,RX,RT)

for i in 1:RX
    for j in 1:RT
        ∂L∂U_quad_values[i,j] = ∂L∂U[1](u_quad_values[i,j],v_quad_values[i,j],w_quad_values[i,j],SineGordon.default_parameters)
        ∂L∂V_quad_values[i,j] = ∂L∂V[1](u_quad_values[i,j],v_quad_values[i,j],w_quad_values[i,j],SineGordon.default_parameters)
        ∂L∂W_quad_values[i,j] = ∂L∂W[1](u_quad_values[i,j],v_quad_values[i,j],w_quad_values[i,j],SineGordon.default_parameters)
    end
end



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
# the symbolics expression should be computed ahead, and eval as a matrix of callable function 


D = 1 #u,v,w, are scaler value functions
∂L∂u = zeros()

expand_derivatives(Du(Ls))

∂L∂u_mat = zeros(ST,RX,RT)
∂L∂v_mat = zeros(ST,RX,RT)
∂L∂w_mat = zeros(ST,RX,RT)

NΘ = length(W)
ST = Float64
∂u∂θ_mat = zeros(ST,D,NΘ)
∂v∂θ_mat = zeros(ST,D,NΘ)
∂w∂θ_mat = zeros(ST,D,NΘ) 

μ₁_t_coes = zeros(ST,RT)
μ₂_t_coes = zeros(ST,RT)
λ₁_x_coes = zeros(ST,RX)
λ₂_x_coes = zeros(ST,RX) 






# components

