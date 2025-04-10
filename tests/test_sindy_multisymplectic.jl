using MultiSymplectic
using Symbolics

@variables P[1:6] t x[1]
u_expr = P[1] * sin(P[2] * t[1] + P[3]) * cos(P[4] * x[1] + P[5]) + P[6]

sindy_base = MultiSymplectic.SindyPDEBasis([u_expr], P, t, x)

function lagrangian(t, x, u, v, w, params)
    @unpack c = params
    1 / 2 * (c * v[1]^2 - w[1,1]^2) + (1 + cos(u[1]))
end
t,x,U,V,W = lagrangianPDE_variables(1, 1) # U,V,W does not have t,x dependence 
const default_parameters = (
    c=1.0,
)
sparams = symbolize(default_parameters)
lag_sys = LagrangianPDESystem(lagrangian(t,x,U,V,W, sparams), t,x,U,V,W, sparams)
