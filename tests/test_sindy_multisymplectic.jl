# cd("..")
using Pkg
Pkg.activate(".")
using GeometricIntegrators
# using QuadratureRules
# using CompactBasisFunctions
using MultiSymplectic
using Symbolics

@variables t x p[1:3]
u_expr = p[1] * atan(exp(p[2] * (x - p[3] * t)))
sindy_basis = SindyPDEBasis([u_expr], [p], t, [x])

RT = 8
RX = 18 # when x interval is quite large, getting slow, and unstable
init_p = [4.02,1.15470,1.05]

t_step = 2.0
x_span = (0.0,3.0)
sindy_int = Sindy_PDE_Integrator(sindy_basis,RT,RX,init_p,x_span,t_step)

lpde = MultiSymplectic.SineGordon.lpdeproblem(tstep = t_step,tspan =(0.0,8.0),xspan = x_span)
sol = MultiSymplectic.integrate(lpde,sindy_int)


# using GeometricProblems.HarmonicOscillator

# lode = HarmonicOscillator.lodeproblem()
# # sol = integrate(lode, ImplicitMidpoint())

# QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
# BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))
# HO_CGVI = integrate(lode, CGVI(BGau4,QGau4))

using Plots

c=4.0
velocity = 1.0
γ = 1 / sqrt(1 - velocity^2 / c)



ics(x,t) = 4 * atan(exp(γ * (x - velocity * t)))
x_ls = collect(0.0:0.01:1)

plot(x_ls,ics.(x_ls,1.0),label="Analytic solution")
plot!(x_ls,sol.sol.u[1],label = "Sindy")
title!("t=1.0")

plot(x_ls,ics.(x_ls,5.0),label="Analytic solution")
plot!(x_ls,sol.sol.u[5],label = "Sindy")
title!("t=5.0")

plot(x_ls,ics.(x_ls,6.0),label="Analytic solution")
plot!(x_ls,sol.sol.u[6],label = "Sindy")
title!("t=6.0")

plot(x_ls,ics.(x_ls,9.0),label="Analytic solution")
plot!(x_ls,sol.sol.u[9],label = "Sindy")
title!("t=9.0")


c2 = 3.96116055279388
v2 = 1.0001766110236021
γ2 = 1.1514154675252721 # small error but obvious in plot

fun2(x,t) = 4 * atan(exp(γ2 * (x - v2 * t)))
plot(x_ls,ics.(x_ls,1.0),label="t=1.0")
plot!(x_ls,fun2.(x_ls,1.0),label="predict t=1.0")
plot!(x_ls,sol.sol.u[1],label="sindy t=1.0")

fun2.(x_ls,3.0)

t_ls = collect(1.0:0.1:2.0)
ics.(0.0,t_ls)
