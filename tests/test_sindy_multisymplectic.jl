cd("..")
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

RT = 4
RX = 12
init_p = [4,1.15470,1.02]

sindy_int = Sindy_PDE_Integrator(sindy_basis,RT,RX,init_p)

lpde = MultiSymplectic.SineGordon.lpdeproblem()
sol = MultiSymplectic.integrate(lpde,sindy_int)


# using GeometricProblems.HarmonicOscillator

# lode = HarmonicOscillator.lodeproblem()
# # sol = integrate(lode, ImplicitMidpoint())

# QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
# BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))
# HO_CGVI = integrate(lode, CGVI(BGau4,QGau4))

using Plots
plot(sol.sol.u[1])
plot(sol.sol.u[2])
plot(sol.sol.u[3])

c=4.0
velocity = 1.0
γ = 1 / sqrt(1 - velocity^2 / c)



ics(x,t) = 4 * atan(exp(γ * (x - velocity * t)))
x_ls = collect(0.0:0.01:1)
plot(x_ls,ics.(x_ls,0.0),label="t=0")
plot!(x_ls,sol.sol.u[1])

plot(x_ls,ics.(x_ls,1.0),label="t=1.0")
plot!(x_ls,sol.sol.u[2])


c2 = 3.999999325464563
v2 = 1.1546994127974766
γ2 = 1.0197314232874848

fun2(x,t) = 4 * atan(exp(γ2 * (x - v2 * t)))
plot(x_ls,ics.(x_ls,1.0),label="t=1.0")
plot!(x_ls,fun2.(x_ls,1.0),label="predict t=1.0")