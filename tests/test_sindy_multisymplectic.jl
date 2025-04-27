cd("..")
using Pkg
Pkg.activate(".")
using GeometricIntegrators

using MultiSymplectic
using Symbolics

@variables t x p[1:3]
u_expr = p[1] * atan(exp(p[2] * (x - p[3] * t)))
sindy_basis = SindyPDEBasis([u_expr], [p], t, [x])

RT = 4
RX = 8
init_p = [4,1.15470,1.02]

sindy_int = Sindy_PDE_Integrator(sindy_basis,RT,RX,init_p)

lpde = MultiSymplectic.SineGordon.lpdeproblem()
sol = MultiSymplectic.integrate(lpde,sindy_int)


# using GeometricProblems.HarmonicOscillator

# lode = HarmonicOscillator.lodeproblem()
# sol = integrate(lode, ImplicitMidpoint())