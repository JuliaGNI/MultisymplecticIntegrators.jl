using GeometricIntegrators
using QuadratureRules
using MultiSymplectic
using Symbolics

@variables t x p[1:3]
u_expr = p[1] * atan(exp(p[2] * (x - p[3] * t)))
sindy_basis = SindyPDEBasis([u_expr], [p], t, [x])


RT = 12
RX = 36 

c2 = 3.9999951547393993
v2 = 1.0397741828102778
γ2 = 1.1546989298112151
init_p = [c2,γ2,v2]

t_step = 1.0
x_span = (-1.,1.0)
sindy_int = Sindy_PDE_Integrator(sindy_basis,init_p,RT = RT,RX = RX, xspan = x_span, tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,k_μ = 4,k_λ₀_x = 4)

lpde = MultiSymplectic.SineGordon.lpdeproblem(tstep = t_step,tspan =(0.0,10.0),xspan = x_span,tspan = (0.0,2.0))
sol = MultiSymplectic.integrate(lpde,sindy_int)

using Plots

c=4.0
velocity = 1.0
γ = 1 / sqrt(1 - velocity^2 / c)



ics(x,t) = 4 * atan(exp(γ * (x - velocity * t)))
x_ls = collect(-1:0.01:1.0)

plot(x_ls,ics.(x_ls,1.0),label="Analytic Solution")
plot!(x_ls,sol.sol.u[1],label = "SINDy Solution")
title!("t=1.0")
savefig("sine_gordon_t=1.pdf")


plot(x_ls,ics.(x_ls,2.0),label="Analytic Solution")
plot!(x_ls,sol.sol.u[2],label = "SINDy Solution")
title!("t=2.0")
savefig("sine_gordon_t=2.pdf")



# # Wave equation 
# using Symbolics

# @variables t x p[1:4]

# # u_expr = (A1 * cos((pi*c*t)/xL) + B1 * sin((pi*c*t)/xL)) * sin((pi*x)/xL) + (A2 * cos((2*pi*c*t)/xL) + B2 * sin((2*pi*c*t)/xL)) * sin((2*pi*x)/xL)
# u_expr_wave =  (p[1] * cos((pi* t)/1.6) + p[2] * sin((pi *t)/1.6 + pi/6)) * sin((pi*x)/1.6) + (p[3] * cos((2*pi*t)/1.6) + p[4] * sin((2*pi*t)/1.6 + pi/6)) * sin((2pi*x)/1.6)
# init_p_wave = [1.0002,1.0003,-0.500001,0.5002]
# sindy_basis_wave = SindyPDEBasis([u_expr_wave], [p], t, [x])

# RT = 8
# RX = 20

# t_step = 1.0
# x_span = (-0.8,0.8)
# sindy_int_wave = Sindy_PDE_Integrator(sindy_basis_wave,init_p_wave,RT = RT,RX = RX, xspan = x_span, tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,k_μ = 6,k_λ₀_x = 6)

# lpde_wave = MultiSymplectic.Wave.lpdeproblem(tstep = t_step,tspan =(0.0,10.0),xspan = x_span)
# wave_sol = MultiSymplectic.integrate(lpde_wave,sindy_int_wave)



# xstep = 0.01
# xspan = (-0.8, 0.8)
# xL = xspan[2] - xspan[1]
# c=1.0

# A1 = 1.0
# B1 = 1.0

# A2 = -0.5
# B2 = 0.5
# u_exact_sol(x,t) = (A1 * cos((pi*c*t)/xL) + B1 * sin((pi*c*t)/xL)) * sin((pi*x)/xL) + (A2 * cos((2*pi*c*t)/xL) + B2 * sin((2*pi*c*t)/xL)) * sin((2*pi*x)/xL)
