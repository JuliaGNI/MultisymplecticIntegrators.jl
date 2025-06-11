using GeometricIntegrators
using QuadratureRules
using MultiSymplectic
using Symbolics
using Parameters


@variables t x p[1:3]
u_expr = p[1] * atan(exp(p[2] * (x - p[3] * t)))
sindy_basis = SindyPDEBasis([u_expr], [p], t, [x])


RT = 4
RX = 4

c2 = 3.9999951547393993
v2 = 1.0397741828102778
γ2 = 1.1546989298112151
init_p = [c2,γ2,v2]

t_step = 0.05
x_span = (0.,0.1)
sindy_int = Sindy_PDE_Integrator(sindy_basis,init_p,RT = RT,RX = RX, xspan = x_span, tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,k_μ = 4,k_λ₀_x = 4)

lpde = MultiSymplectic.SineGordon.lpdeproblem(tstep = t_step,tspan =(0.0,5.5),xspan = x_span)
sol = MultiSymplectic.integrate(lpde,sindy_int)

using Plots

c=4.0
velocity = 1.0
γ = 1 / sqrt(1 - velocity^2 / c)
analytic_sol(x,t) = c * atan(exp(γ * (x - velocity * t)))

function u_SindySol(x,t,params)
    local c_sol,γ_sol,v_sol =  params
    c_sol * atan(exp(γ_sol * (x - v_sol * t)))
end


x_ls = collect(-5:0.1:5)

x_vertics = [x_span[1], x_span[2], x_span[2], x_span[1]]
y_vertics = [-0.2, -0.2, 7.0, 7.0]

plot(x_ls,analytic_sol.(x_ls,0.05),label="Analytic Solution",ylims = (-0.2,7.00),size = (1000,400))
plot!(x_ls,[u_SindySol(x,0.05,sol.internal.x[2][1:3]) for x in x_ls],label = "SINDy Solution")
plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
# scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
title!("t = 0.05")
xlabel!("x")
ylabel!("u")

plot(x_ls,analytic_sol.(x_ls,5.0),label="Analytic Solution",ylims = (-0.2,7.00),size = (1000,400))
plot!(x_ls,[u_SindySol(x,5.0,sol.internal.x[100][1:3]) for x in x_ls],label = "SINDy Solution")
plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
# scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
title!("t = 5.0")
xlabel!("x")
ylabel!("u")


anim = @animate for (i, t) in enumerate(0:t_step:5.0)
    plot(x_ls,analytic_sol.(x_ls,t),label="Analytic Solution",ylims = (-0.2,7.00),size = (1000,400))
    plot!(x_ls,[u_SindySol(x,t,sol.internal.x[i+1][1:3]) for x in x_ls],label = "SINDy Solution")
    plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
    # scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
    title!("t = $t")
    xlabel!("x")
    ylabel!("u")
end 
gif(anim, "figures/sine_gordon.gif",fps = 5)


# Wave equation 
A1 = 0.3
B1 = 0.2
A2 = 0.4
B2 = 0.5
c = 1.
u_exact_sol(x,t)=  (A1 * cos((pi*c*t)) + B1 * sin((pi*c*t) + pi/6)) * sin((pi*x)) + (A2 * cos((2*pi*c*t)) + B2 * sin((2*pi*c*t) + pi/6)) * sin((2*pi*x))
x_ls = collect(0.2:0.01:0.8)
plot(x_ls,u_exact_sol.(x_ls,0.0),label="Analytic Solution")
function wave_u_SindySol(x,t,p)
    (p[1] * cos((pi*t)) + p[2] * sin((pi*t) + pi/6)) * sin((pi*x)) + (p[3] * cos((2*pi*t)) + p[4] * sin((2*pi*t) + pi/6)) * sin((2*pi*x))
end


@variables t x p[1:4]
c=1.0
u_expr_wave =  (p[1] * cos((pi*t)) + p[2] * sin((pi*t) + pi/6)) * sin((pi*x)) + (p[3] * cos((2*pi*t)) + p[4] * sin((2*pi*t) + pi/6)) * sin((2*pi*x))
init_p_wave = [0.300001, 0.200001, 0.400001, 0.500001] # initial guess for the parameters
sindy_basis_wave = SindyPDEBasis([u_expr_wave], [p], t, [x])
RT = 4
RX = 6
t_step = 0.05
xspan = (0.2, 0.8)
sindy_int_wave = Sindy_PDE_Integrator(sindy_basis_wave,init_p_wave,RT = RT,RX = RX, xspan = xspan, tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,k_μ = 4,k_λ₀_x = 4)

lpde_wave = MultiSymplectic.Wave.lpdeproblem(tspan =(0.0,1.0),tstep = t_step, xspan = xspan)
wave_sol = MultiSymplectic.integrate(lpde_wave,sindy_int_wave)

x_ls = collect(0.2:0.01:0.8)
plot(x_ls,u_exact_sol.(x_ls,0.05),label="Analytic Solution",ylims = (-1.,1.50),size = (1000,400))
plot!(x_ls,[wave_u_SindySol(x,0.05,wave_sol.internal.x[2][1:4]) for x in x_ls],label = "SINDy Solution")
title!("t = 0.05")
xlabel!("x")
ylabel!("u")


x_vertics = [xspan[1], xspan[2], xspan[2], xspan[1]]
y_vertics = [-1,-1, 1.5, 1.5]

wave_anim = @animate for (i, t) in enumerate(0:t_step:0.8)
    plot(x_ls,u_exact_sol.(x_ls,t),label="Analytic Solution",size = (1000,400), ylims = (-1.,1.50))
    plot!(x_ls,[wave_u_SindySol(x,t,wave_sol.internal.x[i+1][1:4]) for x in x_ls],label = "SINDy Solution")
    plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
    # scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
    title!("t = $t")
    xlabel!("x")
    ylabel!("u")
end 
gif(wave_anim, "figures/wave_eqn.gif",fps = 5)