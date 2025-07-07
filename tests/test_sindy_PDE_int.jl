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





# Linear Transport PDE

# y3(x₁) = exp((x₁ + (x₁ + 0.31313)) * ((x₁ + 3.8436) * -0.49997)) * 0.0094303
# y4(x₁) = exp(((x₁ + x₁) + (((x₁ - -0.86449) * (x₁ - -1.1355)) + 0.048086)) * -1) * 0.014468
# y5(x₁) = exp(x₁ * ((x₁ - -4) * -1)) * 0.0051667
# y6(x₁) = exp((x₁ - -3.9999999915018885) * (x₁ * -1.000000000573464)) * 0.005166746413755448


@variables t x p[1:5]
c = 0.2
u_expr = exp((p[1] * (x - c*t) + p[2])*(x - c * t + p[3])*p[4]) * p[5]
init_p = [2.00, 0.31313, 3.8436, -0.49997, 0.0094303]
sindy_basis_linear_transport = SindyPDEBasis([u_expr], [p], t, [x])
RT = 4
RX = 6
t_step= 0.1
xspan = (-0.5, 0.0)
sindy_int_lt = Sindy_PDE_Integrator(sindy_basis_linear_transport,init_p,RT = RT,RX = RX, xspan = (-0.5, 0.0), tstep = t_step,μ =:BSplineDirichlet,λ =:BSplineDirichlet,k_μ = 4,k_λ₀_x = 4)
lpde_lt = MultiSymplectic.LinearTransport.lpdeproblem(tstep = t_step,tspan =(0.0,15.0),xspan = (-0.5, 0.0))
sol_lt = MultiSymplectic.integrate(lpde_lt,sindy_int_lt)

function exact_u(t,x)
    c = 0.2
    0.5 * exp(-(x+2 - c*t)^2) / sqrt(π)   
end

function exact_v(t,x)
    c = 0.2
    c*(2 + x - c*t)*exp(-((2 + x - c*t)^2)) / sqrt(π)
end

function exact_w(t,x)
    c = 0.2
    -(2 + x - c*t)*exp(-((2 + x - c*t)^2)) / sqrt(π)
end 


u0 = exact_u.(0.0,collect(-5:0.01:5.0))
v0 = exact_v.(0.0,collect(-5:0.01:5.0))
w0 = exact_w.(0.0,collect(-5:0.01:5.0))
H0_ls = [MultiSymplectic.LinearTransport.hamiltonian(0.0,0.0,u0i,v0i,w0i,(c=0.2,)) for (u0i,v0i,w0i) in zip(u0,v0,w0)]
H0 = sum(H0_ls)

function lt_u_SindySol(x,t,p)
    exp((p[1] * (x - 0.2*t) + p[2])*(x - 0.2 * t + p[3])*p[4]) * p[5]
end

function lt_v_SindySol(x,t,p)
    -0.2 * (p[1] * (x - 0.2*t) + p[2]) * exp((p[1] * (x - 0.2*t) + p[2])*(x - 0.2 * t + p[3])*p[4]) * p[5]
end

function lt_w_SindySol(x,t,p)
    (x - 0.2 * t + p[3]) * exp((p[1] * (x - 0.2*t) + p[2])*(x - 0.2 * t + p[3])*p[4]) * p[5]
end


x_vertics = [xspan[1], xspan[2], xspan[2], xspan[1]]
y_vertics = [-0.05,-0.05, 0.3, 0.3]
x_ls = collect(-5:0.01:5.0)

plot(x_ls,[lt_u_SindySol(xx,0.0,sol_lt.internal.x[2][1:5]) for xx in x_ls],label = "SINDy Solution")
plot!(x_ls,[lt_u_SindySol(xx,0.1,init_p) for xx in x_ls],label = "init Solution")
plot!(x_ls,exact_u.(0,x_ls),label="Analytic Solution",size = (1000,400))

tem_p = sol_lt.internal.x[2][1:5]
u_sol_ls = [lt_u_SindySol(xx,0,tem_p) for xx in x_ls] 
v_sol_ls = [lt_v_SindySol(xx,0,tem_p) for xx in x_ls]
w_sol_ls = [lt_w_SindySol(xx,0,tem_p) for xx in x_ls]
ham_ls = [MultiSymplectic.LinearTransport.hamiltonian(0.0,0.0,ui,vi,wi,(c=0.2,)) for (ui,vi,wi) in zip(u_sol_ls,v_sol_ls,w_sol_ls)]
total_sum = sum(ham_ls)


t_ls = 0:t_step:9.2
ham_ls = zeros(length(t_ls))
current_domain_ham = zeros(length(x_ls))
lp_anim = @animate for (i, t) in enumerate(t_ls)
    p = plot(layout=@layout([a;b]), label="", size=(1200,400))# d;e

    tem_p = sol_lt.internal.x[i+1][1:5]
    u_sol_ls = [lt_u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [lt_v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [lt_w_SindySol(xx,t,tem_p) for xx in x_ls]
    plot!(p[1],x_ls,u_sol_ls,label = "VISE Solution")
    plot!(p[1],x_ls,exact_u.(t,x_ls),label="Analytic Solution",size = (800,400),ylims = (-0.05,0.3),linestyle = :dash)

    title!(p[1],"t = $t,\n exp(($(tem_p[1]) * (x -c*t) + $(tem_p[2]))*(x - c * t + $(tem_p[3]))*$(tem_p[4])) * $(tem_p[5])",titlefontsize = 7)
    xlabel!(p[1],"x")
    ylabel!(p[1],"u")


    current_domain_ham = [MultiSymplectic.LinearTransport.hamiltonian(0.0,0.0,ui,vi,wi,(c=0.2,)) for (ui,vi,wi) in zip(u_sol_ls,v_sol_ls,w_sol_ls)]
    total_sum = sum(current_domain_ham)
    ham_ls[i] = total_sum
    plot!(p[2],t_ls[1:i],ham_ls[1:i],label = "VISE Hamiltonian",size = (800,400), xlims = (0, 9.2),ylims = (-1.8388,-1.8380))
    # plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
    xlabel!(p[2],"t")
    ylabel!(p[2],"Hamiltonian")
end
gif(lp_anim, "figures/linear_transport.gif",fps = 5)
