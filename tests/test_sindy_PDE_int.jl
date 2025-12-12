using QuadratureRules
using MultiSymplectic
using Symbolics
using Parameters
using Plots
# using CairoMakie


# Sine-Gordon Equation
begin
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

    c=4.0
    t_step = 0.05

    velocity = 1.0
    γ = 1 / sqrt(1 - velocity^2 / c)
    analytic_sol(x,t) = c * atan(exp(γ * (x - velocity * t)))
    analytic_v(x,t) = (-c * γ * velocity * exp(γ * (x - velocity * t))) / (1 + exp(γ * (x - velocity * t))^2)
    analytic_w(x,t) = (c * γ * exp(γ * (x - velocity * t))) / (1 + exp(γ * (x - velocity * t))^2)


    function u_SindySol(x,t,params)
        local c_sol,γ_sol,v_sol =  params
        c_sol * atan(exp(γ_sol * (x - v_sol * t)))
    end

    function v_SindySol(x,t,params)
        local c_sol,γ_sol,v_sol =  params
        γ_sol * c_sol * exp(γ_sol * (x - v_sol * t)) / (1 + exp(γ_sol * (x - v_sol * t))^2)
    end

    function w_SindySol(x,t,params)
        local c_sol,γ_sol,v_sol =  params
        -v_sol * γ_sol * c_sol * exp(γ_sol * (x - v_sol * t)) / (1 + exp(γ_sol * (x - v_sol * t))^2)
    end


    x_ls = collect(-5:0.1:5)

    # x_vertics = [x_span[1], x_span[2], x_span[2], x_span[1]]
    # y_vertics = [-0.2, -0.2, 7.0, 7.0]

    # plot(x_ls,analytic_sol.(x_ls,0.05),label="Analytic Solution",ylims = (-0.2,7.00),size = (1000,400))
    # plot!(x_ls,[u_SindySol(x,0.05,sol.internal.x[2][1:3]) for x in x_ls],label = "SINDy Solution")
    # plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
    # # scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
    # title!("t = 0.05")
    # xlabel!("x")
    # ylabel!("u")

    # plot(x_ls,analytic_sol.(x_ls,5.0),label="Analytic Solution",ylims = (-0.2,7.00),size = (1000,400))
    # plot!(x_ls,[u_SindySol(x,5.0,sol.internal.x[100][1:3]) for x in x_ls],label = "SINDy Solution")
    # plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
    # # scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
    # title!("t = 5.0")
    # xlabel!("x")
    # ylabel!("u")


    # anim = Plots.@animate for (i, t) in enumerate(0:t_step:5.0)
    #     Plots.plot(x_ls,analytic_sol.(x_ls,t),label="Analytic Solution",ylims = (-0.2,7.00),size = (1000,400))
    #     Plots.plot!(x_ls,[u_SindySol(x,t,sol.internal.x[i+1][1:3]) for x in x_ls],label = "SINDy Solution")
    #     Plots.plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
    #     # scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
    #     Plots.title!("t = $t")
    #     Plots.xlabel!("x")
    #     Plots.ylabel!("u")
    # end 
    # gif(anim,fps = 5)

    sine_gordon_fig = Figure(linewidth = 2, size = (2200,1600))
    x_ls = collect(-5:0.1:5)
    sg_u_axis = Axis(sine_gordon_fig[1, 1], xlabel = "x", ylabel = "u", xticks = -3:1:10,xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
    # sg_ham_axis = Axis(sine_gordon_fig[2, 1], xlabel = "t", ylabel = "Hamiltonian",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = 0:0.5:5.0)
    sg_ham_axis = Axis(sine_gordon_fig[2, 1], xlabel = "t", ylabel = "Relative Hamiltonian Error",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = 0:0.5:5.0)

    x_ls = collect(-3:0.1:10)
    t = 0.05
    t_step = 0.05
    lin0 = lines!(sg_u_axis, x_ls, analytic_sol.(x_ls, t), label = "Analytic Solution", color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol.internal.x[Int(t/t_step)+1][1:3]
    u_sol_ls = [u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin1 =lines!(sg_u_axis, x_ls, u_sol_ls, label = "t = $t, VISE Solution,")

    t = 1.0
    lines!(sg_u_axis, x_ls, analytic_sol.(x_ls, t), label = false, color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol.internal.x[Int(t/t_step)+1][1:3]
    u_sol_ls = [u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin2 = lines!(sg_u_axis, x_ls, u_sol_ls, label = "t = $t, VISE Solution,")

    t = 2.0
    lines!(sg_u_axis, x_ls, analytic_sol.(x_ls, t), label = false, color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol.internal.x[Int(t/t_step)+1][1:3]
    u_sol_ls = [u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin3= lines!(sg_u_axis, x_ls, u_sol_ls, label = "t = $t, VISE Solution,")

    t = 3.0
    lines!(sg_u_axis, x_ls, analytic_sol.(x_ls, t), label = false, color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol.internal.x[Int(t/t_step)+1][1:3]
    u_sol_ls = [u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin4= lines!(sg_u_axis, x_ls, u_sol_ls, label = "t = $t, VISE Solution,")

    t = 4.0
    lines!(sg_u_axis, x_ls, analytic_sol.(x_ls, t), label = false, color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol.internal.x[Int(t/t_step)+1][1:3]
    u_sol_ls = [u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin5= lines!(sg_u_axis, x_ls, u_sol_ls, label = "t = $t, VISE Solution,")

    t = 5.0
    lines!(sg_u_axis, x_ls, analytic_sol.(x_ls, t), label = false, color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol.internal.x[Int(t/t_step)+1][1:3]
    u_sol_ls = [u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin6= lines!(sg_u_axis, x_ls, u_sol_ls, label = "t = $t, VISE Solution,")

    Legend(sine_gordon_fig[1, 2],[lin0, lin1, lin2, lin3, lin4, lin5, lin6],
            ["Analytical Solution",
                "t = 0.1, VISE Solution", "t = 1.0, VISE Solution", "t = 2.0, VISE Solution",
                "t = 3.0, VISE Solution", "t = 4.0, VISE Solution", "t = 5.0, VISE Solution"]
        ,labelsize = 30,framevisible = false)
    # Calculate Hamiltonian
    ham_ls = zeros(length(0:t_step:5.0))
    analytic_ham = zeros(length(0:t_step:5.0))
    for (i, t) in enumerate(0:t_step:5.0)
        tem_p = sol.internal.x[i+1][1:3]
        u_sol_ls = [u_SindySol(xx,t,tem_p) for xx in x_ls]
        v_sol_ls = [v_SindySol(xx,t,tem_p) for xx in x_ls]
        w_sol_ls = [w_SindySol(xx,t,tem_p) for xx in x_ls]
        current_domain_ham = [MultiSymplectic.SineGordon.hamiltonian(0.0,0.0,ui,vi,wi,(c=4.0,)) for (ui,vi,wi) in zip(u_sol_ls,v_sol_ls,w_sol_ls)]
        ham_ls[i] = sum(current_domain_ham)


        analytic_u_values = analytic_sol.(x_ls, t)
        analytic_v_values = analytic_v.(x_ls, t)
        analytic_w_values = analytic_w.(x_ls, t)
        current_ham = [MultiSymplectic.SineGordon.hamiltonian(0.0,0.0,ui,vi,wi,(c=4.0,)) for (ui,vi,wi) in zip(analytic_u_values,analytic_v_values,analytic_w_values)]
        analytic_ham[i] = sum(current_ham)

    end

    # lines!(sg_ham_axis, 0:t_step:5.0, ham_ls, label = "VISE Hamiltonian")
    # lines!(sg_ham_axis, 0:t_step:5.0, analytic_ham, label = "Analytic Hamiltonian", linestyle = :dash)

    lines!(sg_ham_axis, 0:t_step:5.0, abs.((ham_ls .- analytic_ham)./analytic_ham))

    # Legend(sine_gordon_fig[2, 2],sg_ham_axis,labelsize = 30,framevisible = false)
    Label(sine_gordon_fig[1:2, 0], "Sine-Gordon Equation", rotation = pi/2,
                fontsize = 30,tellheight = false)
    # sine_gordon_fig
    # save("figures/static_sine_gordon.pdf", sine_gordon_fig)

end






# # Wave equation 
# begin
    A1 = 0.4
    B1 = 0.3
    A2 = 0.0
    B2 = 0.0
    c = 1.
    u_exact_sol(x,t)=  (A1 * cos((pi*c*t)) + B1 * sin((pi*c*t) + pi/6)) * sin((pi*x)) + (A2 * cos((2*pi*c*t)) + B2 * sin((2*pi*c*t) + pi/6)) * sin((2*pi*x)) + 1.0 
    x_ls = collect(-2:0.01:2)
    Plots.plot(x_ls,u_exact_sol.(x_ls,0.0),label="Analytic Solution")
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

    wave_anim = @animate for (i, t) in enumerate(0:t_step:5.0)
        plot(x_ls,u_exact_sol.(x_ls,t),label="Analytic Solution",size = (1000,400), ylims = (-1.,1.50))
        # plot!(x_ls,[wave_u_SindySol(x,t,wave_sol.internal.x[i+1][1:4]) for x in x_ls],label = "SINDy Solution")
        # plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
        # scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
        title!("t = $t")
        xlabel!("x")
        ylabel!("u")
    end 
    gif(wave_anim, "figures/wave_eqn.gif",fps = 5)

# end

# Linear Transport 
begin
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
    lpde_lt = MultiSymplectic.LinearTransport.lpdeproblem(tstep = t_step,tspan =(0.0,9.0),xspan = (-0.5, 0.0))
    sol_lt = MultiSymplectic.integrate(lpde_lt,sindy_int_lt)



    c = 0.2
    t_step= 0.1

    function lt_u_SindySol(x,t,p)
        exp((p[1] * (x - 0.2*t) + p[2])*(x - 0.2 * t + p[3])*p[4]) * p[5]
    end

    function lt_v_SindySol(x,t,p)
        (-0.2p[1]*(p[3] - 0.2t + x)*p[4] - 0.2(p[2] + p[1]*(-0.2t + x))*p[4])*p[5]*exp((p[2] + p[1]*(-0.2t + x))*(p[3] - 0.2t + x)*p[4])   
    end
    function lt_w_SindySol(x,t,p)
        (p[1]*(p[3] - 0.2t + x)*p[4] + (p[2] + p[1]*(-0.2t + x))*p[4])*p[5]*exp((p[2] + p[1]*(-0.2t + x))*(p[3] - 0.2t + x)*p[4])
    end

    function exact_u(t,x)
        0.5 * exp(-(x+2 - 0.2*t)^2) / sqrt(π)   
    end

    function exact_v(t,x)
        0.2 *(2 + x - c*t)*exp(-((2 + x - c*t)^2)) / sqrt(π)
    end

    function exact_w(t,x)
        -(2 + x - c*t)*exp(-((2 + x - c*t)^2)) / sqrt(π)
    end

    # x_vertics = [xspan[1], xspan[2], xspan[2], xspan[1]]
    # y_vertics = [-0.05,-0.05, 0.3, 0.3]
    x_ls = collect(-5:0.01:5.0)

    # plot(x_ls,[lt_u_SindySol(xx,0.0,sol_lt.internal.x[2][1:5]) for xx in x_ls],label = "SINDy Solution")
    # plot!(x_ls,[lt_u_SindySol(xx,0.1,init_p) for xx in x_ls],label = "init Solution")
    # plot!(x_ls,exact_u.(0,x_ls),label="Analytic Solution",size = (1000,400))

    # tem_p = sol_lt.internal.x[2][1:5]
    # u_sol_ls = [lt_u_SindySol(xx,0,tem_p) for xx in x_ls] 
    # v_sol_ls = [lt_v_SindySol(xx,0,tem_p) for xx in x_ls]
    # w_sol_ls = [lt_w_SindySol(xx,0,tem_p) for xx in x_ls]
    # ham_ls = [MultiSymplectic.LinearTransport.hamiltonian(0.0,0.0,ui,vi,wi,(c=0.2,)) for (ui,vi,wi) in zip(u_sol_ls,v_sol_ls,w_sol_ls)]
    # total_sum = sum(ham_ls)

    total_sum = 0.0
    t_ls = t_step:t_step:4.0
    ham_ls = zeros(length(t_ls))
    lp_anim = @animate for (i, t) in enumerate(t_ls)
        p = plot(layout=@layout([a;b]), label="", size=(1200,400))# d;e

        tem_p = sol_lt.internal.x[i+1][1:5]
        u_sol_ls = [lt_u_SindySol(xx,t,tem_p) for xx in x_ls]
        v_sol_ls = [lt_v_SindySol(xx,t,tem_p) for xx in x_ls]
        w_sol_ls = [lt_w_SindySol(xx,t,tem_p) for xx in x_ls]
        plot!(p[1],x_ls,u_sol_ls,label = "VISE Solution")
        plot!(p[1],x_ls,exact_u.(t,x_ls),label="Analytic Solution",size = (800,400),ylims = (-0.05,0.3),xlims = (-5,5),linestyle = :dash)

        title!(p[1],"t = $t,\n exp(($(tem_p[1]) * (x -c*t) + $(tem_p[2]))*(x - c * t + $(tem_p[3]))*$(tem_p[4])) * $(tem_p[5])",titlefontsize = 7)
        xlabel!(p[1],"x")
        ylabel!(p[1],"u")

        current_domain_ham = [MultiSymplectic.LinearTransport.hamiltonian(0.0,0.0,ui,vi,wi,(c=0.2,)) for (ui,vi,wi) in zip(u_sol_ls,v_sol_ls,w_sol_ls)]
        total_sum = sum(current_domain_ham)
        ham_ls[i] = total_sum

        analytic_u_values = exact_u.(t,x_ls)
        analytic_v_values = exact_v.(t,x_ls)
        analytic_w_values = exact_w.(t,x_ls)

        exact_u_ls[i] = sum(analytic_u_values)
        exact_v_ls[i] = sum(analytic_v_values)
        exact_w_ls[i] = sum(analytic_w_values)
        current_ham = [MultiSymplectic.LinearTransport.hamiltonian(0.0,0.0,ui,vi,wi,(c=0.2,)) for (ui,vi,wi) in zip(analytic_u_values,analytic_v_values,analytic_w_values)]
        analytic_ham[i] = sum(current_ham)

        plot!(p[2],t_ls[1:i],abs.((ham_ls .- analytic_ham)./analytic_ham)[1:i],label = "Hamiltonian Error",size = (800,400), xlims = (0, 4),ylims = (-0.001,0.02))#
        # plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
        xlabel!(p[2],"t")
        ylabel!(p[2],"Hamiltonian")
    end
    gif(lp_anim, "figures/linear_transport2.gif",fps = 5)
    gif(lp_anim, fps = 5)
    
    t_ls = t_step:t_step:5.0
    VISE_ham = zeros(length(t_ls))
    analytic_ham = zeros(length(t_ls))

    VISE_u_ls = zeros(length(t_ls))
    VISE_v_ls = zeros(length(t_ls))
    VISE_w_ls = zeros(length(t_ls))

    exact_u_ls = zeros(length(t_ls))
    exact_v_ls = zeros(length(t_ls))
    exact_w_ls = zeros(length(t_ls))

    for (i, t) in enumerate(t_ls)
        tem_p = sol_lt.internal.x[i+1][1:5]
        u_sol_ls = [lt_u_SindySol(xx,t,tem_p) for xx in x_ls]
        v_sol_ls = [lt_v_SindySol(xx,t,tem_p) for xx in x_ls]
        w_sol_ls = [lt_w_SindySol(xx,t,tem_p) for xx in x_ls]

        VISE_u_ls[i] = sum(u_sol_ls)
        VISE_v_ls[i] = sum(v_sol_ls)
        VISE_w_ls[i] = sum(w_sol_ls)

        current_domain_ham = [MultiSymplectic.LinearTransport.hamiltonian(0.0,0.0,ui,vi,wi,(c=0.2,)) for (ui,vi,wi) in zip(u_sol_ls,v_sol_ls,w_sol_ls)]
        VISE_ham[i] = sum(current_domain_ham)

        analytic_u_values = exact_u.(t,x_ls)
        analytic_v_values = exact_v.(t,x_ls)
        analytic_w_values = exact_w.(t,x_ls)

        exact_u_ls[i] = sum(analytic_u_values)
        exact_v_ls[i] = sum(analytic_v_values)
        exact_w_ls[i] = sum(analytic_w_values)
        current_ham = [MultiSymplectic.LinearTransport.hamiltonian(0.0,0.0,ui,vi,wi,(c=0.2,)) for (ui,vi,wi) in zip(analytic_u_values,analytic_v_values,analytic_w_values)]
        analytic_ham[i] = sum(current_ham)
    end


    # makie_fig = Figure(linewidth = 2, size = (2200,800))
    # u_axis = Axis(makie_fig[1, 1],xlabel = "x", ylabel = "u",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = -4:0.5:2,)
    # ham_axis = Axis(makie_fig[2, 1],xlabel = "t", ylabel = "Hamiltonian",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = 0:0.5:5,)
    u_axis = Axis(sine_gordon_fig[3, 1],xlabel = "x", ylabel = "u",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = -4:0.5:2,)
    # ham_axis = Axis(sine_gordon_fig[4, 1],xlabel = "t", ylabel = "Hamiltonian",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = 0:0.5:5,)
    ham_axis = Axis(sine_gordon_fig[4, 1],xlabel = "t", ylabel = "Relative Hamiltonian Error",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = 0:0.5:5,)

    linewidth =3 
    x_ls = collect(-4:0.01:2.0)
    t = 0.1
    lin0 = lines!(u_axis, x_ls, exact_u.(t,x_ls), label="Analytical Solution",color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol_lt.internal.x[Int(t/t_step)+1][1:5]
    u_sol_ls = [lt_u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [lt_v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [lt_w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin1 =lines!(u_axis, x_ls, u_sol_ls, label = "t = 0.1, VISE Solution,")

    t = 1.0
    lines!(u_axis, x_ls, exact_u.(t,x_ls), label=false,color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol_lt.internal.x[Int(t/t_step)+1][1:5]
    u_sol_ls = [lt_u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [lt_v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [lt_w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin2 = lines!(u_axis, x_ls, u_sol_ls, label = "t = $(t), VISE Solution,")

    t = 2.0
    lines!(u_axis, x_ls, exact_u.(t,x_ls), label=false,color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol_lt.internal.x[Int(t/t_step)+1][1:5]
    u_sol_ls = [lt_u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [lt_v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [lt_w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin3= lines!(u_axis, x_ls, u_sol_ls, label = "t = $(t), VISE Solution,")

    t = 3.0
    lines!(u_axis, x_ls, exact_u.(t,x_ls), label=false,color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol_lt.internal.x[Int(t/t_step)+1][1:5]
    u_sol_ls = [lt_u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [lt_v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [lt_w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin4 = lines!(u_axis, x_ls, u_sol_ls, label = "t = $(t), VISE Solution,")

    t = 4.0
    lines!(u_axis, x_ls, exact_u.(t,x_ls), label=false,color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol_lt.internal.x[Int(t/t_step)+1][1:5]
    u_sol_ls = [lt_u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [lt_v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [lt_w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin5 = lines!(u_axis, x_ls, u_sol_ls, label = "t = $(t), VISE Solution,")

    t = 5.0
    lines!(u_axis, x_ls, exact_u.(t,x_ls), label=false,color = :black,linestyle = :dash,linewidth = 3)
    tem_p = sol_lt.internal.x[Int(t/t_step)+1][1:5]
    u_sol_ls = [lt_u_SindySol(xx,t,tem_p) for xx in x_ls]
    v_sol_ls = [lt_v_SindySol(xx,t,tem_p) for xx in x_ls]
    w_sol_ls = [lt_w_SindySol(xx,t,tem_p) for xx in x_ls]
    lin6 = lines!(u_axis, x_ls, u_sol_ls, label = "t = $(t), VISE Solution,")

    Legend(sine_gordon_fig[3, 2], [lin0, lin1, lin2, lin3, lin4, lin5,lin6],
        ["Analytical Solution",
            "t = 0.1, VISE Solution", "t = 1.0, VISE Solution", "t = 2.0, VISE Solution",
            "t = 3.0, VISE Solution", "t = 4.0, VISE Solution", "t = 5.0, VISE Solution"]#
    ,labelsize = 30,framevisible = false)

    # lines!(ham_axis,collect(t_step:t_step:5.0),VISE_ham,label = "VISE Hamiltonian")
    # lines!(ham_axis,collect(t_step:t_step:5.0),analytic_ham,label = "Analytic Hamiltonian",linestyle = :dash)
    # lines!(ham_axis, collect(t_step:t_step:4.0), abs.((VISE_ham .- analytic_ham)./analytic_ham)[1:40])
    lines!(ham_axis, collect(t_step:t_step:5.0), abs.((VISE_ham .- analytic_ham)./analytic_ham))

    # Legend(sine_gordon_fig[4, 2], ham_axis,labelsize = 30,framevisible = false)
    
    Label(sine_gordon_fig[3:4, 0], "Linear Transport Equation", rotation = pi/2,
            fontsize = 30,tellheight = false)
    # makie_fig
    # save("figures/static_linear_transport.pdf", makie_fig)
    save("figures/static_full_hamerror1_5.pdf", sine_gordon_fig)
end


# transport_ele_fig = Figure(linewidth = 2, size = (2200,800))
# transport_u_axis = Axis(transport_ele_fig[1, 1], xlabel = "x", ylabel = "u", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
# lines!(transport_u_axis,VISE_u_ls, label = "VISE u", color = :blue, linewidth = 2)
# lines!(transport_u_axis,exact_u_ls, label = "Exact u", color = :red, linestyle = :dash, linewidth = 2)

# transport_v_axis = Axis(transport_ele_fig[2, 1], xlabel = "x", ylabel = "v", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
# lines!(transport_v_axis,VISE_v_ls, label = "VISE v", color = :blue, linewidth = 2)
# lines!(transport_v_axis,exact_v_ls, label = "Exact v", color = :red, linestyle = :dash, linewidth = 2)

# transport_w_axis = Axis(transport_ele_fig[3, 1], xlabel = "x", ylabel = "w", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
# lines!(transport_w_axis,VISE_w_ls, label = "VISE w", color = :blue, linewidth = 2)
# lines!(transport_w_axis,exact_w_ls, label = "Exact w", color = :red, linestyle = :dash, linewidth = 2)
# Label(transport_ele_fig[1:3, 0], "Linear Transport Equation", rotation = pi/2,
#             fontsize = 30,tellheight = false)
# Legend(transport_ele_fig[1, 2], transport_u_axis)
# Legend(transport_ele_fig[2, 2], transport_v_axis)
# Legend(transport_ele_fig[3, 2], transport_w_axis)
# save("figures/linear_transport_element.pdf", transport_ele_fig)

# ham_relative_error = abs.(VISE_ham .- analytic_ham) ./ abs.(analytic_ham)
# ham_relative_error_fig = Figure(linewidth = 2, size = (2200,800))
# ham_axis = Axis(ham_relative_error_fig[1, 1], xlabel = "t", ylabel = "Relative Error", xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
# lines!(ham_axis,    collect(t_step:t_step:4.0), ham_relative_error[1:40], label = "Relative Error", color = :blue, linewidth = 2)
# Label(ham_relative_error_fig[1, 0], "Relative Error in Hamiltonian", rotation = pi/2,
#             fontsize = 30,tellheight = false)
# save("figures/ham_relative_error.pdf", ham_relative_error_fig)