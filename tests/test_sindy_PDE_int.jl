using QuadratureRules
using MultiSymplectic
using Symbolics
using Parameters
using Plots
using CairoMakie
using Base
using Infiltrator

# Sine-Gordon Equation
begin
    @variables t x p[1:3]
    u_expr = p[1] * atan(exp(p[2] * (x - p[3] * t)))
    sindy_basis = SindyPDEBasis([u_expr], [p], t, [x])


    RT = 4
    RX = 12

    c2 = 3.9999951547393993
    v2 = 1.0397741828102778
    γ2 = 1.1546989298112151
    init_p = [c2,γ2,v2]

    t_step = 0.05
    x_span = (-2.0,5.0)
    sindy_int = Sindy_PDE_Integrator(sindy_basis,init_p,RT = RT,RX = RX, xspan = x_span,
        Nbasis_μ_t = 4,μ =:Lagrange,k_μ_t = 3,
        Nbasis_λ_x = 4,λ =:Lagrange,k_λ_x = 3,
        show_status = false,)
    lpde = MultiSymplectic.SineGordon.lpdeproblem(timestep = t_step,timespan =(0.0,5.0),xspan = x_span)
    # log_file="logs/sindy_pde.txt"
    # open(log_file, "w") do io
    #     redirect_stdout(io) do
    sol =  MultiSymplectic.integrate(lpde,sindy_int)
    #     end
    # end


    # x_ls = collect(-5:0.1:5)

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
    x_ls = collect(x_span[1]:0.01:x_span[2])
    sg_u_axis = Axis(sine_gordon_fig[1, 1], xlabel = "x", ylabel = "u", xticks = x_span[1]:0.5:x_span[2],xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
    # sg_ham_axis = Axis(sine_gordon_fig[2, 1], xlabel = "t", ylabel = "Hamiltonian",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = 0:0.5:5.0)
    sg_ham_axis = Axis(sine_gordon_fig[2, 1], xlabel = "t", ylabel = "Relative Hamiltonian Error",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = 0:0.5:5.0)

    t = 0.05
    t_step = 0.05
    lin0 = lines!(sg_u_axis, x_ls, lpde.exact_u.(t, x_ls), label = "Analytic Solution", color = :black,linestyle = :dash,linewidth = 3)
    lin1 =lines!(sg_u_axis, x_ls, sol.s.u[Int(t/t_step)], label = "t = $t, VISE Solution,")

    t = 1.0
    lines!(sg_u_axis, x_ls, lpde.exact_u.(t, x_ls), label = false, color = :black,linestyle = :dash,linewidth = 3)
    lin2 = lines!(sg_u_axis, x_ls, sol.s.u[Int(t/t_step)], label = "t = $t, VISE Solution,")

    t = 2.0
    lines!(sg_u_axis, x_ls, lpde.exact_u.(t, x_ls), label = false, color = :black,linestyle = :dash,linewidth = 3)
    lin3= lines!(sg_u_axis, x_ls, sol.s.u[Int(t/t_step)], label = "t = $t, VISE Solution,")

    t = 3.0
    lines!(sg_u_axis, x_ls, lpde.exact_u.(t, x_ls), label = false, color = :black,linestyle = :dash,linewidth = 3)
    lin4= lines!(sg_u_axis, x_ls, sol.s.u[Int(t/t_step)], label = "t = $t, VISE Solution,")

    t = 4.0
    lines!(sg_u_axis, x_ls, lpde.exact_u.(t, x_ls), label = false, color = :black,linestyle = :dash,linewidth = 3)
    lin5= lines!(sg_u_axis, x_ls, sol.s.u[Int(t/t_step)], label = "t = $t, VISE Solution,")

    Legend(sine_gordon_fig[1, 2],[lin0, lin1, lin2, lin3, lin4, lin5,],
            ["Analytical Solution",
                "t = 0.1, VISE Solution", "t = 1.0, VISE Solution", "t = 2.0, VISE Solution",
                "t = 3.0, VISE Solution", "t = 4.0, VISE Solution",]
    ,labelsize = 30,framevisible = false)

    c = 1.0
    sine_gordon_ham(u,v,w) = 1 / 2 * (c * v^2 + w^2) - (1 + cos(u))

    ham_ls = zeros(length(0:t_step:4.0))
    analytic_ham = zeros(length(0:t_step:4.0))
    for (i, t) in enumerate(0:t_step:4.0)
        current_domain_ham = [sine_gordon_ham(ui,vi,wi) for (ui,vi,wi) in zip(sol.s.u[i],sol.s.v[i],sol.s.w[i])]
        ham_ls[i] = sum(current_domain_ham)

        analytic_u_values = lpde.exact_u.(t, x_ls)
        analytic_v_values = lpde.exact_v.(t, x_ls)
        analytic_w_values = lpde.exact_w.(t, x_ls)
        current_ham = [sine_gordon_ham(ui,vi,wi) for (ui,vi,wi) in zip(analytic_u_values,analytic_v_values,analytic_w_values)]
        analytic_ham[i] = sum(current_ham)
    end

    # lines!(sg_ham_axis, 0:t_step:5.0, ham_ls, label = "VISE Hamiltonian")
    # lines!(sg_ham_axis, 0:t_step:5.0, analytic_ham, label = "Analytic Hamiltonian", linestyle = :dash)

    lines!(sg_ham_axis, 0:t_step:4.0, abs.((ham_ls .- analytic_ham)./analytic_ham))

    # Legend(sine_gordon_fig[2, 2],sg_ham_axis,labelsize = 30,framevisible = false)
    Label(sine_gordon_fig[1:2, 0], "Sine-Gordon Equation", rotation = pi/2,
                fontsize = 30,tellheight = false)
    sine_gordon_fig
    save("logs/static_sine_gordon.pdf", sine_gordon_fig)

end

# # Wave equation 
begin

    @variables t x p[1:4]
    c = 0.5
    l = 1.0
    u_expr_wave = (p[1] * cos((pi*c*t)/l) + p[2] * sin((pi*c*t)/l + pi/6)) * sin((pi*x)/l) + (p[3] * cos((2*pi*c*t)/l) + p[4] * sin((2*pi*c*t)/l + pi/6)) * sin((2*pi*x)/l)
    init_p_wave = [0.803001, 0.800401, 0.400001, 0.300001] # initial guess for the parameters
    sindy_basis_wave = SindyPDEBasis([u_expr_wave], [p], t, [x])
    RT = 4
    RX = 12
    t_step = 0.25
    xspan = (0.5,0.8)
    sindy_int_wave = Sindy_PDE_Integrator(sindy_basis_wave,init_p_wave,RT = RT,RX = RX, xspan = xspan,
        Nbasis_μ_t = 4,μ =:Lagrange,k_μ_t = 3,
        Nbasis_λ_x = 4,λ =:Lagrange,k_λ_x = 3,
        show_status = false,)
    lpde_wave = MultiSymplectic.Wave.lpdeproblem(timespan =(0.0,5.0),timestep = t_step, xspan = xspan,params = (c=c,A1 = 0.8,A2 = 0.4,B1 = 0.8,B2 = 0.3,l = l,))
    log_file="logs/wave_sindy_pde.txt"
    open(log_file, "w") do io
        redirect_stdout(io) do
            wave_sol = MultiSymplectic.integrate(lpde_wave,sindy_int_wave)

            wave_fig = Figure(linewidth = 2, size = (2200,1600))
            x_ls = collect(xspan[1]:0.01:xspan[2])
            sg_u_axis = Axis(wave_fig[1, 1], xlabel = "x", ylabel = "u", xticks = xspan[1]:0.5:xspan[2],xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20)
            # sg_ham_axis = Axis(wave_fig[2, 1], xlabel = "t", ylabel = "Hamiltonian",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = 0:0.5:5.0)
            sg_ham_axis = Axis(wave_fig[2, 1], xlabel = "t", ylabel = "Relative Hamiltonian Error",xlabelsize = 25, ylabelsize = 25,yticklabelsize = 20,xticklabelsize = 20,xticks = 0:0.5:5.0)

            t1 = 0.5
            lin1 = lines!(sg_u_axis, x_ls, lpde_wave.exact_u.(t1, x_ls), label = false, color = :black,linestyle = :dash,linewidth = 3)
            lin2 = lines!(sg_u_axis, x_ls, wave_sol.s.u[Int(t1/t_step)], label = "t = $t1, VISE Solution,")

            t2 = 1.0
            lines!(sg_u_axis, x_ls, lpde_wave.exact_u.(t2, x_ls), label = false, color = :black,linestyle = :dash,linewidth = 3)
            lin3= lines!(sg_u_axis, x_ls, wave_sol.s.u[Int(t2/t_step)], label = "t = $t2, VISE Solution,")
            
            t3 = 1.5
            lines!(sg_u_axis, x_ls, lpde_wave.exact_u.(t3, x_ls), label = false, color = :black,linestyle = :dash,linewidth = 3)
            lin4= lines!(sg_u_axis, x_ls, wave_sol.s.u[Int(t3/t_step)], label = "t = $t3, VISE Solution,")

            t4 = 2.0
            lines!(sg_u_axis, x_ls, lpde_wave.exact_u.(t4, x_ls), label = false, color = :black,linestyle = :dash,linewidth = 3)
            lin5= lines!(sg_u_axis, x_ls, wave_sol.s.u[Int(t4/t_step)], label = "t = $t4, VISE Solution,")

            Legend(wave_fig[1, 2],[lin1, lin2, lin3, lin4, lin5,],
                    ["Analytical Solution",
                        "t = $t1, VISE Solution", "t = $t2, VISE Solution",
                        "t = $t3, VISE Solution", "t = $t4, VISE Solution",]
            ,labelsize = 30,framevisible = false)
            
            wave_ham(u,v,w) = 1 / 2 * (c * v^2 + w^2)

            ham_ls = zeros(length(0:t_step:4.0))
            analytic_ham = zeros(length(0:t_step:4.0))
            for (i, t) in enumerate(0:t_step:4.0)
                current_domain_ham = [wave_ham(ui,vi,wi) for (ui,vi,wi) in zip(wave_sol.s.u[i],wave_sol.s.v[i],wave_sol.s.w[i])]
                ham_ls[i] = sum(current_domain_ham)

                analytic_u_values = lpde_wave.exact_u.(t, x_ls)
                analytic_v_values = lpde_wave.exact_v.(t, x_ls)
                analytic_w_values = lpde_wave.exact_w.(t, x_ls)
                current_ham = [wave_ham(ui,vi,wi) for (ui,vi,wi) in zip(analytic_u_values,analytic_v_values,analytic_w_values)]
                analytic_ham[i] = sum(current_ham)
            end

            lines!(sg_ham_axis, 0:t_step:4.0, abs.((ham_ls .- analytic_ham)./analytic_ham))

            # Legend(wave_fig[2, 2],sg_ham_axis,labelsize = 30,framevisible = false)
            Label(wave_fig[1:2, 0], "Wave Equation", rotation = pi/2,
                        fontsize = 30,tellheight = false)
            wave_fig
            save("logs/static_wave.pdf", wave_fig)
        end
    end
     # x_ls = collect(0.2:0.01:0.8)
    # plot(x_ls,u_exact_sol.(x_ls,0.05),label="Analytic Solution",ylims = (-1.,1.50),size = (1000,400))
    # plot!(x_ls,[wave_u_SindySol(x,0.05,wave_sol.internal.x[2][1:4]) for x in x_ls],label = "SINDy Solution")
    # title!("t = 0.05")
    # xlabel!("x")
    # ylabel!("u")


    # x_vertics = [xspan[1], xspan[2], xspan[2], xspan[1]]
    # y_vertics = [-1,-1, 1.5, 1.5]

    # wave_anim = @animate for (i, t) in enumerate(0:t_step:5.0)
    #     plot(x_ls,u_exact_sol.(x_ls,t),label="Analytic Solution",size = (1000,400), ylims = (-1.,1.50))
    #     # plot!(x_ls,[wave_u_SindySol(x,t,wave_sol.internal.x[i+1][1:4]) for x in x_ls],label = "SINDy Solution")
    #     # plot!(Shape(x_vertics, y_vertics), label = "Spatial Domain", color = :lightblue, alpha = 0.2, linestyle = :dash)
    #     # scatter!(x_span[2] .* QuadratureRules.GaussLegendreQuadrature(RX).nodes,-0.18 * ones(RX),label = "Quadrature Points", color = :red, markersize = 5, markershape = :x)
    #     title!("t = $t")
    #     xlabel!("x")
    #     ylabel!("u")
    # end 
    # gif(wave_anim, "figures/wave_eqn.gif",fps = 5)


end
