using MultiSymplectic
using Infiltrator
using Base
using GeometricIntegratorsBase
using Plots
using JLD2
GeometricIntegratorsBase.default_options(::Galerkin_Bspline_Integrator) = (
    x_suctol = 8eps(),
    f_abstol = 8eps(),
    max_iterations = 3,
)

k = parse(Int, ARGS[1])
t_knot_interval = parse(Float64, ARGS[2])
x_knot_interval = parse(Float64, ARGS[3])
t_step = parse(Float64, ARGS[4])
# k = 4
# t_knot_interval = 0.1
# x_knot_interval = 0.1
t_span = (0.,20.0)
xspan = (0.0, 1.0)
# for t_step in [0.1,0.2,0.4]
    # t_step = 0.2
    lpde = MultiSymplectic.Wave.lpdeproblem(timestep=t_step, timespan=t_span, xspan=xspan, xstep=0.01)
    wave_ham(u,v,w) = 1 / 2 * (c * v^2 + w^2)
    x_ls = xspan[1]:lpde.xstep:xspan[2]
    c = lpde.params.c

    # log_file="Spline_int_logs/SplineInt_h=$(t_step)_k=$(k)_t_knot_interval=$(t_knot_interval)_x_knot_interval=$(x_knot_interval).txt"
    # open(log_file, "w") do io
        # redirect_stdio(stdout=log_file, stderr=log_file) do
            record_results = Dict()

            spline_basis = BSpline2D(k,xspan = xspan, t_knot_interval = t_knot_interval,x_knot_interval = x_knot_interval)
            spline_int = Galerkin_Bspline_Integrator(spline_basis,xspan=xspan,RT_per_interval = k,RX_per_interval = k,show_status = false)
            println("Start Spline Integrator")
            sol = MultiSymplectic.integrate(lpde, spline_int)
            println("End Spline Integrator with h=$(t_step), k=$(k), t_knot_interval=$(t_knot_interval), x_knot_interval=$(x_knot_interval)")

            ham_ls = zeros(length(t_span[1]:t_step:t_span[2]))
            analytic_ham = zeros(length(t_span[1]:t_step:t_span[2]))
            for (i, t) in enumerate(t_span[1]:t_step:t_span[2])
                current_domain_ham = [wave_ham(ui,vi,wi) for (ui,vi,wi) in zip(sol.u[i-1],sol.v[i-1],sol.w[i-1])]
                ham_ls[i] = sum(current_domain_ham)

                analytic_u_values = lpde.exact_u.(t, x_ls)
                analytic_v_values = lpde.exact_v.(t, x_ls)
                analytic_w_values = lpde.exact_w.(t, x_ls)
                current_ham = [wave_ham(ui,vi,wi) for (ui,vi,wi) in zip(analytic_u_values,analytic_v_values,analytic_w_values)]
                analytic_ham[i] = sum(current_ham)
            end
            relative_ham_err = abs.((ham_ls .-  analytic_ham) ./ analytic_ham)
            plot(t_span[1]:t_step:t_span[2], relative_ham_err, xlabel="Time", ylabel="Relative Hamiltonian Error")
            savefig("Spline_int_logs2/SplineInt_Hamiltonian_Error_h=$(t_step)_k=$(k)_t_knot_interval=$(t_knot_interval)_x_knot_interval=$(x_knot_interval).png")
            
            record_results["maximum_relative_ham_err"] = maximum(relative_ham_err)
            record_results["sol_u"] = sol.u
            record_results["sol_v"] = sol.v
            record_results["sol_w"] = sol.w
            record_results["sol_hamiltonian"] = ham_ls
            record_results["analytic_hamiltonian"] = analytic_ham
            save("Spline_int_logs2/SplineInt_h=$(t_step)_k=$(k)_t_knot_interval=$(t_knot_interval)_x_knot_interval=$(x_knot_interval).jld2", record_results)
            println("results saved: h=$(t_step), k=$(k), t_knot_interval=$(t_knot_interval), x_knot_interval=$(x_knot_interval)")
    #     end
    # end
# end


error_table = zeros(3,4)
end_time = t_span[2]
for (i, t_step) in enumerate([0.1,0.2,0.4])
    for (ki,k) in enumerate([3,4,5,6])
        jld2_file = "Spline_int_logs2/SplineInt_h=$(t_step)_k=$(k)_t_knot_interval=0.1_x_knot_interval=0.1.jld2"
        record_results = load(jld2_file)
        ham_ls = record_results["sol_hamiltonian"]
        analytic_ham = record_results["analytic_hamiltonian"]
        relative_ham_err = abs.((ham_ls .-  analytic_ham) ./ analytic_ham)
        error_table[i,ki] = maximum(relative_ham_err[1:Int(floor(end_time / t_step))])
        # error_table[i,ki] = record_results["maximum_relative_ham_err"]
    end
end

fig = Figure()
ax = Axis(fig[1,1], xlabel="h", ylabel="Maximum Relative Hamiltonian Error", yscale = log10)
for (ki, k) in enumerate([3,4,5,6])
    scatterlines!(ax, [0.1,0.2,0.4], error_table[:,ki], label="k=$(k)")
end
axislegend(ax)

save("Spline_int_logs2/SplineInt_Error_Table.pdf", fig)
