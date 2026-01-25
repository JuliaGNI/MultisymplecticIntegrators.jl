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
t_span = (0.,2.0)
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
            savefig("Spline_int_logs/SplineInt_Hamiltonian_Error_h=$(t_step)_k=$(k)_t_knot_interval=$(t_knot_interval)_x_knot_interval=$(x_knot_interval).png")
            
            record_results["maximum_relative_ham_err"] = maximum(relative_ham_err)
            record_results["sol_u"] = sol.u
            record_results["sol_v"] = sol.v
            record_results["sol_w"] = sol.w
            record_results["sol_hamiltonian"] = ham_ls
            record_results["analytic_hamiltonian"] = analytic_ham
            save("Spline_int_logs/SplineInt_h=$(t_step)_k=$(k)_t_knot_interval=$(t_knot_interval)_x_knot_interval=$(x_knot_interval).jld2", record_results)
            println("results saved: h=$(t_step), k=$(k), t_knot_interval=$(t_knot_interval), x_knot_interval=$(x_knot_interval)")
    #     end
    # end
# end
# using BSplineKit
# using CairoMakie
# xspan = (0.0,1.0)
# k = 4
# x_knot_interval = 0.05
# xs = collect(xspan[1]:x_knot_interval:xspan[2])
# Bx = BSplineBasis(BSplineOrder(k), xs)
# basis = RecombinedBSplineBasis(Bx,Derivative(0))
# function plot_knots!(ax, ts; knot_offset = 0.05, kws...)
#     ys = zero(ts)
#     # Add offset to distinguish knots with multiplicity > 1
#     for i in eachindex(ts)[(begin + 1):end]
#         if ts[i] == ts[i - 1]
#             ys[i] = ys[i - 1] + knot_offset
#         end
#     end
#     scatter!(ax, ts, ys; marker = '×', markersize = 24, color = :gray, kws...)
#     ax
# end

# function plot_basis!(ax, B; eval_args = (), kws...)
#     cmap = cgrad(:tab20)
#     N = length(B)
#     ts = knots(B)
#     hlines!(ax, 0; color = :gray)
#     for (n, bi) in enumerate(B)
#         color = cmap[(n - 1) / (N - 1)]
#         i, j = extrema(support(bi))
#         lines!(ax, ts[i]..ts[j], x -> bi(x, eval_args...); color, linewidth = 2.5)
#     end
#     plot_knots!(ax, ts; kws...)
#     ax
# end

# fig = Figure()
# ax = Axis(
#     fig[1, 1];
#     xlabel = rich("x"; font = :italic),
#     ylabel = rich("b", subscript("i"), rich("(x)"; offset = (0.1, 0.0)); font = :italic),
# )
# plot_basis!(ax, basis)
# fig

# fig_original = Figure()
# ax_original = Axis(
#     fig_original[1, 1];
#     xlabel = rich("x"; font = :italic),
#     ylabel = rich("B", subscript("i"), rich("(x)"; offset = (0.1, 0.0)); font = :italic),
# )
# plot_basis!(ax_original,Bx)
# fig_original

# using Random
# rng = MersenneTwister(42)
# Ndata = 20
# xs = range(0, 1; length = Ndata) .+ 0.01 .* randn(rng, Ndata)
# sort!(xs)  # make sure coordinates are sorted
# xs[begin] = 0; xs[end] = 1;   # not strictly necessary; just to set the data limits
# ys = sinpi.(xs) .+ 0.02 .* randn(rng, Ndata)

# S = interpolate(xs, ys, BSplineOrder(k),Natural())
# plot_basis!(ax_original,S.spline.basis)
# fig_original

# scatter(xs, ys; label = "Data", color = :black)
# lines(0..1, S; label = "k = 4", color = Cycled(4 - 3))
