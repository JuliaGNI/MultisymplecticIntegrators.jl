using MultiSymplectic
using Infiltrator
using Base
using GeometricIntegratorsBase
using Plots

GeometricIntegratorsBase.default_options(::Galerkin_Bspline_Integrator) = (
    x_abstol = 8eps(),
    f_abstol = 8eps(),
    max_iterations = 100,
)

xspan = (0.0, 1.0)
spline_basis = BSpline2D(4,xspan = xspan)
spline_int = Galerkin_Bspline_Integrator(spline_basis,xspan=xspan,RT_per_interval = 4,RX_per_interval = 4)
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=0.3, timespan=(0.0, 0.6), xspan=xspan, xstep=0.01)

# log_file="logs/SplineInt_2.txt"
# open(log_file, "w") do io
#     redirect_stdio(stdout=log_file, stderr=log_file) do
        println("Start Spline Integrator")
        sol = MultiSymplectic.integrate(lpde, spline_int)

        p = @layout [a b c; d e f]
        p1 = plot([lpde.exact_u(0.3,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_u")
        plot!(p1, sol.u[1], label="sol.u")
        p2 = plot([lpde.exact_v(0.3,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_v")
        plot!(p2, sol.v[1], label="sol.v")
        p3 = plot([lpde.exact_w(0.3,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_w")
        plot!(p3, sol.w[1], label="sol.w")
        p4 = plot([lpde.exact_u(0.3,xx) for xx in xspan[1]:0.01:xspan[2]] .- sol.u[1], label="error_u")
        p5 = plot([lpde.exact_v(0.3,xx) for xx in xspan[1]:0.01:xspan[2]] .- sol.v[1], label="error_v")
        p6 = plot([lpde.exact_w(0.3,xx) for xx in xspan[1]:0.01:xspan[2]] .- sol.w[1], label="error_w")
        p = plot(p1, p2, p3, p4, p5, p6, layout=p)
        savefig("logs/SplineInt_t=h.pdf")

        p = @layout [a b c; d e f]
        p1 = plot([lpde.exact_u(0.6,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_u")
        plot!(p1, sol.u[2], label="sol.u")
        p2 = plot([lpde.exact_v(0.6,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_v")
        plot!(p2, sol.v[2], label="sol.v")
        p3 = plot([lpde.exact_w(0.6,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_w")
        plot!(p3, sol.w[2], label="sol.w")
        p4 = plot([lpde.exact_u(0.6,xx) for xx in xspan[1]:0.01:xspan[2]] .- sol.u[2], label="error_u")
        p5 = plot([lpde.exact_v(0.6,xx) for xx in xspan[1]:0.01:xspan[2]] .- sol.v[2], label="error_v")
        p6 = plot([lpde.exact_w(0.6,xx) for xx in xspan[1]:0.01:xspan[2]] .- sol.w[2], label="error_w")
        p = plot(p1, p2, p3, p4, p5, p6, layout=p)
        savefig("logs/SplineInt_t=2h.pdf")
#     end
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