
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

xspan = (0.3, 0.8)
spline_basis = BSpline2D(4,xspan = xspan)
spline_int = Galerkin_Bspline_Integrator(spline_basis,xspan=xspan,RT = 32,RX = 32,k_μ_t = 8,k_λ_x = 8)
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=0.3, timespan=(0.0, 0.3), xspan=xspan, xstep=0.01)

# log_file="logs/SplineInt.txt"
# open(log_file, "w") do io
#     redirect_stdio(stdout=log_file, stderr=log_file) do
        println("Start Spline Integrator")
        sol = MultiSymplectic.integrate(lpde, spline_int)

        plot([lpde.exact_u(0.3,xx) for xx in xspan[1]:0.01:xspan[2]])
        plot!(sol.u[1])
        plot([lpde.exact_v(0.3,xx) for xx in xspan[1]:0.01:xspan[2]])
        plot!(sol.v[1])
        plot([lpde.exact_w(0.3,xx) for xx in xspan[1]:0.01:xspan[2]])
        plot!(sol.w[1])
        savefig("logs/SplineInt_u.pdf")
#     end
# end



