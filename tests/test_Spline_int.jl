
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
spline_int = Galerkin_Bspline_Integrator(spline_basis,xspan=xspan,RT = 32,RX = 32)
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=0.3, timespan=(0.0, 0.6), xspan=xspan, xstep=0.01)

log_file="logs/SplineInt.txt"
open(log_file, "w") do io
    redirect_stdio(stdout=log_file, stderr=log_file) do
        println("Start Spline Integrator")
        sol = MultiSymplectic.integrate(lpde, spline_int)

        p = @layout [a b c]
        p1 = plot([lpde.exact_u(0.3,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_u")
        plot!(p1, sol.u[1], label="sol.u")
        p2 = plot([lpde.exact_v(0.3,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_v")
        plot!(p2, sol.v[1], label="sol.v")
        p3 = plot([lpde.exact_w(0.3,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_w")
        plot!(p3, sol.w[1], label="sol.w")
        p = plot(p1, p2, p3, layout=p)
        savefig("logs/SplineInt_t=h.pdf")

        p = @layout [a b c]
        p1 = plot([lpde.exact_u(0.6,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_u")
        plot!(p1, sol.u[2], label="sol.u")
        p2 = plot([lpde.exact_v(0.6,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_v")
        plot!(p2, sol.v[2], label="sol.v")
        p3 = plot([lpde.exact_w(0.6,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_w")
        plot!(p3, sol.w[2], label="sol.w")
        p = plot(p1, p2, p3, layout=p)
        savefig("logs/SplineInt_t=2h.pdf")
    end
end



