
using MultiSymplectic
using Infiltrator
using Base
using GeometricIntegratorsBase
GeometricIntegratorsBase.default_options(::Galerkin_Bspline_Integrator) = (
    x_abstol = 8eps(),
    f_abstol = 8eps(),
    max_iterations = 2,
)

xspan = (0.3, 0.7)
spline_basis = BSpline2D(4,xspan = xspan)
spline_int = Galerkin_Bspline_Integrator(spline_basis,xspan=xspan)
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=0.3, timespan=(0.0, 0.3), xspan=xspan, xstep=0.01)

# log_file="logs/SplineInt.txt"
# open(log_file, "w") do io
#     redirect_stdio(stdout=log_file, stderr=log_file) do
        println("Start Spline Integrator")
        sol = MultiSymplectic.integrate(lpde, spline_int)
#     end
# end


using Plots
plot([lpde.exact_u(0.3,xx) for xx in xspan[1]:0.01:xspan[2]])
plot!(sol.u[1])
