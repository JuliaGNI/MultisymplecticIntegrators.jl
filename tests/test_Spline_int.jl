using MultiSymplectic
using Infiltrator

xspan = (0.3, 0.8)
spline_basis = BSpline2D(3,xspan = xspan)
spline_int = Galerkin_Bspline_Integrator(spline_basis,xspan=xspan)
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=0.3, timespan=(0.0, 0.3), xspan=xspan, xstep=0.01)

log_file="logs/SplineInt.txt"
open(log_file, "w") do io
    redirect_stdout(io) do
        sol = MultiSymplectic.integrate(lpde, spline_int)
    end
end

# [int.problem.lagrangian_system.functions.∂L∂W[1](int.problem.exact_u.(t_quad_nodes[i],int.problem.xspan[1]),int.problem.exact_v.(t_quad_nodes[i],int.problem.xspan[1]),int.problem.exact_w.(t_quad_nodes[i],int.problem.xspan[1]),int.problem.lagrangian_system.params) for i in 1:length(t_quad_nodes)]

# using Plots
# plot([lpde.exact_u(0.3,xx) for xx in 0.0:0.01:1.0])
# plot(0.0:0.01:1.0,sol.sol.u[1])