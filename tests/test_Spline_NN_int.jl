using MultiSymplectic
using GeometricIntegrators
using Infiltrator
spline_basis = BSpline2D(3)
spline_int = Galerkin_Bspline_Integrator(spline_basis)
lpde = MultiSymplectic.SineGordon.lpdeproblem(tstep=0.3, tspan=(0.0, 0.3), xspan=(0.0, 1.0), xstep=0.01)


log_file="logs/SplineInt.txt"
open(log_file, "w") do io
    redirect_stdout(io) do
        sol = MultiSymplectic.integrate(lpde, spline_int)
    end
end

# [int.problem.lagrangian_system.functions.∂L∂W[1](int.problem.exact_u.(t_quad_nodes[i],int.problem.xspan[1]),int.problem.exact_v.(t_quad_nodes[i],int.problem.xspan[1]),int.problem.exact_w.(t_quad_nodes[i],int.problem.xspan[1]),int.problem.lagrangian_system.params) for i in 1:length(t_quad_nodes)]

