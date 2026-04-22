using Revise
using QuadratureRules
using MultiSymplectic
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
using Plots
using ForwardDiff
using Base

using Infiltrator
using Base
using GeometricIntegratorsBase
using BenchmarkTools
# using Gtk4
using ProfileView
# using PProf
# by default, the first dimension is time, the second is space, 
# and just consider 1+1 now. 
relu2(x) = max(0, x) ^2
relu3(x) = max(0, x) ^3

GeometricIntegratorsBase.default_options(::NN_PDE_Integrator) = (
    x_abstol = 8eps(),
    f_abstol = 8eps(),
    max_iterations = 1,
)

S = 5
nn_pde_basis = NetworkPDEBasis(S,tanh,:Partially) # Partially, Fully
xspan = (0.0,1.0)

nn_int = NN_PDE_Integrator(nn_pde_basis,xspan = xspan, μ =:BSplineDirichlet,λ =:BSplineDirichlet,
k_μ_t = 4,k_λ_x = 4,initial_guess_method = OGA2D(), show_status=false)

t_step = 0.1
lpde = MultiSymplectic.Wave.lpdeproblem(timestep=t_step, timespan =(0.0,0.1),xspan = xspan)

# @benchmark sol = MultiSymplectic.integrate($lpde,$nn_int)

# p = @layout [a b c]
# p1 = plot([lpde.exact_u(t_step,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_u")
# plot!(p1, sol[1].u[1], label="sol.u")
# p2 = plot([lpde.exact_v(t_step,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_v")
# plot!(p2, sol[1].v[1], label="sol.v")
# p3 = plot([lpde.exact_w(t_step,xx) for xx in xspan[1]:0.01:xspan[2]], label="exact_w")
# plot!(p3, sol[1].w[1], label="sol.w")
# p = plot(p1, p2, p3, layout=p)
# savefig("NNInt_t=h_partially.pdf")

using GeometricSolutions
sol = GeometricSolution(lpde)
integrator = PDEIntegrator(lpde, nn_int)
import GeometricIntegratorsBase: solutionstep,nlsolution,current,parameters
solstep = solutionstep(integrator, sol[0])

MultiSymplectic.prior_initial_guess!(cache(integrator),solstep,integrator)
Q1 = GeometricIntegratorsBase.current(solstep)
Q2 = GeometricIntegratorsBase.history(solstep)
Q3 = GeometricIntegratorsBase.parameters(solstep)
MultiSymplectic.integrate_step!(Q1, Q2, Q3, integrator)
@profview MultiSymplectic.integrate_step!(Q1, Q2, Q3, integrator)
@profview MultiSymplectic.integrate_step!(Q1, Q2, Q3, integrator)



# @benchmark  MultiSymplectic.components!(nlsolution($integrator),current($solstep), parameters($solstep), $integrator)
# @benchmark MultiSymplectic.residual!(nlsolution($integrator), current($solstep), parameters($solstep), $integrator)