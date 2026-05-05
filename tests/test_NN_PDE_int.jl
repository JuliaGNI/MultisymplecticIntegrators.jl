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
using JLD2

# using Gtk4
# using ProfileView
# using PProf
# by default, the first dimension is time, the second is space, 
# and just consider 1+1 now. 
relu2(x) = max(0, x) ^2
relu3(x) = max(0, x) ^3

t_step = parse(Float64, ARGS[1])
reg_factor = parse(Float64, ARGS[2])
S = parse(Int, ARGS[3])

GeometricIntegratorsBase.default_options(::NN_PDE_Integrator) = (
    max_iterations = 100,
    regularization_factor = reg_factor,
    # f_abstol = 2eps(),
    # x_suctol = 2eps()
)

nn_pde_basis = NetworkPDEBasis(S,tanh,:Fully) # Partially, Fully
xspan = (0.0,1.0)
t_span = (0.0,10.0)
nn_int = NN_PDE_Integrator(nn_pde_basis,xspan = xspan, μ =:BSplineDirichlet,λ =:BSplineDirichlet,
k_μ_t = 4,k_λ_x = 4, show_status=false)

lpde = MultiSymplectic.Wave.lpdeproblem(timestep=t_step, timespan = t_span,xspan = xspan)

sol,internal_values = MultiSymplectic.integrate(lpde,nn_int)

wave_ham(u,v,w) = 1 / 2 * (0.5 * v^2 + w^2)
ham_ls = zeros(length(t_span[1]:t_step:t_span[2]))
analytic_ham = zeros(length(t_span[1]:t_step:t_span[2]))
x_ls = xspan[1]:0.01:xspan[2]

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
max_err = maximum(relative_ham_err)



record = Dict(
    "sol_u" => sol.u,
    "sol_v" => sol.v,
    "sol_w" => sol.w,
    "internal_values" => internal_values,
)
save("NNInt_fully_T$(t_span[2])_h$(t_step)_reg$(reg_factor)_S$(S)_err$(max_err).jld2", record)


# using GeometricSolutions
# sol = GeometricSolution(lpde)
# integrator = PDEIntegrator(lpde, nn_int)
# import GeometricIntegratorsBase: solutionstep,nlsolution,current,parameters
# solstep = solutionstep(integrator, sol[0])

# MultiSymplectic.prior_initial_guess!(cache(integrator),solstep,integrator)
# Q1 = GeometricIntegratorsBase.current(solstep)
# Q2 = GeometricIntegratorsBase.history(solstep)
# Q3 = GeometricIntegratorsBase.parameters(solstep)
# MultiSymplectic.integrate_step!(Q1, Q2, Q3, integrator)
# @profview MultiSymplectic.integrate_step!(Q1, Q2, Q3, integrator)
# @profview MultiSymplectic.integrate_step!(Q1, Q2, Q3, integrator)



# @benchmark  MultiSymplectic.components!(nlsolution($integrator),current($solstep), parameters($solstep), $integrator)
# @benchmark MultiSymplectic.residual!(nlsolution($integrator), current($solstep), parameters($solstep), $integrator)