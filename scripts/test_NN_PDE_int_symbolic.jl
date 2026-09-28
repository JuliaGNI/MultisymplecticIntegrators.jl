using Revise
using QuadratureRules
using MultisymplecticIntegrators
using AbstractNeuralNetworks
using Random
using GeometricMachineLearning
using Zygote
using ForwardDiff
using Base

using Infiltrator
using Base
using GeometricIntegratorsBase
using BenchmarkTools
using JLD2
using Profile
# using ProfView

# using Gtk4
# using ProfileView
# using PProf
# by default, the first dimension is time, the second is space, 
# and just consider 1+1 now. 
relu2(x) = max(0, x) ^ 2
relu3(x) = max(0, x) ^ 3

# t_step = parse(Float64, ARGS[1])
# reg_factor = parse(Float64, ARGS[2])
# S = parse(Int, ARGS[3])

t_step = 0.5
reg_factor = 1e-5
S = 70
max_iters = 100

function GeometricIntegratorsBase.default_options(::NN_PDE_Integrator_Symbolic)
    (
        max_iterations = max_iters,
        warn_iterations = max_iters,
        regularization_factor = reg_factor,
        # f_abstol = 2eps(),
        # x_suctol = 2eps(),
        verbosity = 2
    )
end

nn_pde_basis_sym = NetworkPDEBasis_Symbolic(S, tanh, :Fully) # Partially, Fully

xspan = (0.0, 1.0)
t_span = (0.0, 10.0)
nn_int = NN_PDE_Integrator_Symbolic(nn_pde_basis_sym, xspan = xspan, μ = :BSplineDirichlet,
    λ = :BSplineDirichlet, t_num_interval = 2, x_num_interval = 5,
    k_μ_t = 4, k_λ_x = 4, show_status = false)

lpde = MultisymplecticIntegrators.Wave.lpdeproblem(timestep = t_step, timespan = t_span, xspan = xspan)

sol = MultisymplecticIntegrators.integrate(lpde, nn_int)

c=0.5
A1 = 0.8
A2 = 0.0
B1 = 0.8
B2 = 0.0
l = 1.0

function hamiltonian_density(t, x, u, v, w)
    1 / 2 * (v[1]^2 + c^2 * w[1]^2)
end

# Hamiltonian on a given spatial-temporal domain
function hamiltonian(u_quad_values::Matrix{Float64}, v_quad_values::Matrix{Float64},
        w_quad_values::Matrix{Float64}, grid_quad_node::Matrix{Vector{ST}},
        grid_quad_weights::Matrix{ST}, xspan = xspan, timestep = t_step) where {ST}
    ham = 0.0

    RT = size(u_quad_values, 1)
    RX = size(u_quad_values, 2)
    x_domain = xspan[2] - xspan[1]
    for rt in 1:RT
        for rx in 1:RX
            ham += x_domain * timestep * grid_quad_weights[rt, rx] *
                   hamiltonian_density(
                       grid_quad_node[rt, rx][1], grid_quad_node[rt, rx][2],
                       u_quad_values[rt, rx], v_quad_values[rt, rx], w_quad_values[rt, rx])
        end
    end

    return ham
end

ham_ls = zeros(length(t_span[1]:t_step:(t_span[2] - t_step)))
analytic_ham = zeros(length(t_span[1]:t_step:(t_span[2] - t_step)))
for (i, t) in enumerate(t_span[1]:t_step:(t_span[2] - t_step))
    ham_ls[i] = hamiltonian(sol.u_quad_values[i][1, :, :], sol.v_quad_values[i][1, :, :],
        sol.w_quad_values[i][1, :, :], nn_int.grid_matrix, nn_int.grid_weights)

    analytic_u_values = [lpde.exact_u(t + t_step * nn_int.grid_matrix[i, j][1],
                             nn_int.grid_matrix[i, j][2]; params = lpde.params)
                         for i in 1:size(nn_int.grid_matrix, 1),
    j in 1:size(nn_int.grid_matrix, 2)]
    analytic_v_values = [lpde.exact_v(t + t_step * nn_int.grid_matrix[i, j][1],
                             nn_int.grid_matrix[i, j][2]; params = lpde.params)
                         for i in 1:size(nn_int.grid_matrix, 1),
    j in 1:size(nn_int.grid_matrix, 2)]
    analytic_w_values = [lpde.exact_w(t + t_step * nn_int.grid_matrix[i, j][1],
                             nn_int.grid_matrix[i, j][2]; params = lpde.params)
                         for i in 1:size(nn_int.grid_matrix, 1),
    j in 1:size(nn_int.grid_matrix, 2)]

    analytic_ham[i] = hamiltonian(analytic_u_values, analytic_v_values, analytic_w_values,
        nn_int.grid_matrix, nn_int.grid_weights)
end

relative_ham_err = abs.((ham_ls .- analytic_ham) ./ analytic_ham)
max_err = maximum(relative_ham_err)

record = Dict(
    "sol_u" => sol.sol.u,
    "sol_v" => sol.sol.v,
    "sol_w" => sol.sol.w,
    "sol_u_quad" => sol.u_quad_values,
    "sol_v_quad" => sol.v_quad_values,
    "sol_w_quad" => sol.w_quad_values,
    "internal_solutions" => sol.internal_solutions,
    "ham_ls" => ham_ls,
    "analytic_ham" => analytic_ham,
    "relative_ham_err" => relative_ham_err,
    "max_err" => max_err
)
save("NNInt_fully_Symb_T$(t_span[2])_h$(t_step)_reg$(reg_factor)_S$(S)_err$(max_err).jld2", record)

# using GeometricSolutions
# sol = GeometricSolution(lpde)
# integrator = PDEIntegrator(lpde, nn_int)
# import GeometricIntegratorsBase: solutionstep,nlsolution,current,parameters
# solstep = solutionstep(integrator, sol[0])

# MultisymplecticIntegrators.prior_initial_guess!(cache(integrator),solstep,integrator)
# Q1 = GeometricIntegratorsBase.current(solstep)
# Q2 = GeometricIntegratorsBase.history(solstep)
# Q3 = GeometricIntegratorsBase.parameters(solstep)
# MultisymplecticIntegrators.integrate_step!(Q1, Q2, Q3, integrator)
# VSCodeServer.@profview MultisymplecticIntegrators.integrate_step!(Q1, Q2, Q3, integrator)
