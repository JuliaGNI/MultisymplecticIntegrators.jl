# cd("tests")
# using Pkg
# Pkg.activate(".")
using BSplineKit
using MultiSymplectic
using Infiltrator
using Base
using GeometricIntegratorsBase
using Base.Threads
using Plots
using JLD2

k = parse(Int, ARGS[1])
t_step = parse(Float64, ARGS[2])
t_knot_interval = parse(Float64, ARGS[3])
x_knot_interval = parse(Float64, ARGS[4])
# regularization_factor = parse(Float64, ARGS[5])

# k = 3
# t_step = 0.1
regularization_factor = 0.0
GeometricIntegratorsBase.default_options(::Galerkin_Full_Restriction_Bspline_Integrator) = (
    # x_suctol = 2eps(),
    # f_abstol = 2eps(),
    regularization_factor = regularization_factor,
    max_iterations = 100,
    verbosity = 1
)


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
function hamiltonian(u_quad_values::Matrix{Float64}, v_quad_values::Matrix{Float64}, w_quad_values::Matrix{Float64}, grid_quad_node::Matrix{Vector{ST}}, 
    grid_quad_weights::Matrix{ST}, xspan = xspan, timestep = t_step) where ST
    ham = 0.0

    RT = size(u_quad_values, 1)
    RX = size(u_quad_values, 2)
    x_domain = xspan[2] - xspan[1]
        for rt in 1:RT
            for rx in 1:RX
                ham+= x_domain * timestep * grid_quad_weights[rt,rx] * 
                hamiltonian_density(grid_quad_node[rt,rx][1], grid_quad_node[rt,rx][2], u_quad_values[rt, rx], v_quad_values[rt, rx], w_quad_values[rt, rx])
            end
        end

    return ham
end



t_span = (0.,5.0)
xspan = (0.0, 1.0)
# for t_step in [0.1,0.2,0.4]

lpde = MultiSymplectic.Wave.lpdeproblem(timestep=t_step, timespan=t_span, xspan=xspan, xstep=0.01)
x_ls = xspan[1]:lpde.xstep:xspan[2]
c = lpde.params.c
# x_knot_interval = 0.05
# t_knot_interval = 0.5
# err_table = zeros(length(x_knot_interval_list),length(t_knot_interval_list))
# for (xki,x_knot_interval) in enumerate(x_knot_interval_list)
#     for (tki,t_knot_interval) in enumerate(t_knot_interval_list)

        spline_basis = Dirichlet_BSpline2D(k,timestep =t_step, xspan = xspan, t_knot_interval = t_knot_interval,x_knot_interval = x_knot_interval)
        spline_int = Galerkin_Full_Restriction_Bspline_Integrator(spline_basis,xspan=xspan,RT_per_interval = k,RX_per_interval = k,show_status = false)
        println("Start Spline Integrator")
        sol_set = MultiSymplectic.integrate(lpde, spline_int)
        println("End Spline Integrator with h=$(t_step), k=$(k), t_knot_interval=$(t_knot_interval), x_knot_interval=$(x_knot_interval)")

        ham_ls = zeros(length(t_span[1]:t_step:t_span[2]-t_step))
        analytic_ham = zeros(length(t_span[1]:t_step:t_span[2]-t_step))
        for (i, t) in enumerate(t_span[1]:t_step:t_span[2]-t_step)

            ham_ls[i] = hamiltonian(sol_set.u_quad_values[i][1,:,:], sol_set.v_quad_values[i][1,:,:], sol_set.w_quad_values[i][1,:,:], spline_int.grid_matrix, spline_int.grid_weights)

            analytic_u_values = [lpde.exact_u(t + t_step*spline_int.grid_matrix[i,j][1], spline_int.grid_matrix[i,j][2]; params=lpde.params) for i in 1:size(spline_int.grid_matrix, 1), j in 1:size(spline_int.grid_matrix, 2)]
            analytic_v_values = [lpde.exact_v(t + t_step*spline_int.grid_matrix[i,j][1], spline_int.grid_matrix[i,j][2]; params=lpde.params) for i in 1:size(spline_int.grid_matrix, 1), j in 1:size(spline_int.grid_matrix, 2)]
            analytic_w_values = [lpde.exact_w(t + t_step*spline_int.grid_matrix[i,j][1], spline_int.grid_matrix[i,j][2]; params=lpde.params) for i in 1:size(spline_int.grid_matrix, 1), j in 1:size(spline_int.grid_matrix, 2)]

            analytic_ham[i] = hamiltonian(analytic_u_values, analytic_v_values, analytic_w_values, spline_int.grid_matrix, spline_int.grid_weights)
        end


        relative_ham_err = (ham_ls .-  analytic_ham) ./ analytic_ham
        max_err = maximum(relative_ham_err)
        # err_table[xki,tki] = max_err
        plot(relative_ham_err, xlabel="Time", ylabel="Relative Hamiltonian Error")
        savefig("full_restriction_spline_1106/SplineInt_Hamiltonian_Error_T=5_h=$(t_step)_k=$(k)_t_knot_interval=$(t_knot_interval)_x_knot_interval=$(x_knot_interval)_regulizer=$(regularization_factor)_078.pdf")

        record_results = Dict()
        record_results["maximum_relative_ham_err"] = maximum(relative_ham_err)
        record_results["sol_u"] = sol_set.sol.u
        record_results["sol_v"] = sol_set.sol.v
        record_results["sol_w"] = sol_set.sol.w
        record_results["sol_hamiltonian"] = ham_ls
        record_results["analytic_hamiltonian"] = analytic_ham
        record_results["relative_hamiltonian_error"] = relative_ham_err

        save("full_restriction_spline_1106/SplineInt_Hamiltonian_Error_T=5_h=$(t_step)_k=$(k)_t_knot_interval=$(t_knot_interval)_x_knot_interval=$(x_knot_interval)_regulizer=$(regularization_factor)_078.jld2", record_results)
        println("results saved: h=$(t_step), k=$(k), t_knot_interval=$(t_knot_interval), x_knot_interval=$(x_knot_interval)")
#     end
# end