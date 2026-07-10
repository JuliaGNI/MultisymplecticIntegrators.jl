using QuadratureRules
using MultiSymplectic
using Symbolics
using Parameters
using Base
using Infiltrator
using GeometricIntegratorsBase
using JLD2
using Plots
using Printf
using Test


t_num_interval = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
x_num_interval = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
RT_per_interval = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
RX_per_interval = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 4
k_μ_t = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3
k_λ_x = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 3
reg_factor = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 1e-11
μ_basis = length(ARGS) >= 8 ? Symbol(ARGS[8]) : :BSplineDirichlet
λ_basis = length(ARGS) >= 9 ? Symbol(ARGS[9]) : :BSplineDirichlet
t_step = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 0.02

GeometricIntegratorsBase.default_options(::Sindy_PDE_Integrator) = (
    regularization_factor = reg_factor,
    verbosity = 2,
    # f_suctol = -1.0,
    # f_abstol = 1e-12,
    # x_suctol = -1.0, 
    max_iterations = 100,
)
function hamiltonian(u_quad_values::Matrix{Float64}, v_quad_values::Matrix{Float64}, w_quad_values::Matrix{Float64}, grid_quad_node::Matrix{Vector{ST}},
    grid_quad_weights::Matrix{ST}, xspan, timestep) where {ST}
    ham = 0.0

    RT = size(u_quad_values, 1)
    RX = size(u_quad_values, 2)
    x_domain = xspan[2] - xspan[1]
    for rt in 1:RT
        for rx in 1:RX
            ham += x_domain * timestep * grid_quad_weights[rt, rx] *
                hamiltonian_density(grid_quad_node[rt, rx][1], grid_quad_node[rt, rx][2],
                    u_quad_values[rt, rx], v_quad_values[rt, rx], w_quad_values[rt, rx])
        end
    end

    return ham
end

# Sine-Gordon Equation
# begin
#     function hamiltonian_density(t, x, u, v, w)
#         c = 4.0
#         1 / 2 * (c * v^2 + w^2) - (1 + cos(u))
#     end



#     @variables t x p[1:3]
#     u_expr = p[1] * atan(exp(p[2] * (x - p[3] * t)))
#     sindy_basis = SindyPDEBasis([u_expr], [p], t, [x])

#     t_span = (0.0, 5.0)
#     x_span = (0,1.0)
#     lpde = MultiSymplectic.SineGordon.lpdeproblem(timestep = t_step,timespan = t_span,xspan = x_span)

#     # init_p = [4.0, lpde.params.γ, lpde.params.velocity]

#     c2 = 3.9999951547393993
#     γ2 = 1.1546989298112151
#     v2 = 0.2397741828102778
#     init_p = [c2,γ2,v2]

#     # t_step = 0.05

#     sindy_int = Sindy_PDE_Integrator(sindy_basis,init_p, xspan = x_span,
#         t_num_interval = t_num_interval,μ = μ_basis,k_μ_t = k_μ_t,
#         x_num_interval = x_num_interval,λ = λ_basis,k_λ_x = k_λ_x,
#         RT_per_interval= RT_per_interval,RX_per_interval = RX_per_interval,
#         show_status = false,)
#     sol_set =  MultiSymplectic.integrate(lpde,sindy_int)

#     ham_ls = zeros(length(t_span[1]:t_step:t_span[2]-t_step))
#     analytic_ham = zeros(length(t_span[1]:t_step:t_span[2]-t_step))
#     for (i, t) in enumerate(t_span[1]:t_step:t_span[2]-t_step)
#         analytic_u_values = [lpde.exact_u(t + t_step * sindy_int.grid_matrix[rt, rx][1], sindy_int.grid_matrix[rt, rx][2]; params=lpde.params) for rt in 1:size(sindy_int.grid_matrix, 1), rx in 1:size(sindy_int.grid_matrix, 2)]
#         analytic_v_values = [lpde.exact_v(t + t_step * sindy_int.grid_matrix[rt, rx][1], sindy_int.grid_matrix[rt, rx][2]; params=lpde.params) for rt in 1:size(sindy_int.grid_matrix, 1), rx in 1:size(sindy_int.grid_matrix, 2)]
#         analytic_w_values = [lpde.exact_w(t + t_step * sindy_int.grid_matrix[rt, rx][1], sindy_int.grid_matrix[rt, rx][2]; params=lpde.params) for rt in 1:size(sindy_int.grid_matrix, 1), rx in 1:size(sindy_int.grid_matrix, 2)]

#         analytic_ham[i] = hamiltonian(analytic_u_values, analytic_v_values, analytic_w_values, sindy_int.grid_matrix, sindy_int.grid_weights, x_span, t_step)
#         ham_ls[i] = hamiltonian(sol_set.u_quad_values[i][1, :, :], sol_set.v_quad_values[i][1, :, :], sol_set.w_quad_values[i][1, :, :], sindy_int.grid_matrix, sindy_int.grid_weights, x_span, t_step)
#     end

#     relative_ham_err = ((ham_ls .- analytic_ham) ./ analytic_ham)
#     max_err = maximum(relative_ham_err)

#     record_results = Dict()
#     record_results["sol_u"] = sol_set.sol.s.u
#     record_results["sol_v"] = sol_set.sol.s.v
#     record_results["sol_w"] = sol_set.sol.s.w
#     record_results["sol_u_quad"] = sol_set.u_quad_values
#     record_results["sol_v_quad"] = sol_set.v_quad_values
#     record_results["sol_w_quad"] = sol_set.w_quad_values
#     record_results["ham_ls"] = ham_ls
#     record_results["analytic_ham"] = analytic_ham
#     record_results["relative_ham_err"] = relative_ham_err
#     record_results["max_err"] = max_err
#     record_results["internal_solutions"] = sol_set.internal_solutions
#     record_results["config"] = Dict(
#         "t_num_interval" => t_num_interval,
#         "x_num_interval" => x_num_interval,
#         "RT_per_interval" => RT_per_interval,
#         "RX_per_interval" => RX_per_interval,
#         "k_μ_t" => k_μ_t,
#         "k_λ_x" => k_λ_x,
#         "reg_factor" => reg_factor,
#         "μ" => μ_basis,
#         "λ" => λ_basis,
#     )

#     err_tag = @sprintf("%.3e", max_err)
#     output_file = "sindyint_results/SINDy_T$(t_span[2])_h$(t_step)_reg$(reg_factor)_err$(err_tag)_mu$(μ_basis)_lambda$(λ_basis)_tint$(t_num_interval)_xint$(x_num_interval)_rt$(RT_per_interval)_rx$(RX_per_interval)_kmu$(k_μ_t)_klambda$(k_λ_x).jld2"
#     save(output_file, record_results)
#     println("saved $(output_file)")
# end


# Wave Equation
begin
    c=0.5
    function hamiltonian_density(t, x, u, v, w)
        1 / 2 * (v[1]^2 + c^2 * w[1]^2) 
    end

    @variables t x p[1:6] 
    u_expr = (p[1] * cos((pi*p[6]*t)/p[5]) + p[2] * sin((pi*p[6]*t)/p[5] + pi/6)) * sin((pi*x)/p[5]) + (p[3] * cos((2*pi*p[6]*t)/p[5]) + p[4] * sin((2*pi*p[6]*t)/p[5] + pi/6)) * sin((2*pi*x)/p[5])
            #(A1 * cos((pi*c*t)/l) + B1 * sin((pi*c*t)/l + pi/6)) * sin((pi*x)/l) + (A2 * cos((2*pi*c*t)/l) + B2 * sin((2*pi*c*t)/l + pi/6)) * sin((2*pi*x)/l)

    sindy_basis = SindyPDEBasis([u_expr], [p], t, [x])

    t_span = (0.0, 10.0)
    x_span = (0,1.0)
    lpde = MultiSymplectic.Wave.lpdeproblem(timestep = t_step,timespan = t_span,xspan = x_span,params = (c=0.5, A1=2.0, A2=0.5, B1=0.7, B2=0.35, l=1.0,))

    init_p = [lpde.params.A1, lpde.params.A2, lpde.params.B1, lpde.params.B2, lpde.params.l, lpde.params.c]

    sindy_int = Sindy_PDE_Integrator(sindy_basis,init_p, xspan = x_span,
        t_num_interval = t_num_interval,μ = μ_basis,k_μ_t = k_μ_t,
        x_num_interval = x_num_interval,λ = λ_basis,k_λ_x = k_λ_x,
        RT_per_interval= RT_per_interval,RX_per_interval = RX_per_interval,
        show_status = false,)
    sol_set =  MultiSymplectic.integrate(lpde,sindy_int)

    ham_ls = zeros(length(t_span[1]:t_step:t_span[2]-t_step))
    analytic_ham = zeros(length(t_span[1]:t_step:t_span[2]-t_step))
    for (i, t) in enumerate(t_span[1]:t_step:t_span[2]-t_step)
        analytic_u_values = [lpde.exact_u(t + t_step * sindy_int.grid_matrix[rt, rx][1], sindy_int.grid_matrix[rt, rx][2]; params=lpde.params) for rt in 1:size(sindy_int.grid_matrix, 1), rx in 1:size(sindy_int.grid_matrix, 2)]
        analytic_v_values = [lpde.exact_v(t + t_step * sindy_int.grid_matrix[rt, rx][1], sindy_int.grid_matrix[rt, rx][2]; params=lpde.params) for rt in 1:size(sindy_int.grid_matrix, 1), rx in 1:size(sindy_int.grid_matrix, 2)]
        analytic_w_values = [lpde.exact_w(t + t_step * sindy_int.grid_matrix[rt, rx][1], sindy_int.grid_matrix[rt, rx][2]; params=lpde.params) for rt in 1:size(sindy_int.grid_matrix, 1), rx in 1:size(sindy_int.grid_matrix, 2)]

        analytic_ham[i] = hamiltonian(analytic_u_values, analytic_v_values, analytic_w_values, sindy_int.grid_matrix, sindy_int.grid_weights, x_span, t_step)
        ham_ls[i] = hamiltonian(sol_set.u_quad_values[i][1, :, :], sol_set.v_quad_values[i][1, :, :], sol_set.w_quad_values[i][1, :, :], sindy_int.grid_matrix, sindy_int.grid_weights, x_span, t_step)
    end

    relative_ham_err = ((ham_ls .- analytic_ham) ./ analytic_ham)
    max_err = maximum(relative_ham_err)

    record_results = Dict()
    record_results["sol_u"] = sol_set.sol.s.u
    record_results["sol_v"] = sol_set.sol.s.v
    record_results["sol_w"] = sol_set.sol.s.w
    record_results["sol_u_quad"] = sol_set.u_quad_values
    record_results["sol_v_quad"] = sol_set.v_quad_values
    record_results["sol_w_quad"] = sol_set.w_quad_values
    record_results["ham_ls"] = ham_ls
    record_results["analytic_ham"] = analytic_ham
    record_results["relative_ham_err"] = relative_ham_err
    record_results["max_err"] = max_err
    record_results["internal_solutions"] = sol_set.internal_solutions
    record_results["config"] = Dict(
        "t_num_interval" => t_num_interval,
        "x_num_interval" => x_num_interval,
        "RT_per_interval" => RT_per_interval,
        "RX_per_interval" => RX_per_interval,
        "k_μ_t" => k_μ_t,
        "k_λ_x" => k_λ_x,
        "reg_factor" => reg_factor,
        "μ" => μ_basis,
        "λ" => λ_basis,
    )

    err_tag = @sprintf("%.3e", max_err)
    output_file = "sindyint_results/Wave_SINDy_T$(t_span[2])_h$(t_step)_reg$(reg_factor)_err$(err_tag)_mu$(μ_basis)_lambda$(λ_basis)_tint$(t_num_interval)_xint$(x_num_interval)_rt$(RT_per_interval)_rx$(RX_per_interval)_kmu$(k_μ_t)_klambda$(k_λ_x).jld2"
    save(output_file, record_results)
    println("saved $(output_file)")
end