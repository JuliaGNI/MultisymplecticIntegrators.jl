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
t_step = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 0.05

GeometricIntegratorsBase.default_options(::Sindy_PDE_Integrator) = (
    regularization_factor = reg_factor,
    verbosity = 2,
    f_suctol = -1.0,
    x_suctol = -1.0, 
    max_iterations = 10,
)

function hamiltonian_density(t, x, u, v, w, params)
    c = params.c
    1 / 2 * (c * v^2 + w^2) - (1 + cos(u))
end

function hamiltonian(u_quad_values::Matrix{Float64}, v_quad_values::Matrix{Float64}, w_quad_values::Matrix{Float64}, grid_quad_node::Matrix{Vector{ST}},
    grid_quad_weights::Matrix{ST}, xspan, timestep, params) where {ST}
    ham = 0.0

    RT = size(u_quad_values, 1)
    RX = size(u_quad_values, 2)
    x_domain = xspan[2] - xspan[1]
    for rt in 1:RT
        for rx in 1:RX
            ham += x_domain * timestep * grid_quad_weights[rt, rx] *
                hamiltonian_density(grid_quad_node[rt, rx][1], grid_quad_node[rt, rx][2],
                    u_quad_values[rt, rx], v_quad_values[rt, rx], w_quad_values[rt, rx], params)
        end
    end

    return ham
end

# Sine-Gordon Equation
# begin
    @variables t x p[1:3]
    u_expr = p[1] * atan(exp(p[2] * (x - p[3] * t)))
    sindy_basis = SindyPDEBasis([u_expr], [p], t, [x])

    t_span = (0.0, 5.0)
    x_span = (0,1.0)
    lpde = MultiSymplectic.SineGordon.lpdeproblem(timestep = t_step,timespan = t_span,xspan = x_span)

    init_p = [4.0, lpde.params.γ, lpde.params.velocity]

    c2 = 3.9999951547393993
    v2 = 0.2497741828102778
    γ2 = 1.1546989298112151
    init_p = [c2,γ2,v2]

    sindy_int = Sindy_PDE_Integrator(sindy_basis,init_p, xspan = x_span,
        t_num_interval = t_num_interval,μ = μ_basis,k_μ_t = k_μ_t,
        x_num_interval = x_num_interval,λ = λ_basis,k_λ_x = k_λ_x,
        RT_per_interval= RT_per_interval,RX_per_interval = RX_per_interval,
        show_status = false,)
    # log_file="logs/sindy_pde.txt"
    # open(log_file, "w") do io
    #     redirect_stdout(io) do
    sol_set =  MultiSymplectic.integrate(lpde,sindy_int)
    #     end
    # end

    ham_ls = zeros(length(t_span[1]:t_step:t_span[2]-t_step))
    analytic_ham = zeros(length(t_span[1]:t_step:t_span[2]-t_step))
    for (i, t) in enumerate(t_span[1]:t_step:t_span[2]-t_step)
        analytic_u_values = [lpde.exact_u(t + t_step * sindy_int.grid_matrix[rt, rx][1], sindy_int.grid_matrix[rt, rx][2]; params=lpde.params) for rt in 1:size(sindy_int.grid_matrix, 1), rx in 1:size(sindy_int.grid_matrix, 2)]
        analytic_v_values = [lpde.exact_v(t + t_step * sindy_int.grid_matrix[rt, rx][1], sindy_int.grid_matrix[rt, rx][2]; params=lpde.params) for rt in 1:size(sindy_int.grid_matrix, 1), rx in 1:size(sindy_int.grid_matrix, 2)]
        analytic_w_values = [lpde.exact_w(t + t_step * sindy_int.grid_matrix[rt, rx][1], sindy_int.grid_matrix[rt, rx][2]; params=lpde.params) for rt in 1:size(sindy_int.grid_matrix, 1), rx in 1:size(sindy_int.grid_matrix, 2)]

        analytic_ham[i] = hamiltonian(analytic_u_values, analytic_v_values, analytic_w_values, sindy_int.grid_matrix, sindy_int.grid_weights, x_span, t_step, lpde.params)
        ham_ls[i] = hamiltonian(sol_set.u_quad_values[i][1, :, :], sol_set.v_quad_values[i][1, :, :], sol_set.w_quad_values[i][1, :, :], sindy_int.grid_matrix, sindy_int.grid_weights, x_span, t_step, lpde.params)
    end

    relative_ham_err = abs.((ham_ls .- analytic_ham) ./ analytic_ham)
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
        "sine_gordon_params" => lpde.params,
        "init_p" => init_p,
    )

    output_dir = get(ENV, "SINDY_OUTPUT_DIR", "sindyint_results")
    mkpath(output_dir)
    cd(output_dir)
    err_tag = @sprintf("%.3e", max_err)
    velocity_tag = @sprintf("%.3g", lpde.params.velocity)
    gamma_tag = @sprintf("%.3g", lpde.params.γ)
    output_file = "SINDy_T$(t_span[2])_h$(t_step)_v$(velocity_tag)_gamma$(gamma_tag)_reg$(reg_factor)_err$(err_tag)_mu$(μ_basis)_lambda$(λ_basis)_tint$(t_num_interval)_xint$(x_num_interval)_rt$(RT_per_interval)_rx$(RX_per_interval)_kmu$(k_μ_t)_klambda$(k_λ_x).jld2"
    save(output_file, record_results)
    println("saved $(output_file)")

    # if get(ENV, "SINDY_SAVE_GIF", "0") == "1"
    #     x_nodes = collect(x_span[1]:lpde.xstep:x_span[2])
    #     y_min = minimum(vcat([sol_set.sol.s.u[i] for i in 0:length(sol_set.sol.s.u)-1]...))
    #     y_max = maximum(vcat([sol_set.sol.s.u[i] for i in 0:length(sol_set.sol.s.u)-1]...))
    #     exact_u_all = [lpde.exact_u.(current_t, x_nodes; params=lpde.params) for current_t in t_span[1]:t_step:t_span[2]]
    #     y_min = min(y_min, minimum(vcat(exact_u_all...)))
    #     y_max = max(y_max, maximum(vcat(exact_u_all...)))
    #     y_pad = 0.05 * max(y_max - y_min, eps())

    #     anim = @animate for (frame_idx, current_t) in enumerate(t_span[1]:t_step:t_span[2])
    #         sol_idx = frame_idx - 1
    #         exact_u_values = exact_u_all[frame_idx]

    #         plot(
    #             x_nodes,
    #             sol_set.sol.s.u[sol_idx],
    #             label = "SINDy",
    #             linewidth = 2,
    #             xlabel = "x",
    #             ylabel = "u(t, x)",
    #             title = "Sine-Gordon t = $(round(current_t, digits = 3))",
    #             ylim = (y_min - y_pad, y_max + y_pad),
    #         )
    #         plot!(x_nodes, exact_u_values, label = "Exact", linewidth = 2, linestyle = :dash)
    #     end

    #     gif(anim, replace(output_file, ".jld2" => ".gif"), fps = 20)
    # end

    # if get(ENV, "SINDY_ASSERT", "0") == "1"
    #     @test max_err < 1.0
    # end

# end
