using QuadratureRules
using MultiSymplectic
using Symbolics
using Parameters
using Base
using Infiltrator
using GeometricIntegratorsBase
using JLD2
using Test


reg_factor = 1e-5
GeometricIntegratorsBase.default_options(::Sindy_PDE_Integrator) = (
    regularization_factor = reg_factor,
)


# Sine-Gordon Equation
begin
    @variables t x p[1:3]
    u_expr = p[1] * atan(exp(p[2] * (x - p[3] * t)))
    sindy_basis = SindyPDEBasis([u_expr], [p], t, [x])


    c2 = 3.9999951547393993
    v2 = 1.0397741828102778
    γ2 = 1.1546989298112151
    init_p = [c2,γ2,v2]

    t_step = 0.05
    time_span = 5.0
    x_span = (-1.0,1.0)
    sindy_int = Sindy_PDE_Integrator(sindy_basis,init_p, xspan = x_span,
        Nbasis_μ_t = 4,μ =:Lagrange,k_μ_t = 3,
        Nbasis_λ_x = 4,λ =:Lagrange,k_λ_x = 3,
        RT_per_interval= 4,RX_per_interval = 2,
        show_status = false,)
    lpde = MultiSymplectic.SineGordon.lpdeproblem(timestep = t_step,timespan =(0.0,time_span),xspan = x_span)
    # log_file="logs/sindy_pde.txt"
    # open(log_file, "w") do io
    #     redirect_stdout(io) do
    sol,internal_values =  MultiSymplectic.integrate(lpde,sindy_int)
    #     end
    # end
    c = 4.0
    sine_gordon_ham(u,v,w) = 1 / 2 * (c * v^2 + w^2) - (1 + cos(u))
    x_ls = collect(x_span[1]:0.01:x_span[2])

    ham_ls = zeros(length(0:t_step:time_span))
    analytic_ham = zeros(length(0:t_step:time_span))
    for (i, t) in enumerate(0:t_step:time_span)
        current_domain_ham = [sine_gordon_ham(ui,vi,wi) for (ui,vi,wi) in zip(sol.s.u[i-1],sol.s.v[i-1],sol.s.w[i-1])]
        ham_ls[i] = sum(current_domain_ham)

        analytic_u_values = lpde.exact_u.(t, x_ls)
        analytic_v_values = lpde.exact_v.(t, x_ls)
        analytic_w_values = lpde.exact_w.(t, x_ls)
        current_ham = [sine_gordon_ham(ui,vi,wi) for (ui,vi,wi) in zip(analytic_u_values,analytic_v_values,analytic_w_values)]
        analytic_ham[i] = sum(current_ham)
    end

    ham_err = abs.((ham_ls .- analytic_ham)./analytic_ham)
    @test max_ham_err = maximum(ham_err) < 1.0

    record_results = Dict()
    record_results["sol_u"] = sol.s.u
    record_results["sol_v"] = sol.s.v
    record_results["sol_w"] = sol.s.w
    record_results["ham_ls"] = ham_ls
    record_results["analytic_ham"] = analytic_ham
    record_results["ham_err"] = ham_err
    record_results["internal_values"] = internal_values

    save("sindy_results_sine_gordon_h$(t_step)_reg$(reg_factor)_x$(x_span[1])_to_$(x_span[2]).jld2",record_results)

end
