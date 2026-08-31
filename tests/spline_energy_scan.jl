using MultiSymplectic
using GeometricIntegratorsBase
using JLD2
using LinearAlgebra
using Printf

const CURRENT_REGULARIZATION = Ref(0.0)
const CURRENT_MAX_ITERATIONS = Ref(10)

function GeometricIntegratorsBase.default_options(::Galerkin_Bspline_Integrator)
    (
        regularization_factor = CURRENT_REGULARIZATION[],
        max_iterations = CURRENT_MAX_ITERATIONS[],
        verbosity = 1
    )
end

function hamiltonian_density(c, u, v, w)
    return 0.5 * (v^2 + c^2 * w^2)
end

function slab_hamiltonian(u_quad_values, v_quad_values, w_quad_values, grid_quad_node,
        grid_quad_weights, c, xspan, timestep)
    ham = 0.0
    x_domain = xspan[2] - xspan[1]
    for rt in axes(u_quad_values, 1), rx in axes(u_quad_values, 2)

        ham += x_domain * timestep * grid_quad_weights[rt, rx] *
               hamiltonian_density(c, u_quad_values[rt, rx], v_quad_values[rt, rx], w_quad_values[rt, rx])
    end
    return ham
end

function run_case(; regularization_factor, t_step = 0.1, t_end = 1.0, k = 3,
        t_knot_interval = 0.5, x_knot_interval = 0.5, xspan = (0.0, 1.0),
        max_iterations = 10, multiplier_stride = 1)
    CURRENT_REGULARIZATION[] = regularization_factor
    CURRENT_MAX_ITERATIONS[] = max_iterations

    lpde = MultiSymplectic.Wave.lpdeproblem(
        timestep = t_step,
        timespan = (0.0, t_end),
        xspan = xspan,
        xstep = 0.02
    )
    c = lpde.params.c
    basis = BSpline2D(k; xspan = xspan, t_knot_interval = t_knot_interval,
        x_knot_interval = x_knot_interval)
    method = Galerkin_Bspline_Integrator(
        basis;
        xspan = xspan,
        RT_per_interval = k,
        RX_per_interval = k,
        show_status = false,
        multiplier_stride = multiplier_stride
    )

    elapsed = @elapsed sol_set = Base.invokelatest(MultiSymplectic.integrate, lpde, method)
    nsteps = length(sol_set.u_quad_values)
    ham = zeros(nsteps)
    analytic_ham = zeros(nsteps)
    signed_relerr = zeros(nsteps)
    mult_norm = zeros(nsteps)

    S = basis.S
    for i in 1:nsteps
        t = (i - 1) * t_step
        ham[i] = slab_hamiltonian(
            sol_set.u_quad_values[i][1, :, :],
            sol_set.v_quad_values[i][1, :, :],
            sol_set.w_quad_values[i][1, :, :],
            method.grid_matrix,
            method.grid_weights,
            c,
            xspan,
            t_step
        )
        analytic_u = [lpde.exact_u(t + t_step * method.grid_matrix[rt, rx][1],
                          method.grid_matrix[rt, rx][2]; params = lpde.params)
                      for rt in axes(method.grid_matrix, 1),
        rx in axes(method.grid_matrix, 2)]
        analytic_v = [lpde.exact_v(t + t_step * method.grid_matrix[rt, rx][1],
                          method.grid_matrix[rt, rx][2]; params = lpde.params)
                      for rt in axes(method.grid_matrix, 1),
        rx in axes(method.grid_matrix, 2)]
        analytic_w = [lpde.exact_w(t + t_step * method.grid_matrix[rt, rx][1],
                          method.grid_matrix[rt, rx][2]; params = lpde.params)
                      for rt in axes(method.grid_matrix, 1),
        rx in axes(method.grid_matrix, 2)]
        analytic_ham[i] = slab_hamiltonian(
            analytic_u,
            analytic_v,
            analytic_w,
            method.grid_matrix,
            method.grid_weights,
            c,
            xspan,
            t_step
        )
        signed_relerr[i] = (ham[i] - analytic_ham[i]) / analytic_ham[i]
        mult_norm[i] = isempty(sol_set.internal_solutions[i][(S + 1):end]) ?
                       0.0 : norm(sol_set.internal_solutions[i][(S + 1):end], Inf)
    end

    oscillates = any(signed_relerr .> 0) && any(signed_relerr .< 0)
    return (
        regularization_factor = regularization_factor,
        elapsed = elapsed,
        nsteps = nsteps,
        max_abs_signed_relerr = maximum(abs.(signed_relerr)),
        mean_abs_signed_relerr = sum(abs.(signed_relerr)) / length(signed_relerr),
        final_signed_relerr = signed_relerr[end],
        oscillates = oscillates,
        max_multiplier_norm = maximum(mult_norm),
        multiplier_stride = multiplier_stride,
        ham = ham,
        analytic_ham = analytic_ham,
        signed_relerr = signed_relerr,
        multiplier_norm = mult_norm
    )
end

function main()
    mkpath("debug_results")
    regs = parse.(Float64, split(get(ENV, "SPLINE_REGS", "0,1e-8,1e-6,1e-4,1e-3"), ","))
    k = parse(Int, get(ENV, "SPLINE_K", "3"))
    t_step = parse(Float64, get(ENV, "SPLINE_T_STEP", "0.1"))
    t_end = parse(Float64, get(ENV, "SPLINE_T_END", "1.0"))
    t_knot_interval = parse(Float64, get(ENV, "SPLINE_T_KNOT_INTERVAL", "0.5"))
    x_knot_interval = parse(Float64, get(ENV, "SPLINE_X_KNOT_INTERVAL", "0.5"))
    max_iterations = parse(Int, get(ENV, "SPLINE_MAX_ITERATIONS", "10"))
    multiplier_stride = parse(Int, get(ENV, "SPLINE_MULTIPLIER_STRIDE", "1"))

    results = []
    for reg in regs
        result = run_case(;
            regularization_factor = reg,
            k = k,
            t_step = t_step,
            t_end = t_end,
            t_knot_interval = t_knot_interval,
            x_knot_interval = x_knot_interval,
            max_iterations = max_iterations,
            multiplier_stride = multiplier_stride
        )
        push!(results, result)
        @printf("reg=%8.1e  time=%6.2fs  steps=%3d  maxerr=%9.3e  meanerr=%9.3e  final=%+9.3e  osc=%s  max|mult|=%9.3e\n",
            result.regularization_factor,
            result.elapsed,
            result.nsteps,
            result.max_abs_signed_relerr,
            result.mean_abs_signed_relerr,
            result.final_signed_relerr,
            string(result.oscillates),
            result.max_multiplier_norm)
    end

    save("debug_results/spline_energy_scan.jld2", Dict("results" => results))
end

main()
