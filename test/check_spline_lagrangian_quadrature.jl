using MultiSymplectic
using LinearAlgebra
using Printf
using QuadratureRules
using Random

function parse_tuple(name, default)
    raw = get(ENV, name, "")
    isempty(raw) && return default
    vals = parse.(Float64, split(raw, ","))
    length(vals) == 2 || error("$(name) must have format a,b")
    return (vals[1], vals[2])
end

function composite_gauss_legendre(num_intervals::Int, order::Int)
    rule = QuadratureRules.GaussLegendreQuadrature(order)
    nodes = Float64[]
    weights = Float64[]
    for i in 1:num_intervals
        left = (i - 1) / num_intervals
        right = i / num_intervals
        width = right - left
        append!(nodes, left .+ width .* rule.nodes)
        append!(weights, width .* rule.weights)
    end
    return (nodes = nodes, weights = weights)
end

function gauss_legendre_on_breaks(breaks::AbstractVector, order::Int)
    unique_breaks = sort(unique(Float64.(breaks)))
    rule = QuadratureRules.GaussLegendreQuadrature(order)
    nodes = Float64[]
    weights = Float64[]
    for i in 1:(length(unique_breaks) - 1)
        left = unique_breaks[i]
        right = unique_breaks[i + 1]
        width = right - left
        width > 0 || continue
        append!(nodes, left .+ width .* rule.nodes)
        append!(weights, width .* rule.weights)
    end
    return (nodes = nodes, weights = weights, breaks = unique_breaks)
end

function lagrangian_density(c, v, w)
    return 0.5 * (v^2 - c^2 * w^2)
end

function method_quadrature(lpde, method)
    xspan = lpde.xspan
    x_domain = xspan[2] - xspan[1]
    return (
        t_nodes = method.time_quadrature.nodes,
        t_weights = method.time_quadrature.weights,
        x_nodes = xspan[1] .+ x_domain .* method.spatial_quadrature.nodes,
        x_weights = x_domain .* method.spatial_quadrature.weights
    )
end

function knot_quadrature(basis, order::Int)
    tq = gauss_legendre_on_breaks(basis.ts, order)
    xq = gauss_legendre_on_breaks(basis.xs, order)
    return (
        t_nodes = tq.nodes,
        t_weights = tq.weights,
        x_nodes = xq.nodes,
        x_weights = xq.weights
    )
end

function integrate_exact_wave_lagrangian(lpde, quad, t_step, tn)
    c = lpde.params.c

    action = 0.0
    for i in eachindex(quad.t_nodes), j in eachindex(quad.x_nodes)

        t_phys = tn + t_step * quad.t_nodes[i]
        x_phys = quad.x_nodes[j]
        v = lpde.exact_v(t_phys, x_phys; params = lpde.params)
        w = lpde.exact_w(t_phys, x_phys; params = lpde.params)
        action += t_step * quad.t_weights[i] * quad.x_weights[j] *
                  lagrangian_density(c, v, w)
    end
    return action
end

function integrate_spline_lagrangian(coeffs, basis, lpde, quad, t_step)
    c = lpde.params.c
    coefs = reshape(coeffs, basis.Nbasis_t, basis.Nbasis_x)

    action = 0.0
    for i in eachindex(quad.t_nodes), j in eachindex(quad.x_nodes)

        t_ref = quad.t_nodes[i]
        x_phys = quad.x_nodes[j]
        v = MultiSymplectic.eval_spline2D_dt(coefs, (basis.Basis_t, basis.Basis_x),
            (t_ref, x_phys)) / t_step
        w = MultiSymplectic.eval_spline2D_dx(coefs, (basis.Basis_t, basis.Basis_x),
            (t_ref, x_phys))
        action += t_step * quad.t_weights[i] * quad.x_weights[j] *
                  lagrangian_density(c, v, w)
    end
    return action
end

function print_compare(label, current_value, knot_value, reference_value)
    current_abs = abs(current_value - reference_value)
    current_rel = current_abs / max(abs(reference_value), eps(Float64))
    knot_abs = abs(knot_value - reference_value)
    knot_rel = knot_abs / max(abs(reference_value), eps(Float64))
    @printf("%-28s current=% .16e  knot=% .16e  reference=% .16e\n",
        label, current_value, knot_value, reference_value)
    @printf("%-28s current_abs=%9.3e  current_rel=%9.3e  knot_abs=%9.3e  knot_rel=%9.3e\n",
        "", current_abs, current_rel, knot_abs, knot_rel)
end

function main()
    k = parse(Int, get(ENV, "SPLINE_K", "3"))
    t_step = parse(Float64, get(ENV, "SPLINE_T_STEP", "0.1"))
    tn = parse(Float64, get(ENV, "SPLINE_TN", "0.0"))
    t_knot_interval = parse(Float64, get(ENV, "SPLINE_T_KNOT_INTERVAL", "0.5"))
    x_knot_interval = parse(Float64, get(ENV, "SPLINE_X_KNOT_INTERVAL", "0.1"))
    ref_order = parse(Int, get(ENV, "SPLINE_REF_ORDER", "32"))
    xspan = parse_tuple("SPLINE_XSPAN", (0.0, 1.0))

    lpde = MultiSymplectic.Wave.lpdeproblem(
        timestep = t_step,
        timespan = (tn, tn + t_step),
        xspan = xspan,
        xstep = 0.01
    )
    basis = BSpline2D(k; xspan = xspan, t_knot_interval = t_knot_interval,
        x_knot_interval = x_knot_interval)
    method = Galerkin_Bspline_Integrator(
        basis;
        xspan = xspan,
        RT_per_interval = k,
        RX_per_interval = k,
        show_status = false
    )

    @printf("k=%d  t_step=%g  tn=%g  xspan=(%g,%g)  t_knot_interval=%g  x_knot_interval=%g\n",
        k, t_step, tn, xspan[1], xspan[2], t_knot_interval, x_knot_interval)
    @printf("RT=%d  RX=%d  S=%d  reference_order=%d\n",
        method.RT, method.RX, basis.S, ref_order)
    @printf("stored basis.ts length=%d  unique=%d  stored basis.xs length=%d  unique=%d\n",
        length(basis.ts), length(unique(basis.ts)), length(basis.xs),
        length(unique(basis.xs)))

    current_quad = method_quadrature(lpde, method)
    knot_quad = knot_quadrature(basis, k)
    reference_quad = knot_quadrature(basis, ref_order)

    exact_current = integrate_exact_wave_lagrangian(lpde, current_quad, t_step, tn)
    exact_knot = integrate_exact_wave_lagrangian(lpde, knot_quad, t_step, tn)
    exact_ref = integrate_exact_wave_lagrangian(lpde, reference_quad, t_step, tn)
    print_compare("exact wave Lagrangian", exact_current, exact_knot, exact_ref)

    rng = MersenneTwister(1234)
    coeffs = randn(rng, basis.S)
    spline_current = integrate_spline_lagrangian(coeffs, basis, lpde, current_quad, t_step)
    spline_knot = integrate_spline_lagrangian(coeffs, basis, lpde, knot_quad, t_step)
    spline_ref = integrate_spline_lagrangian(coeffs, basis, lpde, reference_quad, t_step)
    print_compare("random spline Lagrangian", spline_current, spline_knot, spline_ref)

    println()
    println("Interpretation:")
    println("- current = quadrature currently stored by Galerkin_Bspline_Integrator.")
    println("- knot = k-point Gauss quadrature on the unique B-spline breakpoints.")
    println("- reference = high-order Gauss quadrature on the unique B-spline breakpoints.")
    println("- random spline Lagrangian should have knot_abs near machine precision if the quadrature is exact for the discrete spline action.")
end

main()
