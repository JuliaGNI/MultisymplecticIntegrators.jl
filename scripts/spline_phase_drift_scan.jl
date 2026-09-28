using MultisymplecticIntegrators
using GeometricIntegratorsBase
using LinearAlgebra
using Printf

const CURRENT_REGULARIZATION = Ref(0.0)
const CURRENT_MAX_ITERATIONS = Ref(100)
const CURRENT_VERBOSITY = Ref(0)

function GeometricIntegratorsBase.default_options(::Galerkin_Bspline_Integrator)
    (
        regularization_factor = CURRENT_REGULARIZATION[],
        max_iterations = CURRENT_MAX_ITERATIONS[],
        verbosity = CURRENT_VERBOSITY[]
    )
end

function GeometricIntegratorsBase.default_options(::Galerkin_Full_Restriction_Bspline_Integrator)
    (
        regularization_factor = CURRENT_REGULARIZATION[],
        max_iterations = CURRENT_MAX_ITERATIONS[],
        verbosity = CURRENT_VERBOSITY[]
    )
end

function _parse_float_list(name, default)
    raw = get(ENV, name, "")
    isempty(raw) && return default
    return parse.(Float64, split(raw, ","))
end

function _parse_int_list(name, default)
    raw = get(ENV, name, "")
    isempty(raw) && return default
    return parse.(Int, split(raw, ","))
end

function _hamiltonian_error_series(lpde, method, sol_set)
    nsteps = length(sol_set.u_quad_values)
    errs = zeros(nsteps)
    exact_u = zeros(1, method.RT, method.RX)
    exact_v = zeros(1, method.RT, method.RX)
    exact_w = zeros(1, method.RT, method.RX)

    for (step, t) in enumerate(lpde.timespan[1]:lpde.timestep:(lpde.timespan[2] - lpde.timestep))
        numerical_hamiltonian = Wave.hamiltonian(
            sol_set.u_quad_values[step],
            sol_set.v_quad_values[step],
            sol_set.w_quad_values[step],
            method.grid_matrix,
            method.grid_weights,
            lpde.params,
            lpde.xspan,
            lpde.timestep
        )

        for rt in axes(method.grid_matrix, 1), rx in axes(method.grid_matrix, 2)

            tq, xq = method.grid_matrix[rt, rx]
            exact_u[1, rt, rx] = lpde.exact_u(t + lpde.timestep * tq, xq; params = lpde.params)
            exact_v[1, rt, rx] = lpde.exact_v(t + lpde.timestep * tq, xq; params = lpde.params)
            exact_w[1, rt, rx] = lpde.exact_w(t + lpde.timestep * tq, xq; params = lpde.params)
        end

        exact_hamiltonian = Wave.hamiltonian(
            exact_u,
            exact_v,
            exact_w,
            method.grid_matrix,
            method.grid_weights,
            lpde.params,
            lpde.xspan,
            lpde.timestep
        )
        errs[step] = (numerical_hamiltonian - exact_hamiltonian) / exact_hamiltonian
    end
    return errs
end

function _envelope_drift(errs)
    abs_errs = abs.(errs)
    n = length(abs_errs)
    n < 4 && return 0.0
    q = max(1, n ÷ 4)
    head = maximum(@view abs_errs[1:q])
    tail = maximum(@view abs_errs[(end - q + 1):end])
    return tail - head
end

function run_case(; family, k, t_step, t_end, xstep, t_knot_interval,
        x_knot_interval, regularization_factor, max_iterations, multiplier_stride)
    CURRENT_REGULARIZATION[] = regularization_factor
    CURRENT_MAX_ITERATIONS[] = max_iterations

    lpde = Wave.lpdeproblem(
        timestep = t_step,
        timespan = (0.0, t_end),
        xspan = (0.0, 1.0),
        xstep = xstep
    )

    if family == :full_restriction
        basis = Dirichlet_BSpline2D(
            k;
            timestep = t_step,
            xspan = lpde.xspan,
            t_knot_interval = t_knot_interval,
            x_knot_interval = x_knot_interval
        )
        method = Galerkin_Full_Restriction_Bspline_Integrator(
            basis;
            xspan = lpde.xspan,
            RT_per_interval = k,
            RX_per_interval = k,
            show_status = false
        )
    elseif family == :multiplier
        basis = BSpline2D(
            k;
            xspan = lpde.xspan,
            t_knot_interval = t_knot_interval,
            x_knot_interval = x_knot_interval
        )
        method = Galerkin_Bspline_Integrator(
            basis;
            xspan = lpde.xspan,
            RT_per_interval = k,
            RX_per_interval = k,
            show_status = false,
            multiplier_stride = multiplier_stride
        )
    else
        error("Unknown family: $family")
    end

    elapsed = @elapsed sol_set = Base.invokelatest(MultisymplecticIntegrators.integrate, lpde, method)
    errs = _hamiltonian_error_series(lpde, method, sol_set)
    return (
        family = family,
        k = k,
        t_step = t_step,
        t_end = t_end,
        xstep = xstep,
        t_knot_interval = t_knot_interval,
        x_knot_interval = x_knot_interval,
        elapsed = elapsed,
        max_abs = maximum(abs.(errs)),
        drift = _envelope_drift(errs),
        final = errs[end],
        sign_changes = count(!=(0), diff(signbit.(errs)))
    )
end

function main()
    families = Symbol.(split(get(ENV, "SPLINE_FAMILIES", "full_restriction,multiplier"), ","))
    ks = _parse_int_list("SPLINE_KS", [3, 4])
    t_step = parse(Float64, get(ENV, "SPLINE_T_STEP", "0.05"))
    t_end = parse(Float64, get(ENV, "SPLINE_T_END", "20.0"))
    xstep = parse(Float64, get(ENV, "SPLINE_XSTEP", "0.01"))
    t_knot_intervals = _parse_float_list("SPLINE_T_KNOT_INTERVALS", [0.5, 0.25, 0.1])
    x_knot_intervals = _parse_float_list("SPLINE_X_KNOT_INTERVALS", [0.1, 0.05, 0.025])
    regularization_factor = parse(Float64, get(ENV, "SPLINE_REGULARIZATION", "0.0"))
    max_iterations = parse(Int, get(ENV, "SPLINE_MAX_ITERATIONS", "100"))
    multiplier_stride = parse(Int, get(ENV, "SPLINE_MULTIPLIER_STRIDE", "1"))

    @printf("%-17s %2s %8s %8s %10s %10s %10s %8s %5s\n",
        "family", "k", "t_knot", "x_knot", "max_abs", "drift", "final", "time", "sign")
    for family in families, k in ks, tk in t_knot_intervals, xk in x_knot_intervals
        result = run_case(
            family = family,
            k = k,
            t_step = t_step,
            t_end = t_end,
            xstep = xstep,
            t_knot_interval = tk,
            x_knot_interval = xk,
            regularization_factor = regularization_factor,
            max_iterations = max_iterations,
            multiplier_stride = multiplier_stride
        )
        @printf("%-17s %2d %8.3g %8.3g %10.3e %10.3e %+10.3e %8.2f %5d\n",
            string(result.family),
            result.k,
            result.t_knot_interval,
            result.x_knot_interval,
            result.max_abs,
            result.drift,
            result.final,
            result.elapsed,
            result.sign_changes,)
    end
end

main()
