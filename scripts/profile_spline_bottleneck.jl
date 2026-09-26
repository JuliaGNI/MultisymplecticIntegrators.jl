using MultiSymplectic
using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using ForwardDiff
using Printf

const CURRENT_REGULARIZATION = Ref(0.0)
const CURRENT_MAX_ITERATIONS = Ref(1)

function GeometricIntegratorsBase.default_options(::Galerkin_Bspline_Integrator)
    (
        regularization_factor = CURRENT_REGULARIZATION[],
        max_iterations = CURRENT_MAX_ITERATIONS[],
        verbosity = 0
    )
end

function time_call(label, f)
    f()
    GC.gc()
    elapsed = @elapsed f()
    @printf("%-26s %9.4f s\n", label, elapsed)
    return elapsed
end

function main()
    k = parse(Int, get(ENV, "SPLINE_K", "3"))
    t_step = parse(Float64, get(ENV, "SPLINE_T_STEP", "0.1"))
    t_end = parse(Float64, get(ENV, "SPLINE_T_END", "0.1"))
    t_knot_interval = parse(Float64, get(ENV, "SPLINE_T_KNOT_INTERVAL", "0.5"))
    x_knot_interval = parse(Float64, get(ENV, "SPLINE_X_KNOT_INTERVAL", "0.1"))
    multiplier_stride = parse(Int, get(ENV, "SPLINE_MULTIPLIER_STRIDE", "1"))
    xspan = (0.2, 0.8)

    lpde = MultiSymplectic.Wave.lpdeproblem(
        timestep = t_step,
        timespan = (0.0, t_end),
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
        show_status = false,
        multiplier_stride = multiplier_stride
    )
    int = MultiSymplectic.PDEIntegrator(lpde, method)
    sol = GeometricSolution(lpde)
    solstep = MultiSymplectic.solutionstep(int, sol[0])

    reset!(solstep, MultiSymplectic.timestep(int))
    MultiSymplectic.initialize_bcs_ics!(solstep, int)
    MultiSymplectic.prior_initial_guess!(MultiSymplectic.cache(int), solstep, int)

    x = MultiSymplectic.nlsolution(int)
    b = similar(x)
    ydual = zeros(eltype(ForwardDiff.Dual.(x, x)), length(x))
    j = zeros(length(x), length(x))

    @printf("S=%d  Nλ=%d  Nμ=%d  unknowns=%d  RT=%d  RX=%d\n",
        method.basis.S, method.Nbasis_λ_x, method.Nbasis_μ_t,
        length(x), method.RT, method.RX)

    time_call("components! Float64",
        () -> Base.invokelatest(
            MultiSymplectic.components!, x, current(solstep), parameters(solstep), int))
    time_call("residual! Float64",
        () -> Base.invokelatest(
            MultiSymplectic.residual!, b, x, current(solstep), parameters(solstep), int))
    time_call("jacobian! ForwardDiff",
        () -> ForwardDiff.jacobian!(
            j,
            (out, xin) -> Base.invokelatest(
                MultiSymplectic.residual!, out, xin, current(solstep), parameters(solstep), int),
            b,
            x
        ))

    dual_x = ForwardDiff.Dual.(x, x)
    time_call("residual! Dual once",
        () -> Base.invokelatest(
            MultiSymplectic.residual!, ydual, dual_x, current(solstep), parameters(solstep), int))
end

main()
