using GeometricIntegratorsBase: GeometricIntegrator, solver
using LinearAlgebra: BLAS
using MultisymplecticIntegrators
using SimpleSolvers: config

# The manufactured problem of the verification files: the quartic wave density
# `L = ½v² − ½c²w² − ¼βu⁴` on `x ∈ [0, 1]` with a forcing `f = u*_tt − c² u*_xx + β u*³` that makes
#
#     u*(t, x) = sin(πx) cos t + ½ sin(2πx) sin t
#
# its exact solution. `u*` vanishes at both ends, so the couplings that take Dirichlet data in space
# run on it. Component `d` of a problem with `D` components is `MANUFACTURED_SCALES[d] · u*`, on the
# same per-component density, with its own forcing. Every derivative and the forcing are written
# out by hand here, never derived by the code under test.

const MANUFACTURED_C = 0.5
const MANUFACTURED_BETA = 1.0
const MANUFACTURED_SCALES = (1.0, -0.7)

# `2 * (π * x)` keeps the element type of `x`, where the literal `2π` is a `Float64`.
manufactured_u(t, x) = sin(π * x) * cos(t) + sin(2 * (π * x)) * sin(t) / 2
manufactured_v(t, x) = -sin(π * x) * sin(t) + sin(2 * (π * x)) * cos(t) / 2
manufactured_w(t, x) = π * cos(π * x) * cos(t) + π * cos(2 * (π * x)) * sin(t)

"""
    manufactured_load(t, x, a)

The forcing of the component `a · u*`: `a (u*_tt − c² u*_xx) + β (a u*)³`, with
`u*_tt − c² u*_xx = (c²π² − 1) sin(πx) cos t + ½(4c²π² − 1) sin(2πx) sin t`. Its value is a
`Float64` whatever the element type of `t` and `x`.
"""
function manufactured_load(t, x, a)
    c, β = MANUFACTURED_C, MANUFACTURED_BETA
    linear = (c^2 * π^2 - 1) * sin(π * x) * cos(t) +
             (4c^2 * π^2 - 1) / 2 * sin(2 * (π * x)) * sin(t)
    return a * linear + β * (a * manufactured_u(t, x))^3
end

"""
    manufactured_forcing(D)

The forcing of the manufactured problem with `D` components: a number at `D = 1`, and a vector of
one value per component otherwise.
"""
function manufactured_forcing(D::Integer)
    D == 1 && return (t, x, params) -> manufactured_load(t, x, MANUFACTURED_SCALES[1])
    return (t, x, params) -> [manufactured_load(t, x, MANUFACTURED_SCALES[d]) for d in 1:D]
end

"""
    manufactured_problem(; D = 1, timestep, nx, T = Float64, tend = 1, forcing)

The manufactured problem on `[0, tend] × [0, 1]` with `nx` grid cells, in the element type `T`. The
exact functions keep the element type of their arguments. `forcing` is the manufactured forcing by
default; `nothing` gives the same problem unforced.
"""
function manufactured_problem(; D::Integer = 1, timestep, nx::Integer, T = Float64,
        tend = 1, forcing = manufactured_forcing(D))
    exact(g) = field_exact(
        (t, x, params) -> [oftype(x, MANUFACTURED_SCALES[d]) * g(t, x) for d in 1:D], D)
    solutions = (u = exact(manufactured_u), v = exact(manufactured_v),
        w = exact(manufactured_w))
    timespan = (zero(T), T(tend))
    xspan = (zero(T), one(T))
    density = Wave.density(; params = (c = T(MANUFACTURED_C),), β = MANUFACTURED_BETA)
    return LagrangianPDEProblem(density; D = D,
        ics_function = (x, params) -> field_initial_data(
            solutions, x, params, D, timespan[1]),
        bcs_function = (t, params) -> field_side_data(solutions, t, params, D, xspan),
        solutions = solutions, timespan = timespan, timestep = T(timestep), xspan = xspan,
        xstep = one(T) / nx, forcing = forcing)
end

"""
    all_steps_converge(result, int)

Whether every step of the run converged by the rule `residual ≤ f_abstol + f_reltol · r₀`, with
the tolerances read from the integrator's own solver options and the relative term counted only for
a finite `r₀`. The rule is applied here to the recorded residuals, beside the run's own record.
"""
function all_steps_converge(result, int)
    options = config(solver(int))
    (; residuals, initial_residuals, converged) = result.stats
    rule = map(residuals, initial_residuals) do r, r₀
        relative = isfinite(r₀) ? options.f_reltol * r₀ : zero(r)
        r ≤ options.f_abstol + relative
    end
    return !isempty(rule) && all(rule) && all(converged)
end

"""
    l2_norm(weights, values, D)

The `L²` norm of the component-major vector `values` of `D` components on a rule with `weights`.
"""
function l2_norm(weights::AbstractVector, values::AbstractVector, D::Integer)
    return sqrt(sum(repeat(weights, D) .* values .^ 2))
end

"""The observed orders `log₂(e₁ / e₂)` of a ladder whose step halves from entry to entry."""
observed_orders(errors) = [log2(errors[i] / errors[i + 1]) for i in 1:(length(errors) - 1)]

"""The metadata a measured number is written with."""
function run_metadata()
    return (julia = VERSION, machine = Sys.MACHINE, cpu = Sys.CPU_NAME,
        blas_threads = BLAS.get_num_threads(),
        check_bounds = Base.JLOptions().check_bounds)
end
