using GeometricBase: parameters
using GeometricIntegratorsBase: GeometricIntegrator, solver
using LinearAlgebra: BLAS
using MultisymplecticIntegrators
using SimpleSolvers: config
using SimpleSplines: Dirichlet, Free

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

# The runs, the ladders and the long-time settings of `test/verification/manufactured.jl` and
# `test/verification/long_time_energy.jl`, which `scripts/manual_verification.jl` reads too, so the
# tests and the manual's tables run one copy of them.

# The B-spline couplings, each with the space boundary condition it runs on: `u*` and the standing
# wave of `Wave` vanish at both ends, so the two couplings that take Dirichlet data in space run on
# them.
const COUPLINGS = (("CoefficientRestriction", CoefficientRestriction(), Dirichlet()),
    ("WeakInitialCondition", WeakInitialCondition(), Dirichlet()),
    ("WeakInitialAndBoundaryConditions", WeakInitialAndBoundaryConditions(), Free()))

# The end time of a manufactured run.
const TEND = 1.0

# The joint ladder: the space cells `nx` of a slab run at `h = 1/nx`, so `c·h/Δx = 0.5` on every
# rung and the order is `p + 1`.
const JOINT_LADDER = (20, 40, 80)

# The self-difference ladder: the time steps of the slab runs at `SLAB_SELF_DIFFERENCE_NX` cells and
# of the box runs at `BOX_SELF_DIFFERENCE_NX` cells. A ladder of four steps gives three differences
# and two orders.
const SELF_DIFFERENCE_STEPS = (0.1, 0.05, 0.025, 0.0125)
const SLAB_SELF_DIFFERENCE_NX = 20
const BOX_SELF_DIFFERENCE_NX = 40

# The box scheme's space ladder: the space cells `nx` at the time step `BOX_SPACE_STEP`.
const BOX_SPACE_LADDER = (10, 20, 40)
const BOX_SPACE_STEP = 1 / 640

# The long-time runs of `Wave` at its defaults: ten periods of `PERIOD = 2l/c = 4` at the time step
# `h`, the slab runs at degree `p` on `nx` space cells, and the box run at the grid step `box_xstep`.
const PERIOD = 4.0
const LONG_TIME = (h = 0.05, tend = 40.0, nx = 20, p = 2, box_xstep = 0.05)

# One slab run of the manufactured problem to `tend`, one time cell per slab: the top trace at the
# last step, the space rule's weights, the `L²` error of the trace against `u*`, and whether every
# step converged.
function slab_run(coupling, boundary, p, h, nx, D; T = Float64, tend = TEND,
        forcing = manufactured_forcing(D))
    problem = manufactured_problem(;
        D = D, timestep = h, nx = nx, T = T, tend = tend, forcing = forcing)
    ansatz = TensorBSplineAnsatz(degree = p, ncells = (1, nx), xspan = (zero(T), one(T)),
        x_boundary = boundary)
    int = GeometricIntegrator(problem, SpaceTimeGalerkin(ansatz, coupling))
    result = integrate(int)
    nodes = space_quadrature_nodes(ansatz)
    weights = MultisymplecticIntegrators.space_quadrature_weights(ansatz)
    u = result.traces[end].u
    exact = problem.solutions.u(T(tend), nodes, parameters(problem))
    return (; result, u, weights, error = l2_norm(weights, u .- exact, D),
        steps = length(result.traces), converged = all_steps_converge(result, int))
end

# One box run of the manufactured problem to `tend`: the level at the last step, the trapezoid
# weights of the grid, the `L²` error of the level against `u*` at the grid nodes, and whether every
# step converged.
function box_run(h, nx, D; T = Float64, tend = TEND, forcing = manufactured_forcing(D))
    problem = manufactured_problem(;
        D = D, timestep = h, nx = nx, T = T, tend = tend, forcing = forcing)
    int = GeometricIntegrator(problem, BoxScheme())
    result = integrate(int)
    grid = collect(xgrid(problem))
    weights = fill(one(T) / nx, length(grid))
    weights[1] /= 2
    weights[end] /= 2
    u = result.traces[end].u
    exact = problem.solutions.u(T(tend), grid, parameters(problem))
    return (; result, u, weights, error = l2_norm(weights, u .- exact, D),
        steps = length(result.traces), converged = all_steps_converge(result, int))
end

# The `L²` norms of the differences of the last levels of consecutive runs of a ladder.
function self_differences(runs, D)
    return [l2_norm(runs[i].weights, runs[i].u .- runs[i + 1].u, D)
            for i in 1:(length(runs) - 1)]
end

# One slab run of `Wave` at its defaults, `steps` steps of `h` on `nx` space cells.
function wave_run(coupling, boundary, p, h, nx, steps)
    problem = Wave.problem(timespan = (0.0, steps * h), timestep = h, xstep = 1 / nx)
    ansatz = TensorBSplineAnsatz(degree = p, ncells = (1, nx), xspan = (0.0, 1.0),
        x_boundary = boundary)
    int = GeometricIntegrator(problem, SpaceTimeGalerkin(ansatz, coupling))
    return (; problem, ansatz, int, result = integrate(int))
end

# One box run of `Wave` at its defaults to `tend`, at the time step `h` and the grid step `xstep`.
function box_wave_run(h, tend, xstep)
    problem = Wave.problem(timespan = (0.0, tend), timestep = h, xstep = xstep)
    int = GeometricIntegrator(problem, BoxScheme())
    return (; problem, int, result = integrate(int))
end

# The energy of a trace on a rule: `Σ ωᵢ H(uᵢ, vᵢ, wᵢ)` over the rule's nodes.
function quadrature_energy(problem, weights, u, v, w)
    return sum(weights[i] *
               energy_density(problem.density, u[i], v[i], w[i], problem.params)
    for i in eachindex(weights))
end

# The energies of a `wave_run` at its top traces, and the exact energies at `t = 0, h, 2h, …` on
# the same space rule; the first exact entry is the energy at `t = 0`.
function slab_run_energies(run, h)
    (; problem, ansatz, result) = run
    nodes = space_quadrature_nodes(ansatz)
    weights = MultisymplecticIntegrators.space_quadrature_weights(ansatz)
    params = parameters(problem)
    Eh = [quadrature_energy(problem, weights, trace.u, trace.v, trace.w)
          for trace in result.traces]
    Ex = [quadrature_energy(
              problem, weights, problem.solutions.u(n * h, nodes, params),
              problem.solutions.v(n * h, nodes, params),
              problem.solutions.w(n * h, nodes, params))
          for n in 0:length(result.traces)]
    return Eh, Ex
end

# The energies of a `box_wave_run` at its levels, and the exact energies at `t = 0, h, 2h, …` on
# the same grid; the first exact entry is the energy at `t = 0`.
function box_run_energies(run, h)
    (; problem, result) = run
    grid = collect(xgrid(problem))
    params = parameters(problem)
    exact = [(u = problem.solutions.u(n * h, grid, params),
                 v = problem.solutions.v(n * h, grid, params),
                 w = problem.solutions.w(n * h, grid, params))
             for n in 0:length(result.traces)]
    energies(levels) = MultisymplecticIntegrators.box_energies(problem, levels)
    return energies(result.traces), energies(exact)
end

# The energy errors `|E_h(t_n) − E_*(t_n)| / E_*(0)` of a series of run energies `Eh` against the
# exact energies `Ex`, whose first entry is the energy at `t = 0`.
energy_errors(Eh, Ex) = abs.(Eh .- Ex[2:end]) ./ Ex[1]

# The largest energy error over the first period `(0, PERIOD]`, the largest over the last
# `[tend − PERIOD, tend]`, and their ratio.
function energy_periods(Eh, Ex, h, tend)
    errors = energy_errors(Eh, Ex)
    times = h .* (1:length(errors))
    first_period = maximum(errors[times .≤ PERIOD + h / 2])
    last_period = maximum(errors[times .≥ tend - PERIOD - h / 2])
    return (; first_period, last_period, ratio = last_period / first_period)
end
