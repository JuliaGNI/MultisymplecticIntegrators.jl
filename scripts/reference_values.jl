# Reference values of the integrators at commit 294f023, written to
# `test/helpers/reference_values.jl`.
#
# Run it from the repository root, on a checkout whose `src/` and `Project.toml` are those of
# commit 294f023, in an environment that resolves the versions named in the header of the output
# file:
#
#     julia --startup-file=no --threads=1 --check-bounds=yes --project=<that environment> \
#         scripts/reference_values.jl
#
# The values depend on the `--check-bounds` flag. The script runs only under
# `--check-bounds=yes`, the flag of `julia-runtest` in CI and of `Pkg.test()` up to Julia 1.12.
# The values describe the code at that commit, defects included. Every case steps the integrator
# with the loop of `integrate!(::SolutionStep, ...)` of `src/utils/PDE_Integrator.jl`, written
# out so that the Newton state of each step can be read. A case that throws records the
# exception, the step that raised it, and the values of the steps before it.

using MultisymplecticIntegrators
using AbstractNeuralNetworks: Chain, Dense
using GeometricSolutions: GeometricSolution
using LinearAlgebra
using Pkg
using Random
using Symbolics
import SimpleSolvers

const MSI = MultisymplecticIntegrators
const COMMIT = "294f023"
const SEED = 1234
const OUTPUT = joinpath(@__DIR__, "..", "test", "helpers", "reference_values.jl")

LinearAlgebra.BLAS.set_num_threads(1)

Base.JLOptions().check_bounds == 1 || error("run this script with --check-bounds=yes")

# The package source must be that of COMMIT: the values describe that code.
cd(pkgdir(MSI)) do
    run(`git diff --quiet $COMMIT -- src Project.toml`)
end

# --- the problems and the integrators -------------------------------------------------------

# Each problem and each integrator is the call that is recorded beside its values. The script
# evaluates the recorded text, in `Main`, with `bindings` as local variables.
function evaluate(code; bindings...)
    Core.eval(Main,
        Expr(:let, Expr(:block, (:($k = $v) for (k, v) in bindings)...), Meta.parse(code)))
end

const WAVE = "Wave.lpdeproblem(timestep = 0.05, timespan = (0.0, 0.25), xspan = (0.0, 1.0), xstep = 0.05)"
const SINE_GORDON = "SineGordon.lpdeproblem(timestep = 0.05, timespan = (0.0, 0.15), xspan = (0.0, 1.0), xstep = 0.05)"

# The ansatz of `Sindy_PDE_Integrator`: the exact Wave solution with its six parameters
# (A1, B1, A2, B2, l, c) free, from `scripts/test_sindy_PDE_int.jl`. The start of the SINDY case,
# [0.8, 0.8, 0.0, 0.0, 1.0, 0.5], is `Wave.default_parameters`: the exact solution itself.
function sindy_wave_basis()
    @variables t x p[1:6]
    u = (p[1] * cos((pi * p[6] * t) / p[5]) + p[2] * sin((pi * p[6] * t) / p[5] + pi / 6)) *
        sin((pi * x) / p[5]) +
        (p[3] * cos((2 * pi * p[6] * t) / p[5]) +
         p[4] * sin((2 * pi * p[6] * t) / p[5] + pi / 6)) *
        sin((2 * pi * x) / p[5])
    SindyPDEBasis([u], [p], t, [x])
end

const SPLINE = "BSpline2D(3; xspan = (0.0, 1.0), t_knot_interval = 0.5, x_knot_interval = 0.05)"
const DIRICHLET = "Dirichlet_BSpline2D(3; timestep = 0.05, xspan = (0.0, 1.0), t_knot_interval = 0.5, x_knot_interval = 0.05)"
const QUAD = "xspan = (0.0, 1.0), RT_per_interval = 3, RX_per_interval = 3"
const NN_OPTS = "xspan = (0.0, 1.0), μ = :BSplineDirichlet, λ = :BSplineDirichlet, t_num_interval = 2, x_num_interval = 5, k_μ_t = 4, k_λ_x = 4, Nw = 50, Nb = 50"
const ELM_BASIS = "NN_Basis(Chain(Dense(2, 20, tanh), Dense(20, 20, tanh)), 20)"

# (name, constructor call, uses a random number generator)
const WAVE_CASES = [
    ("GALERKIN_BSPLINE", "Galerkin_Bspline_Integrator($SPLINE; $QUAD)", false),
    ("GALERKIN_DIRICHLET_BSPLINE",
        "Galerkin_Dirichlet_Bspline_Integrator($DIRICHLET; $QUAD)", false),
    ("GALERKIN_FULL_RESTRICTION_BSPLINE",
        "Galerkin_Full_Restriction_Bspline_Integrator($DIRICHLET; $QUAD)", false),
    ("SPACETIME_SPLINE_WAVE", "SpaceTime_Spline_Wave_Integrator($DIRICHLET, lpde)", false),
    ("FEM_MULTISYMPLECTIC",
        "FEM_Multisymplectic_Integrator(lpde; startup = :exact)", false),
    ("NN_FULLY", "NN_PDE_Integrator(NetworkPDEBasis(10, tanh, :Fully); $NN_OPTS)", true),
    ("NN_PARTIALLY",
        "NN_PDE_Integrator(NetworkPDEBasis(10, tanh, :Partially); $NN_OPTS)", true),
    ("NN_SYMBOLIC_FULLY",
        "NN_PDE_Integrator_Symbolic(NetworkPDEBasis_Symbolic(10, tanh, :Fully); $NN_OPTS)",
        true),
    ("ELM_LSGD",
        "ELM_PDE_int($ELM_BASIS; $QUAD, initial_guess_method = LSGD(), show_status = false)",
        true),
    ("TRIALNN_OGA2D",
        "TrialNN_PDE_int(Trial_Solution_Basis(10, tanh, (0.0, 1.0)); xspan = (0.0, 1.0), t_num_interval = 2, x_num_interval = 5, Nw = 50, Nb = 50)",
        true),
    ("SINDY",
        "Sindy_PDE_Integrator(sindy_wave_basis(), [0.8, 0.8, 0.0, 0.0, 1.0, 0.5]; xspan = (0.0, 1.0), t_num_interval = 2, x_num_interval = 5, k_λ_x = 4)",
        false)
]

const SINE_GORDON_CASES = [
    ("GALERKIN_BSPLINE", "Galerkin_Bspline_Integrator($SPLINE; $QUAD)", false),
    ("NN_FULLY", "NN_PDE_Integrator(NetworkPDEBasis(10, tanh, :Fully); $NN_OPTS)", true),
    ("ELM_LSGD",
        "ELM_PDE_int($ELM_BASIS; $QUAD, initial_guess_method = LSGD(), show_status = false)",
        true),
    ("TRIALNN_OGA2D",
        "TrialNN_PDE_int(Trial_Solution_Basis(10, tanh, (0.0, 1.0)); xspan = (0.0, 1.0), t_num_interval = 2, x_num_interval = 5, Nw = 50, Nb = 50)",
        true)
]

# --- one step, as `integrate!(::SolutionStep, ::AbstractPDEIntegrator)` takes it -----------

const DIRECT = Union{FEM_Multisymplectic_Integrator, SpaceTime_Spline_Wave_Integrator}

function step!(solstep, int)
    MSI.reset!(solstep, MSI.timestep(int))
    C = MSI.cache(int)
    MSI.copy_internal_variables!(C, solstep)
    MSI.initialize_bcs_ics!(solstep, int)
    MSI.prior_initial_guess!(C, solstep, int)
    sol, params = MSI.current(solstep), MSI.parameters(solstep)
    status = nothing
    if int.method isa DIRECT
        MSI.integrate_step!(sol, MSI.history(solstep), params, int)
    else
        x, s = MSI.nlsolution(int), MSI.solver(int)
        state = SimpleSolvers.NonlinearSolverState(x, SimpleSolvers.value(SimpleSolvers.cache(s)))
        SimpleSolvers.solve!(x, s, state, (sol, params, int))
        status = SimpleSolvers.NonlinearSolverStatus(state, SimpleSolvers.config(s))
        MSI.update!(sol, int)
    end
    MSI.components!(MSI.nlsolution(int), sol, params, int)
    MSI.copy_internal_variables!(solstep, C)
    C.flag_done_initial_guess[1] = 0.0
    return status
end

# The relative error of the quadrature energy of one slab, with the problem's Hamiltonian
# density, against that of the exact solution on the same nodes. The factor (x length) × h of
# the slab energy cancels.
function energy_error(density, int, tn)
    C, lpde, method = MSI.cache(int), int.problem, int.method
    nodes = [(tn + lpde.timestep * τ, x) for (τ, x) in method.grid_matrix]
    energy(u, v, w) = sum(method.grid_weights .*
                          density.(first.(nodes), last.(nodes), u, v, w, Ref(lpde.params)))
    exact(f) = [f(t, x; params = lpde.params) for (t, x) in nodes]
    quad(values) = values[1, axes(method.grid_matrix)...]
    reference = energy(exact(lpde.exact_u), exact(lpde.exact_v), exact(lpde.exact_w))
    numerical = energy(quad(C.u_quad_values), quad(C.v_quad_values), quad(C.w_quad_values))
    return (numerical - reference) / reference
end

function failure(e)
    message = first(split(sprint(showerror, e), '\n'))
    (exception = string(nameof(typeof(e))),
        message = length(message) > 300 ? first(message, 300) * " …" : message)
end

# The options of the Newton solver. Every Newton case must run with the same options, which the
# output records once, as `SOLVER_OPTIONS`.
const OPTIONS = Any[]

function record_options!(int)
    s = MSI.solver(int)
    c = SimpleSolvers.config(s)
    options = (x_abstol = c.x_abstol, x_reltol = c.x_reltol, x_suctol = c.x_suctol,
        f_abstol = c.f_abstol, f_reltol = c.f_reltol, f_suctol = c.f_suctol,
        min_iterations = c.min_iterations, max_iterations = c.max_iterations,
        regularization_factor = c.regularization_factor,
        linesearch = string(nameof(typeof(SimpleSolvers.linesearch(s).method))))
    isempty(OPTIONS) && push!(OPTIONS, options)
    isequal(options, only(OPTIONS)) ||
        error("the solver options of two cases differ: $options")
end

# The problem and the integrators generate functions with `Symbolics.eval` at run time, so each
# stage runs in the latest world (`Base.invokelatest`). An integrator that cannot be built fails
# at step 0.
function run_case(call, problem, density, seeded)
    seeded && Random.seed!(SEED)
    lpde = evaluate(problem)
    int = try
        Base.invokelatest(MSI.PDEIntegrator, lpde, evaluate(call; lpde))
    catch e
        e isa InterruptException && rethrow()
        return (status = :failed, steps = 0, failed_at_step = 0, failure(e)...)
    end
    return Base.invokelatest(simulate, int, density)
end

function simulate(int, density)
    lpde = int.problem
    newton = !(int.method isa DIRECT)
    newton && record_options!(int)
    sol = GeometricSolution(lpde)
    solstep = MSI.solutionstep(int, sol[0])
    errors, statuses, failed = Float64[], [], (;)
    for n in 1:round(Int, (lpde.timespan[2] - lpde.timespan[1]) / lpde.timestep)
        status = try
            step!(solstep, int)
        catch e
            e isa InterruptException && rethrow()
            failed = (failed_at_step = n, failure(e)...)
            break
        end
        push!(statuses, status)
        sol[n] = solstep
        push!(errors, energy_error(density, int, (n - 1) * lpde.timestep))
        any(s -> any(isnan, s), MSI.current(solstep)) && break
    end
    steps = length(errors)
    rows(series) = stack((collect(series[n]) for n in 1:steps); dims = 1)
    solution = steps == 0 ? (;) :
               (u = rows(sol.u), v = rows(sol.v), w = rows(sol.w), energy_error = errors)
    state = steps == 0 || !newton ? (;) :
            (iterations = [s.iterations for s in statuses],
        converged = [SimpleSolvers.isconverged(s) for s in statuses],
        residual = [s.rfₐ for s in statuses])
    return (
        status = isempty(failed) ? :ok : :failed, steps = steps, failed..., solution...,
        state...)
end

# A network case runs twice from `Random.seed!(SEED)`; the script stops if the two runs differ.
function run_cases(prefix, cases, problem, density)
    map(cases) do (name, call, seeded)
        @info "case $name"
        result = run_case(call, problem, density, seeded)
        seeded && !isequal(result, run_case(call, problem, density, seeded)) &&
            error("$name differs between two runs from the same seed")
        prefix * name => (constructor = call, problem = problem,
            seed = seeded ? SEED : nothing, result...)
    end
end

# --- the FEM stencil, written out -----------------------------------------------------------

# The bilinear space-time FEM stencil of `FEM_Multisymplectic_Integrator` for u_tt = c² u_xx on
# Wave, Dirichlet values from the exact solution, u at t = -h from the exact solution. Returns
# max |u - u_exact| over the grid at `T`.
function fem_stencil_error(h, dx, T)
    c = Wave.default_parameters.c
    x = collect(0.0:dx:1.0)
    N = length(x)
    nsteps = round(Int, T / h)
    exact(t) = Wave.exact_u.(t, x)
    I, B = 2:(N - 1), [1, N]
    # rows i of M (u⁺ - 2u + u⁻) / h² = c² L (u⁺ + 4u + u⁻) / (6 dx²), i interior
    M = Tridiagonal(fill(1 / 6, N - 1), fill(4 / 6, N), fill(1 / 6, N - 1))
    L = Tridiagonal(fill(1.0, N - 1), fill(-2.0, N), fill(1.0, N - 1))
    K = M / h^2 - c^2 * L / (6 * dx^2)
    F = lu(K[I, I])
    previous, current = exact(-h), exact(0.0)
    for n in 1:nsteps
        next = exact(n * h)                        # boundary values; interior is overwritten
        rhs = M * (2 .* current .- previous) ./ h^2 .+
              c^2 .* (L * (4 .* current .+ previous)) ./ (6 * dx^2)
        next[I] .= F \ (rhs[I] .- K[I, B] * next[B])
        previous, current = current, next
    end
    return maximum(abs.(current .- exact(nsteps * h)))
end

# The same error from the package's integrator.
function fem_package_error(h, dx, T)
    lpde = Wave.lpdeproblem(timestep = h, timespan = (0.0, T), xspan = (0.0, 1.0), xstep = dx)
    method = FEM_Multisymplectic_Integrator(lpde; startup = :exact)
    sol = Base.invokelatest(MSI.integrate, lpde, method)
    xs = collect(lpde.xspan[1]:lpde.xstep:lpde.xspan[2])
    return maximum(abs.(sol.sol.u[end] .- lpde.exact_u.(lpde.timespan[2], xs)))
end

# `numpy_error` is the error that a reader's independent reimplementation of the stencil in numpy
# gave, where one exists. That code is not in this repository, and its CFL 1 figure states no
# final time.
function fem_case(h, dx, T, numpy)
    (h = h, dx = dx, t = T, cfl = Wave.default_parameters.c * h / dx,
        stencil_error = fem_stencil_error(h, dx, T), package_error = fem_package_error(h, dx, T),
        numpy_error = numpy)
end

# --- the order of the full-restriction integrator ---------------------------------------------

const FULL_RESTRICTION_PROBLEM = "Wave.lpdeproblem(timestep = h, timespan = (0.0, 1.0), xspan = (0.0, 1.0), xstep = 0.025)"
const FULL_RESTRICTION = "Galerkin_Full_Restriction_Bspline_Integrator(Dirichlet_BSpline2D(3; timestep = h, xspan = (0.0, 1.0), t_knot_interval = 0.5, x_knot_interval = knot); $QUAD)"

function full_restriction_error(h, knot)
    lpde = evaluate(FULL_RESTRICTION_PROBLEM; h)
    sol = Base.invokelatest(MSI.integrate, lpde, evaluate(FULL_RESTRICTION; h, knot))
    xs = collect(lpde.xspan[1]:lpde.xstep:lpde.xspan[2])
    return maximum(abs.(sol.sol.u[end] .- lpde.exact_u.(lpde.timespan[2], xs)))
end

function full_restriction_order()
    resolutions = [(0.1, 0.1), (0.05, 0.05), (0.025, 0.025)]
    entry = (constructor = FULL_RESTRICTION, problem = FULL_RESTRICTION_PROBLEM)
    errors = try
        [full_restriction_error(h, knot) for (h, knot) in resolutions]
    catch e
        e isa InterruptException && rethrow()
        return (entry..., status = :failed, failure(e)...)
    end
    return (entry..., status = :ok, h = first.(resolutions),
        x_knot_interval = last.(resolutions), error_at_t1 = errors,
        observed_order = [log2(errors[i] / errors[i + 1]) for i in 1:2])
end

# --- writing the file -------------------------------------------------------------------------

# Each entry is written with `repr`, one line per constant. `.JuliaFormatter.toml` excludes the
# output from formatting.
function header()
    deps = Dict(d.name => d.version for d in values(Pkg.dependencies()))
    direct = sort([d.name for d in values(Pkg.dependencies()) if d.is_direct_dep])
    commit = readchomp(Cmd(`git rev-parse $COMMIT`; dir = pkgdir(MSI)))
    check_bounds = ("auto", "yes", "no")[Base.JLOptions().check_bounds + 1]
    lines = [
        "# Reference values of the integrators at commit $COMMIT. Generated by",
        "# `scripts/reference_values.jl`; do not edit by hand.", "#", "# commit: $commit",
        "# Julia: $VERSION, --check-bounds=$check_bounds, $(Threads.nthreads()) thread(s), " *
        "$(BLAS.get_num_threads()) BLAS thread(s)",
        ["# $name $(deps[name])" for name in [direct; "OpenBLAS_jll"]]...,
        "#",
        "# The values depend on the `--check-bounds` flag; compare them under the flag above.",
        "#",
        "# Each Wave and SineGordon entry: `u`, `v`, `w` hold one row per step, on the nodes",
        "# of the problem's output grid. `energy_error` is the relative error of the quadrature",
        "# energy of each slab, with the problem's Hamiltonian density, against that of the exact",
        "# solution on the same nodes; for FEM_MULTISYMPLECTIC the quadrature is the one time node",
        "# at the end of the step, with finite-difference `v` and `w`. `iterations`, `converged`",
        "# and `residual` (absolute) are the Newton state of each step. `converged` is the",
        "# solver's `isconverged`, which a small change of the iterate meets also at a large",
        "# residual. Every Newton case runs with `SOLVER_OPTIONS`; a direct integrator has no",
        "# Newton fields. A network case ran twice from `Random.seed!(seed)`, with identical",
        "# results. An entry with `status = :failed` holds the exception that step",
        "# `failed_at_step` raised (0: building the integrator) and the values of the steps",
        "# before it."]
    join(lines, "\n") * "\n"
end

function main()
    wave = run_cases("WAVE_", WAVE_CASES, WAVE, Wave.hamiltonian_density)
    sine_gordon = run_cases("SINE_GORDON_", SINE_GORDON_CASES, SINE_GORDON,
        SineGordon.hamiltonian)
    @info "FEM stencil"
    fem = [
        "FEM_STENCIL_H005_T5" => fem_case(0.05, 0.05, 5.0, nothing),
        "FEM_STENCIL_H005_T200" => fem_case(0.05, 0.05, 200.0, 0.130973),
        "FEM_STENCIL_H0025_T5" => fem_case(0.025, 0.025, 5.0, nothing),
        "FEM_STENCIL_H00125_T5" => fem_case(0.0125, 0.0125, 5.0, nothing),
        "FEM_STENCIL_H0025_T200" => fem_case(0.025, 0.025, 200.0, 0.0397),
        "FEM_STENCIL_H00125_T200" => fem_case(0.0125, 0.0125, 200.0, 0.0104),
        "FEM_STENCIL_CFL1_T5" => fem_case(0.05, 0.025, 5.0, 2.1e-12),
        "FEM_STENCIL_CFL1_T200" => fem_case(0.05, 0.025, 200.0, 2.1e-12)
    ]
    @info "full-restriction order"
    constants = [wave; sine_gordon; "SOLVER_OPTIONS" => only(OPTIONS); fem;
                 "WAVE_FULL_RESTRICTION_ORDER" => full_restriction_order()]
    text = header() *
           join("\nconst $name = $(repr(entry))\n" for (name, entry) in constants)
    mkpath(dirname(OUTPUT))
    write(OUTPUT, text)
    @info "wrote $(normpath(OUTPUT)), $(count(==('\n'), text)) lines"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
