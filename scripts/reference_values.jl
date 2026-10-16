# Reference values of the old stack, written to `test/helpers/reference_values.jl`.
#
# Run it on a checkout whose `src/` is that of commit 294f023, with the manifest that resolves
# the old releases (GeometricBase 0.12, SimpleSolvers 0.7, ...), from the repository root:
#
#     julia --startup-file=no --threads=1 --project=<checkout with that manifest> \
#         scripts/reference_values.jl
#
# The output file is written next to this script, under `../test/helpers/`. It is the oracle
# that the rewritten integrators reproduce; the values describe the old code, defects included.
# Every case steps the integrator with the loop of `integrate!(::SolutionStep, ...)` of
# `src/utils/PDE_Integrator.jl`, written out so that the Newton state of each step can be read.
# A case that throws records the exception instead of values.

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
const PACKAGES = ["GeometricBase", "GeometricIntegratorsBase", "GeometricEquations",
    "GeometricSolutions", "SimpleSolvers", "QuadratureRules", "CompactBasisFunctions",
    "AbstractNeuralNetworks", "SymbolicNeuralNetworks", "GeometricMachineLearning", "Symbolics"]

LinearAlgebra.BLAS.set_num_threads(1)

# The package source must be that of COMMIT: the values describe that code.
cd(pkgdir(MSI)) do
    run(`git diff --quiet $COMMIT -- src Project.toml`)
end

# --- the problems --------------------------------------------------------------------------

function wave_problem(nsteps)
    Wave.lpdeproblem(timestep = 0.05, timespan = (0.0, 0.05 * nsteps),
        xspan = (0.0, 1.0), xstep = 0.05)
end
function sine_gordon_problem(nsteps)
    SineGordon.lpdeproblem(timestep = 0.05,
        timespan = (0.0, 0.05 * nsteps), xspan = (0.0, 1.0), xstep = 0.05)
end

const WAVE_CALL = "Wave.lpdeproblem(timestep = 0.05, timespan = (0.0, 0.25), xspan = (0.0, 1.0), xstep = 0.05)"
const SINE_GORDON_CALL = "SineGordon.lpdeproblem(timestep = 0.05, timespan = (0.0, 0.15), xspan = (0.0, 1.0), xstep = 0.05)"

# The ansatz of `Sindy_PDE_Integrator`: the exact Wave solution with its six parameters
# (A1, B1, A2, B2, l, c) free, from `scripts/test_sindy_PDE_int.jl`.
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

construct(call, lpde) = Core.eval(Main, :(let lpde = $lpde
    $(Meta.parse(call))
end))

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
    C.flag_done_initial_guess[1] != 0.0 && (C.flag_done_initial_guess[1] = 0.0)
    return status
end

# The quadrature energy of one slab, the formula of `Wave.hamiltonian` with the problem's
# Hamiltonian density.
function slab_energy(density, u, v, w, int, tn)
    lpde, method = int.problem, int.method
    x_domain = lpde.xspan[2] - lpde.xspan[1]
    h = lpde.timestep
    energy = 0.0
    for rt in axes(method.grid_matrix, 1), rx in axes(method.grid_matrix, 2)

        τ, x = method.grid_matrix[rt, rx]
        energy += x_domain * h * method.grid_weights[rt, rx] *
                  density(tn + h * τ, x, u(rt, rx), v(rt, rx), w(rt, rx), lpde.params)
    end
    return energy
end

# Relative error of the slab energy against that of the exact solution on the same nodes.
function energy_error(density, int, tn)
    C, lpde = MSI.cache(int, Float64), int.problem
    h = lpde.timestep
    node(rt, rx) = (
        tn + h * int.method.grid_matrix[rt, rx][1], int.method.grid_matrix[rt, rx][2])
    numerical = slab_energy(density, (rt, rx) -> C.u_quad_values[1, rt, rx],
        (rt, rx) -> C.v_quad_values[1, rt, rx], (rt, rx) -> C.w_quad_values[1, rt, rx], int, tn)
    exact = slab_energy(density,
        (rt, rx) -> lpde.exact_u(node(rt, rx)...; params = lpde.params),
        (rt, rx) -> lpde.exact_v(node(rt, rx)...; params = lpde.params),
        (rt, rx) -> lpde.exact_w(node(rt, rx)...; params = lpde.params), int, tn)
    return (numerical - exact) / exact
end

function failure(e)
    message = first(split(sprint(showerror, e), '\n'))
    (status = :failed, exception = string(nameof(typeof(e))),
        message = length(message) > 300 ? first(message, 300) * " …" : message)
end

# The problem and the integrators generate functions with `Symbolics.eval` at run time, so each
# stage runs in the latest world (`Base.invokelatest`).
function run_case(call, problem, nsteps, density, seeded)
    seeded && Random.seed!(SEED)
    try
        lpde = problem(nsteps)
        method = Base.invokelatest(construct, call, lpde)
        int = Base.invokelatest(MSI.PDEIntegrator, lpde, method)
        return Base.invokelatest(simulate, int, nsteps, density)
    catch e
        e isa InterruptException && rethrow()
        return failure(e)
    end
end

function simulate(int, nsteps, density)
    lpde = int.problem
    sol = GeometricSolution(lpde)
    solstep = MSI.solutionstep(int, sol[0])
    errors, statuses = Float64[], Any[]
    for n in 1:nsteps
        push!(statuses, step!(solstep, int))
        sol[n] = solstep
        push!(errors, energy_error(density, int, (n - 1) * lpde.timestep))
        any(s -> any(isnan, s), MSI.current(solstep)) && break
    end
    steps = length(errors)
    rows(series) = permutedims(reduce(hcat, [collect(series[n]) for n in 1:steps]))
    newton = if int.method isa DIRECT
        (newton = "not used: one direct linear solve per step",)
    else
        config = SimpleSolvers.config(MSI.solver(int))
        (iterations = [s.iterations for s in statuses],
            converged = [SimpleSolvers.isconverged(s) for s in statuses],
            residual = [s.rfₐ for s in statuses],
            x_abstol = config.x_abstol, x_reltol = config.x_reltol,
            x_suctol = config.x_suctol, f_abstol = config.f_abstol,
            f_reltol = config.f_reltol, f_suctol = config.f_suctol,
            min_iterations = config.min_iterations, max_iterations = config.max_iterations,
            regularization_factor = config.regularization_factor,
            linesearch = string(nameof(typeof(SimpleSolvers.linesearch(MSI.solver(int)).method))))
    end
    return (status = :ok, steps = steps, u = rows(sol.u), v = rows(sol.v),
        w = rows(sol.w), energy_error = errors, newton...)
end

function run_cases(cases, problem, problem_call, nsteps, density)
    entries = Pair{String, Any}[]
    for (name, call, seeded) in cases
        @info "case $name"
        first_run = run_case(call, problem, nsteps, density, seeded)
        entry = (constructor = call, problem = problem_call,
            seed = seeded ? SEED : "none", first_run...)
        if seeded
            second_run = run_case(call, problem, nsteps, density, seeded)
            same = isequal(first_run, second_run)
            entry = (entry..., repeat = same ? "identical" : "differs, see _REPEAT")
            push!(entries, name => entry)
            same || push!(entries, name * "_REPEAT" => (constructor = call, second_run...))
        else
            push!(entries, name => entry)
        end
    end
    return entries
end

# --- the FEM stencil, written out -----------------------------------------------------------

# The bilinear space-time FEM stencil of `FEM_Multisymplectic_Integrator` for u_tt = c² u_xx on
# Wave, Dirichlet values from the exact solution, u at t = -h from the exact solution. Returns
# max |u - u_exact| over the grid at `T`.
function fem_stencil_error(h, dx, T)
    params = Wave.default_parameters
    c = params.c
    x = collect(0.0:dx:1.0)
    N = length(x)
    nsteps = round(Int, T / h)
    exact(t) = Wave.exact_u.(t, x)
    I, B = 2:(N - 1), [1, N]
    # rows i of M (u⁺ - 2u + u⁻) / h² = c² L (u⁺ + 4u + u⁻) / (6 dx²), i interior
    M = Matrix(Tridiagonal(fill(1 / 6, N - 1), fill(4 / 6, N), fill(1 / 6, N - 1)))
    L = Matrix(Tridiagonal(fill(1.0, N - 1), fill(-2.0, N), fill(1.0, N - 1)))
    K = M / h^2 - c^2 * L / (6 * dx^2)
    previous, current = exact(-h), exact(0.0)
    for n in 1:nsteps
        next = exact(n * h)                        # boundary values; interior is overwritten
        rhs = M * (2 .* current .- previous) ./ h^2 .+
              c^2 .* (L * (4 .* current .+ previous)) ./ (6 * dx^2)
        next[I] .= K[I, I] \ (rhs[I] .- K[I, B] * next[B])
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

function fem_case(h, dx, T, numpy)
    stencil, package = fem_stencil_error(h, dx, T), fem_package_error(h, dx, T)
    agreement = numpy === nothing ? "no numpy figure" :
                round(stencil, sigdigits = 3) == round(numpy, sigdigits = 3) ?
                "reproduced to 3 digits" :
                "differs: stencil $(round(stencil, sigdigits = 3)), numpy $numpy"
    (h = h, dx = dx, t = T, cfl = Wave.default_parameters.c * h / dx,
        stencil_error = stencil, package_error = package,
        numpy_error = numpy === nothing ? "none" : numpy, numpy = agreement)
end

# --- the order of the full-restriction integrator ---------------------------------------------

function full_restriction_error(h, knot)
    lpde = Wave.lpdeproblem(timestep = h, timespan = (0.0, 1.0), xspan = (0.0, 1.0),
        xstep = 0.025)
    basis = Dirichlet_BSpline2D(3; timestep = h, xspan = (0.0, 1.0), t_knot_interval = 0.5,
        x_knot_interval = knot)
    method = Galerkin_Full_Restriction_Bspline_Integrator(basis; xspan = (0.0, 1.0),
        RT_per_interval = 3, RX_per_interval = 3)
    sol = Base.invokelatest(MSI.integrate, lpde, method)
    xs = collect(lpde.xspan[1]:lpde.xstep:lpde.xspan[2])
    return maximum(abs.(sol.sol.u[end] .- lpde.exact_u.(1.0, xs)))
end

function full_restriction_order()
    resolutions = [(0.1, 0.1), (0.05, 0.05), (0.025, 0.025)]
    call = "Galerkin_Full_Restriction_Bspline_Integrator(Dirichlet_BSpline2D(3; timestep = h, xspan = (0.0, 1.0), t_knot_interval = 0.5, x_knot_interval = knot); $QUAD) on Wave.lpdeproblem(timestep = h, timespan = (0.0, 1.0), xspan = (0.0, 1.0), xstep = 0.025)"
    try
        errors = [full_restriction_error(h, knot) for (h, knot) in resolutions]
        return (constructor = call, status = :ok, h = first.(resolutions),
            x_knot_interval = last.(resolutions), error_at_t1 = errors,
            observed_order = [log2(errors[i] / errors[i + 1]) for i in 1:2])
    catch e
        e isa InterruptException && rethrow()
        return (constructor = call, failure(e)...)
    end
end

# --- writing the literals ---------------------------------------------------------------------

literal(x::AbstractFloat) = repr(x)
literal(x::Union{Integer, Bool, Symbol, AbstractString}) = repr(x)

# The layout is the fixed point of JuliaFormatter's SciML style, so that the written file is
# formatted as written: a vector on one line where it fits in 92 columns, else one element per
# line; a matrix with one row per line.
function literal(x::AbstractVector, prefix)
    line = "[" * join(literal.(x), ", ") * "]"
    length(prefix) + length(line) + 1 ≤ 92 && return line
    "[\n" * join("        " .* literal.(x), ",\n") * "]"
end

function literal(x::AbstractMatrix, prefix)
    rows = [join(literal.(row), " ") for row in eachrow(x)]
    "[" * join(rows, ";\n" * " "^(length(prefix) + 1)) * "]"
end

literal(x, prefix) = literal(x)

function constant(name, entry)
    fields = String[]
    for (key, value) in pairs(entry)
        prefix = "    $key = "
        push!(fields, prefix * literal(value, prefix))
    end
    "const $name = (\n" * join(fields, ",\n") * ")\n"
end

function header()
    deps = Dict(d.name => d.version for d in values(Pkg.dependencies()))
    commit = readchomp(Cmd(`git rev-parse $COMMIT`; dir = pkgdir(MSI)))
    lines = [
        "# Reference values of the old stack. Generated by `scripts/reference_values.jl`;",
        "# do not edit by hand.", "#", "# commit: $commit", "# Julia: $VERSION",
        ["# $name $(deps[name])" for name in PACKAGES]...,
        "#",
        "# Each Wave and SineGordon entry: `u`, `v`, `w` hold one row per step, on the nodes",
        "# of the problem's output grid; `energy_error` is the relative error of the quadrature",
        "# energy of each slab (the formula of `Wave.hamiltonian`) against that of the exact",
        "# solution on the same nodes; `iterations`, `converged` and `residual` (absolute) are",
        "# the Newton state of each step, with the solver options beside them. A network case is",
        "# run twice from `Random.seed!(seed)`; `repeat` says whether the runs agree. An entry",
        "# with `status = :failed` holds the exception instead of values."]
    join(lines, "\n") * "\n"
end

function main()
    wave = run_cases(WAVE_CASES, wave_problem, WAVE_CALL, 5, Wave.hamiltonian_density)
    sine_gordon = run_cases(SINE_GORDON_CASES, sine_gordon_problem, SINE_GORDON_CALL, 3,
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
    order = full_restriction_order()

    text = header()
    for (name, entry) in wave
        text *= "\n" * constant("WAVE_" * name, entry)
    end
    for (name, entry) in sine_gordon
        text *= "\n" * constant("SINE_GORDON_" * name, entry)
    end
    for (name, entry) in fem
        text *= "\n" * constant(name, entry)
    end
    text *= "\n" * constant("WAVE_FULL_RESTRICTION_ORDER", order)
    mkpath(dirname(OUTPUT))
    write(OUTPUT, text)
    @info "wrote $(normpath(OUTPUT)), $(count(==('\n'), text)) lines"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
