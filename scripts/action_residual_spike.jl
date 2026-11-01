# The slab residual as the gradient of a discrete action: what it costs.
#
#   julia --startup-file=no --project=scripts scripts/action_residual_spike.jl              # the record
#   julia --startup-file=no --project=scripts scripts/action_residual_spike.jl --quick      # the gate
#   julia --startup-file=no --project=scripts scripts/action_residual_spike.jl --selfcheck
#   julia --startup-file=no --project=scripts scripts/action_residual_spike.jl --sizes 60,120
#
# A variational integrator's slab residual is a gradient of the slab's discrete action `S_h`, and
# its Jacobian a block of the Hessian of `S_h`. This script takes `S_h` from the property suite's
# action oracle, `test/helpers/action_oracle.jl`, for `TensorBSplineAnsatz` under
# `CoefficientRestriction` on `Wave` and `SineGordon`, and sets each slab up with
# `test/helpers/problem_generator.jl`. It compares the AD gradient and Hessian of `S_h` with the
# package's `residual!` and the Newton solver's Jacobian: first for equality, then in time and
# allocations.
#
# Under the restriction the unknowns are the time rows `2:Nt` of the coefficient array `ĉ`, and the
# equations are `∂S_h/∂ĉ` on the rows `1:Nt-1`, as the oracle's `oracle_residual` lays them out.
# The AD therefore differentiates `S_h` in the whole array `ĉ`, of `Nt/(Nt-1)` times the solver's
# `N` entries: the residual is the gradient on the equation rows, and the Jacobian the block of the
# Hessian whose rows are the equation rows and whose columns are the unknown rows.
#
# At each size `N`, before any timing, the AD gradient must equal `residual!` within `100 eps`, and
# the AD Hessian block the solver's Jacobian within `1e-8`, both relative to the largest entry of
# the AD value, in the max norm, at a seeded random vector. A failed check stops the process with
# the size and the error, and that process records no number at that size or after it.
#
# Each backend runs in a fresh process per problem, which this script starts itself, and each
# process prints its Julia version, its thread count and `uptime` before it measures. The first call
# of each function compiles and is not timed; `BenchmarkTools` then gives the median time and the
# allocations of one call. The reverse-mode backend is Enzyme, through DifferentiationInterface;
# where its process fails, Mooncake runs in its place. `--quick` runs the smallest size with one
# sample. `--selfcheck` hands the gradient and the Hessian a perturbed vector, which both checks
# must reject: it exits non-zero when they do.

using BenchmarkTools: @benchmark, allocs, median, memory
using DifferentiationInterface: AutoEnzyme, AutoForwardDiff, AutoMooncake, Constant,
                                SecondOrder, gradient, hessian, prepare_gradient,
                                prepare_hessian
using Enzyme: Enzyme
using GeometricBase: parameters
using GeometricIntegratorsBase: solver
using Mooncake: Mooncake
using Printf: @printf
using Random: Xoshiro
using SimpleSolvers: SimpleSolvers

include(joinpath(@__DIR__, "..", "test", "helpers", "problem_generator.jl"))
include(joinpath(@__DIR__, "..", "test", "helpers", "action_oracle.jl"))

const M = MultisymplecticIntegrators

"""The sizes `N` of the solver's vector the record measures at."""
const TARGETS = (60, 120, 250, 500)

"""The largest relative distance of a mesh's `N` from its target."""
const SIZE_TOLERANCE = 0.15

"""The bound of the gradient check, that of the property suite's gradient property."""
const GRADIENT_BOUND = 100 * eps(Float64)

"""The bound of the Hessian check."""
const HESSIAN_BOUND = 1e-8

"""The seed of the random vector at which the checks and the timings run."""
const VECTOR_SEED = 20261101

"""The shift of every entry of the vector that `--selfcheck` hands the gradient and the Hessian."""
const PERTURBATION = 1e-3

const PROBLEMS = ("Wave", "SineGordon")

"""
The gradient and Hessian backends of each name. The two reverse-mode backends take the Hessian
reverse over ForwardDiff. Enzyme runs with runtime activity, because without it Enzyme stops on the
oracle's action with an `EnzymeRuntimeActivityError`.
"""
const BACKENDS = Dict(
    "ForwardDiff" => (gradient = AutoForwardDiff(), hessian = AutoForwardDiff()),
    "Enzyme" => let enzyme = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
        (gradient = enzyme, hessian = SecondOrder(enzyme, AutoForwardDiff()))
    end,
    "Mooncake" => (gradient = AutoMooncake(),
        hessian = SecondOrder(AutoMooncake(), AutoForwardDiff())))

"""A check of the AD derivatives against the package that failed."""
struct CheckFailure <: Exception
    check::String
    context::String
    error::Float64
    bound::Float64
end

function Base.showerror(io::IO, e::CheckFailure)
    @printf(io, "the %s check failed at %s: relative error %.3e > %.3e",
        e.check, e.context, e.error, e.bound)
end

"""
    check(name, ad, package, bound, context)

The largest entry of `package - ad`, relative to the largest entry of `ad`, which throws a
[`CheckFailure`](@ref) where it is above `bound`. A NaN fails.
"""
function check(name, ad, package, bound, context)
    error = maximum(abs, package - ad) / maximum(abs, ad)
    error ≤ bound || throw(CheckFailure(name, context, error, bound))
    return error
end

function spike_problem(name)
    name == "Wave" &&
        return Wave.problem(timespan = (0.0, 1.0), timestep = 0.1, xstep = 0.01)
    name == "SineGordon" &&
        return SineGordon.problem(timespan = (0.0, 1.0), timestep = 0.1, xstep = 0.01)
    throw(ArgumentError("unknown problem `$name`; the spike runs $(join(PROBLEMS, " and "))"))
end

"""
    spike_entry(problem, ncells)

The problem with the ansatz of the spline registry at `ncells` space cells: degree 2 in time and
space, one time cell, three quadrature nodes per cell and direction, and a `Dirichlet` space basis.
"""
function spike_entry(problem, ncells::Integer)
    return GeneratedProblem(problem, :general, 0, (2, 2), (1, ncells), (3, 3), Dirichlet())
end

"""The length of the solver's vector of the problem at `ncells` space cells, from the oracle."""
function unknowns(problem, ncells::Integer)
    method = entry_method(spike_entry(problem, ncells), CoefficientRestriction())
    return unknown_count(ActionOracle(problem, method))
end

"""
    mesh(problem, target)

The space cell count whose `N` is nearest to `target`, and that `N`. Throws where it is more than
`SIZE_TOLERANCE` away from the target.
"""
function mesh(problem, target::Integer)
    ncells = 1
    while unknowns(problem, ncells) < target
        ncells += 1
    end
    if ncells > 1
        below = unknowns(problem, ncells - 1)
        target - below < unknowns(problem, ncells) - target && (ncells -= 1)
    end
    N = unknowns(problem, ncells)
    abs(N - target) ≤ SIZE_TOLERANCE * target ||
        throw(ArgumentError("no mesh gives N within $(SIZE_TOLERANCE) of $target; the nearest " *
                            "gives $N"))
    return ncells, N
end

"""
    action(z, oracle, datum)

The oracle's discrete action of the slab at the whole coefficient array `ĉ = reshape(z, Nt, Nx, D)`,
with the restriction's load term. The restriction has no multipliers, so none is passed.
"""
function action(z, oracle, datum)
    ĉ = reshape(z, oracle.nt, oracle.nx, oracle.D)
    return slab_action(oracle, datum, ĉ, datum.λ₀, nothing, nothing, nothing)
end

"""
    Slab

One slab of a problem at one mesh: the package's residual and the solver's Jacobian, the oracle's
action and its datum, the vector `θ` at which everything is evaluated, and the indices into the
whole coefficient array of the equation rows `1:Nt-1` and of the unknown rows `2:Nt`, each in the
solver's order.
"""
struct Slab{I, S, A, O, D}
    integrator::I
    solver::S
    arguments::A
    oracle::O
    datum::D
    θ::Vector{Float64}
    equations::Vector{Int}
    unknowns::Vector{Int}
end

function Slab(problem, ncells::Integer)
    slab = first_slab(spike_entry(problem, ncells), CoefficientRestriction())
    oracle = ActionOracle(problem, slab.method)
    N = length(cache(slab.int).x)
    N == unknown_count(oracle) ||
        error("the oracle lays out $(unknown_count(oracle)) unknowns, the solver $N")
    θ = 2 .* rand(Xoshiro(VECTOR_SEED), N) .- 1
    arguments = (current(slab.solstep), parameters(problem), slab.int)
    indices = LinearIndices((oracle.nt, oracle.nx, oracle.D))
    equations = vec(indices[1:(oracle.nt - 1), :, :])
    unknown_rows = vec(indices[2:(oracle.nt), :, :])
    datum = first_datum(oracle)
    return Slab(
        slab.int, solver(slab.int), arguments, oracle, datum, θ, equations, unknown_rows)
end

"""The whole coefficient array of the solver's vector `θ`, as a vector."""
coefficients(slab::Slab, θ) = vec(first(decode(slab.oracle, slab.datum, θ)))

function package_residual!(b, slab::Slab, θ)
    M.residual!(b, θ, nothing, slab.arguments[2], slab.integrator)
    return b
end

function package_jacobian!(slab::Slab, θ)
    SimpleSolvers.jacobian!(slab.solver, θ, slab.arguments)
    return SimpleSolvers.jacobianmatrix(slab.solver)
end

contexts(slab::Slab) = (Constant(slab.oracle), Constant(slab.datum))

"""The residual at `θ` as the AD gradient of the action on the equation rows."""
function ad_residual(slab::Slab, preparation, backend, θ)
    g = gradient(action, preparation.gradient, backend.gradient, coefficients(slab, θ),
        contexts(slab)...)
    return g[slab.equations]
end

"""The Jacobian at `θ` as the block of the AD Hessian of the action: equation rows, unknown rows."""
function ad_jacobian(slab::Slab, preparation, backend, θ)
    H = hessian(action, preparation.hessian, backend.hessian, coefficients(slab, θ),
        contexts(slab)...)
    return H[slab.equations, slab.unknowns]
end

function prepare(slab::Slab, backend)
    z = coefficients(slab, slab.θ)
    return (gradient = prepare_gradient(action, backend.gradient, z, contexts(slab)...),
        hessian = prepare_hessian(action, backend.hessian, z, contexts(slab)...))
end

"""
    checks(slab, backend, preparation, context)

The two equality checks at the slab's vector: the AD gradient against `residual!`, then the AD
Hessian block against the solver's Jacobian. Returns the two relative errors.
"""
function checks(slab::Slab, backend, preparation, context)
    θ = slab.θ
    b = package_residual!(similar(θ), slab, θ)
    g = ad_residual(slab, preparation, backend, θ)
    gradient_error = check("gradient", g, b, GRADIENT_BOUND, context)
    J = copy(package_jacobian!(slab, θ))
    H = ad_jacobian(slab, preparation, backend, θ)
    hessian_error = check("Hessian", H, J, HESSIAN_BOUND, context)
    return gradient_error, hessian_error
end

"""The median time in seconds, the allocations and the bytes of one call, as a named tuple."""
function measured(trial)
    return (time = median(trial).time / 1e9, allocs = allocs(trial), bytes = memory(trial))
end

"""
    timings(slab, backend, preparation, samples)

The package's residual and Jacobian, and the AD residual and Jacobian, each benchmarked after its
first call. One sample each where `samples == 1`, otherwise at most `samples` samples in at most
five seconds each.
"""
function timings(slab::Slab, backend, preparation, samples::Integer)
    θ = slab.θ
    b = similar(θ)
    seconds = samples == 1 ? 1e-3 : 5.0
    residual = measured(@benchmark(package_residual!($b, $slab, $θ),
        samples=samples, evals=1, seconds=seconds))
    jacobian = measured(@benchmark(package_jacobian!($slab, $θ),
        samples=samples, evals=1, seconds=seconds))
    g = measured(@benchmark(ad_residual($slab, $preparation, $backend, $θ),
        samples=samples, evals=1, seconds=seconds))
    H = measured(@benchmark(ad_jacobian($slab, $preparation, $backend, $θ),
        samples=samples, evals=1, seconds=seconds))
    return (; residual, jacobian, gradient = g, hessian = H)
end

"""
    worker(problem_name, backend_name, targets, samples)

One process's work: at each target size, the mesh, the two checks and the timings of one backend on
one problem, printed as one `result` line per size.
"""
function worker(problem_name, backend_name, targets, samples)
    println("== $problem_name, $backend_name, Julia $(VERSION), $(Threads.nthreads()) thread(s)")
    println("uptime: ", strip(read(`uptime`, String)))
    backend = BACKENDS[backend_name]
    problem = spike_problem(problem_name)
    println("result\tproblem\tbackend\tN\tncells\tgradient error\tHessian error\t" *
            "residual s\tresidual allocs\tresidual bytes\tgradient s\tgradient allocs\t" *
            "gradient bytes\tgradient factor\tJacobian s\tJacobian allocs\tJacobian bytes\t" *
            "Hessian s\tHessian allocs\tHessian bytes\tHessian factor")
    for target in targets
        ncells, N = mesh(problem, target)
        slab = Slab(problem, ncells)
        preparation = prepare(slab, backend)
        context = "N = $N ($problem_name, $backend_name)"
        gradient_error, hessian_error = checks(slab, backend, preparation, context)
        t = timings(slab, backend, preparation, samples)
        @printf("result\t%s\t%s\t%d\t%d\t%.2e\t%.2e\t%.3e\t%d\t%d\t%.3e\t%d\t%d\t%.1f\t%.3e\t%d\t%d\t%.3e\t%d\t%d\t%.1f\n",
            problem_name, backend_name, N, ncells, gradient_error, hessian_error,
            t.residual.time, t.residual.allocs, t.residual.bytes, t.gradient.time,
            t.gradient.allocs, t.gradient.bytes, t.gradient.time / t.residual.time,
            t.jacobian.time, t.jacobian.allocs, t.jacobian.bytes, t.hessian.time,
            t.hessian.allocs, t.hessian.bytes, t.hessian.time / t.jacobian.time)
        flush(stdout)
    end
    return 0
end

"""
    selfcheck()

Both checks on `SineGordon` at the smallest size with ForwardDiff: first at the slab's vector, where
they must pass, then with the AD derivatives at a vector shifted by `PERTURBATION`, where each must
fail. Returns 1 where both reject the shifted vector, and 0, with the check that accepted it named,
otherwise.
"""
function selfcheck()
    problem = spike_problem("SineGordon")
    ncells, N = mesh(problem, first(TARGETS))
    slab = Slab(problem, ncells)
    backend = BACKENDS["ForwardDiff"]
    preparation = prepare(slab, backend)
    context = "N = $N (SineGordon, ForwardDiff)"
    checks(slab, backend, preparation, context)
    θ = slab.θ
    perturbed = θ .+ PERTURBATION
    accepted = String[]
    try
        check("gradient", ad_residual(slab, preparation, backend, perturbed),
            package_residual!(similar(θ), slab, θ), GRADIENT_BOUND, context * ", perturbed")
        push!(accepted, "gradient")
    catch e
        e isa CheckFailure || rethrow()
        println(stderr, "selfcheck: ", sprint(showerror, e))
    end
    try
        check("Hessian", ad_jacobian(slab, preparation, backend, perturbed),
            copy(package_jacobian!(slab, θ)), HESSIAN_BOUND, context * ", perturbed")
        push!(accepted, "Hessian")
    catch e
        e isa CheckFailure || rethrow()
        println(stderr, "selfcheck: ", sprint(showerror, e))
    end
    isempty(accepted) && return 1
    println(stderr, "selfcheck: the $(join(accepted, " and ")) check accepts a perturbed vector")
    return 0
end

"""
    run_worker(problem_name, backend_name, arguments)

Start one worker in a fresh process, with its output on this process's, and say whether it
succeeded.
"""
function run_worker(problem_name, backend_name, arguments)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project())
               $(@__FILE__) --worker $problem_name $backend_name $arguments`
    return success(pipeline(command; stdout = stdout, stderr = stderr))
end

"""
    driver(arguments)

One ForwardDiff process and one reverse-mode process per problem. A ForwardDiff failure stops the
run; an Enzyme failure starts Mooncake in its place, and a Mooncake failure leaves ForwardDiff alone.
"""
function driver(arguments)
    for problem_name in PROBLEMS
        if !run_worker(problem_name, "ForwardDiff", arguments)
            println(stderr, "ForwardDiff failed on $problem_name; the run stops")
            return 1
        end
        run_worker(problem_name, "Enzyme", arguments) && continue
        println("Enzyme failed on $problem_name; Mooncake runs in its place")
        run_worker(problem_name, "Mooncake", arguments) && continue
        println("neither reverse-mode backend runs on $problem_name; ForwardDiff alone is measured")
    end
    return 0
end

"""The sizes of `--sizes 60,120`, or every target."""
function sizes(arguments)
    i = findfirst(==("--sizes"), arguments)
    i === nothing && return TARGETS
    return Tuple(parse.(Int, split(arguments[i + 1], ",")))
end

function main(arguments)
    "--selfcheck" in arguments && return selfcheck()
    quick = "--quick" in arguments
    targets = quick ? (first(TARGETS),) : sizes(arguments)
    samples = quick ? 1 : 10_000
    i = findfirst(==("--worker"), arguments)
    if i !== nothing
        try
            return worker(arguments[i + 1], arguments[i + 2], targets, samples)
        catch e
            println(stderr, "worker failed: ", sprint(showerror, e))
            return 1
        end
    end
    forwarded = quick ? ["--quick"] : ["--sizes", join(targets, ",")]
    return driver(forwarded)
end

exit(main(ARGS))
