# The entry points of the inference and JET checks: ten methods — nine of the ten pairs of an
# ansatz and a coupling that the slab integrator accepts, with a network's two modes counted apart,
# all but the `:last_layer` network under the hard-constraint lifting, and the box scheme — each
# built as a `GeometricIntegrator` at `T ∈ (Float64, Float32)` and `D ∈ (1, 2)`, unforced and
# forced. The unforced problem is `Wave.problem` for `D = 1` and `VectorWave.problem` for `D = 2`;
# the forced one is the manufactured problem of `manufactured.jl`, whose forcing returns a real
# number at `D = 1` and one value per component at `D = 2`, the two forms a forcing may take. Every
# method is built at a small size, because the checks read types, not values.

using MultisymplecticIntegrators
using SimpleSplines: Dirichlet, Free
using Symbolics: Num, @variables
using GeometricIntegratorsBase: GeometricIntegrator, cache, current, nlsolution,
                                solutionstep
using GeometricBase: parameters
using GeometricEquations: problem as gproblem
using GeometricSolutions: GeometricSolution

include("manufactured.jl")

const QUALITY_TYPES = (Float64, Float32)
const QUALITY_COMPONENTS = (1, 2)
const QUALITY_FORCED = (false, true)

function quality_problem(T, D)
    kwargs = (T = T, timespan = (zero(T), T(0.2)), timestep = T(0.1), xstep = T(0.05))
    D == 1 && return Wave.problem(; kwargs...)
    D == 2 && return VectorWave.problem(; kwargs...)
    throw(ArgumentError("the quality checks run D = 1 and D = 2, got D = $(D)"))
end

function quality_spline(T, x_boundary)
    TensorBSplineAnsatz(degree = 2, ncells = (2, 10), xspan = (zero(T), one(T)),
        x_boundary = x_boundary)
end

@variables quality_τ quality_ξ
@variables quality_q[1:2] quality_s[1:4]

function quality_symbolic(T, D)
    τ, ξ, q, s = quality_τ, quality_ξ, quality_q, quality_s
    D == 1 && return SymbolicAnsatz(q[1] * sin(q[2] * ξ - τ), τ, ξ, collect(q);
        xspan = (zero(T), one(T)), start = T[1, 1], degree = 2, ncells = (2, 4))
    return SymbolicAnsatz(Num[s[1] * sin(s[2] * ξ - τ), s[3] * sin(s[4] * ξ - τ)], τ, ξ,
        collect(s); xspan = (zero(T), one(T)), start = T[1, 1, 1, 1], degree = 2,
        ncells = (2, 4))
end

function quality_network(T, D, mode)
    NetworkAnsatz(
        S = 4, mode = mode, xspan = (zero(T), one(T)), degree = 2, ncells = (2, 4),
        D = D)
end

function quality_fixed_boundary(T, D)
    FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (zero(T), one(T)), ncells = 8, D = D)
end

# Each method by name, as a function of `T` and `D` that builds it.
const QUALITY_METHODS = [
    "B-spline ansatz, coefficient restriction" =>
        (T, D) -> SpaceTimeGalerkin(
            quality_spline(T, Dirichlet()), CoefficientRestriction()),
    "B-spline ansatz, weak initial condition" =>
        (T, D) -> SpaceTimeGalerkin(
            quality_spline(T, Dirichlet()), WeakInitialCondition()),
    "B-spline ansatz, weak initial and boundary conditions" =>
        (T, D) -> SpaceTimeGalerkin(
            quality_spline(T, Free()), WeakInitialAndBoundaryConditions()),
    "B-spline ansatz, hard constraint lifting" =>
        (T, D) -> SpaceTimeGalerkin(
            quality_spline(T, Dirichlet()), HardConstraintLifting()),
    "symbolic ansatz, weak initial and boundary conditions" =>
        (T, D) -> SpaceTimeGalerkin(
            quality_symbolic(T, D), WeakInitialAndBoundaryConditions()),
    "network ansatz :all, weak initial and boundary conditions" =>
        (T, D) -> SpaceTimeGalerkin(
            quality_network(T, D, :all), WeakInitialAndBoundaryConditions()),
    "network ansatz :last_layer, weak initial and boundary conditions" =>
        (T, D) -> SpaceTimeGalerkin(
            quality_network(T, D, :last_layer), WeakInitialAndBoundaryConditions()),
    "network ansatz :all, hard constraint lifting" =>
        (T, D) -> SpaceTimeGalerkin(
            quality_network(T, D, :all), HardConstraintLifting()),
    "fixed-boundary network, fixed boundary" =>
        (T, D) -> SpaceTimeGalerkin(
            quality_fixed_boundary(T, D), FixedBoundary()),
    "box scheme" => (T, D) -> BoxScheme()
]

"""
    quality_forced_problem(T, D)

The manufactured problem with `D` components in the element type `T`, on the spans and steps of
`quality_problem`.
"""
function quality_forced_problem(T, D)
    manufactured_problem(; D = D, timestep = T(0.1), nx = 20, T = T, tend = 0.2)
end

"""
    quality_entry(make, T, D, forced)

The integrator of the method that `make(T, D)` builds on the problem of `T` and `D`, forced or
not, and the arguments `(b, x, sol, params, int)` of its `residual!`: `x` is a copy of the cache's
nonlinear solution and `b` a buffer of its shape. `sol` is `nothing` on the unforced problem, which
does not read it, and the current state of a solution step on the forced one, whose forcing reads
its time.
"""
function quality_entry(make, T, D, forced::Bool)
    problem = forced ? quality_forced_problem(T, D) : quality_problem(T, D)
    int = GeometricIntegrator(problem, make(T, D))
    sol = forced ? current(solutionstep(int, GeometricSolution(problem)[0])) : nothing
    x = copy(nlsolution(cache(int)))
    b = similar(x)
    return int, (b, x, sol, parameters(gproblem(int)), int)
end
