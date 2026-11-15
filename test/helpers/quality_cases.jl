# The entry points of the inference and JET checks: the ten methods that the package constructs —
# the nine pairs of an ansatz and a coupling that the slab integrator accepts, and the box scheme —
# each built as a `GeometricIntegrator` at `T ∈ (Float64, Float32)` and `D ∈ (1, 2)`, on
# `Wave.problem` for `D = 1` and on `VectorWave.problem` for `D = 2`. Every method is built at
# a small size, because the checks read types, not values.

using MultisymplecticIntegrators
using SimpleSplines: Dirichlet, Free
using Symbolics: Num, @variables
using GeometricIntegratorsBase: GeometricIntegrator, cache, nlsolution
using GeometricBase: parameters
using GeometricEquations: problem as gproblem

const QUALITY_TYPES = (Float64, Float32)
const QUALITY_COMPONENTS = (1, 2)

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
    quality_entry(make, T, D)

The integrator of the method that `make(T, D)` builds on the problem of `T` and `D`, and the
arguments `(b, x, nothing, params, int)` of its `residual!`: `x` is a copy of the cache's
nonlinear solution and `b` a buffer of its shape.
"""
function quality_entry(make, T, D)
    int = GeometricIntegrator(quality_problem(T, D), make(T, D))
    x = copy(nlsolution(cache(int)))
    b = similar(x)
    return int, (b, x, nothing, parameters(gproblem(int)), int)
end
