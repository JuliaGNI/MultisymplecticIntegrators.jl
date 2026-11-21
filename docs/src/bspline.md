```@meta
CurrentModule = MultisymplecticIntegrators
```

# The B-spline family

A [`SpaceTimeGalerkin`](@ref) method with a [`TensorBSplineAnsatz`](@ref) discretises the field of
one slab by a tensor product of B-splines in the normalised time `t̂` and in `x`. Three couplings
join consecutive slabs:

| coupling | the slab's data | the space basis below | at `294f023` |
|:--|:--|:--|:--|
| [`CoefficientRestriction`](@ref) | the bottom coefficient row is fixed, the momentum flux is carried | `Dirichlet()` | `Galerkin_Full_Restriction_Bspline_Integrator`, `SpaceTime_Spline_Wave_Integrator`, `FEM_Multisymplectic_Integrator` |
| [`WeakInitialCondition`](@ref) | the initial condition by multipliers | `Dirichlet()` | `Galerkin_Dirichlet_Bspline_Integrator` |
| [`WeakInitialAndBoundaryConditions`](@ref) | the initial and the boundary conditions by multipliers | `Free()` | `Galerkin_Bspline_Integrator` |

At degree 1 under `CoefficientRestriction` the step is the bilinear finite-element scheme of the
old FEM integrator. The step is a one-step form, so it needs no start-up, and its velocity is the
ansatz's own `∂u/∂t̂ / h`. The space basis' boundary condition is SimpleSplines' `Dirichlet()` or
`Free()`, and SimpleSplines counts the **degree** of a basis, not its order.

## The three couplings

Every example on this page integrates `Wave` for two steps of `h = 0.1` on twenty space cells.

```@example bspline
using MultisymplecticIntegrators
using SimpleSplines: Dirichlet, Free

problem = Wave.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05)
ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0),
    x_boundary = Dirichlet())
result = integrate(problem, SpaceTimeGalerkin(ansatz, CoefficientRestriction()))
result.stats
```

```@example bspline
result = integrate(problem, SpaceTimeGalerkin(ansatz, WeakInitialCondition()))
result.stats
```

```@example bspline
free = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0),
    x_boundary = Free())
result = integrate(problem, SpaceTimeGalerkin(free, WeakInitialAndBoundaryConditions()))
result.stats
```

Both weak couplings take the keyword `multiplier_basis`: `:bspline`, the ansatz's own trace basis,
the default, or `:lagrange`, the nodal basis of the same set of functions.

```@example bspline
result = integrate(problem,
    SpaceTimeGalerkin(ansatz, WeakInitialCondition(multiplier_basis = :lagrange)))
result.stats
```

`result` is the named tuple `(; solution, traces, stats)` of [`integrate`](@ref): the
`GeometricSolution` on the output grid, the top trace of every step, and the record of every
step's Newton solve.

## The lifting on a Dirichlet ansatz

[`HardConstraintLifting`](@ref), the coupling of the TrialNN family, also takes a B-spline ansatz
whose space basis is `Dirichlet()`. The slab's field is `P[N] + ℓ`: the ansatz's field `N`
projected so that it vanishes at both ends of `xspan` and at the bottom of the slab, plus the
transfinite interpolant `ℓ` of the slab's data. A Dirichlet basis makes `N` zero at the two ends,
but not at the bottom, so the projection still subtracts the bottom trace of `N`. It samples that
trace at the space nodes of the ansatz's quadrature rule and subtracts the linear blend of the
samples between two nodes, so `P[N]` vanishes on the bottom at those nodes, not between them. The
bottom datum is stored at the same nodes, as the [TrialNN page](trialnn.md) describes. No momentum
is carried. The call below does not converge on this page's problem: the slab's Jacobian is
singular to round-off, as K77 of `KNOWN_ISSUES.md` records, so the call is shown and not run. The
lifting's tests record this arm's solves and do not assert that they converge.

```julia
result = integrate(problem, SpaceTimeGalerkin(ansatz, HardConstraintLifting()))
```

## The direct linear solve

For an ansatz whose step is affine in its unknowns and a linear density, such as `Wave`'s,
`linear_solver = true` replaces Newton's iterations by two solves per step with one factorisation
of the slab's Jacobian, taken at the run's first step.

```@example bspline
result = integrate(problem,
    SpaceTimeGalerkin(ansatz, CoefficientRestriction(); linear_solver = true))
result.stats
```

## The stability bound

The B-spline slab integrator on the wave equation is stable only up to about `c·h/Δx ≤ 1`, with
`c` the wave speed, `h` the time step and `Δx` the width of a space cell. Above the bound the high
space modes grow from round-off, while every step still converges, so a short run looks right. At
degree 1 the step is the Newmark scheme with `β = 1/6` and `γ = 1/2`. The examples above run
`Wave` (`c = 0.5`) at `h = 0.1` and `Δx = 0.05`, which is `c·h/Δx = 1`, at the bound. The
measurements are in K62 of `KNOWN_ISSUES.md`; the package has no stable time rule.
