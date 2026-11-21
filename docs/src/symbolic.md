```@meta
CurrentModule = MultisymplecticIntegrators
```

# The symbolic family

A [`SymbolicAnsatz`](@ref) is a closed-form expression of the normalised time `t̂`, of `x` and of a
parameter vector `p`, written with Symbolics. The parameters are the slab's unknowns, and
[`WeakInitialAndBoundaryConditions`](@ref) is the one coupling the ansatz takes. The hand-built
symbolic network of the old stack and the SINDy integrator, `Sindy_PDE_Integrator`, are both such
an expression.

The time of the expression is the slab's normalised time, so an expression in physical time is
substituted before it is given. `start` is the parameter vector of the first slab's guess, and each
later slab starts from the previous slab's parameters. The trace basis, built from `degree` and
`ncells`, carries the coupling's multipliers and does not constrain the field.

The solve may fail, so the example prints the run's stats and claims nothing about them.

```@example symbolic
using MultisymplecticIntegrators
using Symbolics: @variables

problem = Wave.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05)
@variables τ ξ q[1:2]
ansatz = SymbolicAnsatz(q[1] * sin(q[2] * ξ - τ), τ, ξ, collect(q); xspan = (0.0, 1.0),
    start = [1.0, 1.0], degree = 2, ncells = (2, 4))
result = integrate(problem, SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions()))
result.stats
```
