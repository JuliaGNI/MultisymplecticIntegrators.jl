```@meta
CurrentModule = MultisymplecticIntegrators
```

# The TrialNN family

The trial-solution network of `TrialNN_PDE_int` is a [`NetworkAnsatz`](@ref) under
[`HardConstraintLifting`](@ref). The slab's field is the network's field, projected so that it
vanishes at both ends of `xspan` and at the bottom of the slab, plus a transfinite interpolant of
the slab's data: the problem's boundary function at the two ends, and the initial condition at the
first slab and the previous slab's top after it. The slab's interior weak form alone is the
residual.

The boundary function is met at the two ends at every time. The bottom datum is stored at the space
nodes of the ansatz's quadrature rule and is met at those nodes. Between two nodes the interpolant
takes the linear blend of their two values, and outside the outer nodes it takes the boundary
function's value at that end at the slab's start.

The family carries **no momentum** between slabs, on purpose: the coupling has no unknowns of its
own, no surface term and no load at the bottom of a slab.

The old PINN pre-training branch and `PINN_for_init.jl` are the fit method [`PINNTraining`](@ref),
and the old `OGA2D` seed is [`OrthogonalGreedy`](@ref). `PINNTraining` draws its sample points from
its own generator, `rng = Xoshiro(1234)` by default. Each example prints the run's stats and claims
nothing about them.

```@example trialnn
using MultisymplecticIntegrators

problem = Wave.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05)
ansatz = NetworkAnsatz(S = 4, mode = :all, xspan = (0.0, 1.0), degree = 2, ncells = (2, 4))
method = SpaceTimeGalerkin(ansatz, HardConstraintLifting();
    initial_guess = FitTo(PreviousSlab(); method = PINNTraining()))
result = integrate(problem, method)
result.stats
```

```@example trialnn
method = SpaceTimeGalerkin(ansatz, HardConstraintLifting();
    initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy(Nw = 10, Nb = 10)))
result = integrate(problem, method)
result.stats
```
