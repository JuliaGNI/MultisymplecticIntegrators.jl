```@meta
CurrentModule = MultisymplecticIntegrators
```

# The fixed-boundary network

A [`FixedBoundaryNetwork`](@ref) is a second network method beside the multiplier network: the
configuration of one time level is the parameter vector of a spatial network that carries the
problem's side data, and the slab between two levels carries a bubble of `St` hidden units. There
are no multipliers. The coupling [`FixedBoundary`](@ref) carries the previous level's parameters
and the momentum that level received, and hands the momentum of the solved level on; the ansatz
takes this coupling and nothing else.

The default solver of the method is SimpleSolvers' `Newton()`, which takes the minimum-norm step of
a pivoted QR of the slab Jacobian cut at the family's [`fixed_boundary_qr_cut`](@ref). A caller may
name [`LevenbergMarquardt`](@ref) instead, which damps the step by a shift on the singular values of
the Jacobian; `LevenbergMarquardt(variable_projection = true)` eliminates the bubble's output weights
in an inner solve.

The default guess, [`Continuation`](@ref), starts the solved level at the previous level's
parameters and the bubble at the previous slab's, and at the first slab at the datum and at
[`bubble_start`](@ref). A `FitTo` fits with the ansatz's [`default_fit`](@ref),
[`BubbleGreedy`](@ref), the one fit method a fixed-boundary network takes. Its target
[`Predictor`](@ref) is a Störmer–Verlet run of the slab on a uniform grid: the guess fits the level
to the predictor's top, and the bubble to what the predictor has beyond the interpolation of the two
levels.

The solve may fail, so each example prints the run's stats and claims nothing about them.

```@example fixed_boundary
using MultisymplecticIntegrators

problem = Wave.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05)
ansatz = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0), ncells = 8)
result = integrate(problem, SpaceTimeGalerkin(ansatz, FixedBoundary()))
result.stats
```

```@example fixed_boundary
method = SpaceTimeGalerkin(ansatz, FixedBoundary();
    initial_guess = FitTo(Predictor(); method = BubbleGreedy()))
result = integrate(problem, method)
result.stats
```
