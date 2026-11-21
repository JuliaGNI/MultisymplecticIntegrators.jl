```@meta
CurrentModule = MultisymplecticIntegrators
```

# The initial guess

Each slab starts its solve from a guess, the `initial_guess` keyword of
[`SpaceTimeGalerkin`](@ref). A guess is one of two things:

- [`Continuation`](@ref), **the default**: the previous slab's solution, carried into the new slab.
  At the first slab there is nothing to carry: a B-spline or network ansatz falls back to
  `FitTo(InitialCondition())` with the ansatz's [`default_fit`](@ref), a symbolic ansatz takes its
  `start` vector, and a fixed-boundary network starts at the slab's datum.
- [`FitTo`](@ref)`(target; method)`: fit the ansatz to a target on the slab with a fit method. An
  unnamed method resolves through `default_fit(ansatz)`.

The examples on this page integrate `Wave` for two steps, with the B-spline ansatz of the
[B-spline family](bspline.md) unless a fit method needs a network. A run whose method may fail shows
its stats, and nothing reads them.

```@example guess
using MultisymplecticIntegrators
using SimpleSplines: Dirichlet

problem = Wave.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05)
ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0),
    x_boundary = Dirichlet())
guessed(guess) = integrate(problem,
    SpaceTimeGalerkin(ansatz, CoefficientRestriction(); initial_guess = guess)).stats
guessed(Continuation())
```

## The targets

A target is a [`SlabTarget`](@ref), which [`evaluate!`](@ref) samples at the points of the slab's
quadrature.

| target | gives |
|:--|:--|
| [`InitialCondition`](@ref) | the problem's initial condition `u₀(x)`, constant in time |
| [`PreviousSlab`](@ref) | the previous slab's solution, extrapolated beyond its top; the initial condition at the first slab |
| [`ExactSolution`](@ref) | the problem's exact solution at the slab's physical time; refused at construction for a problem without one |
| [`TargetFunction`](@ref) | a user function `f(t, x, params)` in physical time |
| [`Predictor`](@ref) | a Störmer–Verlet run of the slab; a target of a [`FixedBoundaryNetwork`](@ref) alone |

```@example guess
guessed(FitTo(InitialCondition()))
```

```@example guess
guessed(FitTo(PreviousSlab()))
```

```@example guess
guessed(FitTo(ExactSolution()))
```

```@example guess
guessed(FitTo(TargetFunction((t, x, params) -> sinpi(x) * cos(t))))
```

`Predictor()` is shown with the fit method of its ansatz, `BubbleGreedy`, below.

## The fit methods

A fit method is a [`SlabFitMethod`](@ref).

| fit method | for |
|:--|:--|
| [`Projection`](@ref) | an ansatz linear in its coefficients: the `L²` projection on the slab's quadrature |
| [`OrthogonalGreedy`](@ref) | a network: hidden units chosen from a dictionary, output weights by least squares |
| [`TrainedHiddenLayer`](@ref) | a network in `:last_layer` mode: gradient descent on the hidden layer, output weights by least squares |
| [`PINNTraining`](@ref) | a network: gradient descent on every free parameter against a sampled target |
| [`BubbleGreedy`](@ref) | a [`FixedBoundaryNetwork`](@ref): the level and the bubble fitted to the target |

`TrainedHiddenLayer` and `PINNTraining` draw from their own generators, `Xoshiro(1234)` by default,
never from the global one.

```@example guess
guessed(FitTo(InitialCondition(); method = Projection()))
```

```@example guess
network(mode) = NetworkAnsatz(S = 4, mode = mode, xspan = (0.0, 1.0), degree = 2,
    ncells = (2, 4))
fitted(ansatz, method) = integrate(problem,
    SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
        initial_guess = FitTo(InitialCondition(); method))).stats
fitted(network(:all), OrthogonalGreedy(Nw = 10, Nb = 10))
```

```@example guess
fitted(network(:last_layer), TrainedHiddenLayer(epochs = 5))
```

```@example guess
fitted(network(:all), PINNTraining(epochs = 10))
```

```@example guess
boundary = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0), ncells = 8)
integrate(problem, SpaceTimeGalerkin(boundary, FixedBoundary();
    initial_guess = FitTo(Predictor(); method = BubbleGreedy()))).stats
```

## A target of your own

A user-defined target is a subtype of `SlabTarget` with one method of the unexported
`MultisymplecticIntegrators.evaluate!(sample, target, cache, ansatz, problem, sol)`, which fills
`sample`, of size `(Qt, Qx, D)`, with the target's value at the slab's quadrature points. Here the
quadrature points come from the ansatz's block in the cache; the target is a standing profile.

```@example guess
struct StandingProfile <: SlabTarget end

function MultisymplecticIntegrators.evaluate!(sample::AbstractArray{ST, 3}, ::StandingProfile,
        cache, ansatz::TensorBSplineAnsatz, problem, sol) where {ST}
    tnodes, xnodes = MultisymplecticIntegrators.quadrature_points(cache.ansatz)
    for d in axes(sample, 3), r in eachindex(xnodes), q in eachindex(tnodes)
        sample[q, r, d] = sinpi(xnodes[r])
    end
    return nothing
end

guessed(FitTo(StandingProfile(); method = Projection()))
```

## A fit method of your own

A user-defined fit method is a subtype of `SlabFitMethod` with one method of the unexported

```julia
MultisymplecticIntegrators.fit!(coeffs, method, ansatz, sample, cache)
```

which places the coefficients `coeffs` — of size `(Nt, Nx, D)` for a tensor ansatz, and the
parameter vector for a network — from the sampled target `sample`. `cache` is the slab's
[`SpaceTimeGalerkinCache`](@ref), which a fit may read. The contract is stated with
[`Projection`](@ref).
