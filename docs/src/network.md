```@meta
CurrentModule = MultisymplecticIntegrators
```

# The multiplier network

A [`NetworkAnsatz`](@ref) is a two-layer network of `t̂` and `x` with `S` hidden units. Under
[`WeakInitialAndBoundaryConditions`](@ref) the slab's initial and boundary conditions are imposed by
multipliers on the ansatz's trace basis, a B-spline basis built from `degree` and `ncells`. This is
the family of the old `NN_PDE_Integrator` and `NN_PDE_Integrator_Symbolic`; their `:Fully` and
`:Partially` variants are the two values of `mode`:

- `mode = :all` frees every parameter of the network, and the step is a nonlinear system;
- `mode = :last_layer` frees the output weights alone and keeps the hidden layer where the initial
  guess put it, so the step is affine in its unknowns.

The solve of a network step may fail, so each example prints the run's stats, and nothing on this
page reads them or claims that a step converges.

```@example network
using MultisymplecticIntegrators

problem = Wave.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05)
ansatz = NetworkAnsatz(S = 4, mode = :all, xspan = (0.0, 1.0), degree = 2, ncells = (2, 4))
result = integrate(problem, SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions()))
result.stats
```

```@example network
last = NetworkAnsatz(S = 4, mode = :last_layer, xspan = (0.0, 1.0), degree = 2,
    ncells = (2, 4))
result = integrate(problem, SpaceTimeGalerkin(last, WeakInitialAndBoundaryConditions()))
result.stats
```

## The fits

A network is not linear in its parameters, so a [`Projection`](@ref) is refused on it. Its
[`default_fit`](@ref) is [`OrthogonalGreedy`](@ref), the greedy fit of the old `OGA2D` seed, which
[`Continuation`](@ref) uses at the first slab. [`TrainedHiddenLayer`](@ref) trains the hidden
layer of a `:last_layer` network (the [ELM family](elm.md)), and [`PINNTraining`](@ref) descends
every free parameter (the [TrialNN family](trialnn.md)).

```@example network
method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
    initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy(Nw = 10, Nb = 10)))
result = integrate(problem, method)
result.stats
```

## The regularisation factor

The solver option `regularization_factor = λ`, passed to [`integrate`](@ref) beside the method,
makes SimpleSolvers' `Newton` add `λ`, unscaled, to the diagonal of the slab Jacobian before each
factorisation. It is a shift of the Newton matrix, not a Tikhonov term: the residual that a step
drives to zero is unchanged. It has two exceptions:

- the direct linear solve of `linear_solver = true` factorises the Jacobian without the shift, so
  `λ` has no effect there;
- the fixed-boundary network's [`LevenbergMarquardt`](@ref) solver does not read it.

```@example network
result = integrate(problem, SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions());
    regularization_factor = 1e-8)
result.stats
```

The regularisation scan of `scripts/network_regularisation_model.jl` runs both modes on `Wave` at
four values of `λ`, as rows of the benchmark matrix.
