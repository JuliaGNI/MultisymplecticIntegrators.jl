```@meta
CurrentModule = MultisymplecticIntegrators
```

# The ELM family

The extreme learning machine of `ELM_PDE_int` is a [`NetworkAnsatz`](@ref) in `:last_layer` mode
under [`WeakInitialAndBoundaryConditions`](@ref), whose hidden layer comes from the initial guess:

- the random-feature hidden layer, with the output layer solved, is `mode = :last_layer`: the step
  is affine in the output weights;
- the LSGD training of the hidden layers is the fit method [`TrainedHiddenLayer`](@ref): every
  epoch takes one gradient-descent step on the hidden weights and biases, and then reads the output
  weights by least squares;
- the direct least-squares branch is the direct linear solve `linear_solver = true` on the affine
  step of a linear density: it factorises the Jacobian at the first step, and again only at a step
  whose guess has moved the hidden layer.

`TrainedHiddenLayer` draws its random features from its own generator, `rng = Xoshiro(1234)` by
default. The run may fail, so the example prints its stats, and nothing reads them.

```@example elm
using MultisymplecticIntegrators

problem = Wave.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05)
ansatz = NetworkAnsatz(S = 4, mode = :last_layer, xspan = (0.0, 1.0), degree = 2,
    ncells = (2, 4))
method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
    initial_guess = FitTo(PreviousSlab(); method = TrainedHiddenLayer(epochs = 5)))
result = integrate(problem, method)
result.stats
```

`epochs = 0` fits the output layer alone on the hidden layer that the generator draws. The same
method with the direct linear solve:

```@example elm
method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
    initial_guess = FitTo(PreviousSlab(); method = TrainedHiddenLayer(epochs = 5)),
    linear_solver = true)
result = integrate(problem, method)
result.stats
```
