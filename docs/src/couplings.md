```@meta
CurrentModule = MultisymplecticIntegrators
```

# Couplings and initial guesses

The couplings that close a slab's system, and the guess that seeds its solve: a target, the four
targets this package ships, and the fits that bring an ansatz to one of them.

## The hard-constraint lifting

`HardConstraintLifting` is the coupling of the network family that imposes a slab's data on the
field itself. The field of the slab is `P[N] + ℓ`: the transfinite projection of the ansatz's field
that vanishes at the two ends of `xspan` and at the bottom of the slab, plus the transfinite
interpolant `ℓ` of the slab's data — the problem's boundary function at the two ends, and the
initial condition's bottom on the first slab and the previous slab's top afterwards. The initial and
boundary conditions are therefore met by construction and the interior weak form alone is the
residual.

The slab carries **no momentum** between steps: the coupling adds no surface term and no load at the
bottom of the slab, and `coupling_unknowns` is zero, so the solver's vector is the ansatz's own. That
is this coupling's formulation, kept on purpose; a formulation that carries the incoming momentum
at the bottom of each slab is not this one.

A `NetworkAnsatz` and a `TensorBSplineAnsatz` are both accepted, and an ansatz of any other kind is
refused when the integrator is built. For a network the transform projects the field's derivative
with respect to every free parameter together with the field, so the residual is the gradient of the
slab's discrete action on the free parameters; for a coefficient ansatz there is no parameter
derivative and the transform places the data in the field alone, which needs
`x_boundary = Dirichlet()`: the transform reads the field of the slab's two ends at the time nodes
of the residual, so a basis that is free there is refused.

`SimpleSplines` counts the **degree** of a basis and not its **order**, so a trace basis of order
`4` is built with `degree = 3`.

## The fits

`PINNTraining` descends a network's free parameters on its misfit against a sampled target, with a
fixed step and no optimiser package: the sample's points are drawn once per fit from a copy of the
method's generator, so two fits that share one method object draw the same points, each pass over
them takes one step per batch of `batch` points, and the gradient is the parameter derivative the
ansatz's block already carries.

```@autodocs
Modules = [MultisymplecticIntegrators]
Pages = ["src/couplings/coefficient_restriction.jl", "src/couplings/weak_multipliers.jl",
    "src/couplings/hard_constraint_lifting.jl",
    "src/initial_guess/fit_methods.jl", "src/initial_guess/guess.jl",
    "src/initial_guess/orthogonal_greedy.jl", "src/initial_guess/trained_hidden_layer.jl"]
```
