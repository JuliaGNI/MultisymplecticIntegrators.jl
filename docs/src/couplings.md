```@meta
CurrentModule = MultisymplecticIntegrators
```

# Couplings and initial guesses

The couplings that close a slab's system, and the guess that seeds its solve: a target, the four
targets this package ships, and the fits that bring an ansatz to one of them.

```@autodocs
Modules = [MultisymplecticIntegrators]
Pages = ["src/couplings/coefficient_restriction.jl", "src/couplings/weak_multipliers.jl",
    "src/initial_guess/fit_methods.jl", "src/initial_guess/guess.jl",
    "src/initial_guess/orthogonal_greedy.jl", "src/initial_guess/trained_hidden_layer.jl"]
```
