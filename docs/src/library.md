```@meta
CurrentModule = MultisymplecticIntegrators
```

# MultisymplecticIntegrators Library Functions

The abstract types, the weak form and the slab core with its integrator shell and its diagnostics.

The manual is split by source layer, so that no generated page grows without bound: this page holds
`methods.jl`, `core/` and `diagnostics.jl`; the page *The ansatz families* holds `ansatz/`;
*Couplings and initial guesses* holds `couplings/` and `initial_guess/`; and *Example problems and
diagnostics* holds the problem layer, the submodules of the four example problems and the figures.
The page *The slab seam* says how the parts of the slab integrator fit together.

Each block that documents the top-level module names the source files of its layer, and Documenter
matches a file by the end of its path, so a source file that no block names leaves its docstrings
out of the manual, which the build reports.

```@autodocs
Modules = [MultisymplecticIntegrators]
Pages = ["src/methods.jl", "src/core/integrate.jl", "src/core/slab_integrator.jl",
    "src/core/weak_form.jl", "src/diagnostics.jl"]
```
