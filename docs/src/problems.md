```@meta
CurrentModule = MultisymplecticIntegrators
```

# Example problems and diagnostics

The problem layer: the Lagrangian density, the problem type and the data a slab of a problem hands
out. Then the four example problems, each of which lives in its own submodule, and the figures of a
run.

```@autodocs
Modules = [MultisymplecticIntegrators]
Pages = ["src/problems/lagrangian_density.jl", "src/problems/lagrangian_problem.jl"]
```

```@autodocs
Modules = [MultisymplecticIntegrators.Wave, MultisymplecticIntegrators.VectorWave,
    MultisymplecticIntegrators.SineGordon, MultisymplecticIntegrators.LinearTransport,
    MultisymplecticIntegrators.Diagnostics]
```
