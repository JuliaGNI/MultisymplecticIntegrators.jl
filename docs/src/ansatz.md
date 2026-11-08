```@meta
CurrentModule = MultisymplecticIntegrators
```

# The ansatz families

The space-time ansätze: the tensor B-spline basis, the network and the symbolic field. Each keeps
its own block in the slab cache and supplies the ten seam functions that the page *The slab seam*
describes.

```@autodocs
Modules = [MultisymplecticIntegrators]
Pages = ["src/ansatz/network.jl", "src/ansatz/symbolic.jl", "src/ansatz/tensor_bspline.jl"]
```
