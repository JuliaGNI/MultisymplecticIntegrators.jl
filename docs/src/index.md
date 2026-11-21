```@meta
CurrentModule = MultisymplecticIntegrators
```

# MultisymplecticIntegrators

Documentation for [MultisymplecticIntegrators](https://github.com/JuliaGNI/MultisymplecticIntegrators.jl).

Multi-symplectic schemes for Lagrangian partial differential equations in one space dimension:
the Lagrangian problems with their exact solutions, the space-time Galerkin slab integrator, and
the shell that drives an integrator over the slabs of the time span.

## The manual

- The families, one page each, with a running example: [the B-spline family](bspline.md) under
  its three couplings, [the multiplier network](network.md), [the ELM family](elm.md),
  [the TrialNN family](trialnn.md), [the symbolic family](symbolic.md),
  [the fixed-boundary network](fixed_boundary.md) and [the box scheme](box_scheme.md), the
  baseline.
- [The initial guess](guess.md): the default, the targets and the fit methods, and how to add
  your own.
- [Verification](verification.md): the convergence tables, the long-time energy and the
  stability bound.
- [From the old stack](inventory.md): every feature of `294f023`, with its test and its page.
- The reference: [the slab seam](seam.md), the [library](library.md),
  [the ansatz families](ansatz.md), [the couplings and initial guesses](couplings.md), and
  [the example problems and diagnostics](problems.md).

```@index
```
