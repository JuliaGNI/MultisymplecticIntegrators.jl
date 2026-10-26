# Known issues

Defects found in review and recorded, not fixed. Each entry names its kind and its evidence.

### K6 · The driver scripts still name the removed integrator families.

- location: `scripts/`
- evidence: The package no longer holds the old integrators, and `Project.toml` no longer holds
  their dependencies. The scripts under `scripts/` still call `Galerkin_Bspline_Integrator`,
  `Galerkin_Dirichlet_Bspline_Integrator`, `Galerkin_Full_Restriction_Bspline_Integrator`,
  `FEM_Multisymplectic_Integrator`, `SpaceTime_Spline_Wave_Integrator`, `NN_PDE_Integrator`,
  `NN_PDE_Integrator_Symbolic`, `ELM_PDE_int`, `TrialNN_PDE_int`, `Sindy_PDE_Integrator`,
  `PDEIntegrator` and `OGA2D`, and they still import `BSplineKit`, `Plots`, `Zygote`,
  `Infiltrator` and `GeometricMachineLearning`; `grep -rnw 'Galerkin_Bspline_Integrator' scripts/`
  names them. They cannot run until the family each one drives returns, and `scripts/` gets its
  own environment and a registry in the part that owns that environment.
- kind: defect
- found: 2026-10-26

### K7 · The three problem constructors repeat themselves.

- location: `src/problems/wave.jl`, `src/problems/sine_gordon.jl`,
  `src/problems/linear_transport.jl`
- evidence: each `problem` function converts its spans, builds the two closures and calls
  `LagrangianPDEProblem` in the same way, about twenty lines three times. One shared constructor
  taking the three exact functions would remove about forty lines. Recorded rather than done,
  because the shared form has to be chosen against the boundary interface that the slab core
  defines.
- kind: defect
- found: 2026-10-26
