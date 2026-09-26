# Known issues

Defects found in review and recorded, not fixed. Each entry names its kind and its evidence.

## KI-1 · docs · `CHANGELOG.md` *Open Issues* names the old test files

The `## Open Issues` section still says that `Pkg.test()` is red (21 pass, 2 fail, 1 error), and it
cites `test_full_multiplier_momentum_carry.jl:37/:40` and `test_fem_multisymplectic_wave.jl:61`. The
older `[Unreleased]` bullet names `test_spacetime_spline_wave_int.jl`. After the test-suite
migration those tests are `test/integrator/Galerkin_Bspline_int.jl` (issue #7),
`test/integrator/FEM_Multisymplectic_int.jl` (issue #2) and
`test/integrator/SpaceTime_Spline_Wave_int.jl`, and the suite is green with broken marks.
Found by critic 1a of the migration (part M7).

## KI-2 · dead code · unused test dependencies

`test/Project.toml` keeps BenchmarkTools, CairoMakie, Infiltrator, JLD2, Plots, Printf, Profile and
Revise. No file under `test/` uses them since the experiment scripts moved to `scripts/`, and
`scripts/` has no environment of its own. Fix: remove them from `test/Project.toml`, or give
`scripts/` its own environment. Found by critic 1b of part M7.

## KI-3 · missing test · a broken mark that asserts nothing

`test/integrator/Galerkin_Bspline_int.jl:40` is
`@test_broken (C.λ₁_carry_x_coes .= 3 .* carried.λ₁_x_coes; true)`. It wraps a setup statement
that throws (issue #7: the cache has no field `λ₁_carry_x_coes`) so that lines 48–50 still run.
It adds one broken count that asserts nothing. Once #7 is fixed, it becomes a plain statement.
Found by critic 1b of part M7.
