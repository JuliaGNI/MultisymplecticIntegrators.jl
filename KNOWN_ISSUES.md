# Known issues

Defects found in review and recorded, not fixed. Each entry names its kind and its evidence.

## KI-1 · docs · `CHANGELOG.md` *Open Issues* names the old test files

The `## Open Issues` section says that `Pkg.test()` is red (21 pass, 2 fail, 1 error), and it
cites `test_full_multiplier_momentum_carry.jl:37/:40` and `test_fem_multisymplectic_wave.jl:61`.
Those tests are `test/integrator/Galerkin_Bspline_int.jl` (issue #7) and
`test/integrator/FEM_Multisymplectic_int.jl` (issue #2), and the suite is green with broken marks.
Found 2026-09-26.

## KI-2 · dead code · unused test dependencies

No file under `test/` uses 20 of the 26 dependencies in `test/Project.toml`: every one except
Aqua, GeometricIntegratorsBase, GeometricSolutions, MultiSymplectic, SafeTestsets and Test. The
experiment scripts in `scripts/` use them, and `scripts/` has no environment of its own. Evidence:
`grep -rn "using\|import" test/` names only those six packages. Fix: remove the 20 from
`test/Project.toml`, or give `scripts/` its own environment. Found 2026-09-26.

## KI-3 · missing test · a broken mark that asserts nothing

`test/integrator/Galerkin_Bspline_int.jl:40` is
`@test_broken (C.λ₁_carry_x_coes .= 3 .* carried.λ₁_x_coes; true)`. It wraps a setup statement
that throws (issue #7: the cache has no field `λ₁_carry_x_coes`) so that lines 48–50 still run.
It adds one broken count that asserts nothing. Once #7 is fixed, it becomes a plain statement.
Found 2026-09-26.
