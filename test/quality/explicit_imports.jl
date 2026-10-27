using ExplicitImports
using MultisymplecticIntegrators
using Test

# Fails on a stale explicit import, on an explicit import or a qualified access through a module
# other than the owner of the name, and on a qualified access to a name its owner does not declare
# public. `no_implicit_imports` is off so that the guard has one form in every package of the
# stack; the check of the explicit imports' public-ness is off because the package builds its
# solution step and its slab core on names that `GeometricIntegratorsBase` and `SimpleSolvers` do not
# mark `public` and give no public equivalent of: `solutionstep`, `CacheType`, `IntegratorCache`,
# `initial_guess!`, `residual!`, `solver` and `solverstate` from the first, and `initial_residual`
# and `method` from the second.
test_explicit_imports(
    MultisymplecticIntegrators;
    no_implicit_imports = false,
    all_explicit_imports_are_public = false
)
