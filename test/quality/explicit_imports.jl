using ExplicitImports
using MultisymplecticIntegrators
using Test

# Fails on a stale explicit import, and on an explicit import or a qualified access through a
# module other than the owner of the name. `no_implicit_imports` is off so that the guard has one
# form in every package of the stack; the two `public` checks are off because the package builds
# its solution step with `solutionstep` and gives the extrapolation of its problem type a method on
# `_extrapolate!`, neither of which `GeometricIntegratorsBase` marks `public`, and neither of which
# has a public equivalent there.
test_explicit_imports(
    MultisymplecticIntegrators;
    no_implicit_imports = false,
    all_explicit_imports_are_public = false,
    all_qualified_accesses_are_public = false
)
