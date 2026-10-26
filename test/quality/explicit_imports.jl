using ExplicitImports
using MultisymplecticIntegrators
using Test

# Fails on a stale explicit import, and on an explicit import or a qualified access through a
# module other than the owner of the name. `no_implicit_imports` is off so that the guard has one
# form in every package of the stack; the two `public` checks are off because the package reaches
# `Symbolics.variable` and `Symbolics.build_function`, which `Symbolics` does not mark `public`.
test_explicit_imports(
    MultisymplecticIntegrators;
    no_implicit_imports = false,
    all_explicit_imports_are_public = false,
    all_qualified_accesses_are_public = false
)
