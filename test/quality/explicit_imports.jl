using ExplicitImports
using MultisymplecticIntegrators
using Test

# Fails on a stale explicit import, on an explicit import or a qualified access through a module
# other than the owner of the name, and on a qualified access to a name its owner does not declare
# public. `no_implicit_imports` is off so that the guard has one form in every package of the
# stack; the check of the explicit imports' public-ness is off because the package imports names
# that their owners do not mark `public`: `CacheType`, `IntegratorCache`,
# `copy_internal_variables!`, `default_options`, `initial_guess!`, `initsolver`, `internal`,
# `residual!`, `solutionstep`, `solver` and `solverstate` from `GeometricIntegratorsBase`;
# `initial_residual`, `isconverged`, `jacobian!`, `jacobianmatrix`, `linearsolver` and `method`
# from `SimpleSolvers`; `Jacobian` and `derivative` from `SymbolicNeuralNetworks`; `layers` from
# `AbstractNeuralNetworks`; `ntimesteps` from `GeometricSolutions`; and `gradient`, `jacobian` and
# `value` from `ForwardDiff`.
test_explicit_imports(
    MultisymplecticIntegrators;
    no_implicit_imports = false,
    all_explicit_imports_are_public = false
)
