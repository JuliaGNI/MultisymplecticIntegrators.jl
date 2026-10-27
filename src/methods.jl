"""
    SpaceTimeAnsatz

Abstract type of a space-time ansatz of one slab.

A slab is the space-time strip that one time step spans, `[t_n, t_{n+1}] × xspan`. The ansatz
carries it in the normalised time coordinate `t̂`, where it is the rectangle `[0, 1] × xspan`.

An ansatz discretises the field `u` and its derivatives `v = ∂ₜu` and `w = ∂ₓu` on the slab, and
gives the integrator the basis functions and the Jacobian from which it assembles the residual. It
is a `GeometricMethod`, so `GeometricIntegrator(problem, method)` accepts it.
"""
abstract type SpaceTimeAnsatz <: GeometricMethod end

# A slab is advanced from its bottom trace alone, so there is no past to extrapolate to: the history
# states of the solution step hold the initial data, which `NoExtrapolation` copies into them.
default_extrapolation(::SpaceTimeAnsatz) = NoExtrapolation()

"""
    SlabCoupling

Abstract type of a coupling of one slab.

A coupling adds the terms that the interior weak form alone does not enforce: the initial
condition, the boundary conditions, or both.
"""
abstract type SlabCoupling end

"""
    SlabInitialGuess

Abstract type of a construction of the initial guess of one slab.
"""
abstract type SlabInitialGuess end
