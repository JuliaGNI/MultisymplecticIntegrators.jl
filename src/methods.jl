"""
    SpaceTimeAnsatz

Abstract type of a space-time ansatz of one slab.

An ansatz discretises the field `u` and its derivatives `v = ∂ₜu` and `w = ∂ₓu` on the slab, and
gives the integrator the basis functions and the Jacobian from which it assembles the residual. It
is a `GeometricMethod`, so `GeometricIntegrator(problem, method)` accepts it.
"""
abstract type SpaceTimeAnsatz <: GeometricMethod end

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
