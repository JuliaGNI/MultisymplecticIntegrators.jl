"""
    SpaceTimeAnsatz

Abstract type of a space-time ansatz of one slab.

A slab is the space-time strip that one time step spans, `[t_n, t_{n+1}] × xspan`. The ansatz
carries it in the normalised time coordinate `t̂`, where it is the rectangle `[0, 1] × xspan`.

An ansatz discretises the field `u` and its derivatives `v = ∂ₜu` and `w = ∂ₓu` on the slab, and
gives the integrator the basis functions and the Jacobian from which it assembles the residual. It
is not a method: a [`SpaceTimeMethod`](@ref), such as [`SpaceTimeGalerkin`](@ref), holds one.
"""
abstract type SpaceTimeAnsatz end

"""
    SpaceTimeMethod

Abstract type of a space-time method: an integrator that advances one slab per step, with an
[`SpaceTimeAnsatz`](@ref) inside.

It is a `GeometricMethod`, so `GeometricIntegrator(problem, method)` accepts it, and
[`integrate`](@ref) takes one.
"""
abstract type SpaceTimeMethod <: GeometricMethod end

# A slab is advanced from its bottom trace alone, so there is no past to extrapolate to: the history
# states of the solution step hold the initial data, which `NoExtrapolation` copies into them.
default_extrapolation(::SpaceTimeMethod) = NoExtrapolation()

"""
    SlabCoupling

Abstract type of a coupling of one slab.

A coupling adds the terms that the interior weak form alone does not enforce: the initial
condition, the boundary conditions, or both.
"""
abstract type SlabCoupling end

"""
    SlabInitialGuess

Abstract type of an initial guess of one slab: [`Continuation`](@ref) or [`FitTo`](@ref).

The initial guess is one of three roles of the construction of a guess. The other two have types of
their own, so that a value of the wrong role is refused where it is given: a [`SlabTarget`](@ref)
is what a [`FitTo`](@ref) fits, and a [`SlabFitMethod`](@ref) is how it fits.
"""
abstract type SlabInitialGuess end

"""
    SlabTarget

Abstract type of a target of a [`FitTo`](@ref): the function on the slab that the ansatz is fitted
to. The targets of the package are [`InitialCondition`](@ref), [`PreviousSlab`](@ref),
[`ExactSolution`](@ref) and [`TargetFunction`](@ref).
"""
abstract type SlabTarget end

"""
    SlabFitMethod

Abstract type of a fit method of a [`FitTo`](@ref): the way the ansatz is brought to a target. The
package ships [`Projection`](@ref).
"""
abstract type SlabFitMethod end
