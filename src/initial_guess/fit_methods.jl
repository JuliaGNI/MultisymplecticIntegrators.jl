"""
    Projection() <: SlabInitialGuess

The fit method of an ansatz that is linear in its coefficients: the `L²` projection of the target
onto the slab's ansatz space, on the quadrature of the slab.

The projection solves the mass-matrix system of the ansatz, which `SimpleSplines` factorises once
per quadrature rather than once per step. An ansatz whose coefficients enter a nonlinearity has no
such system, and a `Projection()` on one is refused at construction — see
[`SpaceTimeGalerkin`](@ref).

A user-defined fit method places the coefficients of `coeffs`, of size `(Nt, Nx, D)`, from the
sampled target, in one method of

```julia
fit!(coeffs, method, ansatz, sample)
```
"""
struct Projection <: SlabInitialGuess end

"""
    default_fit(ansatz)

The fit method `FitTo(target)` uses when the caller names none.
"""
default_fit(::SpaceTimeAnsatz) = Projection()

"""
    is_linear_in_coefficients(ansatz)

Whether the ansatz is linear in its coefficients. A `Projection()` needs one that is; an ansatz
whose coefficients enter a nonlinearity reports `false` and is refused.
"""
is_linear_in_coefficients(::SpaceTimeAnsatz) = false

function fit!(coeffs::AbstractArray{ST, 3}, ::Projection, ansatz::TensorBSplineAnsatz,
        sample::AbstractArray{ST, 3}) where {ST}
    q = quadrature(ansatz)
    for d in axes(coeffs, 3)
        coeffs[:, :, d] .= l2_projection(q, view(sample, :, :, d))
    end
    return nothing
end
