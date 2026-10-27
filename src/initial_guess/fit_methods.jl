"""
    Projection() <: SlabFitMethod

The fit method of an ansatz that is linear in its coefficients: the `L²` projection of the target
onto the slab's ansatz space, on the quadrature of the slab.

The projection solves the mass-matrix system of the ansatz, which `SimpleSplines` factorises once
per quadrature rather than once per step. An ansatz whose coefficients enter a nonlinearity has no
such system, and a `Projection()` on one is refused at construction — see
[`SpaceTimeGalerkin`](@ref).

A user-defined fit method is a subtype of `SlabFitMethod`, and places the coefficients of `coeffs`,
of size `(Nt, Nx, D)`, from the sampled target, in one method of

```julia
fit!(coeffs, method, ansatz, sample, cache)
```

`cache` is the [`SpaceTimeGalerkinCache`](@ref), which owns the buffers a step reuses: a fit that
allocates nothing works in them.
"""
struct Projection <: SlabFitMethod end

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
        sample::AbstractArray{ST, 3}, c::SpaceTimeGalerkinCache) where {ST}
    q = quadrature(ansatz)
    for d in axes(coeffs, 3)
        l2_projection!(view(coeffs, :, :, d), q, view(sample, :, :, d), c)
    end
    return nothing
end

"""
    l2_projection!(û, q::TensorProductQuadrature, f, cache::SpaceTimeGalerkinCache)

The `L²` projection of `f`, sampled on the quadrature grid of `q` in two directions, written into
`û`, in the buffers of `cache`: nothing is allocated. The load `Φₜ (w ⊙ f) Φₓᵀ` is formed by two
sparse products and the mass system is solved along each direction, as `SimpleSplines`'
`l2_projection` does, which allocates the weighted sample, the products and the solution.
"""
function l2_projection!(û::AbstractMatrix, q::TensorProductQuadrature{T, 2},
        f::AbstractMatrix, c::SpaceTimeGalerkinCache) where {T}
    qt, qx = quadratures(q)
    wt = quadrature_weights(qt)
    wx = quadrature_weights(qx)
    for j in axes(f, 2), i in axes(f, 1)

        c.weighted[i, j] = f[i, j] * wt[i] * wx[j]
    end
    mul!(c.load, basis_values(qt, 0), c.weighted)
    mul!(û, c.load, basis_values(qx, 0)')
    mt = mass_operator(qt)
    for j in axes(û, 2)
        column = view(û, :, j)
        mass_solve!(column, mt, column)
    end
    mx = mass_operator(qx)
    for i in axes(û, 1)
        c.fibre .= view(û, i, :)
        mass_solve!(c.fibre, mx, c.fibre)
        û[i, :] .= c.fibre
    end
    return û
end
