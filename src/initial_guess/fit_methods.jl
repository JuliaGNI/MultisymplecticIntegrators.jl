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

`cache` is the [`SpaceTimeGalerkinCache`](@ref) of the slab, which a fit may read; it holds no
scratch of the fit.
"""
struct Projection <: SlabFitMethod end

"""
    OrthogonalGreedy(; bias_interval = (-π, π), Nw = 50, Nb = 50) <: SlabFitMethod

The fit method of an ansatz whose parameters enter a nonlinearity, such as a
[`NetworkAnsatz`](@ref): a dictionary of `σ(w₁ t̂ + w₂ x + b)` atoms, from which the fit selects one
atom per hidden unit and reads the output weights by least squares.

The dictionary is `Nw + 1` directions of equal angle on the half-open circle `[0, 2π)`, crossed with
`Nb + 1` biases over `bias_interval`. The fit is linear in the output weights alone, so the hidden
weights and the biases stay where the dictionary put them; a `Projection()` is refused on such an
ansatz, and this fit replaces it as the ansatz's [`default_fit`](@ref).

```julia
method = SpaceTimeGalerkin(NetworkAnsatz(S = 10, xspan = (0.0, 1.0), ncells = (2, 5),
        degree = 3), WeakInitialAndBoundaryConditions();
    initial_guess = FitTo(ExactSolution(); method = OrthogonalGreedy(Nw = 50, Nb = 50)))
```
"""
struct OrthogonalGreedy <: SlabFitMethod
    bias_interval::Tuple{Float64, Float64}
    Nw::Int
    Nb::Int

    function OrthogonalGreedy(; bias_interval = (-π, π), Nw::Integer = 50, Nb::Integer = 50)
        length(bias_interval) == 2 ||
            throw(ArgumentError("bias_interval must hold two values, got $bias_interval"))
        all(isfinite, bias_interval) ||
            throw(ArgumentError("bias_interval must be finite, got $bias_interval"))
        float(bias_interval[1]) < float(bias_interval[2]) ||
            throw(ArgumentError("bias_interval must increase, got $bias_interval"))
        Nw ≥ 1 || throw(ArgumentError("Nw must be at least 1, got $Nw"))
        Nb ≥ 1 || throw(ArgumentError("Nb must be at least 1, got $Nb"))
        return new((Float64(bias_interval[1]), Float64(bias_interval[2])), Int(Nw), Int(Nb))
    end
end

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
        l2_projection!(view(coeffs, :, :, d), q, view(sample, :, :, d))
    end
    return nothing
end
