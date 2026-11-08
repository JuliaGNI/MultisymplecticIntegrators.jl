"""
    Projection() <: SlabFitMethod

The fit method of an ansatz that is linear in its coefficients: the `L²` projection of the target
onto the slab's ansatz space, on the quadrature of the slab.

The projection solves the mass-matrix system of the ansatz, which `SimpleSplines` factorises once
per quadrature rather than once per step. An ansatz whose coefficients enter a nonlinearity has no
such system, and a `Projection()` on one is refused at construction — see
[`SpaceTimeGalerkin`](@ref).

A user-defined fit method is a subtype of `SlabFitMethod`, and places the coefficients of `coeffs`,
of size `(Nt, Nx, D)` for a tensor ansatz and the parameter vector of a network, from the sampled
target, in one method of

```julia
fit!(coeffs, method, ansatz, sample, cache)
```

`cache` is the [`SpaceTimeGalerkinCache`](@ref) of the slab, which a fit may read; a projection
keeps no scratch in it, and a network's block holds the workspace of its greedy fit.
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
    TrainedHiddenLayer(; epochs = 100, learning_rate = 1e-5, rng = Xoshiro(1234)) <: SlabFitMethod

The fit method of a [`NetworkAnsatz`](@ref) in `:last_layer` mode that trains the hidden layer
instead of taking it from a dictionary: every epoch takes one gradient-descent step on the hidden
weights and the hidden biases, and then reads the output weights by least squares on the hidden
features that step left.

The misfit the step descends is the field's distance from the sample of the slab, weighted as the
slab's own measure weights it on the normalised time: the ansatz's interior rule by `ωₜ ωₓ`, the two
surfaces at the space rule by `ωₓ`, and the two surfaces at the time rule by `ωₜ`. The features the
solve reads are the hidden layer's activations at those same points, so the four surfaces carry the
initial and boundary data into the fit.

`epochs = 0` fits the output layer alone on the hidden layer the `rng` draws; a larger `epochs`
descends first. The draw is taken from `rng` and never from the process's global generator, and the
generator is copied per call, so one method object gives the same fit to every integrator that
shares it. The buffers of the fit live in the ansatz's block, not in the method.

A network in `:all` mode is refused, because Newton determines the hidden layer there.

```julia
method = SpaceTimeGalerkin(NetworkAnsatz(S = 20, mode = :last_layer, xspan = (0.0, 1.0),
        ncells = (2, 5), degree = 3), WeakInitialAndBoundaryConditions();
    initial_guess = FitTo(InitialCondition(); method = TrainedHiddenLayer(epochs = 20)))
```
"""
struct TrainedHiddenLayer <: SlabFitMethod
    epochs::Int
    learning_rate::Float64
    rng::AbstractRNG
end

function TrainedHiddenLayer(; epochs::Integer = 100, learning_rate::Real = 1e-5,
        rng::AbstractRNG = Xoshiro(1234))
    epochs ≥ 0 || throw(ArgumentError("epochs must be at least 0, got $epochs"))
    isfinite(learning_rate) && learning_rate ≥ 0 || throw(ArgumentError(
        "learning_rate must be finite and at least 0, got $learning_rate"))
    return TrainedHiddenLayer(Int(epochs), Float64(learning_rate), rng)
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
