# The fit of a `:last_layer` network that trains its hidden layer: the misfit it descends, the
# weighted least squares that reads the output weights, and the workspace that keeps its buffers.
#
# The block of the network is a `NetworkSlab`, which this file is loaded before; the functions below
# take it untyped and reach it only through its fields and the seam functions, so that the block
# can hold this file's workspace as one of its own fields.

"""
    TrainedFit{ST}

The workspace of a [`TrainedHiddenLayer`](@ref) fit in the block of a network ansatz: the sample
the fit is taken on and the buffers of its two steps, which a fit of other sizes builds again.

`points` holds the normalised time and the space coordinate of every point of the sample, `measure`
the weight the slab's measure gives that point, and `target` and `values` the target and the field
there, one row per point and one column per component. `features` holds the hidden layer's
activations at the points, weighted by the square root of `measure`, and `rhs` the target weighted
the same way; `features \\ rhs` is the least-squares problem whose solution is the output layer.
`gradient` is the misfit's derivative with respect to the hidden parameters, and `misfit` the misfit
of the fitted field.

The buffers are of the problem's element type. `points` and `measure` depend on the block's rules
alone, and every fit overwrites the rest, so the method holds no state of the fit.
"""
mutable struct TrainedFit{ST}
    points::Matrix{ST}
    measure::Vector{ST}
    target::Matrix{ST}
    values::Matrix{ST}
    features::Matrix{ST}
    rhs::Matrix{ST}
    gradient::Vector{ST}
    misfit::ST
end

function TrainedFit(::Type{ST}) where {ST}
    return TrainedFit{ST}(zeros(ST, 2, 0), ST[], zeros(ST, 0, 0), zeros(ST, 0, 0),
        zeros(ST, 0, 0), zeros(ST, 0, 0), ST[], zero(ST))
end

"""
    misfit(f::TrainedFit)

The weighted misfit of the field at the sample: `Σ_k ω_k Σ_d (u_kd − f_kd)²`.
"""
function misfit(f::TrainedFit{ST}) where {ST}
    total = zero(ST)
    for k in axes(f.target, 1), d in axes(f.target, 2)

        total += f.measure[k] * abs2(f.values[k, d] - f.target[k, d])
    end
    return total
end

# Whether the workspace has to be built again for these sizes.
function fit_stale(f::TrainedFit, S::Integer, D::Integer, N::Integer)
    length(f.measure) == N || return true
    size(f.target) == (N, D) || return true
    return size(f.features) != (N, S)
end

"""
    fit_layout!(f, ansatz, s)

Build the sample the fit is taken on: the points of the ansatz's own interior rule, then the four
surfaces at the nodes of the trace basis' rules, and the weight the slab's measure gives each of
them — `ωₜ ωₓ` in the interior, `ωₓ` at the two surfaces in time and `ωₜ` at the two in space.
"""
function fit_layout!(f::TrainedFit{ST}, ansatz, s) where {ST}
    S, D = ansatz.S, ansatz.D
    tq, xq = s.tabulation.time, s.tabulation.space
    tt, xt = s.trace_tabulation.time, s.trace_tabulation.space
    qt, qx, qtt, qxt = nquadrature(tq), nquadrature(xq), nquadrature(tt), nquadrature(xt)
    N = qt * qx + 2 * qxt + 2 * qtt
    if fit_stale(f, S, D, N)
        f.points, f.measure = zeros(ST, 2, N), zeros(ST, N)
        f.target, f.values = zeros(ST, N, D), zeros(ST, N, D)
        f.rhs, f.features = zeros(ST, N, D), zeros(ST, N, S)
        f.gradient = zeros(ST, ansatz.np - S * D)
        k = 0
        for q in 1:qt, r in 1:qx

            k += 1
            f.points[1, k], f.points[2, k] = tq.node[q], xq.node[r]
            f.measure[k] = tq.weight[q] * xq.weight[r]
        end
        for t̂ in (zero(ST), one(ST)), r in 1:qxt

            k += 1
            f.points[1, k], f.points[2, k] = t̂, xt.node[r]
            f.measure[k] = xt.weight[r]
        end
        for r in 1:2, q in 1:qtt

            k += 1
            f.points[1, k] = tt.node[q]
            f.points[2, k] = xspan(ansatz)[r]
            f.measure[k] = tt.weight[q]
        end
    end
    return nothing
end

# The target of every point of the sample, from the fit's sample `(; interior, faces)`: the interior
# one as the guess sampled it, and then the four surfaces.
function sample_target!(f::TrainedFit{ST}, sample::NamedTuple, s) where {ST}
    (; interior, faces) = sample
    qt, qx, D = size(interior)
    size(f.target) == (qt * qx + size(faces, 1), D) && size(faces, 2) == D ||
        throw(DimensionMismatch("the sample has $(qt * qx) interior and $(size(faces, 1)) face " *
                                "points of $D component(s); the fit is laid out for " *
                                "$(size(f.target, 1)) points of $(size(f.target, 2))"))
    k = 0
    for q in 1:qt, r in 1:qx

        k += 1
        for d in 1:D
            f.target[k, d] = interior[q, r, d]
        end
    end
    for row in axes(faces, 1)
        k += 1
        for d in 1:D
            f.target[k, d] = faces[row, d]
        end
    end
    for k in 1:length(f.measure), d in 1:D

        f.rhs[k, d] = sqrt(f.measure[k]) * f.target[k, d]
    end
    return nothing
end

# The hidden features at the sample points, weighted by the square root of the measure, from a
# working parameter vector in the order `L1.W`, `L1.b`.
function hidden_features!(f::TrainedFit{ST}, w::AbstractVector{ST}, S::Integer,
        activation) where {ST}
    for k in axes(f.points, 2), s in 1:S

        f.features[k, s] = sqrt(f.measure[k]) * activation(
            w[s] * f.points[1, k] + w[S + s] * f.points[2, k] + w[2S + s])
    end
    return nothing
end

# The output layer the features leave: the least-squares solution of `features * W ≈ rhs`, written
# into `w` in the order of the output weights, the component index running fastest inside a unit,
# and into the block's parameters, which every later evaluation reads.
function solve_output_layer!(f::TrainedFit{ST}, w::AbstractVector{ST}, ansatz, s) where {ST}
    S, D = ansatz.S, ansatz.D
    hidden_features!(f, w, S, activation_function(ansatz))
    W = f.features \ f.rhs
    for unit in 1:S, d in 1:D

        w[3S + (unit - 1) * D + d] = W[unit, d]
    end
    copyto!(s.params, w)
    return nothing
end

# One pass over the sample: the field at every point, the misfit, and its derivative with respect to
# the hidden parameters, from the parameter derivative the network's block carries.
function misfit_gradient!(f::TrainedFit{ST}, w::AbstractVector{ST}, nh::Integer, s) where {ST}
    copyto!(s.params, w)
    fill!(f.gradient, zero(ST))
    D = size(f.values, 2)
    for k in axes(f.points, 2)
        evaluate_at!(s, f.points[1, k], f.points[2, k])
        ω = f.measure[k]
        for d in 1:D
            u = s.values[d]
            f.values[k, d] = u
            ζ = 2 * ω * (u - f.target[k, d])
            for θ in 1:nh
                f.gradient[θ] += ζ * s.gradient[d, θ]
            end
        end
    end
    f.misfit = misfit(f)
    return nothing
end

"""
    fit_trained!(f, method, w, ansatz, sample, s)

Fit the hidden layer of a `:last_layer` network and its output layer to the sample: draw the hidden
layer from the method's generator, read the output weights of that draw by least squares, and then,
once per epoch, take one gradient-descent step on the hidden parameters and read the output weights
again. `epochs = 0` leaves the drawn hidden layer in place and fits the output layer alone.

`sample` is the fit's sample `(; interior, faces)` of [`fit_sample`](@ref). The draw is taken once
in `Float64` and rounded to the element type, so that one generator gives one hidden layer in every
precision, up to rounding.

The parameters are written into `w` in the order `L1.W`, `L1.b`, `L2.W`; `w`'s hidden part — the
entries before `first_free`, which the mode does not free — is the draw's, whatever it held before.
"""
function fit_trained!(f::TrainedFit{ST}, method::TrainedHiddenLayer, w::AbstractVector{ST},
        ansatz, sample::NamedTuple, s) where {ST}
    S, D = ansatz.S, ansatz.D
    nh = ansatz.np - S * D
    rng = copy(method.rng)
    for θ in 1:nh
        w[θ] = convert(ST, 2 * rand(rng) - 1)
    end
    fit_layout!(f, ansatz, s)
    sample_target!(f, sample, s)
    solve_output_layer!(f, w, ansatz, s)
    if method.epochs > 0
        δ = convert(ST, method.learning_rate)
        for epoch in 1:method.epochs
            misfit_gradient!(f, w, nh, s)
            for θ in 1:nh
                w[θ] -= δ * f.gradient[θ]
            end
            solve_output_layer!(f, w, ansatz, s)
        end
    end
    misfit_gradient!(f, w, nh, s)
    return w
end
