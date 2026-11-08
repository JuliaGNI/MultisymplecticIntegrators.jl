# The fit of a `:last_layer` network that trains its hidden layer: the misfit it descends, the
# weighted least squares that reads the output weights, and the workspace that keeps its buffers.
#
# The block of the network is a `NetworkSlab`, which this file is loaded before; the functions below
# take it untyped and reach it only through its fields and the seam functions, so that the block
# can hold this file's workspace as one of its own fields.

"""
    TrainedFit{ST}

The workspace of a [`TrainedHiddenLayer`](@ref) fit in the block of a network ansatz.

`initial` is the hidden layer the method's generator drew, which is what the buffers were built
for: a fit whose method, sizes or draw differ from these builds them again, so that two methods of
one size on one block give the two fits that fresh blocks give. `face_target` is the target at the
four surfaces of the slab, which the guess fills beside the interior one. The remaining fields are
the sample the fit is taken on and the buffers of its two steps: `points` holds the normalised time
and the space coordinate of every point of the sample, `measure` the weight the slab's measure gives
that point, and `target` and `values` the target and the field there, one row per point and one
column per component. `features` holds the hidden layer's activations at the points, weighted by the
square root of `measure`, and `rhs` the target weighted the same way; `features \\ rhs` is the
least-squares problem whose solution is the output layer. `gradient` is the misfit's derivative with
respect to the hidden parameters, and `misfit` the misfit of the fitted field.

The buffers are of the problem's element type, and the method holds no state of the fit.
"""
mutable struct TrainedFit{ST}
    method::Union{Nothing, TrainedHiddenLayer}
    initial::Vector{ST}
    face_target::Matrix{ST}
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
    return TrainedFit{ST}(nothing, ST[], zeros(ST, 0, 0), zeros(ST, 2, 0), ST[],
        zeros(ST, 0, 0), zeros(ST, 0, 0), zeros(ST, 0, 0), zeros(ST, 0, 0), ST[], zero(ST))
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

# Whether the workspace has to be built again for this method, this draw and these sizes.
function fit_stale(f::TrainedFit{ST}, method::TrainedHiddenLayer, draw::AbstractVector{ST},
        S::Integer, D::Integer, N::Integer) where {ST}
    f.method === nothing && return true
    f.method.epochs == method.epochs || return true
    f.method.learning_rate == method.learning_rate || return true
    length(f.initial) == length(draw) || return true
    f.initial == draw || return true
    length(f.measure) == N || return true
    size(f.target) == (N, D) || return true
    return size(f.features) != (N, S)
end

"""
    fit_layout!(f, method, draw, ansatz, s)

Build the sample the fit is taken on: the points of the ansatz's own interior rule, then the four
surfaces at the nodes of the trace basis' rules, and the weight the slab's measure gives each of
them — `ωₜ ωₓ` in the interior, `ωₓ` at the two surfaces in time and `ωₜ` at the two in space.
"""
function fit_layout!(f::TrainedFit{ST}, method::TrainedHiddenLayer,
        draw::AbstractVector{ST}, ansatz, s) where {ST}
    S, D = ansatz.S, ansatz.D
    tq, xq = s.tabulation.time, s.tabulation.space
    tt, xt = s.trace_tabulation.time, s.trace_tabulation.space
    qt, qx, qtt, qxt = nquadrature(tq), nquadrature(xq), nquadrature(tt), nquadrature(xt)
    N = qt * qx + 2 * qxt + 2 * qtt
    if fit_stale(f, method, draw, S, D, N)
        f.method, f.initial = method, collect(draw)
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
    size(f.face_target) == (2 * qxt + 2 * qtt, D) && return nothing
    f.face_target = zeros(ST, 2 * qxt + 2 * qtt, D)
    return nothing
end

# The target of every point of the sample: the interior one as the guess sampled it, and the four
# surfaces from the target the fit was given.
function fit_sample!(f::TrainedFit{ST}, sample::AbstractArray{ST, 3}, s) where {ST}
    tq, xq = s.tabulation.time, s.tabulation.space
    qt, qx, D = nquadrature(tq), nquadrature(xq), size(sample, 3)
    k = 0
    for q in 1:qt, r in 1:qx

        k += 1
        for d in 1:D
            f.target[k, d] = sample[q, r, d]
        end
    end
    for row in axes(f.face_target, 1)
        k += 1
        for d in 1:D
            f.target[k, d] = f.face_target[row, d]
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
    for s in 1:S, d in 1:D

        w[3S + (s - 1) * D + d] = W[s, d]
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

The parameters are written into `w` in the order `L1.W`, `L1.b`, `L2.W`; `w`'s hidden part is the
draw's, whatever it held before.
"""
function fit_trained!(f::TrainedFit{ST}, method::TrainedHiddenLayer, w::AbstractVector{ST},
        ansatz, sample::AbstractArray{ST, 3}, s) where {ST}
    S, D = ansatz.S, ansatz.D
    nh = ansatz.np - S * D
    rng = copy(method.rng)
    draw = Vector{ST}(undef, nh)
    for θ in eachindex(draw)
        draw[θ] = 2 * rand(rng, ST) - 1
    end
    fit_layout!(f, method, draw, ansatz, s)
    fit_sample!(f, sample, s)
    copyto!(w, 1, draw, 1, nh)
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
