"""
    greedy_atoms(method::OrthogonalGreedy, ::Type{T}, N)

The dictionary of the orthogonal greedy fit: `Nw + 1` directions of equal angle on the half-open
circle `[0, 2π)` crossed with `Nb + 1` biases over `bias_interval`, as rows `(w₁, w₂, b)` in
precision `T`.

The angles are `2πk / (Nw + 1)`, so the first direction is not repeated at the end of the grid and
every direction of the circle appears exactly once. The biases are `Nb + 1` equally spaced values
from the first end of `bias_interval` to the second, both ends included.
"""
function greedy_atoms(method::OrthogonalGreedy, ::Type{T}) where {T}
    nd, nb = method.Nw + 1, method.Nb + 1
    a, b = convert(T, method.bias_interval[1]), convert(T, method.bias_interval[2])
    atoms = Matrix{T}(undef, nd * nb, 3)
    for k in 1:(nd * nb)
        θ = 2 * T(π) * T((k - 1) ÷ nb) / T(nd)
        atoms[k, 1] = cos(θ)
        atoms[k, 2] = sin(θ)
        atoms[k, 3] = a + (b - a) * T((k - 1) % nb) / T(nb - 1)
    end
    return atoms
end

"""
    fit!(w, method::OrthogonalGreedy, ansatz::NetworkAnsatz, sample, cache)

Fit a network to the sampled target of a slab, by the orthogonal greedy algorithm of
[`OrthogonalGreedy`](@ref).

The dictionary is [`greedy_atoms`](@ref). At every selection the atom whose projection on the
residual has the largest modulus over the components is taken, and the output weights of every
component are refitted by least squares on the atoms taken so far. An atom whose remaining direction
is shorter than `√eps` of its own length adds no rank and is passed over, so the selection stays well
conditioned. The fit is linear in the output weights alone: the hidden weights and the biases stay
where the dictionary put them, and the components share one hidden layer but not its output weights.

`w` is the network's whole parameter vector, in the order `L1.W`, `L1.b`, `L2.W`, and it is written in
full; the caller takes the free part of it. The target and the rule are the slab's own, from `sample`
and the cache's block, and the workspace is that block's, sized and tabulated at its first fit.
"""
function fit!(w::AbstractVector{ST}, method::OrthogonalGreedy, ansatz::NetworkAnsatz,
        sample::AbstractArray{ST, 3}, c::SpaceTimeGalerkinCache) where {ST}
    f = prepare_fit!(c.ansatz.fit, method, c.ansatz.tabulation, ansatz.S, size(sample, 3),
        activation_function(ansatz))
    S, D = ansatz.S, size(sample, 3)
    N = size(f.table, 2)
    n = 0
    for q in 1:nquadrature(c.ansatz.tabulation.time),
        r in 1:nquadrature(c.ansatz.tabulation.space)

        n += 1
        for d in 1:D
            f.targets[n, d] = sample[q, r, d]
        end
    end
    copyto!(f.residual, f.targets)
    fill!(f.taken, false)
    fill!(f.selected, 0)
    fill!(w, zero(ST))
    floor = sqrt(eps(ST))
    s = 0
    while s < S
        for d in 1:D, i in 1:N

            f.weighted[i, d] = f.weights[i] * f.residual[i, d]
        end
        mul!(f.correlations, f.table, f.weighted)
        k, best = 0, zero(ST)
        for a in 1:size(f.table, 1)
            f.taken[a] && continue
            value = zero(ST)
            for d in 1:D
                value = max(value, abs(f.correlations[a, d]))
            end
            value > best && ((k, best) = (a, value))
        end
        k == 0 && break
        f.taken[k] = true
        norm2 = select_direction!(f, k, s, N)
        gain = √(f.gram[s + 1, s + 1])
        gain > floor * √(norm2) || continue
        s += 1
        select_atom!(f, k, gain, s, N, D)
    end
    return fit_parameters!(w, f, S, D)
end
