"""The workspace of one fit: the dictionary, its table at the slab's nodes, its buffers."""
mutable struct NetworkFit{ST}
    atoms::Matrix{ST}
    table::Matrix{ST}
    chosen::Matrix{ST}
    gram::Matrix{ST}
    correlations::Matrix{ST}
    residual::Matrix{ST}
    coefficients::Matrix{ST}
    selected::Vector{Int}
    taken::Vector{Bool}
end

function NetworkFit(::Type{ST}) where {ST}
    NetworkFit{ST}((zeros(ST, 0, 0) for _ in 1:7)..., Int[], Bool[])
end

"""Fit `S` atoms and the output weights of `D` components to `sample` by the orthogonal greedy
algorithm of [`OrthogonalGreedy`](@ref), in place throughout, and return the number of atoms taken:
a fit after the first allocates nothing, and an atom shorter than `√eps` adds no rank."""
function fit_network!(f::NetworkFit{ST}, method::OrthogonalGreedy,
        sample::AbstractArray{ST, 3}, tabulation::SlabTabulation, S::Integer, D::Integer,
        activation) where {ST}
    t, x = tabulation.time, tabulation.space
    qt, qx = nquadrature(t), nquadrature(x)
    m, N = (method.Nw + 1) * (method.Nb + 1), qt * qx
    if size(f.table) != (m, N) || size(f.chosen, 2) != S || size(f.residual, 2) != D
        f.atoms, f.selected, f.taken = greedy_atoms(method, ST), zeros(Int, S), falses(m)
        f.table, f.chosen = zeros(ST, m, N), zeros(ST, N, S)
        f.gram, f.residual = zeros(ST, S, S), zeros(ST, N, D)
        f.coefficients, f.correlations = zeros(ST, S, D), zeros(ST, m, D)
        for k in 1:m, n in 1:N

            q, r = (n - 1) ÷ qx + 1, (n - 1) % qx + 1
            f.table[k, n] = sqrt(t.weight[q] * x.weight[r]) * activation(
                f.atoms[k, 1] * t.node[q] + f.atoms[k, 2] * x.node[r] + f.atoms[k, 3])
        end
    end
    for q in 1:qt, r in 1:qx, d in 1:D
        f.residual[(q - 1) * qx + r, d] = sqrt(t.weight[q] * x.weight[r]) * sample[q, r, d]
    end
    foreach(v -> fill!(v, 0), (f.selected, f.taken))
    s = 0
    while s < S
        mul!(f.correlations, f.table, f.residual)
        best, k = zero(ST), 0
        for a in axes(f.table, 1), d in 1:D

            f.taken[a] && continue
            value = abs(f.correlations[a, d])
            value > best && ((k, best) = (a, value))
        end
        iszero(best) && break
        f.taken[k] = true
        norm2 = zero(ST)
        for n in 1:N
            f.chosen[n, s + 1] = f.table[k, n]
            norm2 += abs2(f.chosen[n, s + 1])
        end
        for j in 1:s
            dot = zero(ST)
            for n in 1:N
                dot += f.chosen[n, j] * f.chosen[n, s + 1]
            end
            f.gram[j, s + 1] = dot
            for n in 1:N
                f.chosen[n, s + 1] -= dot * f.chosen[n, j]
            end
        end
        gain2 = zero(ST)
        for n in 1:N
            gain2 += abs2(f.chosen[n, s + 1])
        end
        gain2 > eps(ST) * norm2 || continue
        gain, s = sqrt(gain2), s + 1
        f.selected[s], f.gram[s, s] = k, gain
        for n in 1:N
            f.chosen[n, s] /= gain
        end
        for d in 1:D
            value = zero(ST)
            for n in 1:N
                value += f.chosen[n, s] * f.residual[n, d]
            end
            f.coefficients[s, d] = value
            for n in 1:N
                f.residual[n, d] -= f.chosen[n, s] * value
            end
        end
    end
    for i in s:-1:1
        for d in 1:D
            value = f.coefficients[i, d]
            for j in (i + 1):s
                value -= f.gram[i, j] * f.coefficients[j, d]
            end
            f.coefficients[i, d] = value / f.gram[i, i]
        end
    end
    return s
end
