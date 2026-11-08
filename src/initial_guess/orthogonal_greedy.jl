"""The workspace of one fit: its dictionary, the table of that dictionary at the slab's nodes, the
buffers of the selection, and the method whose parameters built them."""
mutable struct NetworkFit{ST}
    method::Union{Nothing, OrthogonalGreedy}
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
    NetworkFit{ST}(nothing, (zeros(ST, 0, 0) for _ in 1:7)..., Int[], Bool[])
end

"""
    greedy_atoms(method::OrthogonalGreedy, ::Type{T})

The dictionary of the fit, as `(w₁, w₂, b)` rows of precision `T`: `Nw + 1` directions of equal
angle on the half-open circle `[0, 2π)`, crossed with `Nb + 1` biases over `bias_interval`. The
grid is half-open, so no direction is repeated and every direction of the circle appears once.
"""
function greedy_atoms(method::OrthogonalGreedy, ::Type{T}) where {T}
    nd, nb = method.Nw + 1, method.Nb + 1
    a, b = convert(T, method.bias_interval[1]), convert(T, method.bias_interval[2])
    atoms = Matrix{T}(undef, nd * nb, 3)
    for k in 1:(nd * nb)
        θ = 2 * T(π) * T((k - 1) ÷ nb) / T(nd)
        atoms[k, 1], atoms[k, 2] = cos(θ), sin(θ)
        atoms[k, 3] = a + (b - a) * T((k - 1) % nb) / T(nb - 1)
    end
    return atoms
end

"""Fit `S` atoms of `D` components to `sample` by the orthogonal greedy algorithm of
[`OrthogonalGreedy`](@ref), in place throughout, and return the number of atoms taken: the
dictionary and its table are built only when the sizes or the method's parameters change, a fit
after that allocates nothing, and an atom shorter than `√eps` adds no rank. `coefficients` leaves
the triangular solve of the fit's Gram matrix, that is the output weights of the atoms taken."""
function fit_network!(f::NetworkFit{ST}, method::OrthogonalGreedy,
        sample::AbstractArray{ST, 3}, tabulation::SlabTabulation, S::Integer, D::Integer,
        activation) where {ST}
    t, x = tabulation.time, tabulation.space
    qt, qx = nquadrature(t), nquadrature(x)
    m, N = (method.Nw + 1) * (method.Nb + 1), qt * qx
    if f.method != method || size(f.table) != (m, N) || size(f.chosen, 2) != S ||
       size(f.residual, 2) != D
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
    f.method = method
    for q in 1:qt, r in 1:qx, d in 1:D
        f.residual[(q - 1) * qx + r, d] = sqrt(t.weight[q] * x.weight[r]) * sample[q, r, d]
    end
    fill!(f.taken, false)
    s = 0
    while s < S
        mul!(f.correlations, f.table, f.residual)
        best, k = zero(ST), 0
        for a in axes(f.table, 1), d in 1:D

            f.taken[a] && continue
            abs(f.correlations[a, d]) > best && ((k, best) = (a, abs(f.correlations[a, d])))
        end
        iszero(best) && break
        f.taken[k] = true
        column = view(f.chosen, :, s + 1)
        copyto!(column, view(f.table, k, :))
        norm2 = sum(abs2, column)
        for j in 1:s
            f.gram[j, s + 1] = dot(view(f.chosen, :, j), column)
            axpy!(-f.gram[j, s + 1], view(f.chosen, :, j), column)
        end
        gain2 = sum(abs2, column)
        gain2 > eps(ST) * norm2 || continue
        gain, s = sqrt(gain2), s + 1
        f.selected[s], f.gram[s, s] = k, gain
        rdiv!(column, gain)
        for d in 1:D
            f.coefficients[s, d] = dot(column, view(f.residual, :, d))
            axpy!(-f.coefficients[s, d], column, view(f.residual, :, d))
        end
    end
    ldiv!(UpperTriangular(view(f.gram, 1:s, 1:s)), view(f.coefficients, 1:s, :))
    return s
end
