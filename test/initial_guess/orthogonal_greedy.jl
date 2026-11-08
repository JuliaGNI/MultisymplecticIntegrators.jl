using GeometricIntegratorsBase
using LinearAlgebra: dot, norm
using MultisymplecticIntegrators
using Test

import GeometricBase: timestep, timespan
import GeometricIntegratorsBase: GeometricIntegrator, cache

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")

const M = MultisymplecticIntegrators

"""
The factor by which the greedy fit may exceed the old `OGA2D` fit's error on the same slab, with the
values this tree measures beside it: the fit reaches `9.17e-2` at `S = 8` on the target
`sin(2πx) cos(πt)` over `[0, 1] × [0, 0.05]`, in both precisions, where the old dictionary and pick
rule reach `8.72e-2` and so bound it at `9.60e-2`. The dictionary's directions are of unit length,
so a ridge spans at most one radian across `[0, 1]`, and eight of them hold a fundamental and not
the second mode that `sin(2πx)` is. The old grid's direction list spans `[-π, π]` with both ends,
so one direction is listed there twice and every atom also appears negated, `tanh` being odd.
"""
const GREEDY_L2_FACTOR = 11 // 10

"""The default dictionary of the fit: `51 × 51` atoms at `Nw = Nb = 50`."""
const GREEDY_ATOMS = 51 * 51

"""
    greedy_slab(::Type{T}; S, D)

The cache of a network slab of `S` hidden units and `D` components on `x ∈ [0, 1]`, whose own rule
the fit reads its nodes and weights from.
"""
function greedy_slab(::Type{T}; S = 8, D = 1, quadrature_order = 4) where {T}
    problem = Wave.problem(T = T, D = D, timespan = (zero(T), T(0.05)), timestep = T(0.05),
        xstep = T(0.05))
    ansatz = NetworkAnsatz(S = S, D = D, xspan = (zero(T), one(T)), ncells = (2, 5),
        degree = 3, quadrature_order = quadrature_order)
    method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
    return ansatz, cache(GeometricIntegrator(problem, method)), problem
end

"""
The sample of a target `f(t, x)` at the points of a network slab's own rule, weighted.

The nodes of that rule are the slab's normalised time and its physical space coordinate, while a
target is a function of the physical time, as `network_target_values!` reads it: the point of node
`q` is `slab start + timestep * t̂[q]`.
"""
function greedy_sample(::Type{T}, c, problem, f) where {T}
    tnodes, xnodes = M.quadrature_points(c.ansatz)
    h = T(timestep(problem))
    t0 = T(first(timespan(problem)))
    D = M.noutputs(c.ansatz.ansatz)
    sample = Array{T, 3}(undef, length(tnodes), length(xnodes), D)
    for q in eachindex(tnodes), r in eachindex(xnodes)

        value = T(f(t0 + h * T(tnodes[q]), T(xnodes[r])))
        for d in 1:D
            sample[q, r, d] = value
        end
    end
    weights = [T(c.ansatz.tabulation.time.weight[q]) *
               T(c.ansatz.tabulation.space.weight[r])
               for q in eachindex(tnodes), r in eachindex(xnodes)]
    return sample, weights
end

"""The field of a parameter vector at the nodes of the slab's own rule."""
function greedy_field(::Type{T}, c, w) where {T}
    tnodes, xnodes = M.quadrature_points(c.ansatz)
    values = Array{T, 2}(undef, length(tnodes), length(xnodes))
    for q in eachindex(tnodes), r in eachindex(xnodes)

        values[q, r] = c.ansatz.ansatz.field(T[T(tnodes[q]), T(xnodes[r])], w)[1]
    end
    return values
end

"""The relative `L²` error of a fitted parameter vector against a sampled target."""
function greedy_error(::Type{T}, c, w, sample, weights) where {T}
    values = greedy_field(T, c, w)
    error = zero(T)
    scale = zero(T)
    for r in axes(values, 2), q in axes(values, 1)

        error += weights[q, r] * (values[q, r] - sample[q, r, 1])^2
        scale += weights[q, r] * sample[q, r, 1]^2
    end
    return sqrt(error / scale)
end

"""
    greedy_nodes(::Type{T}, c, sample, weights)

The slab's own rule as the flat vectors an atom of the old dictionary reads: a `2 × N` matrix
whose first row is the normalised time and whose second is the space coordinate of each node, the
node weights, and the first component of the sampled target, in the order the fit lays them out.
"""
function greedy_nodes(::Type{T}, c, sample, weights) where {T}
    tnodes, xnodes = M.quadrature_points(c.ansatz)
    N = length(tnodes) * length(xnodes)
    nodes = Matrix{T}(undef, 2, N)
    node_weights = Vector{T}(undef, N)
    target = Vector{T}(undef, N)
    n = 0
    for q in eachindex(tnodes), r in eachindex(xnodes)

        n += 1
        nodes[1, n] = T(tnodes[q])
        nodes[2, n] = T(xnodes[r])
        node_weights[n] = T(weights[q, r])
        target[n] = T(sample[q, r, 1])
    end
    return nodes, node_weights, target
end

"""
    old_oga2d_error(::Type{T}, nodes, weights, target; Nw = 50, Nb = 50, S = 8)

The relative `L²` error of the `OGA2D` dictionary and pick rule on the sample `target`: the bound
the greedy fit is held to, measured here instead of read from a reference.

Its dictionary is `Nw + 1` directions of `[-π, π]` with both ends, so the last entry repeats the
first, crossed with `Nb + 1` biases over the same interval; its atoms are
`tanh(w₁ t̂ + w₂ x + b)` at the nodes. It selects `S` atoms by the modulus of their projection on
the weighted residual, refitting the output weights by weighted least squares after each selection,
and it shares no code with the fit it measures.

`nodes` is a `2 × N` matrix of the slab's own nodes, and `weights` and `target` are vectors of
length `N`.
"""
function old_oga2d_error(::Type{T}, nodes::AbstractMatrix{T}, weights::AbstractVector{T},
        target::AbstractVector{T}; Nw = 50, Nb = 50, S = 8) where {T}
    nd, nb = Nw + 1, Nb + 1
    N = size(nodes, 2)
    table = Matrix{T}(undef, nd * nb, N)
    k = 0
    for θ in range(-T(π), T(π); length = nd), b in range(-T(π), T(π); length = nb)

        k += 1
        c, s = T(cos(θ)), T(sin(θ))
        for n in 1:N
            table[k, n] = tanh(c * nodes[1, n] + s * nodes[2, n] + b)
        end
    end
    chosen = Matrix{T}(undef, N, S)
    root_weights = sqrt.(weights)
    residual = copy(target)
    correlations = zeros(T, size(table, 1))
    coefficients = zeros(T, S)
    for s in 1:S
        for a in axes(table, 1)
            projection = zero(T)
            for n in 1:N
                projection += table[a, n] * residual[n] * weights[n]
            end
            correlations[a] = abs(projection)
        end
        chosen[:, s] = table[argmax(correlations), :]
        scaled = chosen[:, 1:s] .* root_weights
        coefficients[1:s] = scaled \ (target .* root_weights)
        residual = target - chosen[:, 1:s] * coefficients[1:s]
    end
    fitted = chosen * coefficients
    error = zero(T)
    scale = zero(T)
    for n in 1:N
        error += weights[n] * (fitted[n] - target[n])^2
        scale += weights[n] * target[n]^2
    end
    return sqrt(error / scale)
end

"""
The greedy fit as its definition states it, recomputed here with no code of the fit: the
closed-form dictionary of `Nw + 1` directions of `[0, 2π)` crossed with `Nb + 1` biases over
`[-π, π]`, the pick of each step by the modulus of the projection on the weighted residual, and the
Gram–Schmidt step that makes the rest of the residual orthogonal to the atom taken. It returns the
`S` picks as `(w₁, w₂, b)` triples and the modulus of every atom's projection at each step.

`nodes` is a `2 × N` matrix of the slab's own nodes, and `weights` and `target` are vectors of
length `N`.
"""
function closed_greedy(::Type{T}, nodes::AbstractMatrix{T}, weights::AbstractVector{T},
        target::AbstractVector{T}; Nw = 50, Nb = 50, S = 8) where {T}
    nd, nb = Nw + 1, Nb + 1
    m, N = nd * nb, size(nodes, 2)
    root = sqrt.(weights)
    dictionary = Matrix{T}(undef, m, 3)
    table = Matrix{T}(undef, m, N)
    for k in 1:m
        θ = 2 * T(π) * T((k - 1) ÷ nb) / T(nd)
        b = -T(π) + 2 * T(π) * T((k - 1) % nb) / T(nb - 1)
        dictionary[k, :] .= (cos(θ), sin(θ), b)
        for n in 1:N
            table[k, n] = root[n] * tanh(cos(θ) * nodes[1, n] + sin(θ) * nodes[2, n] + b)
        end
    end
    chosen = zeros(T, N, S)
    residual = root .* target
    taken = falses(m)
    picks = NTuple{3, T}[]
    correlations = Vector{T}[]
    for s in 1:S
        corr = abs.(table * residual)
        push!(correlations, corr)
        k = argmax(ifelse.(taken, zero(T), corr))
        push!(picks, (dictionary[k, 1], dictionary[k, 2], dictionary[k, 3]))
        taken[k] = true
        column = table[k, :] .* one(T)
        norm2 = dot(table[k, :], table[k, :])
        for j in 1:(s - 1)
            column .-= dot(view(chosen, :, j), column) .* view(chosen, :, j)
        end
        gain = sqrt(dot(column, column))
        gain > sqrt(eps(T) * norm2) || break
        chosen[:, s] .= column ./ gain
        residual .-= dot(view(chosen, :, s), residual) .* view(chosen, :, s)
    end
    return picks, correlations
end

"""The allocations of one fit of a fresh workspace, which build its tables, through a function
barrier."""
function first_fit_allocations(w, ansatz, sample, c)
    return @allocated M.fit!(w, OrthogonalGreedy(), ansatz, sample, c)
end

"""The allocations of one fit of a workspace whose tables the first fit built, through a function
barrier."""
function fit_allocations(w, ansatz, sample, c)
    M.fit!(w, OrthogonalGreedy(), ansatz, sample, c)
    return @allocated M.fit!(w, OrthogonalGreedy(), ansatz, sample, c)
end

@testset "Orthogonal greedy" begin
    @testset "G1 the fit reproduces the seed on the §A case" for T in (Float64, Float32)
        ansatz, c, problem = greedy_slab(T)
        sample, weights = greedy_sample(T, c, problem,
            (t, x) -> sin(2 * π * x) * cos(π * t))
        w = zeros(T, ansatz.np)
        M.fit!(w, OrthogonalGreedy(), ansatz, sample, c)
        error = greedy_error(T, c, w, sample, weights)
        nodes, node_weights, target = greedy_nodes(T, c, sample, weights)
        @test error ≤ GREEDY_L2_FACTOR * old_oga2d_error(T, nodes, node_weights, target)
        # no atom is selected twice
        @test allunique(M.fit_atoms(w, ansatz))
        # the selection recomputed from its definition, on the closed form of the same dictionary:
        # the eight atoms written are the copy's picks in order, and each step's pick carries the
        # copy's largest correlation against the atoms the earlier steps left
        picks, correlations = closed_greedy(T, nodes, node_weights, target)
        atoms = M.fit_atoms(w, ansatz)
        dictionary = M.greedy_atoms(OrthogonalGreedy(), T)
        position(atom) = argmin([maximum(abs.(collect(atom) .- dictionary[k, :]))
                                 for k in axes(dictionary, 1)])
        @test length(atoms) == length(picks)
        for s in eachindex(picks)
            @test maximum(abs.(collect(atoms[s]) .- picks[s])) ≤ 4 * eps(T)
            left = setdiff(axes(dictionary, 1), position.(atoms[1:(s - 1)]))
            @test abs(correlations[s][position(atoms[s])]) ≥
                  (1 - 64 * eps(T)) * maximum(abs, correlations[s][left])
        end
        assert_no_upcast(w, T)
    end

    # The workspace of a fit is keyed on the parameters of its method as well as on its sizes: a
    # second fit on the same block, with a dictionary of the same size and another bias interval,
    # reads its own dictionary — a workspace keyed on sizes alone returns the first one's biases.
    @testset "G5 a second fit on one block reads its own dictionary" for T in (Float64, Float32)
        ansatz, c, problem = greedy_slab(T)
        sample, _ = greedy_sample(T, c, problem, (t, x) -> sin(2 * π * x) * cos(π * t))
        narrow = OrthogonalGreedy(bias_interval = (-1, 1))
        w = zeros(T, ansatz.np)
        M.fit!(w, OrthogonalGreedy(), ansatz, sample, c)
        M.fit!(w, narrow, ansatz, sample, c)
        _, fresh, _ = greedy_slab(T)
        expected = zeros(T, ansatz.np)
        M.fit!(expected, narrow, ansatz, sample, fresh)
        S = ansatz.S
        @test all(b -> -1 ≤ b ≤ 1, w[(2S + 1):(3S)])
        @test w == expected
    end

    @testset "G2 the dictionary holds no duplicate atom" for T in (Float64, Float32)
        method = OrthogonalGreedy()
        atoms = M.greedy_atoms(method, T)
        @test size(atoms) == (GREEDY_ATOMS, 3)
        @test eltype(atoms) === T
        # the 51 directions are the half-open circle: no direction appears twice, and each one is
        # crossed with every bias
        keys = [(atoms[k, 1], atoms[k, 2]) for k in axes(atoms, 1)]
        directions = unique(keys)
        @test length(directions) == 51
        @test all(count(==(d), keys) == 51 for d in directions)
        # every direction lies on the unit circle and on the closed-form half-open grid, and no two
        # of them is closer than half the grid's chord — a duplicate of the two ends is separated by
        # rounding alone, about `2.4e-16` in `Float64` and `1.7e-7` in `Float32`
        @test all(abs(hypot(d[1], d[2]) - one(T)) ≤ 4 * eps(T) for d in directions)
        closed_form = [(cospi(2 * T(j - 1) / T(51)), sinpi(2 * T(j - 1) / T(51)))
                       for j in eachindex(directions)]
        @test all(
            pair -> abs(pair[1][1] - pair[2][1]) ≤ 4 * eps(T) &&
                    abs(pair[1][2] - pair[2][2]) ≤ 4 * eps(T),
            zip(directions, closed_form))
        separations = [hypot(d[1] - e[1], d[2] - e[2])
                       for (i, d) in enumerate(directions), (k, e) in enumerate(directions)
                       if i < k]
        @test minimum(separations) ≥ sin(T(π) / T(51))
        # and no two atoms coincide in their two weights and their bias together
        @test allunique([(atoms[k, 1], atoms[k, 2], atoms[k, 3]) for k in axes(atoms, 1)])
    end

    @testset "G3 one greedy fit per component, and the memory bound" for T in (Float64, Float32)
        ansatz, c, problem = greedy_slab(T; D = 2)
        sample, _ = greedy_sample(T, c, problem, (t, x) -> sin(2 * π * x) * cos(π * t))
        sample[:, :, 2] .= T(0)
        w = zeros(T, ansatz.np)
        M.fit!(w, OrthogonalGreedy(), ansatz, sample, c)
        weights1 = M.output_weights(w, ansatz, 1)
        weights2 = M.output_weights(w, ansatz, 2)
        @test weights1 != weights2
        @test norm(weights2) ≤ sqrt(eps(T)) * max(norm(weights1), one(T))
        # the workspace of the default dictionary, measured on a fit of a fresh slab: the first fit
        # builds the dictionary's tables, and no fit after it allocates anything
        _, c2, _ = greedy_slab(T; D = 2)
        w2 = zeros(T, ansatz.np)
        @test first_fit_allocations(w2, ansatz, sample, c2) < 100_000_000
        @test fit_allocations(w2, ansatz, sample, c2) == 0
    end

    # A dictionary of four atoms fills at most four of the ten units, and the units the selection
    # passes over stay at zero whatever the vector held before the fit: the fit clears it and writes
    # only the atoms it took.
    @testset "G4 a short dictionary leaves the units it did not fill at zero" for T in (Float64,
        Float32)
        ansatz, c, problem = greedy_slab(T; S = 10)
        sample, _ = greedy_sample(T, c, problem, (t, x) -> sin(2 * π * x) * cos(π * t))
        w = ones(T, ansatz.np)
        M.fit!(w, OrthogonalGreedy(Nw = 1, Nb = 1), ansatz, sample, c)
        atoms = M.fit_atoms(w, ansatz)
        @test count(a -> a != (zero(T), zero(T), zero(T)), atoms) ≤ 4
        @test all(a -> a == (zero(T), zero(T), zero(T)), atoms[5:end])
        @test all(iszero, M.output_weights(w, ansatz, 1)[5:end])
    end
end
