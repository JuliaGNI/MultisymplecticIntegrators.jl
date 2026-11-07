using GeometricIntegratorsBase
using LinearAlgebra: norm
using MultisymplecticIntegrators
using Test

import GeometricBase: timestep, timespan
import GeometricIntegratorsBase: GeometricIntegrator, cache

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")

const M = MultisymplecticIntegrators

"""
The relative `L²` bound of the greedy fit on the slab — `5e-2`, the bound a local fit's own error is
allowed — with the value this tree measures beside it: `9.2e-2` at `S = 8` on the target
`sin(2πx) cos(πt)` over `[0, 1] × [0, 0.05]`, in both precisions, and `1.5e-2` on the §A case's own
target at the same size. The two differ because the dictionary's directions are of unit length, so a
ridge spans at most one radian across `[0, 1]`: eight of them hold the fundamental of the §A case,
and not the second mode that `sin(2πx)` is. The best eight atoms of that dictionary, found by a
search over its own grid, reach `5.4e-2` on this target, so no selection passes the bound here.
`Float32` keeps the same bound, whose square is far above its round-off.
"""
const GREEDY_L2_BOUND = 5.0e-2

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

"""The allocations of one fit of a fresh workspace, through a function barrier."""
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
        @test error ≤ GREEDY_L2_BOUND
        # every selected atom is a direction and a bias of the dictionary, and no atom repeats
        @test length(M.fit_atoms(w, ansatz)) == ansatz.S
        @test allunique(M.fit_atoms(w, ansatz))
        assert_no_upcast(w, T)
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
        # every direction lies on the unit circle, and 0 and 2π are the same direction
        @test all(abs(hypot(d[1], d[2]) - one(T)) ≤ 4 * eps(T) for d in directions)
        @test !((one(T), zero(T)) in directions[2:end])
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
        # the workspace of the default dictionary, measured on a fit of a fresh slab
        _, c2, _ = greedy_slab(T; D = 2)
        w2 = zeros(T, ansatz.np)
        allocations = fit_allocations(w2, ansatz, sample, c2)
        @test allocations < 100_000_000
    end
end
