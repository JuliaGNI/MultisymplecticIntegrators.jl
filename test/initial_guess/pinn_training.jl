using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS, norm
using MultisymplecticIntegrators
using Random
using Test

import GeometricEquations: problem
import GeometricIntegratorsBase: GeometricIntegrator, cache, solutionstep

include("../helpers/testsetup.jl")

const M = MultisymplecticIntegrators

"""The target of the fit's scenarios, `u(t, x) = sin(πx) cos(πt)`, in physical time."""
target(t, x, params) = sinpi(x) * cospi(t)

"""
The slab of a `PINNTraining` fit on `Wave` at `h = 0.05`: the cache, its block, the interior sample
the other fits take, and the sample the fit takes from it.
"""
function pinn_slab(::Type{T}; S = 10, D = 1, mode = :all, epochs = 100, step = 1e-2,
        nsamples = 5_000, batch = 100) where {T}
    h = T(0.05)
    problem = Wave.problem(T = T, D = D, timespan = (zero(T), h), timestep = h,
        xstep = T(0.05))
    ansatz = NetworkAnsatz(S = S, mode = mode, xspan = (zero(T), one(T)), degree = 3,
        ncells = (2, 5), D = D)
    method = PINNTraining(epochs = epochs, step = step, batch = batch, nsamples = nsamples)
    int = GeometricIntegrator(problem,
        SpaceTimeGalerkin(ansatz, HardConstraintLifting();
            initial_guess = FitTo(TargetFunction(target); method = method)))
    c = cache(int)
    solstep = solutionstep(int, GeometricSolution(problem)[0])
    M.slab_bottom!(c, c.coupling, problem, ansatz, solstep)
    tgt = TargetFunction(target)
    tnodes, xnodes = M.quadrature_points(c.ansatz)
    interior = zeros(T, length(tnodes), length(xnodes), D)
    M.evaluate!(interior, tgt, c, ansatz, problem, solstep)
    return (; problem, ansatz, int, c, solstep, method, tgt, interior,
        sample = M.fit_sample(method, interior, tgt, c, ansatz, problem, solstep))
end

"""
The relative `L²` error of the block's field against the target at the nodes of the ansatz's own
rule: the misfit the fit descends, read from the block's own evaluation and not from the fit.
"""
function misfit(c, problem, ::Type{T}, h) where {T}
    tq, xq = M.quadrature_points(c.ansatz)
    a = c.ansatz
    err = zero(T)
    scale = zero(T)
    t₀ = timespan(problem)[1]
    for (q, t̂) in enumerate(tq), (r, x) in enumerate(xq)

        M.evaluate_at!(a, t̂, x)
        want = target(t₀ + h * t̂, x, parameters(problem))
        err += abs2(a.values[1] - want)
        scale += abs2(want)
    end
    return sqrt(err / scale)
end

"""A seeded start for the fit, in `[-1, 1]`, of the length of the solver's vector."""
function seeded_start(::Type{T}, n; seed = 20261113) where {T}
    return 2 .* rand(Xoshiro(seed), T, n) .- 1
end

"""The misfit of a fitted vector, read from the block after the vector is written into it."""
function fitted_misfit(s, w, ::Type{T}) where {T}
    M.set_parameters!(s.c.ansatz, copy(w))
    return misfit(s.c, s.problem, T, T(0.05))
end

# `@allocated` through a function barrier of concrete argument types, after one warm-up call.
function allocations(w, method, s)
    M.fit!(w, method, s.ansatz, s.sample, s.c)
    return @allocated M.fit!(w, method, s.ansatz, s.sample, s.c)
end

"""
The bound of P1's factor, per precision: the misfit falls by at least this factor over the 100
epochs of the default step. The factors this tree measures from a seeded start in `[-1, 1]` are
`2.9` in `Float64` and `4.3` in `Float32`, on `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia
1.13.3; the bound sits below each of them.
"""
pinn_factor(::Type{Float64}) = 2.0
pinn_factor(::Type{Float32}) = 2.0

@testset "The PINN fit" begin
    @testset "P1 the fit improves the network on a sampled target, $T" for T in (Float64,
        Float32)
        s = pinn_slab(T)
        n = length(s.c.x)
        w₀ = seeded_start(T, n)
        M.set_parameters!(s.c.ansatz, copy(w₀))
        before = misfit(s.c, s.problem, T, T(0.05))
        # the same fit run twice with one method object: bit-identical points and parameters
        first_sample = M.fit_sample(s.method, s.interior, s.tgt, s.c, s.ansatz, s.problem,
            s.solstep)
        second_sample = M.fit_sample(s.method, s.interior, s.tgt, s.c, s.ansatz, s.problem,
            s.solstep)
        @test first_sample.points == second_sample.points
        @test first_sample.values == second_sample.values
        @test size(first_sample.points) == (5_000, 2)
        @test size(first_sample.values) == (5_000, 1)
        @test all(t̂ -> zero(T) ≤ t̂ ≤ one(T), first_sample.points[:, 1])
        @test all(x -> zero(T) ≤ x ≤ one(T), first_sample.points[:, 2])
        w₁ = copy(w₀)
        M.fit!(w₁, s.method, s.ansatz, first_sample, s.c)
        M.set_parameters!(s.c.ansatz, copy(w₀))
        w₂ = copy(w₀)
        M.fit!(w₂, s.method, s.ansatz, second_sample, s.c)
        @test w₁ == w₂
        after = fitted_misfit(s, w₁, T)
        factor = before / after
        @test isfinite(after)
        @test after < before / T(pinn_factor(T))
        @info("the PINN fit on its own sample", :P1, T, before, after, factor, VERSION,
            machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads())
    end

    @testset "P2 epochs and the fit error against the greedy fit, $T" for T in (Float64,
        Float32)
        s = pinn_slab(T)
        n = length(s.c.x)
        w₀ = seeded_start(T, n)
        M.set_parameters!(s.c.ansatz, copy(w₀))
        before = misfit(s.c, s.problem, T, T(0.05))
        # `epochs = 0` leaves the parameters exactly at their initial values
        none = PINNTraining(epochs = 0)
        w = copy(w₀)
        M.fit!(w, none, s.ansatz,
            M.fit_sample(none, s.interior, s.tgt, s.c, s.ansatz,
                s.problem, s.solstep),
            s.c)
        @test w == w₀
        # the default's error is below that value
        w = copy(w₀)
        M.fit!(w, s.method, s.ansatz, s.sample, s.c)
        after = fitted_misfit(s, w, T)
        @test after < before
        # and the greedy fit's error on the same target and nodes, beside it
        greedy = copy(w₀)
        M.fit!(greedy, OrthogonalGreedy(), s.ansatz, s.interior, s.c)
        greedy_error = fitted_misfit(s, greedy, T)
        @test isfinite(greedy_error)
        @info("the PINN fit against the greedy fit",
            T, before, after, greedy_error, VERSION,
            machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
    end

    @testset "P3 the mode keeps its fixed parameters, $T" for T in (Float64,)
        # a `:last_layer` network frees its output weights alone; the hidden layer stays put
        s = pinn_slab(T; mode = :last_layer)
        w₀ = seeded_start(T, M.nparameters(s.ansatz))
        M.set_parameters!(s.c.ansatz, copy(w₀))
        w = copy(w₀)
        M.fit!(w, s.method, s.ansatz, s.sample, s.c)
        lo = s.ansatz.first_free
        @test w[1:(lo - 1)] == w₀[1:(lo - 1)]
        @test any(w[lo:end] .!= w₀[lo:end])
        # and an `:all` network moves every parameter
        a = pinn_slab(T)
        wₐ₀ = seeded_start(T, M.nparameters(a.ansatz); seed = 20261114)
        M.set_parameters!(a.c.ansatz, copy(wₐ₀))
        wₐ = copy(wₐ₀)
        M.fit!(wₐ, a.method, a.ansatz, a.sample, a.c)
        @test all(wₐ .!= wₐ₀)
    end

    @testset "P4 the descent allocates nothing per epoch, $T" for T in (Float64,)
        s = pinn_slab(T)
        w = seeded_start(T, length(s.c.x))
        M.set_parameters!(s.c.ansatz, copy(w))
        one_epoch = allocations(w, PINNTraining(epochs = 1), s)
        ten_epochs = allocations(w, PINNTraining(epochs = 10), s)
        # 50 batches per epoch: a buffer allocated inside the batch loop would make the second
        # count the larger of the two
        @test one_epoch == ten_epochs
        @test one_epoch > 0
    end

    @testset "P5 the fit seeds a zero network, $T" for T in (Float64, Float32)
        # the block of a fresh integrator holds zero parameters, where a `tanh` network is
        # stationary: the guess' fit draws its start from the sample's generator, so the first
        # slab of a run moves
        s = pinn_slab(T)
        @test all(iszero, s.c.ansatz.params)
        zero_misfit = misfit(s.c, s.problem, T, T(0.05))
        M.initial_guess!(s.solstep, nothing, parameters(s.problem), s.int)
        n = s.ansatz.nfree
        fitted = s.ansatz.first_free:(s.ansatz.first_free + n - 1)
        @test all(!iszero, s.c.ansatz.params[fitted])
        after = misfit(s.c, s.problem, T, T(0.05))
        @test isfinite(after)
        @test after < zero_misfit
        @info("the PINN fit of a zero network", T, zero_misfit, after, VERSION,
            machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
    end
end
