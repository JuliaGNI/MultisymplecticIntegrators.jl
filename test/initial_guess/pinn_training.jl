using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS
using MultisymplecticIntegrators
using Random
using Test

import GeometricEquations: problem
import GeometricIntegratorsBase: GeometricIntegrator, cache, current, reset!, solutionstep

include("../helpers/testsetup.jl")

const M = MultisymplecticIntegrators

"""The target of the fit's scenarios, `u(t, x) = sin(πx) cos(πt)`, in physical time."""
target(t, x, params) = sinpi(x) * cospi(t)

"""
The two-component target of the scalar-and-vector scenarios, `u(t, x) = (sin(πx) cos(πt),
sin(2πx) cos(πt) / 2)`: its components differ by up to `√3/2 · 3/2 = 1.299` on the slab, so a fit
that reads one component's target value for both components is not the fit this target asks for.
"""
target₂(t, x, params) = (sinpi(x) * cospi(t), sinpi(2x) * cospi(t) / 2)

"""The target of a fit with `D` components: the scalar one, or the two-component one."""
pinn_target(D) = TargetFunction(D === 1 ? target : target₂)

"""
    pinn_problem(::Type{T}, D)

The problem of the fit's slab at `h = 0.05`: `Wave` for a scalar field and `VectorWave` for a
two-component one, so that the scenario runs on the scalar problem and on the vector problem alike.
"""
function pinn_problem(::Type{T}, D) where {T}
    h = T(0.05)
    if D === 1
        return Wave.problem(T = T, D = D, timespan = (zero(T), h), timestep = h,
            xstep = T(0.05))
    end
    return VectorWave.problem(T = T, timespan = (zero(T), h), timestep = h,
        xstep = T(0.05))
end

"""
    pinn_slab(::Type{T}; S, D, mode, epochs, step, nsamples, batch)

The slab of a `PINNTraining` fit: the cache, its block, the interior sample the other fits take, and
the sample the fit takes from it. The solution step is reset to the time the first step of the run
ends, as the integration loop resets it, so the slab starts at the problem's own `timespan[1]` and
the sample's target is read over this slab's own interval.
"""
function pinn_slab(::Type{T}; S = 10, D = 1, mode = :all, epochs = 100, step = 1e-2,
        nsamples = 5_000, batch = 100) where {T}
    problem = pinn_problem(T, D)
    ansatz = NetworkAnsatz(S = S, mode = mode, xspan = (zero(T), one(T)), degree = 3,
        ncells = (2, 5), D = D)
    method = PINNTraining(epochs = epochs, step = step, batch = batch, nsamples = nsamples)
    tgt = pinn_target(D)
    int = GeometricIntegrator(problem,
        SpaceTimeGalerkin(ansatz, HardConstraintLifting();
            initial_guess = FitTo(tgt; method = method)))
    c = cache(int)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    reset!(solstep, timesteps(solution)[1])
    now = current(solstep)
    M.slab_bottom!(c, c.coupling, problem, ansatz, now)
    tnodes, xnodes = M.quadrature_points(c.ansatz)
    interior = zeros(T, length(tnodes), length(xnodes), D)
    M.evaluate!(interior, tgt, c, ansatz, problem, now)
    return (; problem, ansatz, int, c, solstep, now, method, tgt, interior,
        sample = M.fit_sample(method, interior, tgt, c, ansatz, problem, now))
end

"""
The relative `L²` error of the block's field against the pointwise target `f` at the nodes of the
ansatz's own rule: the misfit the fit descends, read from the block's own evaluation and not from
the fit, over every component of the field, each against its own entry of the target.
"""
function misfit(c, problem, ::Type{T}, h, f) where {T}
    tq, xq = M.quadrature_points(c.ansatz)
    a = c.ansatz
    err = zero(T)
    scale = zero(T)
    t₀ = timespan(problem)[1]
    for (q, t̂) in enumerate(tq), (r, x) in enumerate(xq)

        M.evaluate_at!(a, t̂, x)
        want = f(t₀ + h * t̂, x, parameters(problem))
        for d in 1:a.ansatz.D
            value = want isa Number ? want : want[d]
            err += abs2(a.values[d] - value)
            scale += abs2(value)
        end
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
    return misfit(s.c, s.problem, T, T(0.05), s.tgt.f)
end

# `@allocated` through a function barrier of concrete argument types, after one warm-up call.
function allocations(w, method, s)
    M.fit!(w, method, s.ansatz, s.sample, s.c)
    return @allocated M.fit!(w, method, s.ansatz, s.sample, s.c)
end

"""
The bound of P1's factor, per precision: the misfit falls by at least this factor over the 100
epochs of the default step, at both `D = 1` and `D = 2`. The factors this tree measures from a
seeded start in `[-1, 1]` are `3.92` in `Float64` and `16.86` in `Float32` at `D = 1`, and `3.85`
and `3.47` at `D = 2`, on `arm64-apple-darwin27.0.0` (`apple-m4`) at Julia 1.13.3 with four BLAS
threads; the same four fits at Julia 1.12.7 round to the same four factors. The bound sits below
each of them.
"""
pinn_factor(::Type{Float64}) = 2.0
pinn_factor(::Type{Float32}) = 2.0

"""
The record of P1's fit: its misfit before and after the fit and the factor between them, beside the
metadata every record of this file carries — the element type, the number of components, the target
of the fit, the Julia version, the machine and CPU, the BLAS thread count and the `--check-bounds`
setting. The testset "P7 the records carry the metadata that makes them readable" checks that
metadata; the factor itself is recorded and never compared with another machine's.
"""
function record_p1(tgt, ::Type{T}, D, before, after, factor) where {T}
    @info("the PINN fit on its own sample", :P1, T, D, before, after, factor, VERSION,
        target=tgt, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

"""
The record of P2: the fit's two misfits and the greedy fit's error on the same target and nodes,
beside the metadata that P1's record carries. No bound is asserted against the greedy fit.
"""
function record_p2(tgt, ::Type{T}, D, before, after, greedy_error) where {T}
    @info("the PINN fit against the greedy fit", :P2, T, D, before, after, greedy_error,
        VERSION, target=tgt, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

"""
The record of P5: the misfit of a fresh block's zero parameters and the fit's misfit after the guess,
beside the metadata that P1's record carries.
"""
function record_p5(tgt, ::Type{T}, D, zero_misfit, after) where {T}
    @info("the PINN fit of a zero network", :P5, T, D, zero_misfit, after, VERSION,
        target=tgt, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

@testset "The PINN fit" begin
    @testset "P1 the fit improves the network on a sampled target, $T, D = $D" for T in (
            Float64, Float32),
        D in (1, 2)

        s = pinn_slab(T; D = D)
        n = length(s.c.x)
        w₀ = seeded_start(T, n)
        M.set_parameters!(s.c.ansatz, copy(w₀))
        before = misfit(s.c, s.problem, T, T(0.05), s.tgt.f)
        # the slab starts at the problem's own first time, so the sample is read over this slab
        @test M.slab_start(s.problem, s.now) == timespan(s.problem)[1]
        # the same fit run twice with one method object: bit-identical points and parameters
        first_sample = M.fit_sample(s.method, s.interior, s.tgt, s.c, s.ansatz, s.problem,
            s.now)
        second_sample = M.fit_sample(s.method, s.interior, s.tgt, s.c, s.ansatz, s.problem,
            s.now)
        @test first_sample.points == second_sample.points
        @test first_sample.values == second_sample.values
        @test first_sample.start == second_sample.start
        @test size(first_sample.points) == (5_000, 2)
        @test size(first_sample.values) == (5_000, D)
        # the two components' samples are separated, so a sample that carries one component's
        # target value in both is not this sample: the two differ by up to `T(1.299)` on the slab
        if D > 1
            @test maximum(abs, first_sample.values[:, 1] - first_sample.values[:, 2]) ≥
                  T(1 // 2)
        end
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
        record_p1(s.tgt, T, D, before, after, factor)
    end

    @testset "P2 epochs and the fit error against the greedy fit, $T, D = $D" for T in (Float64,
            Float32),
        D in (1, 2)

        s = pinn_slab(T; D = D)
        n = length(s.c.x)
        w₀ = seeded_start(T, n)
        M.set_parameters!(s.c.ansatz, copy(w₀))
        before = misfit(s.c, s.problem, T, T(0.05), s.tgt.f)
        # `epochs = 0` leaves the parameters exactly at their initial values
        none = PINNTraining(epochs = 0)
        w = copy(w₀)
        M.fit!(w, none, s.ansatz,
            M.fit_sample(none, s.interior, s.tgt, s.c, s.ansatz,
                s.problem, s.now),
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
        record_p2(s.tgt, T, D, before, after, greedy_error)
    end

    @testset "P3 the mode keeps its fixed parameters, $T, D = $D" for T in (Float64,),
        D in (1, 2)
        # a `:last_layer` network frees its output weights alone; the hidden layer stays put
        s = pinn_slab(T; mode = :last_layer, D = D)
        w₀ = seeded_start(T, M.nparameters(s.ansatz))
        M.set_parameters!(s.c.ansatz, copy(w₀))
        w = copy(w₀)
        M.fit!(w, s.method, s.ansatz, s.sample, s.c)
        lo = s.ansatz.first_free
        @test w[1:(lo - 1)] == w₀[1:(lo - 1)]
        @test any(w[lo:end] .!= w₀[lo:end])
        # and an `:all` network moves every parameter
        a = pinn_slab(T; D = D)
        wₐ₀ = seeded_start(T, M.nparameters(a.ansatz); seed = 20261114)
        M.set_parameters!(a.c.ansatz, copy(wₐ₀))
        wₐ = copy(wₐ₀)
        M.fit!(wₐ, a.method, a.ansatz, a.sample, a.c)
        @test all(wₐ .!= wₐ₀)
    end

    @testset "P4 the descent allocates nothing per epoch, $T, D = $D" for T in (Float64,),
        D in (1, 2)

        s = pinn_slab(T; D = D)
        w = seeded_start(T, length(s.c.x))
        M.set_parameters!(s.c.ansatz, copy(w))
        one_epoch = allocations(w, PINNTraining(epochs = 1), s)
        ten_epochs = allocations(w, PINNTraining(epochs = 10), s)
        # 50 batches per epoch: a buffer allocated inside the batch loop would make the second
        # count the larger of the two
        @test one_epoch == ten_epochs
        @test one_epoch > 0
    end

    @testset "P5 the fit seeds a zero network, $T, D = $D" for T in (Float64, Float32),
        D in (1, 2)
        # the block of a fresh integrator holds zero parameters, where a `tanh` network is
        # stationary: the guess' fit draws its start from the sample's generator, so the first
        # slab of a run moves
        s = pinn_slab(T; D = D)
        @test all(iszero, s.c.ansatz.params)
        zero_misfit = misfit(s.c, s.problem, T, T(0.05), s.tgt.f)
        M.initial_guess!(s.solstep, nothing, parameters(s.problem), s.int)
        n = s.ansatz.nfree
        fitted = s.ansatz.first_free:(s.ansatz.first_free + n - 1)
        @test all(!iszero, s.c.ansatz.params[fitted])
        after = misfit(s.c, s.problem, T, T(0.05), s.tgt.f)
        @test isfinite(after)
        @test after < zero_misfit
        record_p5(s.tgt, T, D, zero_misfit, after)
    end

    @testset "P6 one epoch is one step per batch, $T" for T in (Float64, Float32)
        # `nsamples = 5 000` at `batch = 300` is 16 batches of 300 points and a last one of 200,
        # taken in the sample's own drawn order: the pass is the 17 single-batch fits below, so a
        # step taken once per epoch, a rate over the sample's size instead of the batch's, and a
        # rate over `batch` on the last batch each move the fitted parameters away from them
        s = pinn_slab(T)
        n = length(s.c.x)
        w₀ = seeded_start(T, n)
        M.set_parameters!(s.c.ansatz, copy(w₀))
        whole = copy(w₀)
        M.fit!(whole, PINNTraining(epochs = 1, batch = 300), s.ansatz, s.sample, s.c)
        M.set_parameters!(s.c.ansatz, copy(w₀))
        parts = copy(w₀)
        points, values = s.sample.points, s.sample.values
        for first in 1:300:size(points, 1)
            finish = min(first + 299, size(points, 1))
            sub = (; points = points[first:finish, :], values = values[first:finish, :],
                start = s.sample.start)
            M.fit!(parts, PINNTraining(epochs = 1, batch = 5_000), s.ansatz, sub, s.c)
        end
        @test whole == parts
    end

    @testset "P7 the records carry the metadata that makes them readable" begin
        # a figure out of an iteration is recorded, never asserted: the record itself carries the
        # element type, the components, the target, the version, the machine, the thread count and
        # the bounds setting, and this testset is what holds them there
        s = pinn_slab(Float64)
        logs, _ = Test.collect_test_logs() do
            record_p1(s.tgt, Float64, 1, 1.0, 0.5, 2.0)
            record_p2(s.tgt, Float64, 1, 1.0, 0.5, 0.4)
            record_p5(s.tgt, Float64, 1, 1.0, 0.5)
        end
        @test length(logs) == 3
        for log in logs
            kw = log.kwargs
            @test kw[:target] === s.tgt
            @test haskey(kw, :machine)
            @test haskey(kw, :cpu)
            @test kw[:blas_threads] == BLAS.get_num_threads()
            @test kw[:check_bounds] == Base.JLOptions().check_bounds
        end
        @test [log.message for log in logs] ==
              ["the PINN fit on its own sample", "the PINN fit against the greedy fit",
            "the PINN fit of a zero network"]
    end

    @testset "the fit's refusal on an ansatz with no parameters" begin
        # a coefficient ansatz has no free parameters for the descent, so a `PINNTraining` fit on
        # one is refused at the first fit, which both fitting `apply_guess!` methods reach, and not
        # when the integrator is built
        problem = Wave.problem(T = Float64, timespan = (0.0, 0.05), timestep = 0.05,
            xstep = 0.05)
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (2, 5), xspan = (0.0, 1.0), n = 3)
        method = SpaceTimeGalerkin(ansatz, HardConstraintLifting();
            initial_guess = FitTo(TargetFunction(target); method = PINNTraining()))
        int = GeometricIntegrator(problem, method)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        reset!(solstep, timesteps(solution)[1])
        err = try
            M.initial_guess!(solstep, nothing, parameters(problem), int)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("PINNTraining needs a NetworkAnsatz", err.msg)
    end
end
