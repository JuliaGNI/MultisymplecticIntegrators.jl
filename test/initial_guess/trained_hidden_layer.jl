using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS, norm
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using SimpleSplines
using Test

import GeometricIntegratorsBase: cache, current, solutionstep, solver, reset!
import SimpleSolvers: config, jacobian!, jacobianmatrix

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")
include("../helpers/action_oracle.jl")

const M = MultisymplecticIntegrators

# The trained fit is measured on what a fit decides: the misfit it leaves on its own sample, the
# four surfaces that sample carries, the draw it takes from its method, and the block its workspace
# lives in. A run that may not converge is a record (§V, U12, U24); the `:last_layer` controls of
# `Wave` are the runs this file asserts.

"""The steps of the family's run on `Wave`."""
const TRAINED_STEPS = 5

"""The hidden units of the family's run, and of its affine control."""
const TRAINED_S = 20

"""
The iterations a `:last_layer` control is allowed. With `f_abstol = 0` a step converges only once the
iterate settles, so past the exact step the solver keeps stepping until round-off settles and the
count differs between machines; the run asserts the convergence rule after two.
"""
const TRAINED_ITERATIONS = 2

"""The hidden units of the fit that is measured on a misfit or on its sample."""
const FIT_S = 8

"""
The entries of one fit's sample at `degree = 3` and `ncells = (2, 5)` and `D = 1`: the interior
rule's `10 · 20`, the two surfaces in time at the space rule's `2 · 20`, and the two in space at the
time rule's `2 · 8`.
"""
const SAMPLE_ENTRIES = 10 * 20 + 2 * 20 + 2 * 8

"""
The fraction of the drawn hidden layer's misfit that 60 epochs at `learning_rate = 1e-2` must
reach, measured on `arm64-apple-darwin27.0.0` at Julia 1.13.3: `5.55e-1` with the hidden layer drawn
and its output layer solved, and `4.41e-2` after the training, a ratio of `0.079`.
"""
const TRAINED_MISFIT_FRACTION = 1 // 2

"""
The two misfits of E1a's fit, by element type and components: `drawn` is the fit that solves the
output layer alone on the hidden layer `Xoshiro(1234)` draws, and `trained` the same fit after 60
epochs at `learning_rate = 1e-2`. Measured on `arm64-apple-darwin27.0.0` at Julia 1.13.3:
`0.5551 → 0.04409` in `Float64` at both `D = 1` and `D = 2`, `0.5551 → 0.04443` in `Float32` at
`D = 1`, and `0.5551 → 0.04402` in `Float32` at `D = 2`, where the least-squares solve carries a
second, zero column and its QR path differs. The same fits at Julia 1.12.7 leave `0.04485` in
`Float32` at `D = 1` and `0.04438` at `D = 2`.

The values are recorded, not asserted: a `Float32` misfit moves in its third significant digit
between the two Julia versions and under a one-ulp change of its learning rate, so no bound on it
holds on every leg, and the fraction is the one bound E1a asserts on them.
"""
const TRAINED_MISFITS = (
    drawn = Dict{Tuple{DataType, Int}, Float64}(
        (Float64, 1) => 0.5551, (Float64, 2) => 0.5551,
        (Float32, 1) => 0.5551, (Float32, 2) => 0.5551),
    trained = Dict{Tuple{DataType, Int}, Float64}(
        (Float64, 1) => 0.04409, (Float64, 2) => 0.04409,
        (Float32, 1) => 0.04443, (Float32, 2) => 0.04402))

function trained_ansatz(::Type{T}; S = FIT_S, D = 1, mode = :last_layer, degree = 3,
        ncells = (2, 5), xspan = (zero(T), one(T))) where {T}
    return NetworkAnsatz(S = S, D = D, mode = mode, xspan = xspan, degree = degree,
        ncells = ncells)
end

function wave_problem(::Type{T}; timestep = T(0.05), xstep = T(0.05),
        timespan = (zero(T), T(0.25))) where {T}
    return Wave.problem(T = T, timespan = timespan, timestep = timestep, xstep = xstep)
end

function sine_gordon_problem(::Type{T}; timestep = T(0.05)) where {T}
    return SineGordon.problem(T = T, timespan = (zero(T), T(0.15)), timestep = timestep,
        xstep = T(0.05))
end

function linear_transport_problem(::Type{T}; timestep = T(0.05),
        timespan = (zero(T), T(0.15))) where {T}
    return LinearTransport.problem(T = T, timespan = timespan, timestep = timestep,
        xstep = T(0.05), xspan = (T(-0.5), zero(T)))
end

"""A `VectorWave` problem on the vector density of the coupling `β`, which is `D = 2`. Its data
functions and its exact solution are `VectorWave.problem`'s."""
function vector_wave_problem(::Type{T}; β, timestep = T(0.05),
        timespan = (zero(T), T(0.5))) where {T}
    params = VectorWave.default_parameters()
    base = VectorWave.problem(T = T; params = params, timestep = timestep,
        timespan = timespan)
    return LagrangianPDEProblem(VectorWave.vector_density(; params = params, β = T(β));
        D = 2,
        ics_function = base.ics_function,
        bcs_function = base.bcs_function,
        solutions = base.solutions,
        timespan = timespan, timestep = timestep, xspan = base.xspan, xstep = base.xstep,
        params = params)
end

"""A problem rebuilt from another's two data functions, without an exact solution."""
function without_solutions(problem, density, D)
    return LagrangianPDEProblem(density;
        D = D,
        ics_function = problem.ics_function,
        bcs_function = problem.bcs_function,
        timespan = problem.timespan, timestep = problem.timestep,
        xspan = problem.xspan, xstep = problem.xstep, params = problem.params)
end

"""The method of one `:last_layer` network on `Wave` with the guess it is given."""
function trained_method(::Type{T}; S = FIT_S, D = 1, guess, xspan = (zero(T), one(T)),
        mode = :last_layer) where {T}
    return SpaceTimeGalerkin(trained_ansatz(T; S = S, D = D, mode = mode, xspan = xspan),
        WeakInitialAndBoundaryConditions(); initial_guess = guess)
end

"""
    trained_slab(problem, method)

The first slab of a run, set up as the family runs it: its datum in the coupling's block, and the
solver's vector from the method's guess, before the solver runs.
"""
function trained_slab(problem, method)
    int = GeometricIntegrator(problem, method)
    solution = GeometricSolution(problem)
    step = solutionstep(int, solution[0])
    reset!(step, timesteps(solution)[1])
    c = cache(int)
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), current(step))
    M.initial_guess!(step, nothing, parameters(problem), int)
    return int
end

"""
The interior sample the fit reads, of size `(Qt, Qx, D)`, rebuilt from the fit's own workspace in
the order the workspace lays its points out: the space index runs fastest inside a time row.
"""
function interior_sample(f, block)
    qt = length(block.tabulation.time.node)
    qx = length(block.tabulation.space.node)
    D = size(f.target, 2)
    sample = similar(f.target, qt, qx, D)
    k = 0
    for q in 1:qt, r in 1:qx

        k += 1
        for d in 1:D
            sample[q, r, d] = f.target[k, d]
        end
    end
    return sample
end

"""The misfit of a fitted parameter vector on the fit's own sample, `Σ_k ω_k (u_k − f_k)²`, read
back through the network's own evaluation."""
function fitted_misfit(block, w)
    f = block.train
    copyto!(block.params, w)
    total = zero(eltype(w))
    for k in axes(f.points, 2)
        M.evaluate_at!(block, f.points[1, k], f.points[2, k])
        for d in axes(f.target, 2)
            total += f.measure[k] * abs2(block.values[d] - f.target[k, d])
        end
    end
    return total
end

"""
    charge(cache)

`J_h = Σ_j ω_j (u¹_j λ²_j − u²_j λ¹_j)` at the top of the last slab: the trace `u` at the nodes of
the trace basis' space rule, the top multiplier `λ` there, and that rule's weights.
"""
function charge(c)
    s = c.ansatz
    ω = s.trace_tabulation.space.weight
    total = zero(eltype(c.trace_u))
    for r in eachindex(ω)
        total += ω[r] * (c.trace_u[r, 1] * s.lambda_top[r, 2] -
                  c.trace_u[r, 2] * s.lambda_top[r, 1])
    end
    return total
end

"""
    record_charge(record, T, J, deviation)

Write one charge record with the metadata that makes the figure readable on another machine. No
bound is asserted on it: `λ_top` lies in the near-null space of the `:last_layer` saddle point, so
resolving it makes the run unstable and truncating it leaves the slab's equations unsatisfied
there, and the drift is the same at every cutoff the factorisation offers.
"""
function record_charge(record::Symbol, ::Type{T}, J, deviation) where {T}
    @info("the rotation charge of a network run", record, T, J, deviation, VERSION,
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

"""
    record_ratio(record, T, ratio, residuals, initial_residuals)

Write one run's per-step residual ratio, in units of `eps(T)`, with the metadata that makes the
figure readable on another machine. No bound is asserted on it: the ratio of a `:last_layer` step is
the conditioning of its hidden layer, whose smallest singular directions one precision cannot
resolve, and no rank cutoff of the factorisation brings it to a constant multiple of `eps(T)` — the
`Float32` ratio is `1.0e-2` at a cutoff of `1e-3` and up to `7.9e5 eps` at `1e-10`.
"""
function record_ratio(
        record::Symbol, ::Type{T}, ratio, residuals, initial_residuals) where {T}
    @info("the residual ratio of a network solve, in units of eps(T)",
        record, T, ratio, residuals,
        initial_residuals, VERSION, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

"""The sup-norm deviation of a run from the `u` of a §A entry, over the steps it took."""
function sup_deviation(result, entry)
    return maximum(maximum(abs.(result.solution.u[n] .- entry.u[n, :]))
    for n in 1:length(result.traces))
end

"""Whether every state of a solution series is finite."""
finite_states(x) = all(n -> all(isfinite, x[n]), eachindex(x))

"""Whether two solution series are bit-identical on every state they hold."""
same_states(x, y) = axes(x) == axes(y) && all(n -> x[n] == y[n], eachindex(x))

"""
    record_deviation(record, entry, T, deviation)

Write one reference deviation with the metadata that makes the figure readable on another machine.
No bound is asserted on the value.
"""
function record_deviation(record::Symbol, entry::Symbol, ::Type{T}, deviation) where {T}
    @info("the reference deviation of an ELM run", record, entry, T, deviation, VERSION,
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

"""
    record_misfit(record, T, D, misfit, recorded)

Write one misfit of a fit on its own sample with the metadata that makes the figure readable on
another machine, beside the `const` the same fit is recorded as. No bound is asserted on it: a
`Float32` misfit moves in its third significant digit under a one-ulp change of its learning rate.
"""
function record_misfit(record::Symbol, ::Type{T}, D, misfit, recorded) where {T}
    @info("the misfit of a trained fit on its own sample", record, T, D, misfit, recorded,
        VERSION, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

@testset "Trained hidden layer" begin
    # The fit's two ends: the drawn hidden layer with its output layer solved, and the same with
    # the hidden layer trained first. The observable is the misfit on the fit's own sample, in both
    # components of a vector-valued field as well as on a scalar one.
    @testset "E1a the fit at epochs = 0 and at epochs > 0" for T in (Float64, Float32),
        D in (1, 2)

        scalar(t, x, p) = sin(2 * T(π) * x) * cos(T(π) * t)
        target = D === 1 ? scalar : (t, x, p) -> (scalar(t, x, p), zero(T))
        problem = D === 1 ? wave_problem(T) :
                  vector_wave_problem(T; β = zero(T), timespan = (zero(T), T(0.25)))
        make(epochs, lr) = trained_method(T; S = FIT_S, D = D,
            guess = FitTo(TargetFunction(target);
                method = TrainedHiddenLayer(epochs = epochs, learning_rate = lr)))
        drawn = cache(trained_slab(problem, make(0, T(1e-2)))).ansatz
        trained = cache(trained_slab(problem, make(60, T(1e-2)))).ansatz
        @test isfinite(drawn.train.misfit)
        @test isfinite(trained.train.misfit)
        @test drawn.train.misfit > zero(T)
        record_misfit(:drawn, T, D, drawn.train.misfit, TRAINED_MISFITS.drawn[(T, D)])
        record_misfit(:trained, T, D, trained.train.misfit, TRAINED_MISFITS.trained[(T, D)])
        @test trained.train.misfit ≤
              convert(T, TRAINED_MISFIT_FRACTION) * drawn.train.misfit
        # the misfit read back through the network's own evaluation is the fit's
        @test fitted_misfit(trained, copy(trained.params)) ≈ trained.train.misfit rtol = sqrt(eps(T))
        # both fits were taken on the same sample
        @test drawn.train.target == trained.train.target
        @test drawn.train.measure == trained.train.measure
        assert_no_upcast(trained.train.features, T)
        assert_no_upcast(trained.train.measure, T)
    end

    # The sample a fit is taken on carries the four surfaces of the slab beside its interior. A
    # target that vanishes at every node of the interior rule and not at the faces is zero on the
    # interior alone, so a fit that dropped the faces would fit the zero function and its field at
    # the faces would be zero.
    @testset "E1b the fit's sample carries the four faces" for T in (Float64, Float32)
        problem = wave_problem(T)
        ansatz = trained_ansatz(T; S = TRAINED_S)
        probe = cache(trained_slab(problem, trained_method(T; S = TRAINED_S,
            guess = Continuation()))).ansatz
        tn = collect(probe.tabulation.time.node)
        xn = collect(probe.tabulation.space.node)
        tt, xt = probe.trace_tabulation.time, probe.trace_tabulation.space
        h = timestep(problem)
        vanishing(t, x, p) = prod(t / h - q for q in tn) + prod(x - r for r in xn)
        int = trained_slab(problem,
            trained_method(T; S = TRAINED_S,
                guess = FitTo(TargetFunction(vanishing);
                    method = TrainedHiddenLayer(epochs = 0))))
        c = cache(int)
        f = c.ansatz.train
        tq, xq = c.ansatz.tabulation.time, c.ansatz.tabulation.space
        interior = length(tn) * length(xn)
        qxt, qtt = length(xt.weight), length(tt.weight)
        @test length(f.measure) == SAMPLE_ENTRIES
        @test length(f.target) == SAMPLE_ENTRIES
        @test maximum(abs, view(f.target, 1:interior, 1)) < 1 // 10^6
        @test maximum(abs, f.face_target) > zero(T)
        # the weight of every point of the sample: the interior's tensor product of the two rules,
        # and each of the four faces by the weights of the other direction
        expected = T[]
        for q in eachindex(tq.weight), r in eachindex(xq.weight)

            push!(expected, tq.weight[q] * xq.weight[r])
        end
        @test view(f.measure, 1:interior) == expected
        rows = interior
        for _ in 1:2
            @test f.measure[(rows + 1):(rows + qxt)] == xt.weight
            rows += qxt
        end
        for _ in 1:2
            @test f.measure[(rows + 1):(rows + qtt)] == tt.weight
            rows += qtt
        end
        faces = T[]
        for r in eachindex(xt.node)
            M.evaluate_at!(c.ansatz, zero(T), xt.node[r])
            push!(faces, c.ansatz.values[1])
            M.evaluate_at!(c.ansatz, one(T), xt.node[r])
            push!(faces, c.ansatz.values[1])
        end
        for r in 1:2, q in eachindex(tt.node)

            M.evaluate_at!(c.ansatz, tt.node[q], xspan(ansatz)[r])
            push!(faces, c.ansatz.values[1])
        end
        @test maximum(abs, faces) > T(1 // 1000) * maximum(abs, f.face_target)
    end

    # The draw is the method's own: two guesses of one method object agree whatever the process's
    # generator does, a method's generator is never advanced, and another seed draws another hidden
    # layer.
    @testset "E1c the random features come from the method's own RNG" for T in (Float64,)
        problem = wave_problem(T)
        method = TrainedHiddenLayer(epochs = 4)
        make(m) = trained_method(T; guess = FitTo(InitialCondition(); method = m))
        drawn(m) = rand(copy(m.rng), UInt64, 3)
        before = drawn(method)
        hidden = 3 * FIT_S
        first = copy(cache(trained_slab(problem, make(method))).ansatz.params)
        Random.seed!(20261108)
        second = copy(cache(trained_slab(problem, make(method))).ansatz.params)
        @test first == second
        @test drawn(method) == before
        other = TrainedHiddenLayer(epochs = 4, rng = Xoshiro(7))
        third = copy(cache(trained_slab(problem, make(other))).ansatz.params)
        @test third != first
        @test third[1:hidden] != first[1:hidden]
        # a second draw of the same generator reproduces the first fit's hidden parameters
        @test copy(cache(trained_slab(problem, make(method))).ansatz.params)[1:hidden] ==
              first[1:hidden]
    end

    # The mode the fit allows: Newton owns every layer in `:all` mode, so the fit refuses it; and
    # `epochs = 0` completes on the drawn hidden layer with the output layer at its least squares.
    @testset "E1d the mode and the epochs the fit allows" for T in (Float64,)
        problem = wave_problem(T)
        err = try
            trained_slab(problem,
                trained_method(T; mode = :all,
                    guess = FitTo(InitialCondition();
                        method = TrainedHiddenLayer(epochs = 0))))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(":last_layer", err.msg)
        @test occursin(":all", err.msg)
        c = cache(trained_slab(problem,
            trained_method(T;
                guess = FitTo(InitialCondition(); method = TrainedHiddenLayer(epochs = 0)))))
        f = c.ansatz.train
        w = copy(c.ansatz.params)
        @test f.method.epochs == 0
        # the hidden parameters are the draw, one scalar at a time from the method's own generator
        rng = copy(Xoshiro(1234))
        @test w[1:(3 * FIT_S)] ==
              T[T(2) * rand(rng, T) - one(T) for _ in 1:(3 * FIT_S)]
        @test any(x -> !iszero(x), w[(3 * FIT_S + 1):end])
        @test f.misfit ≤ fitted_misfit(c.ansatz, w) * (one(T) + sqrt(eps(T)))
        # the output layer is the least-squares solution, so the misfit's derivative with respect to
        # every output weight vanishes at it, read through the block's own evaluation and parameter
        # derivative: the largest measured is `1944 eps(T)` of the summed magnitudes of the terms
        for θ in (3 * FIT_S + 1):length(w)
            g, scale = zero(T), zero(T)
            for k in axes(f.points, 2)
                M.evaluate_at!(c.ansatz, f.points[1, k], f.points[2, k])
                for d in axes(f.target, 2)
                    ζ = 2 * f.measure[k] * (c.ansatz.values[d] - f.target[k, d])
                    g += ζ * c.ansatz.gradient[d, θ]
                    scale += abs(ζ * c.ansatz.gradient[d, θ])
                end
            end
            @test abs(g) ≤ 10^4 * eps(T) * scale
        end
        assert_no_upcast(f.features, T)
    end

    # The fit's buffers and the block are of the problem's element type, and no evaluation of a
    # `Float32` slab reaches `Float64` — on a scalar field and on a vector-valued one.
    @testset "E1e the element type of the fit and of its workspace" for T in (Float64, Float32),
        (D, problem) in ((1, wave_problem(T; timespan = (zero(T), T(0.1)))),
            (2, vector_wave_problem(T; β = zero(T), timespan = (zero(T), T(0.1)))))

        method = trained_method(T; D = D,
            guess = FitTo(InitialCondition();
                method = TrainedHiddenLayer(epochs = 4)))
        int = GeometricIntegrator(problem, method)
        c = cache(int)
        result = integrate(int)
        assert_no_upcast(c.ansatz.params, T)
        assert_no_upcast(c.ansatz.train.features, T)
        assert_no_upcast(c.ansatz.train.target, T)
        assert_no_upcast(c.ansatz.train.values, T)
        assert_no_upcast(c.ansatz.train.gradient, T)
        @test typeof(c.ansatz.train.misfit) === T
        assert_no_upcast(c.x, T)
        assert_no_upcast(result.solution.u, T)
        @test length(result.traces) == 2
    end

    # The layout of the output layer: the component index runs fastest inside a hidden unit, which
    # is the order `NeuralNetworkParameters` reads. A target whose second component vanishes leaves
    # the second component of the fitted field exactly zero, because the least-squares solution of a
    # zero right-hand side is zero; a fit that wrote the output weights unit-major would fill the
    # second component from the first one's weights.
    @testset "E1f the output layer is written in the parameter order" for T in (Float64,)
        problem = vector_wave_problem(T; β = 0.0, timespan = (zero(T), T(0.05)))
        target = (t, x, p) -> (sin(2 * T(π) * x) * cos(T(π) * t), zero(T))
        int = trained_slab(problem,
            trained_method(T; S = FIT_S, D = 2,
                guess = FitTo(TargetFunction(target);
                    method = TrainedHiddenLayer(epochs = 0))))
        c = cache(int)
        f = c.ansatz.train
        first, second = zero(T), zero(T)
        for k in axes(f.points, 2)
            M.evaluate_at!(c.ansatz, f.points[1, k], f.points[2, k])
            first = max(first, abs(c.ansatz.values[1]))
            second = max(second, abs(c.ansatz.values[2]))
        end
        @test first > zero(T)
        @test iszero(second)
    end

    # The trained fit takes every target of `§Guess`: the exact solution, which only a problem that
    # has one can supply, and the previous slab's field continued past its top. The sample is the
    # target's own values in every row it carries — the exact solution at the physical time of every
    # point, and the previous slab's field at `t̂ = 1`, carried up by its slope there — so the two
    # fits are what a fit that read another target would not be. The interior rows and the rows of
    # the four faces reach the target by two routes, and both continue the previous slab.
    @testset "E1g the trained fit takes every target of the guess" for T in (Float64, Float32)
        problem = wave_problem(T; timespan = (zero(T), T(0.1)))
        h = timestep(problem)
        t₀ = first(problem.timespan)
        exact = cache(trained_slab(problem,
            trained_method(T;
                guess = FitTo(ExactSolution(); method = TrainedHiddenLayer(epochs = 0))))).ansatz
        f = exact.train
        @test all(isfinite, f.target)
        @test maximum(abs, f.target) > zero(T)
        for k in axes(f.points, 2)
            @test f.target[k, 1] ==
                  problem.solutions.u(t₀ + h * f.points[1, k], f.points[2, k],
                parameters(problem))[1]
        end
        # the previous slab's target, read at the second slab, is the first slab's top field
        int = GeometricIntegrator(problem,
            trained_method(T;
                guess = FitTo(PreviousSlab(); method = TrainedHiddenLayer(epochs = 0))))
        c = cache(int)
        solution = GeometricSolution(problem)
        step = solutionstep(int, solution[0])
        reset!(step, timesteps(solution)[1])
        M.integrate_step!(current(step), nothing, parameters(problem), int)
        previous = copy(c.ansatz.params)
        reset!(step, timesteps(solution)[2])
        M.integrate_step!(current(step), nothing, parameters(problem), int)
        f = c.ansatz.train
        copyto!(c.ansatz.params, previous)
        for k in axes(f.points, 2)
            M.evaluate_at!(c.ansatz, one(T), f.points[2, k])
            @test f.target[k, 1] ==
                  c.ansatz.values[1] + f.points[1, k] * c.ansatz.values[2]
        end
    end

    # A method object carries no fit state: two integrators that share one give bit-identical runs,
    # a run repeated from the same start gives the same solution, and two fits on one block give
    # what two fresh blocks give.
    @testset "E2a two integrators, one method" for T in (Float64,)
        problem = wave_problem(T; timespan = (zero(T), T(0.15)))
        method = TrainedHiddenLayer(epochs = 4)
        ansatz = trained_ansatz(T)
        drawn = rand(copy(method.rng), UInt64, 3)
        shared = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = method))
        first_int = GeometricIntegrator(problem, shared)
        second_int = GeometricIntegrator(problem, shared)
        first = integrate(first_int)
        second = integrate(second_int)
        again = integrate(second_int)
        for run in (second, again)
            @test same_states(run.solution.u, first.solution.u)
            @test same_states(run.solution.v, first.solution.v)
            @test run.stats.iterations == first.stats.iterations
        end
        @test method.epochs == 4
        @test rand(copy(method.rng), UInt64, 3) == drawn
        # one block, two fits of two epochs, against the two fits of two fresh blocks, all on the
        # first slab's sample
        fresh(epochs) = copy(cache(trained_slab(problem,
            SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
                initial_guess = FitTo(InitialCondition();
                    method = TrainedHiddenLayer(epochs = epochs))))).ansatz.params)
        c = cache(trained_slab(problem,
            SpaceTimeGalerkin(ansatz,
                WeakInitialAndBoundaryConditions();
                initial_guess = FitTo(InitialCondition();
                    method = TrainedHiddenLayer(epochs = 4)))))
        f = c.ansatz.train
        sample = interior_sample(f, c.ansatz)
        w = copy(c.ansatz.params)
        M.fit!(w, TrainedHiddenLayer(epochs = 4), ansatz, sample, c)
        four = copy(w)
        M.fit!(w, TrainedHiddenLayer(epochs = 8), ansatz, sample, c)
        @test w == fresh(8)
        @test four == fresh(4)
        @test w != four
        # the workspace names the method it was last built for, so the key holds `epochs`
        @test f.method.epochs == 8
    end

    # The direct linear solve is offered to a `:last_layer` network on a linear density, and to no
    # other network; the mode is what the first refusal names and the density the second.
    @testset "E3a the option is admitted for the mode alone" for T in (Float64, Float32)
        transport = linear_transport_problem(T)
        xspan = (T(-0.5), zero(T))
        last_layer = SpaceTimeGalerkin(trained_ansatz(T; S = 20, D = 2, xspan = xspan),
            WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()),
            linear_solver = true)
        @test last_layer.linear_solver === true
        admitted = GeometricIntegrator(transport, last_layer)
        @test nlsolution(admitted) !== nothing
        all_mode = try
            GeometricIntegrator(transport,
                SpaceTimeGalerkin(
                    trained_ansatz(T; S = 20, D = 2, mode = :all,
                        xspan = xspan),
                    WeakInitialAndBoundaryConditions();
                    initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()),
                    linear_solver = true))
            nothing
        catch e
            e
        end
        @test all_mode isa ArgumentError
        @test occursin(":all", all_mode.msg)
        @test occursin("affine", all_mode.msg)
        nonlinear = try
            GeometricIntegrator(vector_wave_problem(T; β = 0.5),
                SpaceTimeGalerkin(trained_ansatz(T; S = 20, D = 2),
                    WeakInitialAndBoundaryConditions();
                    initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()),
                    linear_solver = true))
            nothing
        catch e
            e
        end
        @test nonlinear isa ArgumentError
        @test occursin("density", nonlinear.msg)
        @test occursin("linear", nonlinear.msg)
    end

    # The option's step under a guess that leaves the hidden layer alone: one factorisation over the
    # run, two recorded solves per step, §V's convergence rule under the family's own tolerance, and
    # a solution of the problem's element type. Each step's residual ratio is recorded in units of
    # `eps(T)` and no floor is asserted on it. The Newton arm records its own figures and asserts
    # neither.
    @testset "E3b one factorisation, two iterations and the recorded residual ratio" for T in (
        Float64, Float32)
        transport = linear_transport_problem(T)
        ansatz = trained_ansatz(T; S = 20, D = 2, xspan = (T(-0.5), zero(T)))
        make(linear) = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()),
            linear_solver = linear)
        int = GeometricIntegrator(transport, make(true); f_abstol = zero(T),
            f_reltol = trained_f_reltol(T))
        c = cache(int)
        result = integrate(int)
        options = config(solver(int))
        @test length(result.traces) == 3
        @test M.factorisations(int) == 1
        @test c.iterations == fill(2, 3)
        @test all(c.residuals .≤
                  options.f_abstol .+ options.f_reltol .* c.initial_residuals)
        # the residual of a step is the step's own observable, and a fabricated zero would pass the
        # convergence rule: every one is finite and positive
        @test all(isfinite, c.residuals)
        @test all(>(0), c.residuals)
        for n in 1:3
            @test all(isfinite, result.solution.u[n])
            @test eltype(result.solution.u[n]) === T
        end
        assert_no_upcast(c.x, T)
        record_ratio(:E3b, T, c.residuals ./ c.initial_residuals ./ eps(T), c.residuals,
            c.initial_residuals)
        # the Newton arm is a record of its own iterations and residuals
        newton = GeometricIntegrator(transport, make(false))
        newton_result = integrate(newton)
        @test length(newton_result.traces) == 3
        @test all(isfinite, newton_result.solution.u[1])
        @test length(cache(newton).iterations) == 3
        @info("the Newton arm of the same three steps", T, cache(newton).iterations,
            cache(newton).residuals)
    end

    # The factorisation is the step's Jacobian, and a `:last_layer` step's Jacobian reads the hidden
    # layer the mode holds fixed: a guess that retrains that layer at every step against a target
    # that moves takes the factorisation again at every step, and every step still converges. A path
    # that kept the run's first factorisation would keep it against a Jacobian that has moved.
    @testset "E3c the factorisation follows the hidden layer" for T in (Float64, Float32)
        transport = linear_transport_problem(T)
        moving = (t, x, p) -> (sin(2 * T(π) * x + 200 * t), zero(T))
        method = SpaceTimeGalerkin(
            trained_ansatz(T; S = 20, D = 2,
                xspan = (T(-0.5), zero(T))),
            WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(TargetFunction(moving);
                method = TrainedHiddenLayer(epochs = 60, learning_rate = 0.1)),
            linear_solver = true)
        int = GeometricIntegrator(transport, method; f_abstol = zero(T),
            f_reltol = trained_f_reltol(T))
        c = cache(int)
        result = integrate(int)
        @test length(result.traces) == 3
        @test M.factorisations(int) == 3
        @test all(c.converged)
        # the copy holds the hidden layer alone: a guess that carries the output weights from slab to
        # slab leaves the parameters the Jacobian reads unchanged, so that run factorises once
        carried = SpaceTimeGalerkin(
            trained_ansatz(T; S = 20, D = 2, xspan = (T(-0.5), zero(T))),
            WeakInitialAndBoundaryConditions();
            initial_guess = Continuation(), linear_solver = true)
        other = GeometricIntegrator(transport, carried; f_abstol = zero(T),
            f_reltol = trained_f_reltol(T))
        carried_result = integrate(other)
        @test length(carried_result.traces) == 3
        @test M.factorisations(other) == 1
        @test all(cache(other).converged)
        @info("the moved hidden layer's factorisations", T, M.factorisations(int),
            c.iterations, c.converged, c.residuals ./ c.initial_residuals,
            trained_f_reltol(T))
    end

    # The family runs on a problem that has no exact solution, from one guess: the trained fit of the
    # previous slab's field, which falls back to the trained fit of the initial condition at the
    # first slab. Its first step is that fallback bit for bit.
    @testset "E4a the family on Wave without an exact solution" for T in (Float64, Float32)
        problem = without_solutions(wave_problem(T), Wave.density(), 1)
        @test problem.solutions === nothing
        int = GeometricIntegrator(problem,
            trained_method(T; S = TRAINED_S,
                guess = FitTo(PreviousSlab();
                    method = TrainedHiddenLayer(epochs = 20)));
            f_abstol = zero(T), f_reltol = trained_f_reltol(T),
            max_iterations = TRAINED_ITERATIONS)
        c = cache(int)
        result = integrate(int)
        @test length(result.traces) == TRAINED_STEPS
        @test all(c.converged)
        @test maximum(c.iterations) ≤ TRAINED_ITERATIONS
        @test maximum(c.residuals ./ c.initial_residuals) ≤ trained_f_reltol(T)
        @test finite_states(result.solution.u)
        @test finite_states(result.solution.v)
        assert_no_upcast(result.solution.u, T)
        assert_no_upcast(result.solution.v, T)
        assert_no_upcast(c.x, T)
        assert_no_upcast(c.trace_u, T)
        # the first step is the trained fit of the initial condition, bit for bit
        initial = GeometricIntegrator(problem,
            trained_method(T; S = TRAINED_S,
                guess = FitTo(InitialCondition();
                    method = TrainedHiddenLayer(epochs = 20)));
            f_abstol = zero(T), f_reltol = trained_f_reltol(T),
            max_iterations = TRAINED_ITERATIONS)
        c1 = cache(initial)
        solution = GeometricSolution(problem)
        step = solutionstep(initial, solution[0])
        reset!(step, timesteps(solution)[1])
        M.integrate_step!(current(step), nothing, parameters(problem), initial)
        @test c1.iterations[1] === c.iterations[1]
        @test c1.residuals[1] === c.residuals[1]
        @test c1.initial_residuals[1] === c.initial_residuals[1]
        @info("the family's Wave run", T, c.iterations, c.converged,
            maximum(c.residuals ./ c.initial_residuals), trained_f_reltol(T))
    end

    # The family's run on a density that is not linear completes and records its figures, and
    # nothing about them is asserted: a method that may fail is a record (§V, U12, U24).
    @testset "E4b the family on SineGordon without an exact solution" for T in (Float64, Float32)
        problem = without_solutions(sine_gordon_problem(T), SineGordon.density(), 1)
        @test problem.solutions === nothing
        int = GeometricIntegrator(problem,
            trained_method(T; S = TRAINED_S,
                guess = FitTo(PreviousSlab();
                    method = TrainedHiddenLayer(epochs = 20)));
            f_abstol = zero(T), f_reltol = trained_f_reltol(T))
        c = cache(int)
        result = integrate(int)
        @test length(result.traces) == 3
        @test length(c.iterations) == 3
        @test length(c.residuals) == 3
        @test length(c.initial_residuals) == 3
        @test length(c.converged) == 3
        @test finite_states(result.solution.u)
        @info("the family's SineGordon run", T, c.iterations, c.residuals,
            c.initial_residuals, c.converged)
    end

    # The two records of the old stack's ELM formulation, at its own cells: a run of each completes
    # and its deviation from the entry is written with its metadata. Both entries are unconverged
    # iterates of another architecture, so the deviation is a record (§L, 2026-11-08).
    @testset "E5a the two section A soundings" for T in (Float64,)
        elm_method(problem) = SpaceTimeGalerkin(
            NetworkAnsatz(S = TRAINED_S, mode = :last_layer, xspan = problem.xspan,
                ncells = (10, 10), degree = 3),
            WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition();
                method = TrainedHiddenLayer(epochs = 100, learning_rate = T(1e-5))))
        cases = ((:WAVE_ELM_LSGD, wave_problem(T), WAVE_ELM_LSGD, 5),
            (:SINE_GORDON_ELM_LSGD, sine_gordon_problem(T), SINE_GORDON_ELM_LSGD, 3))
        for (name, problem, entry, steps) in cases
            int = GeometricIntegrator(problem, elm_method(problem))
            result = integrate(int)
            c = cache(int)
            @test length(result.traces) == steps
            @test length(c.iterations) == steps
            @test finite_states(result.solution.u)
            deviation = sup_deviation(result, entry)
            @test isfinite(deviation)
            record_deviation(:E5a, name, T, deviation)
            @info("the ELM record", T, name, c.iterations, c.residuals,
                c.initial_residuals, c.converged, deviation)
        end
    end

    # The action oracle of the mode that frees a part of the parameters reads the fixed ones from
    # the block it is given, and a call without one is refused.
    @testset "S5a the oracle reads every parameter" for T in (Float64,)
        problem = wave_problem(T)
        make(mode) = SpaceTimeGalerkin(trained_ansatz(T; S = 10, mode = mode),
            WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        all_slab = trained_slab(problem, make(:all))
        @test unknown_count(ActionOracle(problem, make(:all))) == 56
        @test length(nlsolution(all_slab)) == 56
        last_slab = trained_slab(problem, make(:last_layer))
        oracle = ActionOracle(problem, make(:last_layer); block = cache(last_slab).ansatz)
        @test unknown_count(oracle) == 26
        @test unknown_count(oracle) ==
              M.nfree(ansatz(make(:last_layer))) + (oracle.nλ + 2 * oracle.nμ) * 1
        @test length(nlsolution(last_slab)) == 26
        err = try
            ActionOracle(problem, make(:last_layer))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test err.msg ==
              "the action oracle of a :last_layer network needs the block that " *
              "holds its fixed parameters"
    end

    # The residual of a slab whose mode frees a part of the parameters is the action's gradient
    # there too, once its free entries are laid out from `first_free` and its fixed ones are the
    # block's.
    @testset "S5b the residual of a last-layer slab is the action's gradient" for T in (Float64,
        Float32)
        problem = wave_problem(T)
        for mode in (:all, :last_layer)
            method = SpaceTimeGalerkin(trained_ansatz(T; S = 10, mode = mode),
                WeakInitialAndBoundaryConditions();
                initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
            int = trained_slab(problem, method)
            c = cache(int)
            oracle = mode === :all ? ActionOracle(problem, method) :
                     ActionOracle(problem, method; block = c.ansatz)
            N = length(c.x)
            @test N == unknown_count(oracle)
            θ = T.(2 .* rand(Xoshiro(11), N) .- 1)
            b = zeros(T, N)
            M.residual!(b, θ, nothing, parameters(problem), int)
            ∇S = oracle_residual(oracle, first_datum(oracle), θ)
            @test all(isfinite, b)
            @test maximum(abs, ∇S) ≥ T(1 // 100)
            @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * maximum(abs, ∇S)
        end
    end

    # The `VectorWave` density at `β ≠ 0`: the run completes, both components are finite, every
    # step's figures are recorded beside its rotation charge, and the first slab's residual is the
    # action's gradient. Neither accuracy nor convergence is asserted, and the charge's deviation is
    # a record (§V, U12, U24).
    @testset "E7a VectorWave at β ≠ 0 is finite and its residual is the action's gradient" for T in (Float64,)
        problem = vector_wave_problem(T; β = 0.5)
        method = trained_method(T; S = TRAINED_S, D = 2,
            guess = FitTo(InitialCondition();
                method = TrainedHiddenLayer(epochs = 20)))
        int = GeometricIntegrator(problem, method;
            f_abstol = zero(T), f_reltol = trained_f_reltol(T))
        c = cache(int)
        solution = GeometricSolution(problem)
        step = solutionstep(int, solution[0])
        J = T[]
        for n in 1:10
            reset!(step, timesteps(solution)[n])
            M.integrate_step!(current(step), nothing, parameters(problem), int)
            copy!(solution, current(step), n)
            push!(J, charge(c))
        end
        @test length(J) == 10
        @test all(isfinite, J)
        @test length(c.iterations) == 10
        @test length(c.residuals) == 10
        @test finite_states(solution.u)
        @test finite_states(solution.v)
        @test all(isfinite, c.trace_u)
        record_charge(:E7a, T, J, maximum(x -> abs(x - J[1]), J))
        first_slab = trained_slab(problem, method)
        c1 = cache(first_slab)
        oracle = ActionOracle(problem, method; block = c1.ansatz)
        N = length(c1.x)
        θ = T.(2 .* rand(Xoshiro(11), N) .- 1)
        b = zeros(T, N)
        M.residual!(b, θ, nothing, parameters(problem), first_slab)
        ∇S = oracle_residual(oracle, first_datum(oracle), θ)
        @test maximum(abs, ∇S) ≥ T(1 // 100)
        @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * maximum(abs, ∇S)
        @info("the VectorWave record", T, c.iterations, c.residuals, c.converged)
    end

    # The affine control of the `β = 0` density: its steps converge in at most two iterations, and
    # its charge is recorded per step with its metadata beside the figures of its steps. The charge
    # is finite and not small; no bound is asserted on its drift, because `λ_top` lies in the
    # near-null space of the mode's saddle point and no cutoff resolves it.
    @testset "E7b the affine control of VectorWave and its charge" for T in (Float64, Float32)
        problem = vector_wave_problem(T; β = 0.0)
        method = trained_method(T; S = TRAINED_S, D = 2,
            guess = FitTo(InitialCondition();
                method = TrainedHiddenLayer(epochs = 20)))
        int = GeometricIntegrator(problem, method;
            f_abstol = zero(T), f_reltol = trained_f_reltol(T),
            max_iterations = TRAINED_ITERATIONS)
        c = cache(int)
        solution = GeometricSolution(problem)
        step = solutionstep(int, solution[0])
        reset!(step, timesteps(solution)[1])
        J = T[]
        for n in 1:10
            M.integrate_step!(current(step), nothing, parameters(problem), int)
            push!(J, charge(c))
        end
        @test length(J) == 10
        @test all(isfinite, J)
        @test length(c.iterations) == 10
        @test all(c.converged)
        @test maximum(c.iterations) ≤ TRAINED_ITERATIONS
        @test maximum(c.residuals ./ c.initial_residuals) ≤ trained_f_reltol(T)
        @test abs(J[1]) ≥ T(1 // 100)
        record_charge(:E7b, T, J, maximum(x -> abs(x - J[1]), J))
        @info("the affine control of VectorWave", T, c.iterations, c.converged, c.residuals,
            c.initial_residuals)
    end
end
