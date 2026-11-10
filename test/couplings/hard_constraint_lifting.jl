using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS, norm
using MultisymplecticIntegrators
using Random
using SimpleSplines
using SimpleSolvers
using Test

import GeometricEquations: problem
import GeometricIntegratorsBase: GeometricIntegrator, cache, current, reset!, solver,
                                 solutionstep

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")
include("../helpers/action_oracle.jl")

const M = MultisymplecticIntegrators

"""
The network of the lifting's scenarios: `S = 10` hidden units of `tanh`, and a trace basis of
`degree = 3` on `ncells = (2, 5)`, which is the §A case's basis.
"""
function lifting_ansatz(::Type{T}; S = 10, D = 1, ncells = (2, 5), degree = 3) where {T}
    return NetworkAnsatz(
        S = S, xspan = (zero(T), one(T)), degree = degree, ncells = ncells,
        D = D)
end

function lifting_problem(::Type{T}; steps = 4, h = 0.05, xstep = 0.05, D = 1,
        timespan = nothing) where {T}
    span = timespan === nothing ? (zero(T), T(h * steps)) : timespan
    return Wave.problem(T = T, D = D, timespan = span, timestep = T(h), xstep = T(xstep))
end

"""The lifting's method on the first slab of a run, with the exact solution as its guess."""
function lifting_method(::Type{T}; S = 10, D = 1,
        guess = FitTo(ExactSolution();
            method = OrthogonalGreedy())) where {T}
    return SpaceTimeGalerkin(lifting_ansatz(T; S = S, D = D), HardConstraintLifting();
        initial_guess = guess)
end

"""
    lifting_slab(problem, method, θ)

The first slab of a run, set up as the residual reads it: the coupling's datum fixed and the
solver's vector the seeded `θ`, with the residual of that vector taken.
"""
function lifting_slab(problem, method, θ)
    int = GeometricIntegrator(problem, method)
    c = cache(int)
    solstep = solutionstep(int, GeometricSolution(problem)[0])
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), solstep)
    b = zeros(eltype(θ), length(θ))
    M.residual!(b, θ, nothing, parameters(problem), int)
    return int, c, solstep, b
end

"""
    lifting_point(cache, problem, T, t̂, x, h)

The transform's field at one point of the slab, beside the ansatz's own field there: the field is
filled with the ansatz's value at the point, as the core fills it at the nodes of the slab, and the
coupling's transform is applied to that one point.
"""
function lifting_point(c, problem, ::Type{T}, t̂, x, h) where {T}
    a = c.ansatz
    D, nf = a.ansatz.D, a.ansatz.nfree
    u = zeros(T, 1, 1, D)
    v = zeros(T, 1, 1, D)
    w = zeros(T, 1, 1, D)
    Ju = zeros(T, 1, 1, D, nf)
    Jv = zeros(T, 1, 1, D, nf)
    Jw = zeros(T, 1, 1, D, nf)
    M.evaluate_at!(a, convert(T, t̂), convert(T, x))
    own = copy(a.values)
    for d in 1:D
        u[1, 1, d] = own[d]
        v[1, 1, d] = own[D + d] / T(h)
        w[1, 1, d] = own[2D + d]
    end
    M.transform_field!(u, v, w, c.coupling, a, (T(t̂),), (T(x),), T(h), problem, Ju, Jv, Jw)
    return (u = u[1, 1, :], v = v[1, 1, :], w = w[1, 1, :], own = own)
end

"""
    spline_field(ansatz, block, t̂, x, h)

The coefficient slab's field and its two derivatives at one point, by `SimpleSplines`' own
evaluation of the two bases: an evaluation of the coefficient array and not of any transform.
"""
function spline_field(ansatz, block, t̂, x, h)
    tb, xb = SimpleSplines.bases(SimpleSplines.basis(ansatz))
    coeffs = block.coeffs
    nt, nb, D = size(coeffs)
    u = zeros(eltype(coeffs), D)
    v = zeros(eltype(coeffs), D)
    w = zeros(eltype(coeffs), D)
    for d in 1:D, i in 1:nt, j in 1:nb
        c = coeffs[i, j, d]
        φ = SimpleSplines.evaluate(tb, i, t̂)
        φ′ = SimpleSplines.evaluate(tb, i, t̂, 1)
        ψ = SimpleSplines.evaluate(xb, j, x)
        ψ′ = SimpleSplines.evaluate(xb, j, x, 1)
        u[d] += c * φ * ψ
        v[d] += c * φ′ * ψ / h
        w[d] += c * φ * ψ′
    end
    return (u = u, v = v, w = w)
end

"""The transform's field of a coefficient slab at one point, beside the spline's own field there."""
function spline_point(c, problem, ansatz, ::Type{T}, t̂, x, h) where {T}
    D = components(problem)
    field = spline_field(ansatz, c.ansatz, convert(T, t̂), convert(T, x), T(h))
    u = reshape(copy(field.u), 1, 1, D)
    v = reshape(copy(field.v), 1, 1, D)
    w = reshape(copy(field.w), 1, 1, D)
    M.transform_field!(u, v, w, c.coupling, c.ansatz, (T(t̂),), (T(x),), T(h), problem)
    return (u = u[1, 1, :], v = v[1, 1, :], w = w[1, 1, :], own = field.u)
end

"""The central differences of the slab's `residual!` at `θ`, one column per unknown."""
function slab_jacobian(int, θ::AbstractVector{T}, params) where {T}
    N = length(θ)
    J = zeros(T, N, N)
    forward = zeros(T, N)
    backward = zeros(T, N)
    x = copy(θ)
    for k in 1:N
        δ = cbrt(eps(T)) * max(one(T), abs(θ[k]))
        x[k] = θ[k] + δ
        M.residual!(forward, x, nothing, params, int)
        up = x[k]
        x[k] = θ[k] - δ
        M.residual!(backward, x, nothing, params, int)
        J[:, k] .= (forward .- backward) ./ (up - x[k])
        x[k] = θ[k]
    end
    return J
end

"""
    lifting_deviation(result, entry)

The sup-norm deviation of a run from the `u` of a §A entry, over the steps both of them hold: an
entry whose run failed at a step records the steps before it, and a run of this tree may take more.
"""
function lifting_deviation(result, entry)
    n = min(size(entry.u, 1), length(result.traces))
    n == 0 && return zero(eltype(result.solution.u[1])), 0
    return maximum(maximum(abs.(result.solution.u[k] .- entry.u[k, :])) for k in 1:n), n
end

"""Whether every state of a solution series is finite."""
finite_states(x) = all(n -> all(isfinite, x[n]), eachindex(x))

"""
    record_run(record, T, name, cache, deviation, compared)

Write one sounding with the metadata that makes the figures readable on another machine. No bound is
asserted on the deviation: it is that of an unconverged iterate of another formulation.
"""
function record_run(record::Symbol, ::Type{T}, name::Symbol, cache, deviation,
        compared) where {T}
    @info("the lifting's section A sounding", record, name, T, cache.iterations,
        cache.residuals, cache.initial_residuals, cache.converged, deviation, compared, VERSION,
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
end

"""
    assert_boundary_data(problem, ansatz, sides, bottom)

The premise of the §A comparison: the boundary function is the exact solution at the two ends, and
the initial condition is the exact solution at the start of the run.
"""
function assert_boundary_data(problem, xa, xb)
    t₀ = timespan(problem)[1]
    params = parameters(problem)
    for k in 0:9
        t = t₀ + k * timestep(problem) / 9
        sides = problem.bcs_function(t, params)
        for d in 1:components(problem)
            @test sides.left.u[d] == problem.solutions.u(t, xa, params)[d]
            @test sides.right.u[d] == problem.solutions.u(t, xb, params)[d]
        end
    end
    initial = problem.ics_function([xa, xb], params)
    for (r, x) in enumerate((xa, xb))
        @test initial.u[r, 1] == problem.solutions.u(t₀, x, params)[1]
    end
    return nothing
end

@testset "Hard constraint lifting" begin
    @testset "L1 the field is the data at the three boundary parts, $T" for T in (Float64,
        Float32)
        problem = lifting_problem(T; steps = 1)
        method = lifting_method(T)
        N = length(cache(GeometricIntegrator(problem, method)).x)
        θ = 2 .* rand(Xoshiro(20261110), T, N) .- 1
        int, c, solstep, b = lifting_slab(problem, method, θ)
        h = timestep(problem)
        t̂, x = M.quadrature_points(c.ansatz)
        xa, xb = xspan(problem)
        t₀ = M.slab_start(problem, current(solstep))
        scale = maximum(abs, c.u)
        tol = tolerance(T) * scale
        own = zero(T)
        # the sides, at the nodes of the quadrature in time
        for (corner, point, which) in ((1, xa, :left), (2, xb, :right))
            for tq in t̂
                field = lifting_point(c, problem, T, tq, point, h)
                sides = problem.bcs_function(t₀ + h * T(tq), parameters(problem))
                want = which === :left ? sides.left.u[1] : sides.right.u[1]
                @test abs(field.u[1] - want) ≤ tol
                own = max(own, abs(field.own[1] - want))
            end
        end
        # the field is the data and not the network's own value: at the sides the two differ
        @test own > T(1e-3) * scale
        # the bottom, at the nodes of the quadrature in space
        values = problem.ics_function(collect(x), parameters(problem))
        for (r, xr) in enumerate(x)
            field = lifting_point(c, problem, T, zero(T), xr, h)
            @test abs(field.u[1] - values.u[r, 1]) ≤ tol
        end
        @test all(isfinite, b)
    end

    @testset "L2 the residual of the lifted slab is the gradient of its action, $T" for T in (
        Float64, Float32)
        problem = lifting_problem(T; steps = 2)
        method = lifting_method(T)
        oracle = ActionOracle(problem, method)
        N = unknown_count(oracle)
        # the first slab
        θ = 2 .* rand(Xoshiro(20261110), T, N) .- 1
        int, c, solstep, b = lifting_slab(problem, method, θ)
        @test N == length(c.x)
        ∇S = oracle_residual(oracle, first_datum(oracle), θ)
        @test maximum(abs, ∇S) ≥ T(1 // 100)
        @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * maximum(abs, ∇S)
        # the second slab: its datum is the first slab's solved top, which the oracle carries
        one = GeometricIntegrator(lifting_problem(T; steps = 1), method)
        integrate(one)
        datum = next_datum(oracle, first_datum(oracle), copy(cache(one).x))
        @test datum.start == timespan(problem)[1] + timestep(problem)
        two = GeometricIntegrator(problem, method)
        integrate(two)
        θ₂ = 2 .* rand(Xoshiro(20261111), T, N) .- 1
        b₂ = zeros(T, N)
        M.residual!(b₂, θ₂, nothing, parameters(problem), two)
        ∇S₂ = oracle_residual(oracle, datum, θ₂)
        @test maximum(abs, ∇S₂) ≥ T(1 // 100)
        @test maximum(abs, b₂ - ∇S₂) ≤ 100 * eps(T) * maximum(abs, ∇S₂)
    end

    @testset "L3 the Jacobian is central differences, $T" for T in (Float64,)
        problem = lifting_problem(T; steps = 1)
        method = lifting_method(T)
        N = length(cache(GeometricIntegrator(problem, method)).x)
        rng = Xoshiro(20261111)
        params = parameters(problem)
        for _ in 1:16
            θ = 2 .* rand(rng, T, N) .- 1
            int, c, solstep, _ = lifting_slab(problem, method, θ)
            s = solver(int)
            SimpleSolvers.jacobian!(s, θ, (current(solstep), params, int))
            J = copy(SimpleSolvers.jacobianmatrix(s))
            scale = maximum(abs, J)
            @test scale ≥ T(1 // 100)
            @test maximum(abs, J - slab_jacobian(int, θ, params)) ≤ sqrt(eps(T)) * scale
        end
    end

    @testset "L4 the lifting acts per component on VectorWave, $T" for T in (Float64, Float32)
        param₂ = (c = T(0.5), l = T(1.0), A1 = (T(0.8), T(0.8)), A2 = (T(0.0), T(0.0)),
            B1 = (T(0.8), T(0.8)), B2 = (T(0.0), T(0.0)))
        param₁ = (c = T(0.5), l = T(1.0), A1 = T(0.8), A2 = T(0.0), B1 = T(0.8),
            B2 = T(0.0))
        problem₂ = VectorWave.problem(
            T = T, params = param₂, timespan = (zero(T), T(0.25)),
            timestep = T(0.05), xstep = T(0.05))
        problem₁ = lifting_problem(T; steps = 5, timespan = (zero(T), T(0.25)))
        method₂ = lifting_method(T; D = 2)
        method₁ = lifting_method(T)
        # the components' data are the scalar problem's, and the transform keeps them apart
        grid = xgrid(problem₁)
        n = length(grid)
        int₂ = GeometricIntegrator(problem₂, method₂)
        c₂ = cache(int₂)
        solstep₂ = solutionstep(int₂, GeometricSolution(problem₂)[0])
        M.slab_bottom!(c₂, c₂.coupling, problem₂, ansatz(method₂), solstep₂)
        M.initial_guess!(solstep₂, nothing, parameters(problem₂), int₂)
        b₂ = zeros(T, length(c₂.x))
        M.residual!(b₂, copy(c₂.x), nothing, parameters(problem₂), int₂)
        @test size(c₂.u, 3) == 2
        # the two components carry the same data, so the transform leaves them equal: the rows of
        # the two go through the block's kernel with the same weights, and the difference this tree
        # measures is `4.2e-15`, about nineteen `eps` of the field's scale, which is the kernel's
        # own association and not a copy of one component into the other
        @test maximum(abs, c₂.u[:, :, 1] .- c₂.u[:, :, 2]) ≤
              100 * eps(T) * maximum(abs, c₂.u)
        # the same seed and the same data give the scalar run's field per component, up to the
        # solver's own step, which is not symmetry preserving on this rank-deficient system
        result₂ = integrate(problem₂, method₂)
        result₁ = integrate(problem₁, method₁)
        @test length(result₂.traces) == 5
        @test length(result₁.traces) == 5
        u₂ = result₂.solution.u[end]
        u₁ = result₁.solution.u[end]
        scale = maximum(abs, u₁)
        deviations = (maximum(abs, u₂[1:n] .- u₁) / scale,
            maximum(abs, u₂[(n + 1):(2n)] .- u₁) / scale)
        @test length(u₂) == 2n
        for deviation in deviations
            @test isfinite(deviation)
        end
        @info("the lifting's vector arm", T, deviations, scale,
            cache(GeometricIntegrator(problem₂, method₂)).iterations, VERSION,
            machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(),
            check_bounds=Base.JLOptions().check_bounds)
    end

    @testset "L5 a B-spline ansatz runs under the lifting, $T" for T in (Float64,)
        problem = lifting_problem(T; steps = 3, timespan = (zero(T), T(0.15)))
        ansatz = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 5), xspan = (zero(T), one(T)),
            n = 3)
        method = SpaceTimeGalerkin(ansatz, HardConstraintLifting();
            initial_guess = FitTo(ExactSolution()))
        N = length(cache(GeometricIntegrator(problem, method)).x)
        θ = 2 .* rand(Xoshiro(20261113), T, N) .- 1
        int, c, solstep, b = lifting_slab(problem, method, θ)
        h = timestep(problem)
        t̂, x = M.quadrature_points(c.ansatz)
        xa, xb = xspan(problem)
        t₀ = M.slab_start(problem, current(solstep))
        scale = maximum(abs, c.u)
        tol = tolerance(T) * scale
        for (corner, point, which) in ((1, xa, :left), (2, xb, :right))
            for tq in t̂
                field = spline_point(c, problem, ansatz, T, tq, point, h)
                sides = problem.bcs_function(t₀ + h * T(tq), parameters(problem))
                want = which === :left ? sides.left.u[1] : sides.right.u[1]
                @test abs(field.u[1] - want) ≤ tol
            end
        end
        values = problem.ics_function(collect(x), parameters(problem))
        for (r, xr) in enumerate(x)
            field = spline_point(c, problem, ansatz, T, zero(T), xr, h)
            @test abs(field.u[1] - values.u[r, 1]) ≤ tol
        end
        @test all(isfinite, b)
        # the run's rows are the block's own basis functions, so no identity is asserted on its
        # residual: the steps and its record are
        int = GeometricIntegrator(problem, method)
        c = cache(int)
        result = integrate(int)
        @test length(result.traces) == 3
        @test length(c.iterations) == 3
        @test finite_states(result.solution.u)
        b = zeros(T, N)
        M.residual!(b, θ, nothing, parameters(problem), int)
        @test all(isfinite, b)
        @info("the lifting's coefficient arm", T, c.iterations, c.residuals, c.converged,
            maximum(abs, b))
    end

    @testset "N1 a problem without an exact solution, $T" for T in (Float64,)
        plain = lifting_problem(T; steps = 2, timespan = (zero(T), T(0.1)))
        problem = LagrangianPDEProblem(Wave.density(; params = plain.params);
            D = 1,
            ics_function = plain.ics_function,
            bcs_function = plain.bcs_function,
            timespan = plain.timespan, timestep = plain.timestep,
            xspan = plain.xspan, xstep = plain.xstep, params = plain.params)
        @test problem.solutions === nothing
        method = lifting_method(T;
            guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        # the residual is the oracle's gradient on a problem that has no exact solution at all
        N = unknown_count(ActionOracle(problem, method))
        θ = 2 .* rand(Xoshiro(20261114), T, N) .- 1
        int, c, solstep, b = lifting_slab(problem, method, θ)
        oracle = ActionOracle(problem, method)
        ∇S = oracle_residual(oracle, first_datum(oracle), θ)
        @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * maximum(abs, ∇S)
        # and the field at the three boundary parts is the problem's own two data functions
        h = timestep(problem)
        t̂, x = M.quadrature_points(c.ansatz)
        xa, xb = xspan(problem)
        t₀ = M.slab_start(problem, current(solstep))
        tol = tolerance(T) * maximum(abs, c.u)
        for (corner, point, which) in ((1, xa, :left), (2, xb, :right))
            for tq in t̂
                field = lifting_point(c, problem, T, tq, point, h)
                sides = problem.bcs_function(t₀ + h * T(tq), parameters(problem))
                want = which === :left ? sides.left.u[1] : sides.right.u[1]
                @test abs(field.u[1] - want) ≤ tol
            end
        end
        values = problem.ics_function(collect(x), parameters(problem))
        for (r, xr) in enumerate(x)
            field = lifting_point(c, problem, T, zero(T), xr, h)
            @test abs(field.u[1] - values.u[r, 1]) ≤ tol
        end
        # both steps complete and record their stats
        run = GeometricIntegrator(problem, method)
        result = integrate(run)
        @test length(result.traces) == 2
        @test length(cache(run).iterations) == 2
    end

    @testset "N3 a corner mismatch is a warning, not an error, $T" for T in (Float64,)
        plain = lifting_problem(T; steps = 1, timespan = (zero(T), T(0.05)))
        base = plain.bcs_function
        offset = function (t, p)
            sides = base(t, p)
            return (
                left = (u = sides.left.u .+ T(1e-2), v = sides.left.v,
                    w = sides.left.w),
                right = sides.right)
        end
        problem = LagrangianPDEProblem(plain.density; D = 1,
            ics_function = plain.ics_function, bcs_function = offset,
            timespan = plain.timespan, timestep = plain.timestep, xspan = plain.xspan,
            xstep = plain.xstep, params = plain.params)
        method = lifting_method(T;
            guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        for _ in 1:2
            logs, _ = Test.collect_test_logs(min_level = Base.CoreLogging.Debug) do
                GeometricIntegrator(problem, method)
            end
            warnings = filter(l -> occursin("corner", l.message), logs)
            @test length(warnings) == 1
            @test occursin("left", first(warnings).message)
        end
        result = integrate(problem, method)
        @test length(result.traces) == 1
    end

    @testset "A1 the Wave sounding, $T" for T in (Float64,)
        params = (c = T(0.5), l = T(1.0), A1 = T(0.8), A2 = T(0.0), B1 = T(0.8),
            B2 = T(0.0))
        problem = lifting_problem(T; steps = 5, timespan = (zero(T), T(0.25)))
        xa, xb = xspan(problem)
        assert_boundary_data(problem, xa, xb)
        method = SpaceTimeGalerkin(lifting_ansatz(T), HardConstraintLifting();
            initial_guess = FitTo(ExactSolution();
                method = OrthogonalGreedy(Nw = 50, Nb = 50)))
        int = GeometricIntegrator(problem, method)
        result = integrate(int)
        c = cache(int)
        @test finite_states(result.solution.u)
        @test length(c.iterations) == length(result.traces)
        deviation, compared = lifting_deviation(result, WAVE_TRIALNN_OGA2D)
        @test isfinite(deviation)
        record_run(:A1, T, :WAVE_TRIALNN_OGA2D, c, deviation, compared)
    end

    @testset "A2 the SineGordon sounding, $T" for T in (Float64,)
        problem = SineGordon.problem(T = T, timespan = (zero(T), T(0.15)),
            timestep = T(0.05), xstep = T(0.05))
        xa, xb = xspan(problem)
        assert_boundary_data(problem, xa, xb)
        method = SpaceTimeGalerkin(lifting_ansatz(T), HardConstraintLifting();
            initial_guess = FitTo(ExactSolution();
                method = OrthogonalGreedy(Nw = 50, Nb = 50)))
        int = GeometricIntegrator(problem, method)
        result = integrate(int)
        c = cache(int)
        @test finite_states(result.solution.u)
        @test length(c.iterations) == 3
        deviation, compared = lifting_deviation(result, SINE_GORDON_TRIALNN_OGA2D)
        @test isfinite(deviation)
        record_run(:A2, T, :SINE_GORDON_TRIALNN_OGA2D, c, deviation, compared)
    end
end
