using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS, norm
using MultisymplecticIntegrators
using Random
using SimpleSplines
using SimpleSolvers
using Symbolics
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
solver's vector the seeded `θ`, with the residual of that vector taken. The solution step is reset
to the time the first step of the run ends, as the integration loop resets it, so the slab starts at
the problem's own `timespan[1]`.
"""
function lifting_slab(problem, method, θ)
    int = GeometricIntegrator(problem, method)
    c = cache(int)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    reset!(solstep, timesteps(solution)[1])
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), current(solstep))
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
        # the second slab: its bottom is the previous slab's top, which the run carried out of the
        # first one, and the slab starts at the time the first one ended
        one_slab = GeometricIntegrator(lifting_problem(T; steps = 1), method)
        integrate(one_slab)
        top = copy(cache(one_slab).trace_u)
        problem₂ = lifting_problem(T; steps = 2)
        two = GeometricIntegrator(problem₂, method)
        integrate(two)
        c₂ = cache(two)
        @test c₂.coupling.start == timespan(problem₂)[1] + h
        tol₂ = tolerance(T) * maximum(abs, top)
        for (r, xr) in enumerate(x)
            field = lifting_point(c₂, problem₂, T, zero(T), xr, h)
            @test abs(field.u[1] - top[r, 1]) ≤ tol₂
        end
        @test all(isfinite, top)
    end

    @testset "L1b the lifted field's derivatives are the derivatives of its value, $T" for T in (
        Float64,)
        # linear data: `u = 0.3 + 0.5x` at the bottom and `0.3 + 0.4t` and `0.8 + 0.4t` at the two
        # sides, with `v = 0.4` and `w = 0.5` there, so the corners agree and the lifted field is a
        # linear function of both arguments. Its two derivatives are then the central differences
        # of the transform's own `u`, whichever way the transform builds them.
        params = (
            c = T(0.5), l = T(1.0), A1 = T(0.8), A2 = T(0.0), B1 = T(0.8), B2 = T(0.0))
        ics = function (x, p)
            return (u = reshape(T(0.3) .+ T(0.5) .* x, :, 1),
                v = zeros(T, length(x), 1), w = fill(T(0.5), length(x), 1))
        end
        bcs = function (t, p)
            return (left = (u = [T(0.3) + T(0.4) * t], v = [T(0.4)], w = [T(0.5)]),
                right = (u = [T(0.8) + T(0.4) * t], v = [T(0.4)], w = [T(0.5)]))
        end
        problem = LagrangianPDEProblem(Wave.density(; params = params); D = 1,
            ics_function = ics, bcs_function = bcs,
            timespan = (zero(T), T(0.1)), timestep = T(0.05), xspan = (zero(T), one(T)),
            xstep = T(0.05), params = params)
        h = timestep(problem)
        δ = cbrt(eps(T))
        points = ((T(0.37), T(0.41)), (T(0.8), T(0.73)), (T(0.15), T(0.22)))
        # the network arm: every point of the slab, in both arguments
        method = lifting_method(T;
            guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        N = length(cache(GeometricIntegrator(problem, method)).x)
        θ = 2 .* rand(Xoshiro(20261115), T, N) .- 1
        _, c, _, _ = lifting_slab(problem, method, θ)
        tol = sqrt(eps(T)) * maximum(abs, c.u)
        for (t̂, x) in points
            at = lifting_point(c, problem, T, t̂, x, h)
            later = lifting_point(c, problem, T, t̂ + δ, x, h)
            earlier = lifting_point(c, problem, T, t̂ - δ, x, h)
            inward = lifting_point(c, problem, T, t̂, x - δ, h)
            outward = lifting_point(c, problem, T, t̂, x + δ, h)
            @test abs(at.v[1] - (later.u[1] - earlier.u[1]) / (2δ) / h) ≤ tol
            @test abs(at.w[1] - (outward.u[1] - inward.u[1]) / (2δ)) ≤ tol
        end
        # the coefficient arm: its `v` at the same points, and its `w` at the top alone, because the
        # bottom it places is blended between the rule's own space nodes
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (2, 5),
            xspan = (zero(T), one(T)), n = 3)
        cmethod = SpaceTimeGalerkin(ansatz, HardConstraintLifting();
            initial_guess = FitTo(InitialCondition()))
        Nc = length(cache(GeometricIntegrator(problem, cmethod)).x)
        θc = 2 .* rand(Xoshiro(20261116), T, Nc) .- 1
        _, cc, _, _ = lifting_slab(problem, cmethod, θc)
        tolc = sqrt(eps(T)) * maximum(abs, cc.u)
        for (t̂, x) in points
            at = spline_point(cc, problem, ansatz, T, t̂, x, h)
            later = spline_point(cc, problem, ansatz, T, t̂ + δ, x, h)
            earlier = spline_point(cc, problem, ansatz, T, t̂ - δ, x, h)
            @test abs(at.v[1] - (later.u[1] - earlier.u[1]) / (2δ) / h) ≤ tolc
        end
        for (_, x) in points
            at = spline_point(cc, problem, ansatz, T, one(T), x, h)
            inward = spline_point(cc, problem, ansatz, T, one(T), x - δ, h)
            outward = spline_point(cc, problem, ansatz, T, one(T), x + δ, h)
            @test abs(at.w[1] - (outward.u[1] - inward.u[1]) / (2δ)) ≤ tolc
        end
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
        # a first slab on a problem whose side data move in time: `Wave`'s sides vanish at both
        # ends at every time, so a side read from the initial condition, or at the normalised time,
        # moves nothing there. The kink travels at the default `velocity`, and the slab starts at
        # the problem's own first time.
        moving = SineGordon.problem(
            T = T, timespan = (zero(T), T(0.1)), timestep = T(0.05), xstep = T(0.05))
        mmethod = lifting_method(T)
        moracle = ActionOracle(moving, mmethod)
        Nm = unknown_count(moracle)
        θm = 2 .* rand(Xoshiro(20261112), T, Nm) .- 1
        _, cm, msolstep, mb = lifting_slab(moving, mmethod, θm)
        @test M.slab_start(moving, current(msolstep)) == timespan(moving)[1]
        mh = timestep(moving)
        mt̂, _ = M.quadrature_points(cm.ansatz)
        xa, xb = xspan(moving)
        mparams = parameters(moving)
        mstart = M.slab_start(moving, current(msolstep))
        initial = moving.ics_function([xa, xb], mparams)
        apart = T(Inf)
        for (r, which) in ((1, :left), (2, :right))
            for tq in mt̂
                side = getproperty(
                    moving.bcs_function(mstart + mh * convert(T, tq), mparams), which)
                apart = min(apart, abs(side.u[1] - initial.u[r, 1]))
            end
        end
        @test apart > tolerance(T) * maximum(abs, cm.u)
        ∇Sₘ = oracle_residual(moracle, first_datum(moracle), θm)
        @test maximum(abs, ∇Sₘ) ≥ T(1 // 100)
        @test maximum(abs, mb - ∇Sₘ) ≤ 100 * eps(T) * maximum(abs, ∇Sₘ)
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
        problem₂ = VectorWave.problem(
            T = T, params = param₂, timespan = (zero(T), T(0.25)),
            timestep = T(0.05), xstep = T(0.05))
        problem₁ = lifting_problem(T; steps = 5, timespan = (zero(T), T(0.25)))
        method₂ = lifting_method(T; D = 2)
        method₁ = lifting_method(T)
        S = 10
        nf₁ = M.nparameters(ansatz(method₁))
        # the seeded scalar vector, and its embedding in the two-component network: one hidden
        # layer, and each component's output weights the scalar's
        θ = 2 .* rand(Xoshiro(20261112), T, nf₁) .- 1
        W = Vector{T}(undef, nf₁ + S)
        W[1:nf₁] = θ
        for s in 1:S, d in 1:2

            W[3S + (s - 1) * 2 + d] = θ[3S + s]
        end
        int₁, c₁, solstep₁, b₁ = lifting_slab(problem₁, method₁, θ)
        int₂, c₂, solstep₂, b₂ = lifting_slab(problem₂, method₂, W)
        # a transform that lifts one component alone leaves the other component's field, and its
        # rows, off the scalar run's
        tol = 10 * eps(T) *
              max(maximum(abs, c₁.u), maximum(abs, c₁.v), maximum(abs, c₁.w))
        for d in 1:2
            @test maximum(abs, c₂.u[:, :, d] .- c₁.u[:, :, 1]) ≤ tol
            @test maximum(abs, c₂.v[:, :, d] .- c₁.v[:, :, 1]) ≤ tol
            @test maximum(abs, c₂.w[:, :, d] .- c₁.w[:, :, 1]) ≤ tol
            @test maximum(abs, c₂.ansatz.Ju[:, :, d, 1:(3S)] .-
                               c₁.ansatz.Ju[:, :, 1, 1:(3S)]) ≤
                  tol
            @test maximum(abs,
                c₂.ansatz.Jv[:, :, d, 1:(3S)] .- c₁.ansatz.Jv[:, :, 1, 1:(3S)]) ≤ tol
            @test maximum(abs,
                c₂.ansatz.Jw[:, :, d, 1:(3S)] .- c₁.ansatz.Jw[:, :, 1, 1:(3S)]) ≤ tol
            for j in 1:S
                @test c₂.ansatz.Ju[:, :, d, 3S + (j - 1) * 2 + d] ≈
                      c₁.ansatz.Ju[:, :, 1, 3S + j] atol = tol
            end
        end
        # the hidden rows sum over both components, and each component's output row is the scalar
        # run's there
        @test maximum(abs, b₂[1:(3S)] .- 2 * b₁[1:(3S)]) ≤ 10 * eps(T) * maximum(abs, b₁)
        for d in 1:2, j in 1:S

            @test b₂[3S + (j - 1) * 2 + d] ≈ b₁[3S + j] atol = 10 * eps(T) *
                                                               maximum(abs, b₁)
        end
        # the five steps of both runs record their deviation, which no bound holds: both are
        # unconverged, and the minimum-norm step of a rank-deficient Jacobian moves the two
        # minimisations apart even in exact arithmetic
        result₂ = integrate(problem₂, method₂)
        result₁ = integrate(problem₁, method₁)
        @test length(result₂.traces) == 5
        @test length(result₁.traces) == 5
        grid = xgrid(problem₁)
        n = length(grid)
        u₂ = result₂.solution.u[end]
        u₁ = result₁.solution.u[end]
        @test length(u₂) == 2n
        scale = maximum(abs, u₁)
        deviations = (maximum(abs, u₂[1:n] .- u₁) / scale,
            maximum(abs, u₂[(n + 1):(2n)] .- u₁) / scale)
        for deviation in deviations
            @test isfinite(deviation)
        end
        @info("the lifting's vector arm", T, deviations, scale,
            cache(GeometricIntegrator(problem₂, method₂)).iterations, VERSION,
            machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
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
            maximum(abs, b), VERSION, machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            blas_threads=BLAS.get_num_threads(), check_bounds=Base.JLOptions().check_bounds)
    end

    @testset "the lifting refuses the ansatz and the basis it cannot lift, $T" for T in (
        Float64,)
        problem = lifting_problem(T; steps = 1)
        # a symbolic ansatz carries no field a coupling can place the data in
        @variables τ ξ q[1:1]
        symbolic = SymbolicAnsatz(q[1] * sin(ξ - τ), τ, ξ, q;
            xspan = (zero(T), one(T)), start = [one(T)])
        err = try
            GeometricIntegrator(problem, SpaceTimeGalerkin(symbolic, HardConstraintLifting()))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("takes a NetworkAnsatz or a TensorBSplineAnsatz", err.msg)
        # the coefficient arm reads the field of the slab's two ends at the time nodes of the
        # residual, so a basis that is free at those ends is refused rather than lifted at the
        # wrong time
        free = TensorBSplineAnsatz(degree = 2, ncells = (2, 5), xspan = (zero(T), one(T)),
            n = 3, x_boundary = Free())
        err = try
            GeometricIntegrator(problem, SpaceTimeGalerkin(free, HardConstraintLifting()))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("needs x_boundary = Dirichlet()", err.msg)
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
        # the warning is the integrator's, at its construction: the cache a solver builds for
        # another element type is built on top of it and says nothing, so building warns once and
        # running the integrator that was already built warns nothing more
        built, int = Test.collect_test_logs(min_level = Base.CoreLogging.Debug) do
            GeometricIntegrator(problem, method)
        end
        @test length(filter(l -> occursin("corner", l.message), built)) == 1
        run_logs, _ = Test.collect_test_logs(min_level = Base.CoreLogging.Debug) do
            integrate(int)
        end
        @test isempty(filter(l -> occursin("corner", l.message), run_logs))
        # one integrator, built and run, warns once
        for _ in 1:2
            logs, _ = Test.collect_test_logs(min_level = Base.CoreLogging.Debug) do
                integrate(problem, method)
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
