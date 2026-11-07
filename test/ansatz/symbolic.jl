using ForwardDiff
using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using SimpleSplines
using Symbolics
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, current, integrate!,
                                 reset!, solutionstep
import SimpleSolvers: jacobian!, jacobianmatrix

include("../helpers/testsetup.jl")
include("../helpers/action_oracle.jl")
include(joinpath(@__DIR__, "..", "..", "scripts", "run_family.jl"))

const M = MultisymplecticIntegrators

# The symbols every expression of this file is written in: the slab's normalised time `t`, the
# space coordinate `x`, three parameter symbols for the scalar cases and six for the vector one,
# two for the affine control, and one scalar parameter for the side-data case.
@variables t x p[1:3] q r[1:2] s[1:6]

const SYMBOLIC_SEED = 20261107

# ---- the fixtures ----------------------------------------------------------
#
# The scalar fixture is the travelling wave `sin(x − ct)` of a wave speed of one half: its period
# is `4π`, so a timestep of `4π` makes every slab the same function of the slab's normalised time
# and the ansatz is the exact solution of all five of them.

"""The travelling wave `sin(x − ct)` on `x ∈ (0, 2π)`, one period per slab of `4π`."""
function travelling_problem(::Type{T}) where {T}
    c = T(1) / 2
    span = (zero(T), T(2) * T(π))
    step = T(π) / 10
    field(t, x, params) = sin(x - c * t)
    velocity(t, x, params) = -c * cos(x - c * t)
    slope(t, x, params) = cos(x - c * t)
    timespan = (zero(T), T(20) * T(π))
    params = (c = c,)
    solutions = (u = field_exact(field, 1), v = field_exact(velocity, 1),
        w = field_exact(slope, 1))
    return LagrangianPDEProblem(Wave.density(; params = params);
        D = 1,
        ics_function = (x, p) -> field_initial_data(solutions, x, p, 1, timespan[1]),
        bcs_function = (t, p) -> field_side_data(solutions, t, p, 1, span),
        solutions = solutions,
        timespan = timespan, timestep = T(4) * T(π), xspan = span, xstep = step,
        params = params)
end

"""The ansatz `p₁ sin(p₂x − p₃t̂)`, whose exact parameters are `[1, 1, 2π]` here."""
function travelling_ansatz(::Type{T}) where {T}
    return SymbolicAnsatz(p[1] * sin(p[2] * x - p[3] * t), t, x, p;
        xspan = (zero(T), T(2) * T(π)), start = T[1, 1, 2π])
end

# The hand-written field map of the travelling wave: the oracle's own, independent of the package.
function travelling_field(t̂, x, params)
    return (u = (params[1] * sin(params[2] * x - params[3] * t̂),),
        ut = (-params[1] * params[3] * cos(params[2] * x - params[3] * t̂),),
        ux = (params[1] * params[2] * cos(params[2] * x - params[3] * t̂),))
end

# The two closed forms of a vector field in the shape the problem's `solutions` are read in: the
# component-major vector at a grid, which `field_initial_data` lays out in `D` columns, and one
# entry per component at a single point, which is the side data's.
function stack(f, g)
    return function (t, x, p)
        x isa AbstractVector || return [f(t, x, p), g(t, x, p)]
        return vcat([f(t, xi, p) for xi in x], [g(t, xi, p) for xi in x])
    end
end

"""Two travelling waves of different speeds on `x ∈ (0, 2π)`, each one period per slab."""
function vector_problem(::Type{T}) where {T}
    c = T(1) / 2
    span = (zero(T), T(2) * T(π))
    step = T(π) / 10
    first(t, x, p) = sin(x - c * t)
    first_velocity(t, x, p) = -c * cos(x - c * t)
    first_slope(t, x, p) = cos(x - c * t)
    second(t, x, p) = T(1) / 2 * sin(2x - 2c * t)
    second_velocity(t, x, p) = -T(1) / 2 * 2c * cos(2x - 2c * t)
    second_slope(t, x, p) = T(1) / 2 * 2 * cos(2x - 2c * t)
    timespan = (zero(T), T(20) * T(π))
    params = (c = c,)
    solutions = (u = stack(first, second), v = stack(first_velocity, second_velocity),
        w = stack(first_slope, second_slope))
    return LagrangianPDEProblem(VectorWave.vector_density(; params = params, D = 2);
        D = 2,
        ics_function = (x, p) -> field_initial_data(solutions, x, p, 2, timespan[1]),
        bcs_function = (t, p) -> field_side_data(solutions, t, p, 2, span),
        solutions = solutions,
        timespan = timespan, timestep = T(4) * T(π), xspan = span, xstep = step,
        params = params)
end

"""The two-component ansatz `(p₁ sin(p₂x − p₃t̂), p₄ sin(p₅x − p₆t̂))`."""
function vector_ansatz(::Type{T}) where {T}
    expressions = Num[s[1] * sin(s[2] * x - s[3] * t), s[4] * sin(s[5] * x - s[6] * t)]
    return SymbolicAnsatz(expressions, t, x, s;
        xspan = (zero(T), T(2) * T(π)), start = T[1, 1, 2π, T(1) / 2, 2, 4π])
end

function vector_field(t̂, x, params)
    u = (params[1] * sin(params[2] * x - params[3] * t̂),
        params[4] * sin(params[5] * x - params[6] * t̂))
    ut = (-params[1] * params[3] * cos(params[2] * x - params[3] * t̂),
        -params[4] * params[6] * cos(params[5] * x - params[6] * t̂))
    ux = (params[1] * params[2] * cos(params[2] * x - params[3] * t̂),
        params[4] * params[5] * cos(params[5] * x - params[6] * t̂))
    return (; u, ut, ux)
end

# ---- the vector fixtures of the affine control and the side data ------------
#
# The two-component problems of the wave density whose fields are a polynomial in `x` of degree
# one: the ramped field `(α₀ + βt)x`, with its own `α₀` and `β` per component, whose data differ
# from slab to slab, and the steady field `slope · x`, whose two ends carry `∓slope`.

"""The two-parameter ansatz `(p₁x, p₂x)` of a steady vector field, on `x ∈ (−1, 1)`."""
function vector_steady_ansatz(::Type{T}; start = T[T(7) / 10, T(-3) / 10]) where {T}
    return SymbolicAnsatz(Num[s[1] * x, s[2] * x], t, x, collect(s[1:2]);
        xspan = (-one(T), one(T)), start = start)
end

"""Two steady fields `slope_d · x`, on `x ∈ (−1, 1)`, `steps` steps of the length one."""
function vector_steady_problem(::Type{T}, slopes; steps = 1) where {T}
    c = T(1) / 2
    span = (-one(T), one(T))
    step = T(1) / 10
    first(t, x, p) = slopes[1] * x
    second(t, x, p) = slopes[2] * x
    flat(t, x, p) = zero(T)
    first_slope(t, x, p) = slopes[1]
    second_slope(t, x, p) = slopes[2]
    params = (c = c,)
    solutions = (u = stack(first, second), v = stack(flat, flat),
        w = stack(first_slope, second_slope))
    return LagrangianPDEProblem(VectorWave.vector_density(; params = params, D = 2);
        D = 2,
        ics_function = (x, p) -> field_initial_data(solutions, x, p, 2, zero(T)),
        bcs_function = (t, p) -> field_side_data(solutions, t, p, 2, span),
        solutions = solutions,
        timespan = (zero(T), T(steps)), timestep = one(T), xspan = span, xstep = step,
        params = params)
end

"""The four-parameter ansatz `((p₁ + p₂t̂)x, (p₃ + p₄t̂)x)` of the ramped vector field."""
function vector_ramp_ansatz(::Type{T}) where {T}
    expressions = Num[(s[1] + s[2] * t) * x, (s[3] + s[4] * t) * x]
    return SymbolicAnsatz(expressions, t, x, collect(s[1:4]);
        xspan = (-one(T), one(T)), start = T[T(1) / 2, T(3) / 5, T(-1) / 2, T(3) / 10])
end

"""Two ramps `(α₀ + βt)x`, one per component, on `x ∈ (−1, 1)`, two slabs of `h = 4 / 10`."""
function vector_ramp_problem(::Type{T}) where {T}
    c = T(1) / 2
    span = (-one(T), one(T))
    step = T(1) / 4
    h = T(4) / 10
    α₀ = (T(3) / 10, T(-1) / 5)
    β = (T(1) / 2, T(-1) / 4)
    first(t, x, p) = (α₀[1] + β[1] * t) * x
    second(t, x, p) = (α₀[2] + β[2] * t) * x
    first_velocity(t, x, p) = β[1] * x
    second_velocity(t, x, p) = β[2] * x
    first_slope(t, x, p) = α₀[1] + β[1] * t
    second_slope(t, x, p) = α₀[2] + β[2] * t
    params = (c = c,)
    solutions = (u = stack(first, second), v = stack(first_velocity, second_velocity),
        w = stack(first_slope, second_slope))
    return LagrangianPDEProblem(VectorWave.vector_density(; params = params, D = 2);
        D = 2,
        ics_function = (x, p) -> field_initial_data(solutions, x, p, 2, zero(T)),
        bcs_function = (t, p) -> field_side_data(solutions, t, p, 2, span),
        solutions = solutions,
        timespan = (zero(T), T(2) * h), timestep = h, xspan = span, xstep = step,
        params = params)
end

# The fixtures of the action-gradient check: a `D = 1` and a `D = 2` problem, each with the
# ansatz and the hand-written field map of its own, and a steady field whose two ends carry
# different data — the travelling wave's carry the same value at both, because its span is one
# period of it — so that the oracle's reading of `left` and `right` is separable.
function symbolic_fixtures(::Type{T}) where {T}
    return ((travelling_problem(T), travelling_ansatz(T), travelling_field),
        (vector_problem(T), vector_ansatz(T), vector_field),
        (side_problem(T, T(7) / 10; steps = 2), side_ansatz(T), steady_field))
end

# ---- the affine control and the side data ----------------------------------
#
# The affine control's data ramp linearly in time, `u(t, x) = (α₀ + βt)x`, which is an exact
# solution of the wave equation and differs from slab to slab, so the continuation guess is never
# the answer of the slab it starts. The side-data fixture is the steady field `slope · x` whose two
# ends carry `∓slope`.

"""The ramp `(α₀ + βt)x`, on two slabs of `h = 4 / 10`."""
function ramp_problem(::Type{T}; α₀ = T(0.3), β = T(1) / 2, h = T(0.4)) where {T}
    c = T(1) / 2
    span = (-one(T), one(T))
    step = T(1) / 4
    field(t, x, params) = (α₀ + β * t) * x
    velocity(t, x, params) = β * x
    slope(t, x, params) = α₀ + β * t
    params = (c = c,)
    solutions = (u = field_exact(field, 1), v = field_exact(velocity, 1),
        w = field_exact(slope, 1))
    return LagrangianPDEProblem(Wave.density(; params = params);
        D = 1,
        ics_function = (x, p) -> field_initial_data(solutions, x, p, 1, zero(T)),
        bcs_function = (t, p) -> field_side_data(solutions, t, p, 1, span),
        solutions = solutions,
        timespan = (zero(T), T(2) * h), timestep = h, xspan = span, xstep = step,
        params = params)
end

"""The steady field `slope · x` on `x ∈ (−1, 1)`, `steps` steps of the length one."""
function side_problem(::Type{T}, slope; steps = 1) where {T}
    c = T(1) / 2
    span = (-one(T), one(T))
    step = T(1) / 10
    field(t, x, params) = slope * x
    velocity(t, x, params) = zero(T)
    gradient(t, x, params) = slope
    params = (c = c,)
    solutions = (u = field_exact(field, 1), v = field_exact(velocity, 1),
        w = field_exact(gradient, 1))
    return LagrangianPDEProblem(Wave.density(; params = params);
        D = 1,
        ics_function = (x, p) -> field_initial_data(solutions, x, p, 1, zero(T)),
        bcs_function = (t, p) -> field_side_data(solutions, t, p, 1, span),
        solutions = solutions,
        timespan = (zero(T), T(steps)), timestep = one(T), xspan = span, xstep = step,
        params = params)
end

"""The one-parameter ansatz `p₁x` of the side-data fixture, on `x ∈ (−1, 1)`."""
function side_ansatz(::Type{T}) where {T}
    return SymbolicAnsatz(q * x, t, x, [q];
        xspan = (-one(T), one(T)), start = [T(7) / 10])
end

# The hand-written field map of the steady field `p₁x`: no velocity, and the slope `p₁`.
function steady_field(t̂, x, params)
    return (u = (params[1] * x,), ut = (zero(params[1]),), ux = (params[1],))
end

# ---- the slab helpers ------------------------------------------------------

"""An integrator of one slab, and the solution and solution step a run of it drives."""
function slab_setup(problem, method; kwargs...)
    int = GeometricIntegrator(problem, method; kwargs...)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    slab = (; problem, method, int, solution, solstep)
    next_slab!(slab, 1)
    return slab
end

"""Move the solution step to step `n` and fix the datum of that slab, as a run does."""
function next_slab!(slab, n::Integer)
    reset!(slab.solstep, timesteps(slab.solution)[n])
    c = cache(slab.int)
    M.slab_bottom!(c, c.coupling, slab.problem, ansatz(slab.method), current(slab.solstep))
    return nothing
end

"""
    random_parameters(T, n; offset = 0)

`n` random parameters in `(−1, 1)`, from the file's fixed seed, so that a fixture is checked at a
point of its parameter space that is neither its start nor its exact parameters.
"""
function random_parameters(::Type{T}, n::Integer; offset = 0) where {T}
    rng = Xoshiro(SYMBOLIC_SEED + offset + n)
    return 2 .* rand(rng, T, n) .- 1
end

"""
    compare_gradient(b, ∇S, T)

The package's residual against the oracle's gradient: both finite, the gradient at least `1/100`
in its largest entry, and the difference at most `100 eps(T)` relative to that entry.
"""
function compare_gradient(b, ∇S, ::Type{T}) where {T}
    @test all(isfinite, b)
    @test all(isfinite, ∇S)
    scale = maximum(abs, ∇S)
    @test scale ≥ T(1 // 100)
    @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * scale
end

"""
    central_differences(int, θ, params)

The central differences of the slab's `residual!` at `θ`, with the step
`δ_k = cbrt(eps(T)) max(1, |θ_k|)` per column.
"""
function central_differences(int, θ::AbstractVector{T}, params) where {T}
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

# ---- the tests -------------------------------------------------------------

@testset "Symbolic ansatz" begin
    @testset "A1a the constructor builds the six evaluators" begin
        ansatz = travelling_ansatz(Float64)
        @test ansatz isa SymbolicAnsatz
        @test ansatz isa SpaceTimeAnsatz
        @test ansatz.num_params == 3
        @test length(ansatz.expr) == 1
        @test ansatz.start == [1.0, 1.0, 2π]
        @test ansatz.trace isa TensorBSplineAnsatz
        @test ansatz.trace.xspan == ansatz.xspan
        for name in (:u, :ut, :ux, :Ju, :Jut, :Jux)
            @test isconcretetype(fieldtype(typeof(ansatz), name))
        end
        # the constructor stores its two coordinates and its span as given
        @test isequal(ansatz.t, t)
        @test isequal(ansatz.x, x)
        @test ansatz.xspan == (zero(Float64), 2π)
        # the parameter vector is the one the expression is written in
        @test all(isequal.(ansatz.p, collect(p)))
    end

    @testset "A1b the field map and its derivatives" begin
        for T in (Float64, Float32), (problem, ansatz, field) in symbolic_fixtures(T)

            slab = M.ansatz_data(ansatz, problem, T)
            tnodes, xnodes = M.quadrature_points(slab)
            h = timestep(problem)
            qt, qx, D = length(tnodes), length(xnodes), length(ansatz.expr)
            params = random_parameters(T, ansatz.num_params)
            u = zeros(T, qt, qx, D)
            v = similar(u)
            w = similar(u)
            M.slab_field!(u, v, w, slab, params, nothing, ansatz.num_params, h)
            # the nodes of the slab's own rule
            for q in 1:qt, r in 1:qx, d in 1:D
                t̂, xr = tnodes[q], xnodes[r]
                values = field(t̂, xr, params)
                @test abs(u[q, r, d] - values.u[d]) ≤
                      16 * eps(T) * max(one(T), abs(values.u[d]))
                @test abs(v[q, r, d] - values.ut[d] / h) ≤
                      16 * eps(T) * max(one(T), abs(values.ut[d] / h))
                @test abs(w[q, r, d] - values.ux[d]) ≤
                      16 * eps(T) * max(one(T), abs(values.ux[d]))
            end
            # the bottom of the slab is `t̂ = 0` at the nodes of the space rule
            for r in 1:qx, d in 1:D

                values = field(zero(T), xnodes[r], params)
                @test abs(slab.bottom_u[r, d] - values.u[d]) ≤
                      16 * eps(T) * max(one(T), abs(values.u[d]))
            end
            # the top of the slab is `t̂ = 1`, at the nodes of the trace evaluation
            utop, vtop, wtop = zeros(T, qx, D), zeros(T, qx, D), zeros(T, qx, D)
            M.top_values!(utop, vtop, wtop, slab, M.trace_evaluation(slab), h)
            for r in 1:qx, d in 1:D

                values = field(one(T), xnodes[r], params)
                @test abs(utop[r, d] - values.u[d]) ≤
                      16 * eps(T) * max(one(T), abs(values.u[d]))
            end
            # the middle of the slab, `t̂ = 1/2`, through the block's own evaluation
            for r in 1:qx, d in 1:D

                M.evaluate_at!(slab, T(1) / 2, xnodes[r])
                values = field(T(1) / 2, xnodes[r], params)
                @test abs(slab.ubuf[d] - values.u[d]) ≤
                      16 * eps(T) * max(one(T), abs(values.u[d]))
            end
            # the two evaluations are the output grid and the space quadrature nodes
            @test M.evaluation_points(M.output_evaluation(slab)) ==
                  collect(xgrid(problem))
            @test M.evaluation_points(M.trace_evaluation(slab)) == collect(xnodes)
            # the rule of the trace basis sums to one in time and to the span in space
            tq, xq = slab.tabulation.time, slab.tabulation.space
            @test abs(sum(tq.weight) - one(T)) ≤ 16 * eps(T)
            span = xspan(problem)[2] - xspan(problem)[1]
            @test abs(sum(xq.weight) - span) ≤
                  16 * eps(T) * max(one(T), abs(span))
            @test M.space_quadrature_weights(ansatz) == xq.weight
        end
    end

    @testset "A1c the parameter Jacobians" begin
        for T in (Float64, Float32), (problem, ansatz, field) in symbolic_fixtures(T)

            slab = M.ansatz_data(ansatz, problem, T)
            tnodes, xnodes = M.quadrature_points(slab)
            h = timestep(problem)
            qt, qx, D = length(tnodes), length(xnodes), length(ansatz.expr)
            params = random_parameters(T, ansatz.num_params; offset = 10)
            u = zeros(T, qt, qx, D)
            v = similar(u)
            w = similar(u)
            M.slab_field!(u, v, w, slab, params, nothing, ansatz.num_params, h)
            for q in 1:qt, r in 1:qx, d in 1:D
                t̂, xr = tnodes[q], xnodes[r]
                of_field = q -> [field(t̂, xr, q).u[d]]
                of_velocity = q -> [field(t̂, xr, q).ut[d] / h]
                of_slope = q -> [field(t̂, xr, q).ux[d]]
                for (expected, stored) in (
                    (ForwardDiff.jacobian(of_field, params), slab.Ju[q, r, d, :]),
                    (ForwardDiff.jacobian(of_velocity, params), slab.Jv[q, r, d, :]),
                    (ForwardDiff.jacobian(of_slope, params), slab.Jw[q, r, d, :]))
                    scale = maximum(abs, expected)
                    @test maximum(abs, stored .- vec(expected)) ≤
                          sqrt(eps(T)) * max(one(T), scale)
                end
            end
        end
    end

    @testset "A1d the refusals of the constructor" begin
        T = Float64
        problem = travelling_problem(T)
        @variables free_symbol
        @test_throws ArgumentError(
            "the expression holds the free symbol free_symbol, which is not t, x or an " *
            "entry of p") SymbolicAnsatz(
            p[1] * sin(p[2] * x - p[3] * t) + free_symbol, t, x, p;
            xspan = (zero(T), T(2) * T(π)), start = T[1, 1, 2π])
        @test_throws ArgumentError(
            "the parameter vector is empty; a SymbolicAnsatz needs at least one parameter " *
            "symbol") SymbolicAnsatz(
            sin(x - t), t, x, Num[]; xspan = (zero(T), T(2) * T(π)), start = T[])
        @test_throws ArgumentError(
            "the start vector holds 2 entries, but the ansatz has 3 parameters") SymbolicAnsatz(
            p[1] * sin(p[2] * x - p[3] * t), t, x, p;
            xspan = (zero(T), T(2) * T(π)), start = T[1, 2])
        # a space interval that does not increase
        decreasing = try
            SymbolicAnsatz(p[1] * sin(p[2] * x - p[3] * t), t, x, p;
                xspan = (T(2) * T(π), zero(T)), start = T[1, 1, 2π])
            nothing
        catch exception
            exception
        end
        @test decreasing isa ArgumentError
        @test occursin("xspan must increase", sprint(showerror, decreasing))
        # an expression count that is not the problem's
        two_components = Wave.problem(T = T, D = 2, timespan = (zero(T), T(8) * T(π)),
            timestep = T(4) * T(π), xspan = (zero(T), T(2) * T(π)), xstep = T(π) / 10)
        method = SpaceTimeGalerkin(travelling_ansatz(T), WeakInitialAndBoundaryConditions())
        @test_throws ArgumentError(
            "the ansatz holds 1 expressions, but the problem has 2 components") (
            GeometricIntegrator(two_components, method))
        # another space interval
        other_span = Wave.problem(T = T, timespan = (zero(T), T(4) * T(π)),
            timestep = T(4) * T(π), xspan = (zero(T), one(T)), xstep = T(0.05))
        @test_throws ArgumentError(
            "the ansatz xspan $(travelling_ansatz(T).xspan) does not match the problem's " *
            "$(xspan(other_span))") GeometricIntegrator(other_span, method)
        # another element type
        other_type = travelling_problem(Float32)
        @test_throws ArgumentError(
            "the ansatz element type Float64 does not match the problem's Float32") (
            GeometricIntegrator(other_type, method))
        # the valid construction of the same call does not throw
        @test GeometricIntegrator(problem, method) isa GeometricIntegrator
    end

    # A one-parameter ansatz that is linear in its parameter, on a steady field whose exact
    # parameter the first slab finds from a start that is neither zero nor it: the second slab's
    # continuation guess is then the first slab's answer, and its initial residual is round-off.
    # The second leg is the two-component field of the same shape.
    @testset "A1e the start vector carries across slabs" begin
        for T in (Float64, Float32)
            scalar = (side_problem(T, T(7) / 10; steps = 2),
                SymbolicAnsatz(q * x, t, x, [q];
                    xspan = (-one(T), one(T)), start = [T(3) / 10]))
            vector = (vector_steady_problem(T, (T(7) / 10, T(-3) / 10); steps = 2),
                vector_steady_ansatz(T; start = T[T(3) / 10, T(2) / 10]))
            for (problem, ansatz) in (scalar, vector)
                np = ansatz.num_params
                method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
                slab = slab_setup(problem, method)
                c = cache(slab.int)
                M.initial_guess!(current(slab.solstep), nothing, nothing, slab.int)
                @test c.x[1:np] == ansatz.start
                integrate!(slab.solstep, slab.int)
                first_initial = c.initial_residuals[1]
                solved = copy(c.x[1:np])
                @test solved != ansatz.start
                next_slab!(slab, 2)
                M.initial_guess!(current(slab.solstep), nothing, nothing, slab.int)
                @test c.x[1:np] == solved
                integrate!(slab.solstep, slab.int)
                # The ansatz is linear in its parameters and one Newton step is exact, so the
                # residual at the carried parameters is the round-off of the residual there:
                # measured, the second step's initial residual is `0.72 eps(T)` of the first
                # step's in `Float64` and `1.91 eps(T)` in `Float32` for the one-component field,
                # and `0.64` and `1.98` for the two-component one, against the bound of
                # `64 eps(T)`.
                @test c.initial_residuals[2] ≤ 64 * eps(T) * first_initial
            end
        end
    end

    @testset "A1f the element type of the block and the run" begin
        source = read(joinpath(pkgdir(M), "src", "ansatz", "symbolic.jl"), String)
        @test !occursin("Float64(", source)
        for T in (Float64, Float32),
            fixture in ((travelling_problem, travelling_ansatz), (
                vector_problem, vector_ansatz))

            problem = fixture[1](T)
            ansatz = fixture[2](T)
            method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
            int = GeometricIntegrator(problem, method)
            c = cache(int)
            @test eltype(c.x) === T
            @test eltype(c.ansatz.params) === T
            @test eltype(c.ansatz.residual) === T
            @test eltype(c.ansatz.u) === T
            @test eltype(c.ansatz.Ju) === T
            result = integrate(problem, method; f_abstol = zero(T),
                f_reltol = T(100) * eps(T))
            @test datatype(result.solution) === T
            for step in result.solution.u
                assert_no_upcast(step, T)
            end
            for trace in result.traces
                assert_no_upcast(trace, T)
            end
        end
    end

    @testset "A2a the residual is the action's gradient" begin
        for T in (Float64, Float32), (problem, ansatz, field) in symbolic_fixtures(T)

            method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
            slab = slab_setup(problem, method)
            oracle = SymbolicActionOracle(problem, ansatz, field)
            symbolic_first_datum!(oracle)
            N = length(cache(slab.int).x)
            rng = Xoshiro(SYMBOLIC_SEED + 1)
            b = zeros(T, N)
            θ = 2 .* rand(rng, T, N) .- 1
            M.residual!(b, θ, nothing, parameters(problem), slab.int)
            compare_gradient(b, symbolic_gradient(oracle, θ), T)
            # the second slab, whose datum the oracle carries from the first slab's solved vector
            integrate!(slab.solstep, slab.int)
            solved = copy(cache(slab.int).x)
            next_slab!(slab, 2)
            symbolic_next_datum!(oracle, solved)
            θ₂ = 2 .* rand(rng, T, N) .- 1
            M.residual!(b, θ₂, nothing, parameters(problem), slab.int)
            compare_gradient(b, symbolic_gradient(oracle, θ₂), T)
        end
    end

    # The solver's Jacobian against central differences of the residual. The differences are taken
    # on the `Float64` slab in both precisions: at the step `cbrt(eps(T))` their truncation error
    # `δ²|∂³F|/6` reaches `6.1e-4` of the scale in `Float32`, above the bound of
    # `sqrt(eps(Float32)) = 3.45e-4`, although the `Float32` Jacobian agrees with the `Float64` one
    # to `1.9e-6`. Measured on this tree, the worst of the sixteen vectors is `0.13` of the bound
    # in `Float64` and `0.005` in `Float32` — margins of `7.8×` and `230×` — for the scalar
    # fixture, and `0.05` and `0.005` for the two-component one.
    @testset "A2b the Jacobian is the derivative of the residual" begin
        for T in (Float64, Float32),
            fixture in ((travelling_problem, travelling_ansatz), (
                vector_problem, vector_ansatz))

            problem = fixture[1](T)
            method = SpaceTimeGalerkin(fixture[2](T), WeakInitialAndBoundaryConditions())
            slab = slab_setup(problem, method)
            reference_problem = fixture[1](Float64)
            reference_method = SpaceTimeGalerkin(fixture[2](Float64),
                WeakInitialAndBoundaryConditions())
            reference = slab_setup(reference_problem, reference_method)
            s = GeometricIntegratorsBase.solver(slab.int)
            N = length(cache(slab.int).x)
            params = parameters(problem)
            rng = Xoshiro(SYMBOLIC_SEED + 2)
            for _ in 1:16
                θ = 2 .* rand(rng, T, N) .- 1
                jacobian!(s, θ, (current(slab.solstep), params, slab.int))
                J = copy(jacobianmatrix(s))
                scale = maximum(abs, J)
                differences = central_differences(reference.int, Float64.(θ),
                    parameters(reference_problem))
                @test scale ≥ T(1 // 100)
                @test maximum(abs, J - differences) ≤ sqrt(eps(T)) * scale
            end
        end
    end

    # The affine control: the field `(p₁ + p₂t̂)x` is linear in its parameters and the wave density
    # is quadratic in them, so the slab's residual is affine in the solver's vector and one Newton
    # step reaches its root. The side data ramp linearly in time, so each slab's solution differs
    # from the previous one and the continuation guess is not the answer. The run bounds the solve
    # at two iterations, because SimpleSolvers' `assess_convergence` asks for a small residual *and*
    # for the iterate to settle (`x_suctol`) or the value to settle (`f_suctol`), and with
    # `f_abstol = zero(T)` no other branch ends the loop: past the exact step the solver keeps
    # stepping until round-off settles, by a count that differs between machines. The solve's
    # tolerance is the family's measured one, `symbolic_f_reltol` of the registry entry's model file.
    @testset "A2c the affine control converges in two iterations" begin
        for T in (Float64, Float32)
            scalar = (ramp_problem(T),
                SymbolicAnsatz((r[1] + r[2] * t) * x, t, x, r;
                    xspan = (-one(T), one(T)), start = T[1] / 2 .* [one(T), T(3) / 5]))
            vector = (vector_ramp_problem(T), vector_ramp_ansatz(T))
            for (problem, ansatz) in (scalar, vector)
                method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
                int = GeometricIntegrator(problem, method; f_abstol = zero(T),
                    f_reltol = symbolic_f_reltol(T), max_iterations = 2)
                result = integrate(int)
                @test all(result.stats.converged)
                @test all(result.stats.iterations .<= 2)
                @test maximum(result.stats.residuals ./ result.stats.initial_residuals) ≤
                      symbolic_f_reltol(T)
            end
        end
    end

    @testset "A3a the couplings the ansatz takes" begin
        for T in (Float64, Float32)
            problem = travelling_problem(T)
            ansatz = travelling_ansatz(T)
            method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
            int = GeometricIntegrator(problem, method)
            unknowns = M.coupling_unknowns(coupling(method), ansatz, problem)
            @test length(cache(int).x) == ansatz.num_params + unknowns
            for c in (WeakInitialCondition(), CoefficientRestriction())
                error_thrown = try
                    GeometricIntegrator(problem, SpaceTimeGalerkin(ansatz, c))
                    nothing
                catch exception
                    exception
                end
                @test error_thrown isa ArgumentError
                @test occursin("WeakInitialAndBoundaryConditions",
                    sprint(showerror, error_thrown))
            end
        end
    end

    # The side data of each end, of a scalar and of a two-component field: the run takes the
    # parameters to the slopes and the field at the two ends to `∓slope`, and a problem whose
    # data are exchanged takes them to the other end's values.
    @testset "A3b the side data of each end" begin
        for T in (Float64, Float32)
            for (slope, expected) in ((T(0.7), T(0.7)), (T(-0.7), T(-0.7)))
                problem = side_problem(T, slope)
                ansatz = SymbolicAnsatz(q * x, t, x, [q];
                    xspan = (-one(T), one(T)), start = [T(0.7)])
                method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
                int = GeometricIntegrator(problem, method)
                result = integrate(int)
                @test all(result.stats.converged)
                @test abs(cache(int).ansatz.params[1] - expected) ≤ 1e3 * eps(T)
                grid = result.solution.u[end]
                @test abs(first(grid) - expected * xspan(problem)[1]) ≤ 1e3 * eps(T)
                @test abs(last(grid) - expected * xspan(problem)[2]) ≤ 1e3 * eps(T)
            end
            for slopes in ((T(0.7), T(-0.3)), (T(-0.7), T(0.3)))
                problem = vector_steady_problem(T, slopes)
                ansatz = vector_steady_ansatz(T; start = T[T(0.7), T(0.3)])
                method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
                int = GeometricIntegrator(problem, method)
                result = integrate(int)
                @test all(result.stats.converged)
                @test maximum(abs.(cache(int).ansatz.params .- collect(slopes))) ≤
                      1e3 * eps(T)
                nx = length(xgrid(problem))
                grid = result.solution.u[end]
                for d in 1:2
                    component = grid[((d - 1) * nx + 1):(d * nx)]
                    @test abs(first(component) - slopes[d] * xspan(problem)[1]) ≤
                          1e3 * eps(T)
                    @test abs(last(component) - slopes[d] * xspan(problem)[2]) ≤
                          1e3 * eps(T)
                end
            end
        end
    end

    # The multiplier guess of a field that moves: the guess writes the `L²` projection of the
    # momentum `∂L/∂v` at the top of the slab onto the multiplier basis, and the projection of the
    # flux `∂L/∂w` at each end onto the boundary basis. The values are the hand-written field's,
    # with the `1/h` of the velocity, so a guess that reads the field wrong, drops the top
    # momentum's `1/h` or writes nothing differs from them.
    @testset "A3c the multiplier guess of a moving field" begin
        for T in (Float64, Float32), (problem, ansatz, field) in symbolic_fixtures(T)

            method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
            slab = slab_setup(problem, method)
            c = cache(slab.int)
            data = c.coupling
            np, D = ansatz.num_params, length(ansatz.expr)
            nλ, nμ = size(data.initial_multiplier, 1), size(data.boundary_basis, 1)
            h = timestep(problem)
            M.initial_guess!(current(slab.solstep), nothing, nothing, slab.int)
            params = c.x[1:np]
            # the top of the slab: `∂L/∂v` at `t̂ = 1`, in the multiplier basis' `L²` projection
            xq = c.ansatz.tabulation.space
            Ωx = Diagonal(xq.weight)
            Λ = data.multiplier_basis
            momentum = zeros(T, length(xq.node), D)
            for r in eachindex(xq.node), d in 1:D

                momentum[r, d] = field(one(T), xq.node[r], params).ut[d] / h
            end
            expected_top = ((Λ * Ωx * Λ') \ (Λ * Ωx)) * momentum
            for d in 1:D, p in 1:nλ

                top = c.x[M.multiplier_index(np, D, nλ, nμ, :top, d, p)]
                @test abs(top - expected_top[p, d]) ≤
                      1e3 * eps(T) * max(one(T), abs(expected_top[p, d]))
            end
            # the two ends: the flux `∂L/∂w = −c² ∂ₓu` of the wave density at the nodes of the
            # time rule, in the boundary basis' projection
            tq = c.ansatz.tabulation.time
            Ωt = Diagonal(tq.weight)
            χ = data.boundary_basis
            c₀ = parameters(problem).c
            for (edge, end_of_span) in ((:left, xspan(problem)[1]),
                (:right, xspan(problem)[2]))
                flux = zeros(T, length(tq.node), D)
                for q in eachindex(tq.node), d in 1:D

                    flux[q, d] = -c₀^2 * field(tq.node[q], end_of_span, params).ux[d]
                end
                expected = ((χ * Ωt * χ') \ (χ * Ωt)) * flux
                for d in 1:D, p in 1:nμ

                    side = c.x[M.multiplier_index(np, D, nλ, nμ, edge, d, p)]
                    @test abs(side - expected[p, d]) ≤
                          1e3 * eps(T) * max(one(T), abs(expected[p, d]))
                end
            end
        end
    end

    @testset "A4a the closed form over five slabs" begin
        for T in (Float64, Float32)
            problem = travelling_problem(T)
            method = SpaceTimeGalerkin(travelling_ansatz(T),
                WeakInitialAndBoundaryConditions())
            int = GeometricIntegrator(problem, method; f_abstol = zero(T),
                f_reltol = T(100) * eps(T))
            result = integrate(int)
            @test length(result.traces) == 5
            @test all(result.stats.converged)
            grid = collect(xgrid(problem))
            for n in 1:5
                exact = sin.(grid .- T(1) / 2 * result.solution.t[n])
                @test maximum(abs.(result.solution.u[n] .- exact)) ≤
                      1e3 * eps(T) * max(one(T), maximum(abs, exact))
            end
            exact_parameters = T[1, 1, 2π]
            @test maximum(abs.(cache(int).ansatz.params .- exact_parameters)) ≤
                  1e3 * eps(T) * maximum(abs, exact_parameters)
            assert_no_upcast(result.solution.u[1], T)
        end
    end

    @testset "A4b the vector field of two expressions" begin
        for T in (Float64, Float32)
            problem = vector_problem(T)
            ansatz = vector_ansatz(T)
            @test ansatz.num_params == 6
            method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
            int = GeometricIntegrator(problem, method; f_abstol = zero(T),
                f_reltol = T(100) * eps(T))
            result = integrate(int)
            @test length(result.traces) == 5
            @test all(result.stats.converged)
            grid = collect(xgrid(problem))
            for n in 1:5
                first = sin.(grid .- T(1) / 2 * result.solution.t[n])
                second = T(1) / 2 .* sin.(2 .* grid .- result.solution.t[n])
                @test maximum(abs.(result.solution.u[n][1:length(grid)] .- first)) ≤
                      1e3 * eps(T)
                @test maximum(abs.(result.solution.u[n][(length(grid) + 1):end] .-
                                   second)) ≤ 1e3 * eps(T)
            end
            exact_parameters = T[1, 1, 2π, T(1) / 2, 2, 4π]
            @test maximum(abs.(cache(int).ansatz.params .- exact_parameters)) ≤
                  1e3 * eps(T) * maximum(abs, exact_parameters)
        end
    end

    # The entry's convergence is recorded and not asserted: the closed-form oracle asserts it
    # (`A4a`), and `symbolic_f_reltol` is a tolerance measured on the family's affine control (`A2c`)
    # which a `Float32` step of this entry does not reach. The stats are still read, so a step that
    # produces no residual, or a residual that is not finite, fails here.
    @testset "A5a the entry's closed form in both precisions" begin
        entry = only(family_entries(:symbolic))
        for T in (Float64, Float32)
            problem = symbolic_problem(entry; T = T)
            method = symbolic_method(entry; T = T)
            ansatz = M.ansatz(method)
            @test ansatz.num_params == 6
            @test length(ansatz.start) == 6
            @test eltype(ansatz.start) === T
            result = integrate(problem, method; f_abstol = zero(T),
                f_reltol = symbolic_f_reltol(T))
            @test length(result.traces) == 2
            @test length(result.stats.iterations) == 2
            @test length(result.stats.residuals) == 2
            @test length(result.stats.initial_residuals) == 2
            @test all(isfinite, result.stats.residuals)
            @test all(isfinite, result.stats.initial_residuals)
            grid = collect(xgrid(problem))
            exact_field = field_exact(Wave.solution_u, 1)
            for n in 1:2
                exact = exact_field(result.solution.t[n], grid, parameters(problem))
                @test maximum(abs.(result.solution.u[n] .- exact)) ≤
                      1e3 * eps(T) * max(one(T), maximum(abs, exact))
            end
            # the archive of the run itself, in this element type: the seven keys that the
            # driver's `store_run!` completes with the entry's `"stem"` (`A5b`).
            archive = symbolic_run(entry; T = T)
            @test sort!(collect(keys(archive))) ==
                  sort!(["kind", "label", "problem_label", "timestep", "t", "x", "u"])
            @test archive["kind"] == "solution"
            @test archive["label"] == entry.label
            @test archive["problem_label"] == "Wave"
            @test archive["timestep"] == entry.timestep
            t, x, u = archive["t"], archive["x"], archive["u"]
            @test eltype(u[1]) === T
            @test size(u[1]) == (length(t), length(x))
            for n in eachindex(t)
                exact = exact_field(t[n], x, parameters(problem))
                @test maximum(abs.(u[1][n, :] .- exact)) ≤
                      1e3 * eps(T) * max(one(T), maximum(abs, exact))
            end
        end
    end

    @testset "A5b the one driver archives the entry" begin
        directory = mktempdir()
        try
            @test main(["symbolic", "--runs-dir", directory]) == 0
            data = load_run("wave-symbolic-h4")
            @test sort!(collect(keys(data))) ==
                  sort!([
                "kind", "stem", "label", "problem_label", "timestep", "t", "x", "u"])
            @test data["stem"] == "wave-symbolic-h4"
            @test data["problem_label"] == "Wave"
            @test data["kind"] == "solution"
            t = data["t"]
            x = data["x"]
            u = data["u"]
            @test length(u) == 1
            @test size(u[1]) == (length(t), length(x))
            exact_field = field_exact(Wave.solution_u, 1)
            params = parameters(symbolic_problem(only(family_entries(:symbolic))))
            for n in eachindex(t)
                exact = exact_field(t[n], x, params)
                @test maximum(abs.(u[1][n, :] .- exact)) ≤
                      1e3 * eps(Float64) * max(1.0, maximum(abs, exact))
            end
        finally
            rm(directory; recursive = true, force = true)
        end
    end

    @testset "A5c an entry's ansatz and start vector have matching lengths" begin
        for entry in family_entries(:symbolic), T in (Float64, Float32)

            problem = symbolic_problem(entry; T = T)
            ansatz = symbolic_ansatz(entry; T = T)
            @test ansatz.num_params == length(ansatz.start) == 6
            @test eltype(ansatz.start) === T
            @test ansatz.xspan == xspan(problem)
            @test length(ansatz.expr) == components(problem)
        end
    end

    # Both precisions. The run's last relative energy deviation is `1.6e-15` in `Float64` and
    # `2.57e-6` in `Float32`, and the bound is `1e3 eps(T)`, a multiple of the element type's own
    # resolution as §V requires: `2.2e-13` in `Float64` and `1.19e-4` in `Float32`.
    @testset "A6 the energy of a symbolic run" begin
        for T in (Float64, Float32),
            fixture in ((travelling_problem, travelling_ansatz), (
                vector_problem, vector_ansatz))

            problem = fixture[1](T)
            method = SpaceTimeGalerkin(fixture[2](T), WeakInitialAndBoundaryConditions())
            result = integrate(problem, method; f_abstol = zero(T),
                f_reltol = T(100) * eps(T))
            errors = energy_error(problem, method, result.traces)
            @test length(errors) == 5
            @test iszero(errors[1])
            @test all(isfinite, errors)
            @test last(errors) ≤ T(1000) * eps(T)
        end
    end
end
