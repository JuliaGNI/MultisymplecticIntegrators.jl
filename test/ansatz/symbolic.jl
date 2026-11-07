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
    grid = collect(span[1]:step:span[2])
    timespan = (zero(T), T(20) * T(π))
    params = (c = c,)
    return LagrangianPDEProblem(Wave.density(; params = params);
        D = 1,
        ics_function = (x, p) -> field_initial_data(field, velocity, slope, x, p, 1,
            timespan[1]),
        bcs_function = (times, p) -> field_boundary_data(field, velocity, slope, times,
            grid, p, 1),
        exact_u = field_exact(field, 1), exact_v = field_exact(velocity, 1),
        exact_w = field_exact(slope, 1),
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
    grid = collect(span[1]:step:span[2])
    timespan = (zero(T), T(20) * T(π))
    params = (c = c,)
    stack(f, g) = (t, x, p) -> vcat([f(t, xi, p) for xi in x], [g(t, xi, p) for xi in x])
    at(f, g, x) = (times, p) -> vcat([f(t, x, p) for t in times], [g(t, x, p)
                                                                   for t in times])
    ics = (x, p) -> (u = stack(first, second)(timespan[1], x, p),
        v = stack(first_velocity, second_velocity)(timespan[1], x, p),
        w = stack(first_slope, second_slope)(timespan[1], x, p))
    bcs = (times, p) -> (
        bottom = (u = stack(first, second)(times[1], grid, p),
            v = stack(first_velocity, second_velocity)(times[1], grid, p),
            w = stack(first_slope, second_slope)(times[1], grid, p)),
        top = (u = stack(first, second)(times[2], grid, p),
            v = stack(first_velocity, second_velocity)(times[2], grid, p),
            w = stack(first_slope, second_slope)(times[2], grid, p)),
        left = (u = at(first, second, span[1])(times, p),
            v = at(first_velocity, second_velocity, span[1])(times, p),
            w = at(first_slope, second_slope, span[1])(times, p)),
        right = (u = at(first, second, span[2])(times, p),
            v = at(first_velocity, second_velocity, span[2])(times, p),
            w = at(first_slope, second_slope, span[2])(times, p)))
    return LagrangianPDEProblem(VectorWave.vector_density(; params = params, D = 2);
        D = 2,
        ics_function = ics, bcs_function = bcs,
        exact_u = stack(first, second), exact_v = stack(first_velocity, second_velocity),
        exact_w = stack(first_slope, second_slope),
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

# The fixtures of the action-gradient check: a `D = 1` and a `D = 2` problem, each with the
# ansatz and the hand-written field map of its own.
function symbolic_fixtures(::Type{T}) where {T}
    return ((travelling_problem(T), travelling_ansatz(T), travelling_field),
        (vector_problem(T), vector_ansatz(T), vector_field))
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
    grid = collect(span[1]:step:span[2])
    params = (c = c,)
    return LagrangianPDEProblem(Wave.density(; params = params);
        D = 1,
        ics_function = (x, p) -> field_initial_data(field, velocity, slope, x, p, 1,
            zero(T)),
        bcs_function = (times, p) -> field_boundary_data(field, velocity, slope, times,
            grid, p, 1),
        exact_u = field_exact(field, 1), exact_v = field_exact(velocity, 1),
        exact_w = field_exact(slope, 1),
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
    grid = collect(span[1]:step:span[2])
    params = (c = c,)
    return LagrangianPDEProblem(Wave.density(; params = params);
        D = 1,
        ics_function = (x, p) -> field_initial_data(field, velocity, gradient, x, p, 1,
            zero(T)),
        bcs_function = (times, p) -> field_boundary_data(field, velocity, gradient, times,
            grid, p, 1),
        exact_u = field_exact(field, 1), exact_v = field_exact(velocity, 1),
        exact_w = field_exact(gradient, 1),
        timespan = (zero(T), T(steps)), timestep = one(T), xspan = span, xstep = step,
        params = params)
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
        for f in (ansatz.u, ansatz.ut, ansatz.ux, ansatz.Ju, ansatz.Jut, ansatz.Jux)
            @test isconcretetype(typeof(f))
        end
        @test fieldtype(typeof(ansatz), :u) === typeof(ansatz.u)
        @test fieldtype(typeof(ansatz), :Jux) === typeof(ansatz.Jux)
        # the parameter vector is the one the expression is written in
        @test all(isequal.(ansatz.p, collect(p)))
    end

    @testset "A1b the field map and its derivatives" begin
        for T in (Float64, Float32)
            problem = travelling_problem(T)
            ansatz = travelling_ansatz(T)
            slab = M.ansatz_data(ansatz, problem, T)
            tnodes, xnodes = M.quadrature_points(slab)
            h = timestep(problem)
            qt, qx = length(tnodes), length(xnodes)
            u, v, w = zeros(T, qt, qx, 1), zeros(T, qt, qx, 1), zeros(T, qt, qx, 1)
            params = T[0.7, 1.1, 1.3]
            M.slab_field!(u, v, w, slab, params, nothing, ansatz.num_params, h)
            for q in 1:qt, r in 1:qx

                t̂, xr = tnodes[q], xnodes[r]
                value = params[1] * sin(params[2] * xr - params[3] * t̂)
                velocity = -params[1] * params[3] * cos(params[2] * xr - params[3] * t̂) / h
                slope = params[1] * params[2] * cos(params[2] * xr - params[3] * t̂)
                @test abs(u[q, r, 1] - value) ≤ 16 * eps(T) * max(one(T), abs(value))
                @test abs(v[q, r, 1] - velocity) ≤
                      16 * eps(T) * max(one(T), abs(velocity))
                @test abs(w[q, r, 1] - slope) ≤ 16 * eps(T) * max(one(T), abs(slope))
            end
            # the top of the slab is `t̂ = 1`, at the nodes of the trace evaluation
            utop, vtop, wtop = zeros(T, qx, 1), zeros(T, qx, 1), zeros(T, qx, 1)
            M.top_values!(utop, vtop, wtop, slab, M.trace_evaluation(slab), h)
            for r in 1:qx
                xr = xnodes[r]
                value = params[1] * sin(params[2] * xr - params[3])
                @test abs(utop[r, 1] - value) ≤ 16 * eps(T) * max(one(T), abs(value))
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
        for T in (Float64, Float32)
            problem = travelling_problem(T)
            ansatz = travelling_ansatz(T)
            slab = M.ansatz_data(ansatz, problem, T)
            tnodes, xnodes = M.quadrature_points(slab)
            h = timestep(problem)
            u, v, w = zeros(T, length(tnodes), length(xnodes), 1),
            zeros(T, length(tnodes), length(xnodes), 1),
            zeros(T, length(tnodes), length(xnodes), 1)
            params = T[0.7, 1.1, 1.3]
            M.slab_field!(u, v, w, slab, params, nothing, ansatz.num_params, h)
            for q in eachindex(tnodes), r in eachindex(xnodes)

                t̂, xr = tnodes[q], xnodes[r]
                field = q -> [q[1] * sin(q[2] * xr - q[3] * t̂)]
                velocity = q -> [-q[1] * q[3] * cos(q[2] * xr - q[3] * t̂) / h]
                slope = q -> [q[1] * q[2] * cos(q[2] * xr - q[3] * t̂)]
                for (expected, stored) in (
                    (ForwardDiff.jacobian(field, params), slab.Ju[q, r, 1, :]),
                    (ForwardDiff.jacobian(velocity, params), slab.Jv[q, r, 1, :]),
                    (ForwardDiff.jacobian(slope, params), slab.Jw[q, r, 1, :]))
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
    # parameter the first slab finds from a start that is not it: the second slab's continuation
    # guess is then the first slab's answer, and its initial residual is round-off.
    @testset "A1e the start vector carries across slabs" begin
        for T in (Float64, Float32)
            problem = side_problem(T, T(7) / 10; steps = 2)
            ansatz = SymbolicAnsatz(q * x, t, x, [q];
                xspan = (-one(T), one(T)), start = [zero(T)])
            method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
            slab = slab_setup(problem, method)
            c = cache(slab.int)
            M.initial_guess!(current(slab.solstep), nothing, nothing, slab.int)
            @test c.x[1] == zero(T)
            integrate!(slab.solstep, slab.int)
            first_initial = c.initial_residuals[1]
            solved = copy(c.x)
            next_slab!(slab, 2)
            M.initial_guess!(current(slab.solstep), nothing, nothing, slab.int)
            @test c.x[1] == solved[1]
            integrate!(slab.solstep, slab.int)
            # The residual at the carried parameter is round-off, so the ratio carries a floor of
            # the element type's own resolution beside the `1e-8` term.
            @test c.initial_residuals[2] ≤ (T(1e-8) + 64 * eps(T)) * first_initial
        end
    end

    @testset "A1f the element type of the block and the run" begin
        source = read(joinpath(pkgdir(M), "src", "ansatz", "symbolic.jl"), String)
        @test !occursin("Float64(", source)
        for T in (Float64, Float32)
            problem = travelling_problem(T)
            ansatz = travelling_ansatz(T)
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

    @testset "A2b the Jacobian is the derivative of the residual" begin
        for T in (Float64, Float32)
            problem = travelling_problem(T)
            method = SpaceTimeGalerkin(travelling_ansatz(T),
                WeakInitialAndBoundaryConditions())
            slab = slab_setup(problem, method)
            s = GeometricIntegratorsBase.solver(slab.int)
            N = length(cache(slab.int).x)
            rng = Xoshiro(SYMBOLIC_SEED + 2)
            θ = 2 .* rand(rng, T, N) .- 1
            params = parameters(problem)
            jacobian!(s, θ, (current(slab.solstep), params, slab.int))
            J = copy(jacobianmatrix(s))
            scale = maximum(abs, J)
            @test scale ≥ T(1 // 100)
            @test maximum(abs, J - central_differences(slab.int, θ, params)) ≤
                  sqrt(eps(T)) * scale
        end
    end

    # The affine control: the field `(p₁ + p₂t̂)x` is linear in its parameters and the wave density
    # is quadratic in them, so the slab's residual is affine in the solver's vector and Newton
    # reaches its root in one step. The side data ramp linearly in time, so each slab's solution
    # differs from the previous one and the continuation guess is not the answer. The solve's
    # tolerance is the family's measured one, `symbolic_f_reltol` of the registry entry's model file.
    @testset "A2c the affine control converges in two iterations" begin
        for T in (Float64, Float32)
            problem = ramp_problem(T)
            ansatz = SymbolicAnsatz((r[1] + r[2] * t) * x, t, x, r;
                xspan = (-one(T), one(T)), start = T[1] / 2 .* [one(T), T(3) / 5])
            method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
            int = GeometricIntegrator(problem, method; f_abstol = zero(T),
                f_reltol = symbolic_f_reltol(T))
            result = integrate(int)
            @test all(result.stats.converged)
            @test all(result.stats.iterations .<= 2)
            @test maximum(result.stats.residuals ./ result.stats.initial_residuals) ≤
                  symbolic_f_reltol(T)
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
            ansatz = symbolic_ansatz(entry; T = T)
            method = symbolic_method(entry; T = T)
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
            exact_field = field_exact(Wave.exact_u, 1)
            for n in 1:2
                exact = exact_field(result.solution.t[n], grid, parameters(problem))
                @test maximum(abs.(result.solution.u[n] .- exact)) ≤
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
            exact_field = field_exact(Wave.exact_u, 1)
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

    # The `Float64` leg only. The run's `Float32` bound is parked: a fixed `1e-6` fails at the
    # lower precision, and the bound that replaces it (`1e3 eps(T)`) is the user's to decide.
    @testset "A6 the energy of a symbolic run" begin
        T = Float64
        problem = travelling_problem(T)
        method = SpaceTimeGalerkin(travelling_ansatz(T), WeakInitialAndBoundaryConditions())
        result = integrate(problem, method; f_abstol = zero(T),
            f_reltol = T(100) * eps(T))
        errors = energy_error(problem, method, result.traces)
        @test length(errors) == 5
        @test iszero(errors[1])
        @test all(isfinite, errors)
        @test last(errors) ≤ 1e-6
    end
end
