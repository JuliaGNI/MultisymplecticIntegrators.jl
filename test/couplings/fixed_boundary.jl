using GeometricBase: parameters, timestep, timespan
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: norm
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, current, integrate!, reset!,
                                 solutionstep, solver
import GeometricBase: timesteps

include("../helpers/problem_generator.jl")
include("../helpers/action_oracle.jl")
include("../helpers/testsetup.jl")

const M = MultisymplecticIntegrators
const WF = M.Wave
const SG = M.SineGordon

# ---- helpers -------------------------------------------------------------------------

"""
    compare_gradient(b, ∇S, T)

The package's residual `b` against the oracle's gradient `∇S`: both finite, the gradient at least
`1/100` in its largest entry, and the difference at most `100 eps(T)` of that entry.
"""
function compare_gradient(b, ∇S, ::Type{T}) where {T}
    @test all(isfinite, b)
    @test all(isfinite, ∇S)
    scale = maximum(abs, ∇S)
    @test scale ≥ T(1 // 100)
    @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * scale
    return nothing
end

"""A fixed-boundary network on a problem's span and component count."""
function fixed_ansatz(problem; Sx = 4, St = 4, form = :abg, mode = :all, ncells = 3,
        nq = (4, 3))
    return FixedBoundaryNetwork(Sx = Sx, St = St, bubble_form = form, mode = mode,
        xspan = xspan(problem), nq = nq, ncells = ncells, D = components(problem))
end

"""The integrator of one slab, and the solution step a run drives."""
function fixed_slab(problem, ansatz; guess = Continuation(), kwargs...)
    method = SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = guess)
    int = GeometricIntegrator(problem, method; kwargs...)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    slab = (; problem, method, int, solution, solstep, ansatz)
    next_fixed_slab!(slab, 1)
    return slab
end

"""Move a slab to step `n` and fix its datum, as a run does before it solves the slab."""
function next_fixed_slab!(slab, n::Integer)
    reset!(slab.solstep, timesteps(slab.solution)[n])
    c = cache(slab.int)
    M.slab_bottom!(c, c.coupling, slab.problem, slab.ansatz, current(slab.solstep))
    return nothing
end

"""The datum of the coupling's block, as the oracle reads it."""
function fixed_datum(c)
    (; phi = copy(c.coupling.phi), momentum = copy(c.coupling.momentum),
        start = c.coupling.start)
end

"""One component of a level's network, written here from its definition alone."""
function level_network(problem, a, φ, x, t, d)
    xa, xb = xspan(problem)
    L = xb - xa
    ŝ = (x - xa) / L
    p = parameters(problem)
    ga = problem.bcs_function(t, p).left.u[d]
    gb = problem.bcs_function(t, p).right.u[d]
    m = zero(eltype(φ))
    for i in 1:a.Sx
        m += φ[2a.Sx + (i - 1) * a.D + d] * tanh(φ[i] * ŝ + φ[a.Sx + i])
    end
    return (1 - ŝ) * ga + ŝ * gb + (x - xa) * (xb - x) * m
end

"""Central differences of a residual at `θ`, one column per unknown."""
function residual_differences(int, θ::AbstractVector{T}, params) where {T}
    n = length(θ)
    J = zeros(T, n, n)
    forward = zeros(T, n)
    backward = zeros(T, n)
    x = copy(θ)
    for k in 1:n
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

"""The residual of a slab at `θ`, in a vector of `θ`'s element type."""
function slab_residual(int, θ, params)
    b = zeros(eltype(θ), length(θ))
    M.residual!(b, θ, nothing, params, int)
    return b
end

const PROBLEMS = Dict(
    :wave =>
        () -> WF.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05,
            xspan = (0.0, 1.0)),
    :sinegordon =>
        () -> SG.problem(timespan = (0.0, 0.2), timestep = 0.1, xstep = 0.05,
            xspan = (0.0, 1.0)))

# ---- FB0 -----------------------------------------------------------------------------

@testset "FB0 the two types and their pair" begin
    problem = PROBLEMS[:wave]()
    a = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0))
    @test FixedBoundary() isa SlabCoupling
    @test a isa SpaceTimeAnsatz
    @test (M.nspatial(a), M.nbubble(a), M.nfree(a)) == (12, 16, 28)
    last_layer = FixedBoundaryNetwork(Sx = 4, St = 4, xspan = (0.0, 1.0), mode = :last_layer)
    @test (M.nspatial(last_layer), length(M.spatial_free(last_layer)),
        M.nfree(last_layer)) == (12, 4, 8)
    network = NetworkAnsatz(S = 4, xspan = (0.0, 1.0), ncells = (2, 5), degree = 3)
    spline = TensorBSplineAnsatz(degree = 3, ncells = (2, 5), xspan = (0.0, 1.0))
    @test_throws r"FixedBoundary takes a FixedBoundaryNetwork, got NetworkAnsatz" GeometricIntegrator(
        problem, SpaceTimeGalerkin(network, FixedBoundary()))
    @test_throws r"FixedBoundary takes a FixedBoundaryNetwork, got TensorBSplineAnsatz" GeometricIntegrator(
        problem, SpaceTimeGalerkin(spline, FixedBoundary()))
    for coupling in (WeakInitialAndBoundaryConditions(), WeakInitialCondition(),
        CoefficientRestriction())
        @test_throws r"a FixedBoundaryNetwork takes the FixedBoundary coupling" GeometricIntegrator(
            problem, SpaceTimeGalerkin(a, coupling))
    end
    method = SpaceTimeGalerkin(a, FixedBoundary())
    int = GeometricIntegrator(problem, method)
    @test default_solver(method) isa LevenbergMarquardt
    @test solver(int) isa M.LevenbergMarquardtSolver
    @test length(cache(int).x) == 28
    @test M.coupling_unknowns(FixedBoundary(), a, problem) == 0
    @test M.free_rows(FixedBoundary(), a) == 28
    @test total_free_coefficients(method, problem) == 28
end

# ---- FB1 -----------------------------------------------------------------------------

@testset "FB1 the residual is the action's gradient" begin
    rng = Xoshiro(7)
    @testset "generated entry, $form, $mode" for entry in generate_problems(Float64;
            rng = Xoshiro(SEED)), form in (:abg, :tsr), mode in (:all, :last_layer)
        ansatz = fixed_ansatz(entry.problem; form = form, mode = mode)
        problem = entry.problem
        slab = fixed_slab(problem, ansatz)
        c = cache(slab.int)
        oracle = ActionOracle(problem, slab.method; block = c.ansatz)
        @test length(c.x) == unknown_count(oracle)
        θ = 2 .* rand(rng, Float64, length(c.x)) .- 1
        compare_gradient(slab_residual(slab.int, θ, parameters(problem)),
            oracle_residual(oracle, fixed_datum(c), θ), Float64)
        integrate!(slab.solstep, slab.int)
        next_fixed_slab!(slab, 2)
        θ₂ = 2 .* rand(rng, Float64, length(c.x)) .- 1
        compare_gradient(slab_residual(slab.int, θ₂, parameters(problem)),
            oracle_residual(oracle, fixed_datum(c), θ₂), Float64)
    end
    @testset "the two problems, first and second slab, $name, $form, $mode" for name in (:wave, :sinegordon),
        form in (:abg, :tsr), mode in (:all, :last_layer)

        problem = PROBLEMS[name]()
        slab = fixed_slab(problem, fixed_ansatz(problem; form = form, mode = mode))
        c = cache(slab.int)
        oracle = ActionOracle(problem, slab.method; block = c.ansatz)
        θ = 2 .* rand(rng, Float64, length(c.x)) .- 1
        compare_gradient(slab_residual(slab.int, θ, parameters(problem)),
            oracle_residual(oracle, fixed_datum(c), θ), Float64)
        integrate!(slab.solstep, slab.int)
        next_fixed_slab!(slab, 2)
        θ₂ = 2 .* rand(rng, Float64, length(c.x)) .- 1
        compare_gradient(slab_residual(slab.int, θ₂, parameters(problem)),
            oracle_residual(oracle, fixed_datum(c), θ₂), Float64)
    end
end

# ---- FB2 -----------------------------------------------------------------------------

@testset "FB2 the Jacobian is the derivative of the residual" begin
    rng = Xoshiro(11)
    # `jacobian!` is SimpleSolvers' and writes the solver's own matrix, so it exists for a
    # `NonlinearSolver`. The family's Levenberg–Marquardt keeps no such matrix — it takes one SVD of
    # the Jacobian at each accepted iterate — so the Jacobian of the residual is read through the
    # `Newton()` solver the family also offers, on the same residual and the same block.
    for name in (:wave, :sinegordon), form in (:abg, :tsr), mode in (:all, :last_layer)
        problem = PROBLEMS[name]()
        slab = fixed_slab(problem, fixed_ansatz(problem; form = form, mode = mode);
            solver = Newton())
        c = cache(slab.int)
        s = solver(slab.int)
        θ = 2 .* rand(rng, Float64, length(c.x)) .- 1
        p = parameters(problem)
        # at the slab's guess, as a run reaches the Jacobian: a `:last_layer` slab whose held hidden
        # layer is still zero has a zero Jacobian, whatever its residual is
        M.initial_guess!(current(slab.solstep), nothing, p, slab.int)
        SimpleSolvers.jacobian!(s, θ, (current(slab.solstep), p, slab.int))
        J = copy(SimpleSolvers.jacobianmatrix(s))
        @test maximum(abs, J) ≥ 1 // 100
        @test maximum(abs, J - residual_differences(slab.int, θ, p)) ≤
              sqrt(eps(Float64)) * maximum(abs, J)
    end
end

# ---- FB3 -----------------------------------------------------------------------------

@testset "FB3 the first momentum and the momentum handed on" begin
    for name in (:wave, :sinegordon)
        problem = PROBLEMS[name]()
        slab = fixed_slab(problem, fixed_ansatz(problem))
        c = cache(slab.int)
        p = parameters(problem)
        ansatz = slab.ansatz
        nodes, weights = ansatz.tabulation.xnode, ansatz.tabulation.xweight
        ics = problem.ics_function(collect(nodes), p)
        expect = zeros(length(c.coupling.momentum))
        for r in eachindex(nodes)
            g = ForwardDiff.gradient(
                φ -> sum(level_network(problem, ansatz, φ, nodes[r],
                             0.0, d) for d in 1:ansatz.D),
                c.coupling.phi)
            for (k, i) in enumerate(M.spatial_free(ansatz)), d in 1:ansatz.D

                expect[k] += weights[r] *
                             derivative_v(problem.density, ics.u[r, d],
                                 ics.v[r, d], ics.w[r, d], p) * g[i]
            end
        end
        @test maximum(abs, c.coupling.momentum - expect) ≤
              100 * eps(Float64) * max(1, maximum(abs, expect))
        # the momentum handed on, at the final iterate of a step that is capped at three iterations
        oracle = ActionOracle(problem, slab.method; block = c.ansatz)
        datum = fixed_datum(c)
        int = GeometricIntegrator(problem, slab.method; solver = LevenbergMarquardt(),
            max_iterations = 3, verbosity = 0)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        # A step's own time is the end of the slab it solves — `slab_start` reads one timestep back
        # from it — so a run's first step is placed at `timesteps(solution)[1]`, the first slab's
        # end, as the helper `next_fixed_slab!` places it.
        reset!(solstep, timesteps(solution)[1])
        integrate!(solstep, int)
        cf = cache(int)
        next_fixed_slab!(
            (; problem, method = slab.method, int, solution, solstep,
                ansatz = ansatz), 2)
        @test maximum(abs,
            oracle_momentum(oracle, datum, copy(cf.ansatz.params[M.free_indices(ansatz)])) -
            cf.coupling.p_next) ≤
              1000 * eps(Float64) * max(1, maximum(abs, cf.coupling.p_next))
    end
end

# ---- FB4 -----------------------------------------------------------------------------

@testset "FB4 the solution carries the side data at every step" begin
    # The state is component-major: component `d`'s `nx` output-grid values start at
    # `(d - 1) * nx + 1`, and the grid's ends are the span's.
    function ends(u, D)
        nx = length(u) ÷ D
        return [(u[(d - 1) * nx + 1], u[d * nx]) for d in 1:D]
    end
    for name in (:wave, :sinegordon), form in (:abg,)

        problem = PROBLEMS[name]()
        method = SpaceTimeGalerkin(fixed_ansatz(problem; form = form), FixedBoundary())
        result = integrate(problem, method; max_iterations = 3, verbosity = 0)
        for n in eachindex(result.stats.converged)
            t = result.solution.t[n]
            bcs = problem.bcs_function(t, parameters(problem))
            scale = max(1, maximum(abs, result.solution.u[n]))
            for (d, (left, right)) in enumerate(ends(result.solution.u[n],
                components(problem)))
                @test abs(left - bcs.left.u[d]) ≤ 8 * eps(Float64) * scale
                @test abs(right - bcs.right.u[d]) ≤ 8 * eps(Float64) * scale
            end
        end
    end
    for entry in generate_problems(Float64; rng = Xoshiro(SEED))[13:15]
        problem = entry.problem
        method = SpaceTimeGalerkin(fixed_ansatz(problem), FixedBoundary())
        result = integrate(problem, method; max_iterations = 3, verbosity = 0)
        for n in eachindex(result.stats.converged)
            bcs = problem.bcs_function(result.solution.t[n], parameters(problem))
            scale = max(1, maximum(abs, result.solution.u[n]))
            for (d, (left, _)) in enumerate(ends(result.solution.u[n], components(problem)))
                @test abs(left - bcs.left.u[d]) ≤ 8 * eps(Float64) * scale
            end
        end
    end
end

# ---- FB5 -----------------------------------------------------------------------------

@testset "FB5 variable projection" begin
    for name in (:wave, :sinegordon), form in (:abg, :tsr)

        problem = PROBLEMS[name]()
        ansatz = fixed_ansatz(problem; form = form)
        method = SpaceTimeGalerkin(ansatz, FixedBoundary();
            initial_guess = FitTo(Predictor()))
        int = GeometricIntegrator(problem, method;
            solver = LevenbergMarquardt(variable_projection = true),
            max_iterations = 2, verbosity = 0)
        slot = length(M.variable_projection_indices(method))
        @test slot == ansatz.St * ansatz.D
        @test M.variable_projection_indices(method) ==
              collect((M.nfree(ansatz) - slot + 1):M.nfree(ansatz))
        c = cache(int)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        reset!(solstep, 0.0)
        M.initial_guess!(solstep, nothing, parameters(problem), int)
        b = slab_residual(int, c.x, parameters(problem))
        @test all(isfinite, b)
    end
end

# ---- FB6 -----------------------------------------------------------------------------

@testset "FB6 the record of a step under each form and solve" begin
    problem = PROBLEMS[:wave]()
    small = FixedBoundaryNetwork(
        Sx = 4, St = 4, xspan = (0.0, 1.0), ncells = 3, nq = (4, 3))
    @test default_solver(SpaceTimeGalerkin(small, FixedBoundary())) isa LevenbergMarquardt
    for form in (:abg, :tsr), solve in (:lm, :newton)

        a = FixedBoundaryNetwork(Sx = 4, St = 4, bubble_form = form, xspan = (0.0, 1.0),
            ncells = 3, nq = (4, 3))
        m = SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = FitTo(Predictor()))
        solvermethod = solve === :lm ? LevenbergMarquardt() : Newton()
        int = GeometricIntegrator(problem, m; solver = solvermethod, max_iterations = 3,
            verbosity = 0)
        result = integrate(problem, m; solver = solvermethod, max_iterations = 3,
            verbosity = 0)
        stats = result.stats
        @test length(stats.iterations) == length(result.traces)
        @test all(stats.iterations .≤ 3)
        @test all(isfinite, stats.residuals)
        @test all(isfinite, stats.initial_residuals)
        options = config(solver(int))
        for n in eachindex(stats.converged)
            @test stats.converged[n] ==
                  M.step_converged(stats.residuals[n], stats.initial_residuals[n], options)
        end
    end
end

# ---- FB7 -----------------------------------------------------------------------------

@testset "FB7 the LM routine against a closed form" begin
    f!(b, x) = (b .= atan.(x))
    x = [2.0, -3.0]
    result = M.lm_solve!(x, f!, 2; max_iterations = 300, f_abstol = 0.0, f_reltol = 1e-12)
    @test result.converged
    @test norm(x) ≤ 1e-8
    @test result.residual == norm(atan.(x))
    @test result.initial_residual > 1
    y = [NaN, 0.0]
    nan = M.lm_solve!(y, f!, 2; max_iterations = 300, f_abstol = 0.0, f_reltol = 1e-8)
    @test nan.iterations == 0
    @test !nan.converged
end

# ---- FB8 -----------------------------------------------------------------------------

@testset "FB8 the affine control in two iterations" begin
    for name in (:wave,), form in (:abg, :tsr)

        problem = PROBLEMS[name]()
        a = FixedBoundaryNetwork(Sx = 4, St = 4, bubble_form = form, mode = :last_layer,
            xspan = (0.0, 1.0), ncells = 3, nq = (4, 3))
        m = SpaceTimeGalerkin(a, FixedBoundary(); initial_guess = FitTo(Predictor()))
        result = integrate(problem, m; solver = Newton(), max_iterations = 2,
            f_reltol = fixed_boundary_f_reltol(Float64), verbosity = 0)
        stats = result.stats
        @test all(stats.converged)
        @test all(stats.iterations .≤ 2)
        @test all(stats.initial_residuals .≥ 1 // 100)
    end
    @test fixed_boundary_f_reltol(Float64) == 1e-8
    @test fixed_boundary_f_reltol(Float32) ≥ 1e-8
    @test fixed_boundary_f_reltol(Float32) != Float32(1e-8)
    problem = PROBLEMS[:wave]()
    int = GeometricIntegrator(problem,
        SpaceTimeGalerkin(fixed_ansatz(problem), FixedBoundary()))
    options = config(solver(int))
    @test options.f_abstol == 0
    @test options.f_reltol == fixed_boundary_f_reltol(Float64)
    @test options.max_iterations == 300
end

# ---- FB9 -----------------------------------------------------------------------------

@testset "FB9 the dead branch on Wave over twenty steps" begin
    problem = WF.problem(timespan = (0.0, 1.0), timestep = 0.05, xstep = 0.05,
        xspan = (0.0, 1.0))
    ansatz = FixedBoundaryNetwork(Sx = 8, St = 10, bubble_form = :abg, xspan = (0.0, 1.0),
        ncells = 20, nq = (6, 4))
    method = SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = Continuation())
    result = integrate(problem, method; solver = Newton(), max_iterations = 60,
        f_abstol = 0.0, f_reltol = fixed_boundary_f_reltol(Float64), verbosity = 0)
    stats = result.stats
    @test length(stats.converged) == 20
    @test all(stats.converged)
    t = result.solution.t
    u = result.solution.u
    grid = xspan(problem)
    xs = collect(range(grid[1], grid[2]; length = 21))
    error = maximum(abs(WF.solution_u(t[end], x, parameters(problem)) - u[end][i])
    for (i, x) in enumerate(xs))
    @info "FB9 the dead branch's twenty steps" error=error worst_ratio=maximum(stats.residuals ./
                                                                               stats.initial_residuals)
    @test 5.96e-4 ≤ error ≤ 7.28e-4
end

# ---- FB10 ----------------------------------------------------------------------------

@testset "FB10 the continuation starts at φ₀ and the previous Θ" begin
    for form in (:abg, :tsr)
        problem = PROBLEMS[:wave]()
        ansatz = fixed_ansatz(problem; form = form)
        slab = fixed_slab(problem, ansatz)
        c = cache(slab.int)
        nfp = length(M.spatial_free(ansatz))
        M.initial_guess!(current(slab.solstep), nothing, parameters(problem), slab.int)
        start = M.bubble_start(ansatz, Float64)
        @test copy(c.x[1:nfp]) ≈ c.coupling.phi[M.spatial_free(ansatz)]
        @test copy(c.x[(nfp + 1):end]) ≈ start[M.bubble_free(ansatz)]
        @test all(iszero, view(start, (3ansatz.St + 1):length(start)))
        integrate!(slab.solstep, slab.int)
        solved = copy(c.x)
        next_fixed_slab!(slab, 2)
        M.initial_guess!(current(slab.solstep), nothing, parameters(problem), slab.int)
        @test copy(c.x[1:nfp]) ≈ c.coupling.phi[M.spatial_free(ansatz)]
        @test copy(c.x[(nfp + 1):end]) ≈ solved[(nfp + 1):end]
    end
end

# ---- FB11 ----------------------------------------------------------------------------

@testset "FB11 a unit at r = 0" begin
    problem = PROBLEMS[:wave]()
    a = FixedBoundaryNetwork(Sx = 8, St = 9, bubble_form = :tsr, xspan = (0.0, 1.0),
        ncells = 3, nq = (4, 3))
    @test_throws r"unit 5" M.bubble_start(a, Float64)
    abg = FixedBoundaryNetwork(Sx = 8, St = 9, bubble_form = :abg, xspan = (0.0, 1.0),
        ncells = 3, nq = (4, 3))
    @test length(M.bubble_start(abg, Float64)) == M.nbubble(abg)
    method = SpaceTimeGalerkin(abg, FixedBoundary())
    result = integrate(problem, method; max_iterations = 2, verbosity = 0)
    @test isfinite(norm(result.solution.u[end]))
    atoms = bubble_atoms(BubbleGreedy(), Float64)
    @test size(atoms, 1) == 2184
    @test !any(iszero(atoms[i, 1]) && iszero(atoms[i, 2]) for i in axes(atoms, 1))
end

# ---- FB12 ----------------------------------------------------------------------------

@testset "FB12 the predictor against the test's own copy" begin
    for name in (:wave, :sinegordon)
        problem = PROBLEMS[name]()
        ansatz = fixed_ansatz(problem)
        slab = fixed_slab(problem, ansatz)
        c = cache(slab.int)
        tb = ansatz.tabulation
        sample = zeros(Float64, length(tb.tnode), length(tb.xnode), ansatz.D)
        M.evaluate!(sample, Predictor(), c, ansatz, problem, current(slab.solstep))
        # the test's own Störmer–Verlet, from the same bottom level and the same velocity
        p = parameters(problem)
        grid = collect(range(xspan(problem)[1], xspan(problem)[2]; length = 101))
        dx = grid[2] - grid[1]
        dt = timestep(problem) / 10
        u = [level_network(problem, ansatz, c.coupling.phi, x, 0.0, 1) for x in grid]
        ics = problem.ics_function(grid, p)
        v = collect(ics.v)
        accel(u, v) = begin
            mass = M.predictor_mass(problem.density, p, Float64, 1)[1]
            a = zeros(length(u))
            w = similar(u)
            for j in eachindex(u)
                j1, j2 = max(j - 1, 1), min(j + 1, length(u))
                w[j] = (u[j2] - u[j1]) / ((j2 - j1) * dx)
            end
            du = zeros(length(u))
            dw = zeros(length(u))
            for j in eachindex(u)
                du[j] = derivative_u(problem.density, u[j], v[j], w[j], p)
                dw[j] = derivative_w(problem.density, u[j], v[j], w[j], p)
            end
            for j in 2:(length(u) - 1)
                a[j] = du[j] - (dw[j + 1] - dw[j - 1]) / (2dx)
                a[j] /= mass
            end
            a
        end
        U = zeros(11, length(grid))
        U[1, :] .= u
        a = accel(u, v)
        for k in 1:10
            v .+= dt / 2 .* a
            u .+= dt .* v
            u[1] = problem.bcs_function(k * dt, p).left.u[1]
            u[end] = problem.bcs_function(k * dt, p).right.u[1]
            a = accel(u, v)
            v .+= dt / 2 .* a
            U[k + 1, :] .= u
        end
        at(values, x) = begin
            j = clamp(floor(Int, (x - grid[1]) / dx) + 1, 1, length(grid) - 1)
            λ = (x - grid[j]) / dx
            (1 - λ) * values[j] + λ * values[j + 1]
        end
        for q in eachindex(tb.tnode), r in eachindex(tb.xnode)

            position = tb.tnode[q] * 10
            k = clamp(floor(Int, position) + 1, 1, 10)
            λ = position - (k - 1)
            expect = (1 - λ) * at(U[k, :], tb.xnode[r]) + λ * at(U[k + 1, :], tb.xnode[r])
            @test abs(sample[q, r, 1] - expect) ≤ 1e-5 * max(1, abs(expect))
        end
        @test isfinite(maximum(abs, c.ansatz.top_sample))
    end
    transport = M.LinearTransport.problem(timespan = (0.0, 0.1), timestep = 0.05,
        xstep = 0.05, xspan = (0.0, 1.0))
    ansatz = fixed_ansatz(transport; Sx = 2, St = 2)
    @test_throws r"∂L/∂v" GeometricIntegrator(transport,
        SpaceTimeGalerkin(ansatz, FixedBoundary(); initial_guess = FitTo(Predictor())))
    wave = PROBLEMS[:wave]()
    spline = TensorBSplineAnsatz(degree = 3, ncells = (1, 5), xspan = (0.0, 1.0))
    c = cache(GeometricIntegrator(wave, SpaceTimeGalerkin(fixed_ansatz(wave), FixedBoundary())))
    @test_throws r"Predictor" M.evaluate!(
        zeros(2, 2, 1), Predictor(), c, spline, wave, nothing)
end

# ---- FB13 ----------------------------------------------------------------------------

@testset "FB13 a second bubble fit on one block reads its own dictionary" begin
    problem = PROBLEMS[:wave]()
    ansatz = fixed_ansatz(problem)
    slab = fixed_slab(problem, ansatz)
    c = cache(slab.int)
    tb = ansatz.tabulation
    sample = zeros(Float64, length(tb.tnode), length(tb.xnode), ansatz.D)
    M.evaluate!(sample, InitialCondition(), c, ansatz, problem, current(slab.solstep))
    w = collect(c.ansatz.params)
    M.fit!(w, BubbleGreedy(γ_range = (-6.0, 6.0)), ansatz, sample, c)
    M.fit!(w, BubbleGreedy(γ_range = (-3.0, 3.0)), ansatz, sample, c)
    second = copy(w)
    chosen = c.ansatz.fit.selected[1:ansatz.St]
    @test all(c.ansatz.fit.atoms[i, 3] ≤ 3.0 for i in chosen)
    @test c.ansatz.fit.method.γ_range == (-3.0, 3.0)
    c.ansatz.fit.method = nothing
    w2 = collect(c.ansatz.params)
    M.fit!(w2, BubbleGreedy(γ_range = (-3.0, 3.0)), ansatz, sample, c)
    @test second ≈ w2
end

# ---- FB14 ----------------------------------------------------------------------------

@testset "FB14 no silent upcast" begin
    problem = WF.problem(T = Float32, timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05,
        xspan = (0.0, 1.0))
    for form in (:abg, :tsr), solve in (:newton, :lm)

        a = fixed_ansatz(problem; form = form)
        m = SpaceTimeGalerkin(a, FixedBoundary())
        solvermethod = solve === :lm ? LevenbergMarquardt() : Newton()
        int = GeometricIntegrator(problem, m; solver = solvermethod, max_iterations = 2,
            verbosity = 0)
        result = integrate(problem, m; solver = solvermethod, max_iterations = 2,
            verbosity = 0)
        assert_no_upcast(result.solution.u, Float32)
        c = cache(int)
        assert_no_upcast(c.x, Float32)
        assert_no_upcast(c.coupling.momentum, Float32)
        assert_no_upcast(c.coupling.phi, Float32)
        assert_no_upcast(c.ansatz.params, Float32)
        assert_no_upcast(c.ansatz.momentum, Float32)
        assert_no_upcast(c.ansatz.fit.atoms, Float32)
    end
end
