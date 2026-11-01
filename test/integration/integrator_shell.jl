using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using MultisymplecticIntegrators
using SimpleSolvers
using Test

import GeometricIntegratorsBase: Cache, CacheType, IntegratorCache, cache, integrate_step!,
                                 method, solutionstep, solver

include("../helpers/testsetup.jl")

# A fixture method: it solves nothing, because this package ships no integrator. It exists to
# exercise the shell — the solution the integrator builds, and the trace it stores beside it. Its
# cache holds the trace at `n` points of the space interval, as a method's cache holds it at the
# nodes of a space quadrature. Each step adds one to `u`, so the solution at step `n` is the initial
# data plus `n`, and the step `nan_at` writes a NaN into the state. It solves nothing, so it keeps
# no record of a solve.
struct FixtureMethod{T} <: SpaceTimeMethod
    nodes::Vector{T}
    nan_at::Int
end

function FixtureMethod(xspan; n::Integer = 1, nan_at::Integer = 0)
    FixtureMethod(collect(range(xspan...; length = n)), Int(nan_at))
end

struct FixtureCache{DT} <: IntegratorCache{DT}
    u::Vector{DT}
    v::Vector{DT}
    w::Vector{DT}
    steps::Base.RefValue{Int}
end

function Cache{ST}(problem::LagrangianPDEProblem, method::FixtureMethod; kwargs...) where {ST}
    nodes = length(method.nodes)
    return FixtureCache{ST}([problem.ics.u[1] for _ in 1:nodes],
        [problem.ics.v[1] for _ in 1:nodes],
        [problem.ics.w[1] for _ in 1:nodes],
        Ref(0))
end

CacheType(ST, ::LagrangianPDEProblem, ::FixtureMethod) = FixtureCache{ST}

function MultisymplecticIntegrators.top_trace(cache::FixtureCache)
    (u = cache.u, v = cache.v,
        w = cache.w)
end

function integrate_step!(sol, history, params,
        int::GeometricIntegrator{<:FixtureMethod, <:LagrangianPDEProblem})
    # The fixture counts its steps in the cache, so that a stored trace that aliases the cache
    # repeats the last value instead of holding each step's own.
    c = cache(int)
    c.u .+= 1
    c.steps[] += 1
    sol.u .+= 1
    c.steps[] == method(int).nan_at && (sol.u[1] = NaN)
    return nothing
end

@testset "Integrator shell" begin
    problem = Wave.problem(timespan = (0.0, 0.4), timestep = 0.1, xstep = 0.25)
    method = FixtureMethod((-0.5, 0.0); n = 2)
    result = integrate(problem, method)

    @testset "the named tuple of the shell" begin
        @test keys(result) == (:solution, :traces, :stats)
        # a method whose cache keeps no record of a solve has no stats
        @test solve_stats(cache(GeometricIntegrator(problem, method))) === nothing
        @test result.stats === nothing
        @test result.solution isa GeometricSolution
        @test datatype(result.solution) === Float64
    end

    @testset "the trace of every step, at the quadrature nodes" begin
        @test length(result.traces) == ntime(result.solution)
        @test !isempty(result.traces)
        @test isconcretetype(eltype(result.traces))
        quadrature_nodes = length(method.nodes)
        interior_nodes = length(xgrid(problem))
        @test quadrature_nodes != interior_nodes
        for trace in result.traces, entry in values(trace)

            @test length(entry) == quadrature_nodes
        end
        @test result.traces[1].u != result.traces[2].u
        @test result.traces[2].u != result.traces[end].u
    end

    @testset "the solution carries the problem's state" begin
        @test keys(result.solution.dataser) == (:t, :u, :v, :w)
        @test result.solution.u[0] ≈ problem.ics.u
        for n in 1:ntime(result.solution)
            @test result.solution.u[n] ≈ problem.ics.u .+ n
        end
        @test result.solution.v[ntime(result.solution)] ≈ problem.ics.v
    end

    @testset "the history states hold the initial data" begin
        @test default_extrapolation(method) isa NoExtrapolation
        solstep = solutionstep(GeometricIntegrator(problem, method),
            GeometricSolution(problem)[0])
        for state in history(solstep)
            @test collect(state.u) == problem.ics.u
            @test collect(state.v) == problem.ics.v
            @test collect(state.w) == problem.ics.w
        end
    end

    @testset "a NaN in the state stops the run" begin
        failing = FixtureMethod((-0.5, 0.0); n = 2, nan_at = 2)
        stopped = @test_logs (:warn, r"NaN at timestep n=2") integrate(problem, failing)
        @test length(stopped.traces) == 2
        @test stopped.stats === nothing
        @test isnan(stopped.solution.u[2][1])
        @test stopped.solution.u[1] ≈ problem.ics.u .+ 1
    end

    @testset "a failed nonlinear solve stops the run, $T" for T in (Float64, Float32)
        # The guess is finite on the first slab and NaN on the second, so the second Newton solve
        # meets a non-finite direction and raises `NonlinearSolverException`.
        short = Wave.problem(T = T, timespan = (zero(T), T(0.15)), timestep = T(0.05),
            xstep = T(0.05))
        guess = FitTo(TargetFunction((t, x, p) -> t > T(0.05) ? T(NaN) :
                                                  Wave.exact_u(t, x, p)))
        slab = SpaceTimeGalerkin(
            TensorBSplineAnsatz(degree = 2, ncells = (2, 20), xspan = (zero(T), one(T)), n = 3);
            initial_guess = guess)
        stopped = @test_logs (:warn, r"nonlinear solver failed at timestep n=2") integrate(
            short, slab)
        @test length(stopped.traces) == 1
        @test all(isfinite, stopped.solution.u[1])
        # the stats of a run that stopped early hold one entry per stored trace
        @test all(entries -> length(entries) == 1, values(stopped.stats))
        @test stopped.stats.converged == [true]
    end

    @testset "R1a a step that does not converge, $T" for T in (Float64, Float32)
        # One Newton iteration on the nonlinear sine-Gordon density leaves a residual of about
        # `3e-5` of the initial one, which the relative tolerance `100 eps(T)` does not accept.
        short = SineGordon.problem(
            T = T, timespan = (zero(T), T(0.15)), timestep = T(0.05),
            xstep = T(0.05))
        slab = SpaceTimeGalerkin(TensorBSplineAnsatz(degree = 2, ncells = (2, 20),
            xspan = (zero(T), one(T)), n = 3))
        options = (max_iterations = 1, f_reltol = 100 * eps(T))
        warned = @test_logs (
            :warn, r"^The nonlinear solve did not converge at timestep n=1: ") (
            :warn, r"^The nonlinear solve did not converge at timestep n=2: ") (
            :warn, r"^The nonlinear solve did not converge at timestep n=3: ") integrate(
            short, slab; options...)
        @test warned.stats.converged == [false, false, false]
        @test warned.stats.iterations == [1, 1, 1]
        @test all(r -> r > zero(T), warned.stats.residuals)
        @test eltype(warned.stats.residuals) === T
        @test eltype(warned.stats.initial_residuals) === T
        # `:warn` is the default, and names the step as the default does
        explicit = @test_logs (
            :warn, r"^The nonlinear solve did not converge at timestep n=1: ") (
            :warn, r"^The nonlinear solve did not converge at timestep n=2: ") (
            :warn, r"^The nonlinear solve did not converge at timestep n=3: ") integrate(
            short, slab; on_failure = :warn, options...)
        @test explicit.stats.converged == [false, false, false]
        # `:error` throws at the first step that does not converge, and names it
        @test_throws r"^The nonlinear solve did not converge at timestep n=1: " integrate(
            short, slab; on_failure = :error, options...)
        @test_throws ErrorException integrate(
            GeometricIntegrator(short, slab; options...); on_failure = :error)
        # a run that converges warns of nothing, under either policy
        converged = @test_logs integrate(short, slab; on_failure = :error)
        @test converged.stats.converged == [true, true, true]
        @test all(≥(2), converged.stats.iterations)
    end

    @testset "R1d the stats belong to the run, $T" for T in (Float64, Float32)
        short = SineGordon.problem(
            T = T, timespan = (zero(T), T(0.15)), timestep = T(0.05),
            xstep = T(0.05))
        slab = SpaceTimeGalerkin(TensorBSplineAnsatz(degree = 2, ncells = (2, 20),
            xspan = (zero(T), one(T)), n = 3))
        int = GeometricIntegrator(short, slab)
        first = integrate(int)
        kept = deepcopy(first.stats)
        # the cache's record is emptied and filled again by every run, so the stats hold copies
        c = cache(int)
        for entries in values(first.stats), name in fieldnames(typeof(c))

            @test getfield(c, name) !== entries
        end
        second = integrate(int)
        @test first.stats == kept
        @test second.stats == kept
    end

    @testset "R1b the per-step rule reads the integrator's own tolerances, $T" for T in (
        Float64, Float32)
        # One Newton iteration leaves a residual of `3e-5` to `4.3e-5` of the step's initial one,
        # which only the relative term of the rule accepts: the absolute tolerance is zero and the
        # relative one is `5e-5`. Step 2's residual is `6.7e-5` of step 1's initial residual, so a
        # rule that reads the previous step's initial residual rejects step 2. The default options
        # of the solver reject every step in `Float64`.
        short = SineGordon.problem(
            T = T, timespan = (zero(T), T(0.15)), timestep = T(0.05),
            xstep = T(0.05))
        slab = SpaceTimeGalerkin(TensorBSplineAnsatz(degree = 2, ncells = (2, 20),
            xspan = (zero(T), one(T)), n = 3))
        reltol = T(5e-5)
        int = GeometricIntegrator(short, slab; max_iterations = 1, f_abstol = 0,
            f_reltol = reltol)
        options = SimpleSolvers.config(solver(int))
        @test options.f_abstol == 0
        @test options.f_reltol == reltol
        result = @test_logs integrate(int)
        stats = result.stats
        @test stats.converged == [true, true, true]
        @test all(r -> r > zero(T), stats.residuals)
        @test all(stats.residuals .≤ reltol .* stats.initial_residuals)
        @test stats.residuals[2] > reltol * stats.initial_residuals[1]
        if T === Float64
            plain = GeometricIntegrator(short, slab; max_iterations = 1)
            rejected = @test_logs (:warn,) (:warn,) (:warn,) integrate(plain)
            @test rejected.stats.converged == [false, false, false]
        end
    end

    @testset "R1c an unknown failure policy" begin
        short = Wave.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05)
        slab = SpaceTimeGalerkin(TensorBSplineAnsatz(degree = 2, ncells = (2, 20),
            xspan = (0.0, 1.0), n = 3))
        message = "on_failure must be :warn or :error, got :ignore"
        @test_throws ArgumentError(message) integrate(short, slab; on_failure = :ignore)
        @test_throws ArgumentError(message) integrate(GeometricIntegrator(short, slab);
            on_failure = :ignore)
        @test_throws ArgumentError("on_failure must be :warn or :error, got \"warn\"") integrate(
            short, slab; on_failure = "warn")
    end

    @testset "a run that starts at a time other than zero" begin
        late = Wave.problem(timespan = (1.0, 1.4), timestep = 0.1, xstep = 0.25)
        steps = integrate(late, method)
        @test ntime(steps.solution) == 4
        @test [steps.solution.t[n] for n in 1:4] ≈ [1.1, 1.2, 1.3, 1.4]
        @test length(steps.traces) == 4
        @test steps.traces[1] !== nothing
        @test steps.solution.u[1] ≈ late.ics.u .+ 1
        @test steps.solution.u[0] ≈ late.ics.u
    end

    @testset "no silent upcast" begin
        small = Wave.problem(T = Float32, timespan = (Float32(0), Float32(0.4)),
            timestep = Float32(0.1), xstep = Float32(0.25))
        small_method = FixtureMethod((Float32(-0.5), Float32(0.0)); n = 2)
        small_result = integrate(small, small_method)
        assert_no_upcast(small.ics.u, Float32)
        assert_no_upcast(small_result.solution.u, Float32)
        for trace in small_result.traces, entry in values(trace)

            assert_no_upcast(entry, Float32)
        end
    end

    @testset "a method without a trace stores nothing" begin
        @test top_trace(nothing) === nothing
    end

    @testset "A1i the two method guards" begin
        small = Wave.problem(T = Float32, timespan = (Float32(0), Float32(0.1)),
            timestep = Float32(0.05), xstep = Float32(0.05))
        large = Wave.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05)
        # the ansatz of each problem is of that problem's element type and space interval
        small_ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
            xspan = (0.0f0, 1.0f0), n = 3)
        large_ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
            xspan = (0.0, 1.0), n = 3)
        wide_ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
            xspan = (0.0, 2.0), n = 3)
        @test MultisymplecticIntegrators.check_method(large, SpaceTimeGalerkin(large_ansatz)) ===
              nothing
        # an element type mismatch, in the problem's and in the ansatz's type
        @test_throws ArgumentError("the ansatz element type Float64 does not match the problem's Float32") MultisymplecticIntegrators.check_method(
            small, SpaceTimeGalerkin(large_ansatz))
        @test_throws ArgumentError("the ansatz element type Float32 does not match the problem's Float64") MultisymplecticIntegrators.check_method(
            large, SpaceTimeGalerkin(small_ansatz))
        # a space interval mismatch, and the element type first where both mismatch
        @test_throws ArgumentError("the ansatz xspan (0.0, 2.0) does not match the problem's (0.0, 1.0)") MultisymplecticIntegrators.check_method(
            large, SpaceTimeGalerkin(wide_ansatz))
        @test_throws ArgumentError("the ansatz element type Float32 does not match the problem's Float64") MultisymplecticIntegrators.check_method(
            large, SpaceTimeGalerkin(TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
                xspan = (0.0f0, 2.0f0), n = 3)))
    end

    @testset "A1j the method and its ansatz have separate supertypes" begin
        problem = Wave.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05)
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0), n = 3)
        method = SpaceTimeGalerkin(ansatz)
        @test method isa SpaceTimeMethod
        @test !(method isa SpaceTimeAnsatz)
        @test ansatz isa SpaceTimeAnsatz
        @test !(ansatz isa SpaceTimeMethod)
        @test !(ansatz isa GeometricIntegratorsBase.GeometricMethod)
        # a method in the place of the ansatz names the method
        @test_throws ArgumentError("SpaceTimeGalerkin needs a SpaceTimeAnsatz, got the method SpaceTimeGalerkin") SpaceTimeGalerkin(method)
        # an ansatz in the place of the method names the ansatz and the wrapper
        @test_throws ArgumentError("TensorBSplineAnsatz is an ansatz, not a method; integrate with SpaceTimeGalerkin(ansatz)") integrate(
            problem, ansatz)
        # an ansatz is no `GeometricMethod`, so the integrator refuses it at construction
        @test_throws MethodError GeometricIntegrator(problem, ansatz)
    end
end
