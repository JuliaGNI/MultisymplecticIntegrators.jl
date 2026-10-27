using GeometricBase
using GeometricIntegratorsBase
using MultisymplecticIntegrators
using SimpleSplines
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")

const M = MultisymplecticIntegrators

"""
An ansatz that is not linear in its coefficients: it stands in for the network ansatz of a later
part, whose `Projection()` must be refused.
"""
struct NonlinearFixtureAnsatz <: SpaceTimeAnsatz end

"""
The fit method the fixture ansatz gives an unnamed [`FitTo`](@ref): a stand-in for the network fits
of the later parts, and not a `Projection()`.
"""
struct FixtureFit <: SlabFitMethod end

M.default_fit(::NonlinearFixtureAnsatz) = FixtureFit()

function guess_run(::Type{T}, guess; timestep = T(0.05), xstep = T(0.05),
        problem = Wave.problem(T = T, timespan = (zero(T), T(0.25)), timestep = timestep,
            xstep = xstep)) where {T}
    ansatz = TensorBSplineAnsatz(degree = 2, ncells = (2, 20),
        xspan = (zero(T), one(T)), n = 3)
    method = SpaceTimeGalerkin(ansatz, CoefficientRestriction(); initial_guess = guess)
    int = GeometricIntegrator(problem, method)
    return int, integrate(int)
end

@testset "Initial guess" begin
    @testset "A4a every option converges to the same solution" for T in (Float64, Float32)
        options = [Continuation(), FitTo(InitialCondition()), FitTo(PreviousSlab()),
            FitTo(ExactSolution()),
            FitTo(TargetFunction((t, x, p) -> Wave.exact_u(t, x, p)))]
        solutions = Vector{Vector{T}}()
        initial_residuals = Vector{Vector{T}}()
        for option in options
            int, result = guess_run(T, option)
            c = cache(int)
            @test length(c.iterations) == 5
            @test length(c.initial_residuals) == 5
            @test all(≥(1), c.iterations)
            @test all(r -> isfinite(r) && r ≥ zero(T), c.initial_residuals)
            push!(solutions, result.solution.u[end])
            push!(initial_residuals, copy(c.initial_residuals))
        end
        for solution in solutions[2:end]
            @test maximum(abs.(solution .- solutions[1])) ≤
                  T(100) * newton_tolerance(T, SOLVER_OPTIONS)
        end
        # The residual is linear, so every guess converges in one iteration and only `r₀` tells a
        # target at the right time from a wrong one: the two exact targets start below
        # `Continuation()` at every step, and a target shifted by one step starts above it.
        for exact in initial_residuals[4:5]
            @test all(exact .< initial_residuals[1])
        end
    end

    @testset "A6b the tolerance helper reads the oracle" for T in (Float64, Float32)
        # the oracle's own field, carried to `T` as the same number of `eps(T)`
        @test newton_tolerance(T, SOLVER_OPTIONS) / eps(T) ==
              T(SOLVER_OPTIONS.f_abstol / eps(Float64))
        @test typeof(newton_tolerance(T, SOLVER_OPTIONS)) === T
        # the one-field options the clause names are read as they stand, not as the oracle's value
        @test newton_tolerance(T, (; f_abstol = 16 * eps(Float64))) == T(16) * eps(T)
    end
    # in `Float64`, where the oracle's entry is a whole number of `eps`, the helper returns it
    @test newton_tolerance(Float64, SOLVER_OPTIONS) === SOLVER_OPTIONS.f_abstol

    @testset "A4c the continuation's guess and the extrapolation's value" for T in (Float64, Float32)
        continuation, _ = guess_run(T, Continuation())
        initial, _ = guess_run(T, FitTo(InitialCondition()))
        rc = cache(continuation).initial_residuals
        ri = cache(initial).initial_residuals
        @test length(rc) == 5
        @test length(ri) == 5
        # the carried top row is the better guess from the second step on
        for n in 2:5
            @test rc[n] < ri[n]
        end
        # the extrapolation is the polynomial piece of the top time cell, about its top
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (3, 20),
            xspan = (zero(T), one(T)), n = 3)
        problem = Wave.problem(T = T, timespan = (zero(T), T(0.25)), timestep = T(0.05),
            xstep = T(0.05))
        c = cache(GeometricIntegrator(problem,
            SpaceTimeGalerkin(ansatz, CoefficientRestriction();
                initial_guess = FitTo(PreviousSlab()))))
        field(t̂, x) = (one(T) + t̂ + t̂^2) * x * (one(T) - x)
        c.coeffs[:, :, 1] .= SimpleSplines.l2_projection(quadrature(ansatz),
            p -> field(p[1], p[2]))
        c.steps = ncells(ansatz)[1]
        qt = M.nquadrature(c.tabulation.time)
        qx = M.nquadrature(c.tabulation.space)
        sample = Array{T, 3}(undef, qt, qx, 1)
        M.evaluate!(sample, PreviousSlab(), c, ansatz, problem, nothing)
        xq = space_quadrature_nodes(ansatz)
        scale = T(100) * eps(T) * maximum(abs, sample)
        for q in 1:qt, r in 1:qx

            u = one(T) + c.tabulation.time.node[q]
            @test abs(sample[q, r, 1] - field(u, xq[r])) ≤ scale
        end
    end

    @testset "A4d the three roles fail at construction" begin
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0), n = 3)
        galerkin(guess) = SpaceTimeGalerkin(ansatz, CoefficientRestriction();
            initial_guess = guess)
        targets = (InitialCondition(), PreviousSlab(), ExactSolution(),
            TargetFunction((t, x, p) -> 0.0))
        # the roles are disjoint abstract types
        for target in targets
            @test target isa SlabTarget
            @test !(target isa SlabInitialGuess)
            @test !(target isa SlabFitMethod)
        end
        @test Projection() isa SlabFitMethod
        @test !(Projection() isa SlabInitialGuess)
        @test !(Projection() isa SlabTarget)
        @test Continuation() isa SlabInitialGuess
        @test FitTo(InitialCondition()) isa SlabInitialGuess
        # a target or a fit method is no guess
        for wrong in (targets..., Projection())
            @test_throws TypeError galerkin(wrong)
        end
        # a guess or a fit method is no target
        @test_throws MethodError FitTo(Continuation())
        @test_throws MethodError FitTo(FitTo(InitialCondition()))
        @test_throws MethodError FitTo(Projection())
        # a guess or a target is no fit method
        for wrong in (Continuation(), InitialCondition(), FitTo(InitialCondition()))
            @test_throws TypeError FitTo(InitialCondition(); method = wrong)
        end
        # the valid forms still build
        @test galerkin(Continuation()) isa SpaceTimeGalerkin
        @test galerkin(FitTo(PreviousSlab(); method = Projection())) isa SpaceTimeGalerkin
    end

    @testset "A4e the in-place projection is the allocating one" for T in (Float64, Float32)
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (3, 20),
            xspan = (zero(T), one(T)), n = 3)
        problem = Wave.problem(T = T, timespan = (zero(T), T(0.25)), timestep = T(0.05),
            xstep = T(0.05))
        c = cache(GeometricIntegrator(problem,
            SpaceTimeGalerkin(ansatz, CoefficientRestriction())))
        q = quadrature(ansatz)
        field(p) = (one(T) + p[1]^2) * sin(T(π) * p[2]) + p[1] * p[2]
        sample = SimpleSplines.quadrature_sample(q, field)
        expected = SimpleSplines.l2_projection(q, sample)
        coeffs = similar(expected)
        SimpleSplines.l2_projection!(coeffs, q, sample, c)
        @test maximum(abs, coeffs .- expected) ≤ T(16) * eps(T) * maximum(abs, expected)
    end

    @testset "A4f every guess allocates nothing" for T in (Float64, Float32)
        # `Pkg.test()` runs the floor under `--check-bounds=yes`, which allocates in the loops, so
        # the count is asserted under `--check-bounds=auto` only
        allocations(f::F, a::A, b::B, s::S) where {F, A, B, S} = (
            f(a, b, s); @allocated f(a, b, s))
        run!(c, int, sol) = M.initial_guess!(sol, nothing, nothing, int)
        # the helper sees an allocation, so a zero below is the guess's and not the helper's
        @test allocations((c, int, sol) -> collect(1:10), nothing, nothing, nothing) > 0
        guesses = [Continuation(), FitTo(InitialCondition()), FitTo(PreviousSlab()),
            FitTo(ExactSolution()),
            FitTo(TargetFunction((t, x, p) -> Wave.exact_u(t, x, p)))]
        for guess in guesses
            problem = Wave.problem(
                T = T, timespan = (zero(T), T(0.25)), timestep = T(0.05),
                xstep = T(0.05))
            ansatz = TensorBSplineAnsatz(degree = 2, ncells = (2, 20),
                xspan = (zero(T), one(T)), n = 3)
            int = GeometricIntegrator(problem,
                SpaceTimeGalerkin(ansatz, CoefficientRestriction(); initial_guess = guess))
            c = cache(int)
            sol = (; t = problem.timestep)
            if Base.JLOptions().check_bounds != 1
                # the first slab of a run, and every later one
                @test c.steps == 0
                @test allocations(run!, c, int, sol) == 0
                c.steps = 2
                @test allocations(run!, c, int, sol) == 0
            end
        end
    end

    @testset "A4b the two refusals and the element type of a target" begin
        T = Float64
        plain = Wave.problem(T = T, timespan = (zero(T), T(0.1)), timestep = T(0.05),
            xstep = T(0.05))
        without_exact = LagrangianPDEProblem(plain.density; D = 1,
            ics_function = plain.ics_function, bcs_function = plain.bcs_function,
            timespan = plain.timespan, timestep = plain.timestep, xspan = plain.xspan,
            xstep = plain.xstep, params = plain.params)
        @test without_exact.exact_u === nothing
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
            xspan = (zero(T), one(T)), n = 3)
        exact_message = "FitTo(ExactSolution()) needs a problem with an exact solution; " *
                        "this problem has exact_u = nothing"
        @test_throws ArgumentError(exact_message) GeometricIntegrator(without_exact,
            SpaceTimeGalerkin(ansatz, CoefficientRestriction();
                initial_guess = FitTo(ExactSolution())))
        # the projection's refusal names the reason: an ansatz that is not linear in its
        # coefficients. The fit is named, because the fixture's own default fit method is not a
        # `Projection()`; the fixture also makes the cache refuse its ansatz, so an assertion on the
        # type alone would pass on the cache's own `ArgumentError`.
        @test M.fit_method(FitTo(InitialCondition()), NonlinearFixtureAnsatz()) isa
              FixtureFit
        @test M.fit_method(FitTo(InitialCondition()), ansatz) isa Projection
        @test M.fit_method(FitTo(InitialCondition(); method = Projection()), ansatz) isa
              Projection
        projection_message = "Projection() needs an ansatz that is linear in its coefficients; " *
                             "NonlinearFixtureAnsatz is not"
        nonlinear = SpaceTimeGalerkin(NonlinearFixtureAnsatz(), CoefficientRestriction();
            initial_guess = FitTo(InitialCondition(); method = Projection()))
        @test_throws ArgumentError(projection_message) GeometricIntegrator(plain, nonlinear)
        # the control: the fixture's own default fit is no projection, so the same construction
        # passes its check
        @test M.check_method(plain,
            SpaceTimeGalerkin(NonlinearFixtureAnsatz(), CoefficientRestriction();
                initial_guess = FitTo(InitialCondition()))) === nothing
        # a target that returns another element type gives the problem's element type
        small = Float32
        int, result = guess_run(small,
            FitTo(TargetFunction((t, x, p) -> Float64(Wave.exact_u(Float64(t), Float64(x), p))));
            timestep = small(0.05), xstep = small(0.05))
        @test eltype(cache(int).x) === small
        assert_no_upcast(cache(int).coeffs, small)
        assert_no_upcast(result.solution.u, small)
        # a target is evaluated in physical time: the step that ends at `t = 0.1` starts at
        # `t = 0.05`, so the sample of `f(t, x) = t` is `0.05 + 0.05 t̂`
        c = cache(GeometricIntegrator(plain,
            SpaceTimeGalerkin(ansatz, CoefficientRestriction())))
        qt, qx = M.nquadrature(c.tabulation.time), M.nquadrature(c.tabulation.space)
        sample = Array{T, 3}(undef, qt, qx, 1)
        M.target_values!(sample, (t, x, p) -> t, c.tabulation, ansatz, plain, (t = T(0.1),))
        for q in 1:qt, r in 1:qx

            @test abs(sample[q, r, 1] - (T(0.05) + T(0.05) * c.tabulation.time.node[q])) ≤
                  tolerance(T)
        end
    end
end
