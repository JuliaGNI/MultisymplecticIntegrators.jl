using GeometricBase: parameters
using GeometricIntegratorsBase: GeometricIntegrator, cache
using MultisymplecticIntegrators
using Random: Xoshiro
using SimpleSplines: Dirichlet
using Test

const M = MultisymplecticIntegrators
const SEED = 20261115

include("../helpers/testsetup.jl")
include("../helpers/manufactured.jl")

function check_ladder(runs, values, expected, label; steps)
    @test all(run -> run.converged, runs)
    @test [run.steps for run in runs] == steps
    orders = observed_orders(values)
    @test length(orders) == 2
    @test all(order -> abs(order - expected) ≤ 0.1, orders)
    @info(label, values=Tuple(values), orders=Tuple(orders), expected,
        run_metadata()...)
    return nothing
end

@testset "Manufactured solutions" begin
    @testset "the joint ladder of $name, p = $p, D = $D" for (name, coupling, boundary) in COUPLINGS,
        p in (1, 2), D in (1, 2)
        # `h = 1/nx`, so `c·h/Δx = 0.5` on every rung: the order is `p + 1`.
        ladder = JOINT_LADDER
        runs = [slab_run(coupling, boundary, p, 1 / nx, nx, D) for nx in ladder]
        check_ladder(runs, [run.error for run in runs], p + 1,
            "manufactured joint ladder, $name, p = $p, D = $D, nx = $ladder";
            steps = collect(ladder))
    end

    @testset "the self-difference ladder of $name, p = $p, D = $D" for (name, coupling, boundary) in COUPLINGS,
        p in (1, 2), D in (1, 2)
        # The same ansatz at four steps, `c·h/Δx = 1, 0.5, 0.25, 0.125`: the order in `h` is `2p`.
        steps, nx = SELF_DIFFERENCE_STEPS, SLAB_SELF_DIFFERENCE_NX
        runs = [slab_run(coupling, boundary, p, h, nx, D) for h in steps]
        check_ladder(runs, self_differences(runs, D), 2p,
            "manufactured self-difference ladder, $name, p = $p, D = $D, nx = $nx, h = $steps";
            steps = [10, 20, 40, 80])
    end

    @testset "the box scheme's space ladder, D = $D" for D in (1, 2)
        ladder = BOX_SPACE_LADDER
        runs = [box_run(BOX_SPACE_STEP, nx, D) for nx in ladder]
        check_ladder(runs, [run.error for run in runs], 2,
            "manufactured box space ladder, D = $D, h = 1/640, nx = $ladder";
            steps = fill(640, 3))
    end

    @testset "the box scheme's self-difference ladder, D = $D" for D in (1, 2)
        steps, nx = SELF_DIFFERENCE_STEPS, BOX_SELF_DIFFERENCE_NX
        runs = [box_run(h, nx, D) for h in steps]
        check_ladder(runs, self_differences(runs, D), 2,
            "manufactured box self-difference ladder, D = $D, nx = $nx, h = $steps";
            steps = [10, 20, 40, 80])
    end

    @testset "a zero forcing changes no entry, D = $D" for D in (1, 2)
        zero_forcing = (t, x, params) -> 0
        for (name, coupling, boundary) in COUPLINGS
            forced = slab_run(coupling, boundary, 2, 0.1, 10, D; tend = 0.3,
                forcing = zero_forcing)
            unforced = slab_run(coupling, boundary, 2, 0.1, 10, D; tend = 0.3,
                forcing = nothing)
            @test forced.steps == unforced.steps == 3
            @test all(n -> forced.result.traces[n] == unforced.result.traces[n], 1:3)
            @test all(n -> forced.result.solution.u[n] == unforced.result.solution.u[n], 0:3)
        end
        forced = box_run(0.1, 10, D; tend = 0.3, forcing = zero_forcing)
        unforced = box_run(0.1, 10, D; tend = 0.3, forcing = nothing)
        @test forced.steps == unforced.steps == 3
        @test all(n -> forced.result.traces[n] == unforced.result.traces[n], 1:3)
        @test all(n -> forced.result.solution.u[n] == unforced.result.solution.u[n], 0:3)
    end

    @testset "a Float64 forcing keeps a Float32 run in Float32, D = $D" for D in (1, 2)
        T = Float32
        @test manufactured_forcing(D)(T(0.1), T(0.3), nothing) isa
              (D == 1 ? Float64 : Vector{Float64})
        runs = [
            [slab_run(coupling, boundary, 2, 0.05, 10, D; T = T, tend = 0.1)
             for (_, coupling, boundary) in COUPLINGS]...,
            box_run(0.05, 10, D; T = T, tend = 0.1)]
        for run in runs
            @test run.steps == 2
            @test run.converged
            for field in (:u, :v, :w)
                assert_no_upcast(getproperty(run.result.solution, field), T)
            end
            for trace in run.result.traces
                assert_no_upcast(trace, T)
            end
        end
    end

    @testset "a non-finite forcing ends the run unconverged: $value, D = $D" for value in (NaN, Inf),
        D in (1, 2)
        # The forcing is not guarded: the residual is not finite, so the first step does not
        # converge, and the run returns its record, which is empty. The two weak couplings raise
        # from the linear solve instead (KNOWN_ISSUES.md).
        forcing = (t, x, params) -> value
        problem = manufactured_problem(;
            D = D, timestep = 0.1, nx = 10, tend = 0.3, forcing = forcing)
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 10), xspan = (0.0, 1.0),
            x_boundary = Dirichlet())
        for method in (SpaceTimeGalerkin(ansatz, CoefficientRestriction()), BoxScheme())
            int = GeometricIntegrator(problem, method)
            result = @test_logs (:warn,) match_mode=:any integrate(int)
            @test isempty(result.traces)
            @test isempty(result.stats.converged)
        end
    end

    @testset "an unforced residual takes sol = nothing" begin
        problem = manufactured_problem(; timestep = 0.1, nx = 10, forcing = nothing)
        @test problem.forcing === nothing
        ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 10), xspan = (0.0, 1.0),
            x_boundary = Dirichlet())
        for method in (SpaceTimeGalerkin(ansatz, CoefficientRestriction()), BoxScheme())
            int = GeometricIntegrator(problem, method)
            x = rand(Xoshiro(SEED), length(cache(int).x))
            b = similar(x)
            @test M.residual!(b, x, nothing, parameters(problem), int) === nothing
            @test all(isfinite, b)
        end
    end
end
