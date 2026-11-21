# The benchmark matrix: a run that throws, or whose child process ends without its archive, is a
# result. One of the files of `test/integration/benchmark*.jl`, which share
# `test/helpers/benchmark_setup.jl`.
using JLD2
using MultisymplecticIntegrators
using Test

include(joinpath(@__DIR__, "..", "helpers", "benchmark_setup.jl"))

@testset "Benchmark matrix: failures" begin
    @testset "a run that throws is a result" begin
        # A row whose boundary data raises at the third step of three: the archive holds the type,
        # the message and the step, and the stats of the steps before it.
        T = Float64
        base = Wave.problem(timespan = (zero(T), T(3) / 20), timestep = T(1) / 20,
            xspan = (zero(T), one(T)), xstep = T(1) / 20)
        failing = (t, params) -> t >= T(3) / 20 ?
                                 throw(DomainError(t, "the boundary data end")) :
                                 base.bcs_function(t, params)
        broken = LagrangianPDEProblem(base.density; D = base.D,
            ics_function = base.ics_function, bcs_function = failing,
            timespan = base.timespan, timestep = base.timestep, xspan = base.xspan,
            xstep = base.xstep, params = base.params)
        entry = row("throwing-row", :box, :wave, :matrix, (timestep = T(1) / 20,))
        data = benchmark_run(broken, BoxScheme(), entry)
        @test data["status"] == :failed
        @test data["exception"] == "DomainError"
        @test !isempty(data["message"])
        @test data["failed_at_step"] == 3
        @test data["steps"] == 2
        @test length(data["converged"]) == 2
        @test length(data["residuals"]) == 2
        @test length(data["wall_per_step"]) == 1
        # A row that throws at its first step archives empty stats and an empty time vector.
        early = (t, params) -> throw(DomainError(t, "the boundary data end at once"))
        first_failure = LagrangianPDEProblem(base.density; D = base.D,
            ics_function = base.ics_function, bcs_function = early,
            timespan = base.timespan, timestep = base.timestep, xspan = base.xspan,
            xstep = base.xstep, params = base.params)
        empty = benchmark_run(first_failure, BoxScheme(), entry)
        @test empty["failed_at_step"] == 1
        @test empty["steps"] == 0
        @test isempty(empty["residuals"])
        @test isempty(empty["wall_per_step"])
        # A row whose method the problem does not support throws before its first step: the box
        # scheme refuses a density outside its class when the integrator is built, so the archive
        # holds no step and records the failure at step `0`.
        transport = LinearTransport.problem(timespan = (zero(T), T(2) / 10),
            timestep = T(1) / 10, xspan = (zero(T), one(T)), xstep = T(1) / 20)
        construction = row("construction-row", :box, :linear_transport, :matrix,
            (timestep = T(1) / 10,))
        unbuilt = benchmark_run(transport, BoxScheme(), construction)
        @test unbuilt["status"] == :failed
        @test unbuilt["exception"] == "ArgumentError"
        @test unbuilt["failed_at_step"] == 0
        @test unbuilt["steps"] == 0
        # A slab method whose coupling the ansatz does not support is refused when its integrator is
        # built, and its cost count throws a `MethodError` as well: the row is archived as the
        # construction's failure at step `0`, and the count never runs.
        spline = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 10), xspan = (zero(T), one(T)),
            n = 3)
        uncounted = benchmark_run(base, SpaceTimeGalerkin(spline, FixedBoundary()),
            row("uncounted-row", :spline_restriction,
                :wave, :matrix, (timestep = T(1) / 20,)))
        @test uncounted["status"] == :failed
        @test uncounted["exception"] == "ArgumentError"
        @test_throws MethodError row_cost(SpaceTimeGalerkin(spline, FixedBoundary()), base)
        @test uncounted["failed_at_step"] == 0
        @test uncounted["steps"] == 0
        @test isempty(uncounted["cost"])
        # A child process that ends without its archive is archived by the parent, with its exit
        # code: the row below is one the child's registry does not hold.
        directory = mktempdir()
        RUNS_DIR[] = directory
        unknown = row("unknown-row", :box, :wave, :matrix, (timestep = T(1) / 20,))
        @test run_row((family = :box, entry = unknown, benchmark = box_benchmark))
        on_disk = load(joinpath(directory, "unknown-row.jld2"))
        @test on_disk["status"] == :failed
        @test occursin("code", on_disk["message"])
        @test on_disk["steps"] == 0
        @test benchmark_archive("unknown-row", on_disk) === on_disk
        # The driver's own selection over a failed row: the row no child's registry holds is
        # archived as failed and the row beside it is archived, so the driver returns 0 for a
        # selection that holds a failure.
        selection_dir = mktempdir()
        RUNS_DIR[] = selection_dir
        ghost = row("ghost-row", :box, :wave, :matrix,
            (timestep = T(1) / 10, xstep = T(1) / 10))
        register_family!(:ghost, (ghost,); benchmark = box_benchmark)
        @test main(["--runs-dir", selection_dir, "ghost-row", "wave-box-h0.1"]) == 0
        ghosted = load(joinpath(selection_dir, "ghost-row.jld2"))
        @test ghosted["status"] == :failed
        @test occursin("code", ghosted["message"])
        @test ghosted["steps"] == 0
        @test load(joinpath(selection_dir, "wave-box-h0.1.jld2"))["status"] == :ok
        # The child's own exit code reaches the archive. A row whose benchmark ends its process
        # cannot be registered in the tree's own registry, so the driver reads the row from a copy
        # of the scripts, and its child ends on code 3 — which no guess of the parent's makes.
        copy = mktempdir()
        cp(joinpath(@__DIR__, "..", "..", "scripts"), joinpath(copy, "scripts"))
        mkpath(joinpath(copy, "test"))
        cp(joinpath(@__DIR__, "..", "helpers"), joinpath(copy, "test", "helpers"))
        write(joinpath(copy, "scripts", "zz_exiting_model.jl"), """
            const EXITING = ((stem = "exits-three", label = "exits", method = :exiting,
                problem = :wave, problem_label = "Wave", tag = :matrix,
                resolution = (timestep = $(T(1) / 20),), parameters = (k = 1,)),)
            register_family!(:exiting, EXITING; benchmark = entry -> exit(3))
            """)
        Base.include(ExitingRow, joinpath(copy, "scripts", "run_benchmark.jl"))
        copied_runs = joinpath(copy, "runs")
        mkpath(copied_runs)
        @test Base.invokelatest(ExitingRow.main, [
            "--runs-dir", copied_runs, "exits-three"]) == 0
        exited = load(joinpath(copied_runs, "exits-three.jld2"))
        @test exited["status"] == :failed
        @test occursin("code 3", exited["message"])
        @test exited["steps"] == 0
        # A child the operating system kills is archived as a signal rather than as a code, and the
        # driver still returns 0. Windows has no signal to raise, so the arm is Unix's.
        if !Sys.iswindows()
            write(
                joinpath(copy, "scripts", "zz_signalled_model.jl"), """
              const SIGNALLED = ((stem = "killed-nine", label = "killed", method = :signalled,
                  problem = :wave, problem_label = "Wave", tag = :matrix,
                  resolution = (timestep = $(T(1) / 20),), parameters = (k = 1,)),)
              register_family!(:signalled, SIGNALLED;
                  benchmark = entry -> ccall(:raise, Cint, (Cint,), 9))
              """)
            Base.include(SignalledRow, joinpath(copy, "scripts", "run_benchmark.jl"))
            @test Base.invokelatest(SignalledRow.main, [
                "--runs-dir", copied_runs, "killed-nine"]) == 0
            killed = load(joinpath(copied_runs, "killed-nine.jld2"))
            @test killed["status"] == :failed
            @test occursin("killed by signal 9", killed["message"])
            @test killed["steps"] == 0
        end
    end
end
