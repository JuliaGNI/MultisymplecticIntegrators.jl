using GeometricBase: ntime, timesteps
using GeometricIntegratorsBase: GeometricIntegrator, solver
using JLD2
using LinearAlgebra: BLAS
using MultisymplecticIntegrators
using MultisymplecticIntegrators: space_quadrature_nodes, space_quadrature_weights
using Printf: @sprintf
using Random: Xoshiro
using SimpleSolvers: SimpleSolvers, config
using SimpleSplines: Dirichlet
using Test

# The scripts are no part of the package, so the whole harness is tested through its files. The
# reference records are included before the runner, which loads them itself only when its caller has
# not, and the driver then includes the archive layer, the registry, the matrix's runner and every
# model file in the order it includes them itself.
include(joinpath(@__DIR__, "..", "helpers", "reference_values.jl"))
include(joinpath(@__DIR__, "..", "..", "scripts", "run_benchmark.jl"))
include(joinpath(@__DIR__, "..", "helpers", "problem_generator.jl"))

# The module a copy of the driver is included into, for the one row whose child process ends on an
# exit code the tree's own registry cannot produce: the parent reads the child's code, so the row
# that ends on a code of its own needs a registry, and therefore a copy, of its own.
module ExitingRow end

# The same, for the row whose child the operating system kills: the parent reads the signal, and the
# arm does not exist on Windows.
module SignalledRow end

# The display names of the problems a test's rows name, which a family's own rows carry and a table
# is keyed by.
const PROBLEM_LABELS = Dict(:wave => "Wave", :vector_wave => "VectorWave",
    :sine_gordon => "SineGordon", :linear_transport => "LinearTransport",
    :generated => "generated")

# One row's metadata, which a family's rows carry and a test builds for a run of its own.
function row(stem, method, problem, tag, resolution; parameters = (k = 1,), kwargs...)
    merge(
        (stem = stem, label = stem, method = method, problem = problem,
            problem_label = get(PROBLEM_LABELS, problem, string(problem)),
            tag = tag, resolution = resolution, parameters = parameters),
        (; kwargs...))
end

# A problem whose density is the same but whose runs have no exact solution: the matrix measures
# such a run against the finest run of its method instead.
function unsolved_problem(base, xstep; timestep = base.timestep)
    return LagrangianPDEProblem(base.density; D = base.D, ics_function = base.ics_function,
        bcs_function = base.bcs_function, timespan = base.timespan,
        timestep, xspan = base.xspan, xstep, params = base.params)
end

# The largest absolute difference of two archived runs' fields at the times the coarse run stored,
# on the nodes their grids share — computed here, from the archives, and not by the code under
# test. An archive's `"error"` holds one entry per stored step, so the comparison is per step too.
# `by_index` compares the two runs at the same step index instead, the reading the archive must not
# make once the resolutions refine the time step as well.
function archived_difference(coarse, fine; by_index = false)
    shared = [searchsortedfirst(fine["x"], node) for node in coarse["x"]]
    rows = by_index ? collect(1:(coarse["steps"] + 1)) :
           [searchsortedfirst(fine["t"], time) for time in coarse["t"]]
    return [maximum(abs(coarse["u"][d][n + 1, j] - fine["u"][d][rows[n + 1], k])
            for d in eachindex(coarse["u"]), (j, k) in enumerate(shared))
            for n in 1:coarse["steps"]]
end

@testset "Benchmark matrix" begin
    @testset "a selection by method, problem or tag" begin
        # Registered in reverse stem order, so that the stem order a selection returns is not the
        # order the registry holds its rows in.
        entries = (
            # `:tag_one` is a prefix of this tag, so a substring match on `:tag_one` takes this row
            # with the rows that carry that tag, and a substring match on this tag takes them all.
            row("sel-e", :method_two, :problem_one, :tag_one_more, (k = 5,)),
            row("sel-d", :method_two, :problem_two, :tag_two, (k = 4,)),
            row("sel-c", :method_two, :problem_one, :tag_one, (k = 3,)),
            row("sel-b", :method_one, :problem_two, :tag_two, (k = 2,)),
            row("sel-a", :method_one, :problem_one, :tag_one, (k = 1,)))
        register_family!(:selection, entries; benchmark = _ -> Dict{String, Any}())
        stems(selected) = [row.entry.stem for row in selected]
        @test stems(select_rows(method = :method_one)) == ["sel-a", "sel-b"]
        @test stems(select_rows(problem = :problem_one)) == ["sel-a", "sel-c", "sel-e"]
        @test stems(select_rows(tag = :tag_one)) == ["sel-a", "sel-c"]
        @test stems(select_rows(tag = :tag_one_more)) == ["sel-e"]
        @test stems(select_rows(method = :method_two, tag = :tag_one)) == ["sel-c"]
        @test !isempty(select_rows(tag = :tag_one))
        @test_throws ArgumentError select_rows(method = :no_such_method)
        thrown = try
            select_rows(tag = :no_such_tag)
            nothing
        catch exception
            exception
        end
        @test thrown isa ArgumentError
        @test occursin("no_such_tag", sprint(showerror, thrown))
        # The refusal comes before a row runs, so a mistyped tag leaves no archive behind.
        directory = mktempdir()
        RUNS_DIR[] = directory
        @test_throws ArgumentError main(["--tag", "no_such_tag", "--runs-dir", directory])
        @test isempty(filter(file -> endswith(file, ".jld2"), readdir(directory)))
        # A family that registers neither a run nor a benchmark names itself in the refusal, and a
        # row of the matrix that names no tag is refused by its stem and the field it lacks.
        nameless = try
            register_family!(:nameless, entries)
            nothing
        catch exception
            exception
        end
        @test nameless isa ArgumentError
        @test occursin(
            Regex("^ArgumentError: the family `nameless` registers neither a `run` " *
                  "nor a `benchmark`; `run_family\\.jl` and `run_benchmark\\.jl` would " *
                  "both skip it\$"),
            sprint(showerror, nameless))
        untagged = (stem = "sel-f", label = "sel-f", method = :method_one,
            problem = :problem_one, resolution = (k = 6,))
        tagless = try
            register_family!(:tagless, (untagged,);
                benchmark = _ -> Dict{String, Any}())
            nothing
        catch exception
            exception
        end
        @test tagless isa ArgumentError
        @test occursin(
            Regex("^ArgumentError: the row `sel-f` of the family `tagless` has no " *
                  "`tag`; a row of the matrix names method, problem, tag and " *
                  "resolution\$"),
            sprint(showerror, tagless))
    end

    @testset "a run archives R's per-step stats" begin
        # R1a's non-converging case, as `test/integration/integrator_shell.jl` builds it: the sine-
        # Gordon equation, whose density is not linear, with one Newton iteration allowed.
        T = Float64
        problem = SineGordon.problem(
            timespan = (zero(T), T(15) / 100), timestep = T(5) / 100,
            xspan = (zero(T), one(T)), xstep = T(5) / 100)
        ansatz = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 20), xspan = (zero(T), one(T)),
            n = 3)
        method = SpaceTimeGalerkin(ansatz, CoefficientRestriction())
        entry = row("stats-row", :spline_restriction, :sine_gordon,
            :matrix, (timestep = T(5) / 100,);
            solver = (max_iterations = 1, f_reltol = 100 * eps(T)))
        data = benchmark_run(problem, method, entry)
        steps = data["steps"]
        @test steps == ntime(problem)
        for key in ("converged", "residuals", "initial_residuals", "iterations", "error",
            "energy_drift", "cost")
            @test length(data[key]) == steps
        end
        @test any(!, data["converged"])
        @test !all(isequal(first(data["residuals"])), data["residuals"])
        @test data["initial_residuals"] != data["residuals"]
        @test length(unique(data["residuals"])) > 1
    end

    @testset "a problem without an exact solution is measured against its finest run" begin
        # One generated problem of the property suite, which passes no `solutions`, at three
        # resolutions that refine the time step and the space step together, as the box scheme's
        # rows do: a coarse run's nodes are the finer run's, and so are its times.
        entry = first(filter(held -> held.problem.D == 1,
            generate_problems(Float64; rng = Xoshiro(SEED))))
        base = entry.problem
        @test base.solutions === nothing
        span = base.xspan
        cells = (12, 24, 48)
        factors = (1, 2, 4)
        xsteps = [(span[2] - span[1]) / m for m in cells]
        problems = [unsolved_problem(base, xsteps[i]; timestep = base.timestep /
                                                                 factors[i])
                    for i in eachindex(cells)]
        entries = [row("generated-box-$(m)", :box, :generated, :matrix,
                       (timestep = base.timestep / f, xstep = x))
                   for (m, f, x) in zip(cells, factors, xsteps)]
        runs = [benchmark_run(problems[i], BoxScheme(), entries[i])
                for i in eachindex(cells)]
        # A second method on the same problem, at two resolutions of its own: a run's reference is
        # the finest run of its own method, whose stem the other method's rows never carry.
        spline = SpaceTimeGalerkin(
            TensorBSplineAnsatz(
                degree = entry.degree, ncells = (entry.ncells[1], cells[1]),
                xspan = span, n = entry.n, x_boundary = Dirichlet()),
            CoefficientRestriction())
        spline_problems = [unsolved_problem(base, xsteps[i]) for i in (1, 2)]
        spline_entries = [row("generated-spline-$(cells[i])",
                              :spline_restriction, :generated,
                              :matrix, (timestep = base.timestep, xstep = xsteps[i]))
                          for i in (1, 2)]
        spline_runs = [benchmark_run(spline_problems[i], spline, spline_entries[i])
                       for i in (1, 2)]
        append!(runs, spline_runs)
        append!(entries, spline_entries)
        @test all(archive -> archive["error_kind"] == "reference", runs)
        for (archive, held) in zip(runs, entries)
            archive["stem"] = held.stem
        end
        attach_references!(runs)
        finest = runs[3]
        @test finest["reference_stem"] == "generated-box-48"
        @test all(isnan, finest["error"])
        spline_finest = runs[5]
        @test spline_finest["reference_stem"] == "generated-spline-24"
        for coarse in runs[1:2]
            @test coarse["reference_stem"] == finest["stem"]
            # The comparison is at the same TIME: the coarse run's step `n` is at `t₀ + n·h`, which
            # the finer run reaches four steps later, and the two readings are different numbers.
            @test coarse["error"] ≈ archived_difference(coarse, finest)
            @test !isapprox(coarse["error"], archived_difference(coarse, finest; by_index = true))
        end
        @test runs[1]["error"][1] > 0
        @test runs[4]["reference_stem"] == "generated-spline-24"
        @test runs[4]["error"] ≈ archived_difference(runs[4], spline_finest)
        # A time that the two grids compute a unit in the last place apart is the same time: the
        # nearer of the two sorted neighbours is the row the coarse run's step sits on.
        nudged = copy(finest)
        nudged["t"] = copy(finest["t"])
        nudged["t"][5] = prevfloat(nudged["t"][5])
        @test reference_error(runs[1], nudged) ≈ runs[1]["error"]
        # A time of the coarse run that the finer run does not hold is refused rather than matched
        # to a neighbouring step: in the middle of the finer grid, past its end, and where the finer
        # run holds no time at all.
        shifted = copy(finest)
        shifted["t"] = finest["t"] .+ (finest["t"][2] - finest["t"][1]) / 2
        truncated = copy(finest)
        truncated["t"] = finest["t"][1:(end - 2)]
        absent = copy(finest)
        absent["t"] = Float64[]
        for other in (shifted, truncated, absent)
            @test_throws ArgumentError reference_error(runs[1], other)
        end
        # A row whose integrator the package refused before its first step is archived with no time
        # and no field row at all, so a pair of such rows has no step to compare and its difference
        # is empty rather than an index on a time vector that holds nothing.
        blank = [failed_archive(entries[i], "the integrator was refused") for i in (1, 2)]
        for (archive, held) in zip(blank, entries[1:2])
            archive["stem"] = "blank-$(held.stem)"
            archive["error_kind"] = "reference"
        end
        @test all(archive -> isempty(archive["t"]), blank)
        @test isempty(reference_error(blank[2], blank[1]))
        # Two runs on one output grid whose time steps differ, as the spline rows refine the time
        # step and the ansatz on the family's fixed grid: the grids tie, so the finest run is the
        # one with the smaller time step, and the coarser one arrives first here.
        tied_entries = [row("tied-box-$(f)", :box, :generated, :matrix,
                            (timestep = base.timestep / f, xstep = xsteps[1]))
                        for f in (1, 2)]
        tied = [benchmark_run(
                    unsolved_problem(base, xsteps[1]; timestep = base.timestep / f),
                    BoxScheme(), held)
                for (f, held) in zip((1, 2), tied_entries)]
        for (archive, held) in zip(tied, tied_entries)
            archive["stem"] = held.stem
        end
        attach_references!(tied)
        @test all(archive -> archive["reference_stem"] == "tied-box-2", tied)
        @test tied[1]["error"] ≈ archived_difference(tied[1], tied[2])
    end

    @testset "the drift and the charge a run archives" begin
        # The four example problems, run by a method that integrates all four, so that the charge of
        # the two-component field and the absence of a charge everywhere else are both the run's own.
        T = Float64
        problem(name) = name === :wave ?
                        Wave.problem(timespan = (zero(T), T(1) / 5),
            timestep = T(1) / 20, xspan = (zero(T), one(T)), xstep = T(1) / 20) :
                        name === :vector_wave ?
                        VectorWave.problem(timespan = (zero(T), T(1) / 5),
            timestep = T(1) / 20, xspan = (zero(T), one(T)), xstep = T(1) / 20) :
                        name === :sine_gordon ?
                        SineGordon.problem(timespan = (zero(T), T(1) / 5),
            timestep = T(1) / 20, xspan = (zero(T), one(T)), xstep = T(1) / 20) :
                        LinearTransport.problem(
            timespan = (zero(T), T(1) / 5), timestep = T(1) / 20,
            xspan = (zero(T), one(T)), xstep = T(1) / 20)
        ansatz = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 10), xspan = (zero(T), one(T)),
            n = 3)
        method = SpaceTimeGalerkin(ansatz, CoefficientRestriction())
        archives = Dict{Symbol, Dict{String, Any}}()
        for name in (:wave, :vector_wave, :sine_gordon, :linear_transport)
            entry = row("charge-$(name)", :spline_restriction,
                name, :matrix, (timestep = T(1) / 20,))
            archives[name] = benchmark_run(problem(name), method, entry)
        end
        vector = archives[:vector_wave]
        @test vector["charge"] isa AbstractVector
        @test length(vector["charge"]) == vector["steps"]
        # The charge of the first stored step, from the exact data on the run's own nodes and
        # weights, is the one the archive holds.
        weights = space_quadrature_weights(ansatz)
        # The trace is stored at the ansatz's space quadrature nodes, so the closed form of the
        # charge is read at those nodes and not on the output grid.
        nodes = collect(space_quadrature_nodes(ansatz))
        nq = length(nodes)
        balanced = problem(:vector_wave)
        exact_u = balanced.solutions.u(vector["t"][2], nodes, balanced.params)
        exact_v = balanced.solutions.v(vector["t"][2], nodes, balanced.params)
        closed = sum(weights[j] *
                     (exact_u[j] * exact_v[nq + j] - exact_u[nq + j] * exact_v[j])
        for j in 1:nq)
        @test isapprox(first(vector["charge"]), closed; rtol = 1.0e-3)
        for name in (:wave, :sine_gordon, :linear_transport)
            # A `nothing` charge has to come from the density's symmetry and not from a run that
            # took no step: the transport row is the one whose density has no rotation symmetry,
            # so its archive must hold real steps for the assertion to mean anything.
            @test archives[name]["status"] !== :failed
            @test archives[name]["steps"] > 0
            @test archives[name]["charge"] === nothing
        end
        @test all(isfinite, archives[:vector_wave]["energy_drift"])
        @test first(archives[:vector_wave]["energy_drift"]) == 0

        # The box scheme's energy is the midpoint rule over its boxes and not the trapezoidal rule
        # at their nodes: `H` at the mean of each box's two nodes, `Σ Δx · H(½(zⱼ + zⱼ₊₁))`. The
        # run's first level is the problem's own initial data, and on it the trapezoidal reading is
        # the closed form's energy at `t₀` to round-off, while the midpoint reading is `6.2e-3`
        # relative below it: the assertion is of the rule and not of the closed form.
        wave = problem(:wave)
        boxed = integrate(wave, BoxScheme())
        initial = boxed.solution[0]
        grid = collect(xgrid(wave))
        t₀ = boxed.solution.t[0]
        c = wave.params.c
        # The wave density's Hamiltonian is `½v² + ½(c w)²`, written here from the density's closed
        # form: the oracle is never the code under test, and `energy_density` is that code.
        hamiltonian(u, v, w) = (v^2 + (c * w)^2) / 2
        midpoint(z, j) = (z[j] + z[j + 1]) / 2
        exact = [wave.solutions.u(t₀, grid, wave.params),
            wave.solutions.v(t₀, grid, wave.params), wave.solutions.w(t₀, grid, wave.params)]
        closed = sum((grid[j + 1] - grid[j]) *
                     hamiltonian(midpoint(exact[1], j), midpoint(exact[2], j),
                         midpoint(exact[3], j)) for j in 1:(length(grid) - 1))
        energies = MultisymplecticIntegrators.box_energies(wave, [initial])
        @test isapprox(first(energies), closed; rtol = 8 * eps(T))
        drift = energy_error(wave, BoxScheme(), boxed.traces)
        @test length(drift) == length(boxed.traces)
        @test first(drift) == 0
        @test all(isfinite, drift)
    end

    @testset "the timing record" begin
        # The matrix's numbers are its own, from a child process that records what they mean; no
        # wall time is asserted, and the schema and the metadata are.
        directory = mktempdir()
        @test main(["--tag", "smoke", "--runs-dir", directory]) == 0
        files = filter(file -> endswith(file, ".jld2"), readdir(directory))
        @test length(files) >= 2
        for file in files
            data = load(joinpath(directory, file))
            steps = data["steps"]
            @test length(data["wall_per_step"]) == max(steps - 1, 0)
            @test all(time -> isfinite(time) && time > 0, data["wall_per_step"])
            @test data["wall_time"] ≈ sum(data["wall_per_step"]; init = 0.0)
            @test length(data["load"]) == 3 &&
                  all(value -> isfinite(value) && value >= 0, data["load"])
            @test data["threads"] >= 1
            @test data["blas_threads"] >= 1
            @test data["julia"] isa VersionNumber
            @test data["check_bounds"] isa Integer
        end
    end

    @testset "each run in its own process" begin
        directory = mktempdir()
        @test main(["--tag", "smoke", "--runs-dir", directory]) == 0
        files = filter(file -> endswith(file, ".jld2"), readdir(directory))
        pids = [load(joinpath(directory, file))["pid"] for file in files]
        @test length(pids) >= 2
        @test length(unique(pids)) == length(pids)
        @test !(getpid() in pids)
    end

    @testset "the method's parameters" begin
        # Two rows of one method that differ in one parameter each archive their own value, and not
        # the method's defaults.
        T = Float64
        problem = Wave.problem(timespan = (zero(T), T(1) / 10), timestep = T(1) / 20,
            xspan = (zero(T), one(T)), xstep = T(1) / 20)
        ansatz(degree) = TensorBSplineAnsatz(degree = degree, ncells = (2, 10),
            xspan = (zero(T), one(T)), n = 3)
        first_entry = row(
            "parameter-2", :spline_restriction, :wave, :matrix, (timestep = T(1) / 20,);
            parameters = (degree = 2, ncells = (2, 10), n = 3))
        second_entry = row(
            "parameter-3", :spline_restriction, :wave, :matrix, (timestep = T(1) / 20,);
            parameters = (degree = 3, ncells = (2, 10), n = 3))
        first_run = benchmark_run(
            problem, SpaceTimeGalerkin(ansatz(2), CoefficientRestriction()),
            first_entry)
        second_run = benchmark_run(
            problem, SpaceTimeGalerkin(ansatz(3), CoefficientRestriction()),
            second_entry)
        @test first_run["parameters"].degree == 2
        @test second_run["parameters"].degree == 3
        @test first_run["parameters"] != second_run["parameters"]
    end

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

    @testset "a run that does not converge is a result" begin
        # The spline family's sine-Gordon row on R1a's case: the steps' solve records a step that
        # did not converge, and the run is not a failure for it.
        T = Float64
        problem = SineGordon.problem(
            timespan = (zero(T), T(15) / 100), timestep = T(5) / 100,
            xspan = (zero(T), one(T)), xstep = T(5) / 100)
        ansatz = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 20), xspan = (zero(T), one(T)),
            n = 3)
        method = SpaceTimeGalerkin(ansatz, CoefficientRestriction())
        entry = row("unconverged-row", :spline_restriction, :sine_gordon, :matrix,
            (timestep = T(5) / 100,); solver = (
                max_iterations = 1, f_reltol = 100 * eps(T)))
        data = benchmark_run(problem, method, entry)
        @test data["status"] == :unconverged
        @test data["steps"] == ntime(problem)
        @test any(!, data["converged"])
        @test !isempty(data["residuals"])
        # The table's row for it shows the status.
        tables = benchmark_tables([merge(data, Dict("stem" => "unconverged-row"))])
        table = tables["SineGordon"]
        line = only(filter(row -> occursin("unconverged-row", row), split(table, '\n')))
        @test occursin("unconverged", line)
    end

    @testset "one table per problem, with the historical rows" begin
        directory = mktempdir()
        @test main(["--tag", "smoke", "--runs-dir", directory]) == 0
        runs = [load(joinpath(directory, file))
                for file in filter(file -> endswith(file, ".jld2"), readdir(directory))]
        tables = benchmark_tables(runs)
        @test haskey(tables, "Wave") && haskey(tables, "SineGordon")
        page = join(values(tables), "\n")
        for archive in runs
            @test length(collect(eachmatch(Regex(archive["stem"]), page))) == 1
        end
        wave = split(tables["Wave"], '\n')
        for name in ("WAVE_NN_FULLY", "WAVE_NN_PARTIALLY", "WAVE_NN_SYMBOLIC_FULLY",
            "WAVE_ELM_LSGD", "WAVE_TRIALNN_OGA2D")
            line = only(filter(text -> occursin(name, text), wave))
            @test occursin("historical", line)
            @test occursin("unconverged", line)
        end
        sine = split(tables["SineGordon"], '\n')
        for name in ("SINE_GORDON_NN_FULLY", "SINE_GORDON_ELM_LSGD", "SINE_GORDON_TRIALNN_OGA2D")
            line = only(filter(text -> occursin(name, text), sine))
            @test occursin("historical", line) && occursin("unconverged", line)
        end
        # A record's own series is its energy error, and the table's error column holds the
        # solution error: the record's value stands in the energy-drift column, where a live row's
        # drift stands, and its error cell is a dash.
        for (problem, name) in HISTORICAL
            lines = split(tables[problem], '\n')
            header = strip.(split(strip(lines[3], '|'), '|'))
            cells = strip.(split(
                strip(only(filter(text -> occursin(string(name), text), lines)), '|'), '|'))
            @test cells[findfirst(==("error"), header)] == "—"
            @test cells[findfirst(==("energy drift"), header)] ==
                  @sprintf("%.3e", last(getfield(@__MODULE__, name).energy_error))
        end
        @test !any(text -> occursin("WAVE_SINDY", text), wave)
        @test !any(text -> occursin("WAVE_TRIALNN_OGA2D", text), sine)
        @test !any(text -> occursin("SINE_GORDON_NN_FULLY", text), wave)
    end

    @testset "the rows of the merged families" begin
        # The registry's rows, read as the driver reads them: the spline family's 36 and the box
        # scheme's, each of which builds its method and its problem without running.
        rows = select_rows()
        spline = [row.entry
                  for row in rows
                  if row.entry.method in
                     (:spline_restriction, :spline_initial, :spline_initial_boundary)]
        @test length(spline) == 36
        stems = [entry.stem for entry in spline]
        @test length(unique(stems)) == 36
        for method in (:spline_restriction, :spline_initial, :spline_initial_boundary),
            problem in (:wave, :vector_wave, :sine_gordon, :linear_transport)

            group = [entry
                     for entry in spline
                     if entry.method === method && entry.problem === problem]
            @test length(group) == 3
            @test length(unique(entry.resolution for entry in group)) == 3
        end
        box = [row.entry for row in rows if row.entry.method === :box]
        @test length(box) >= 3
        for problem in (:wave, :vector_wave, :sine_gordon)
            @test any(entry -> entry.problem === problem, box)
        end
        for entry in vcat(spline, box)
            @test entry.resolution isa NamedTuple && !isempty(entry.resolution)
            problem = entry.method === :box ? box_problem(entry) :
                      spline_matrix_problem(entry)
            method = entry.method === :box ? box_method(entry) : spline_matrix_method(entry)
            @test method isa SpaceTimeMethod
            @test ntime(problem) >= 1
            # The row's cache is built and no step runs: a row whose ansatz and coupling disagree
            # is a row the driver cannot run, and a count of rows is no evidence that it can.
            @test GeometricIntegrator(problem, method) isa GeometricIntegrator
            if entry.problem === :linear_transport
                @test problem.D == 2
            end
        end
        # The rows of the network, symbolic, ELM, TrialNN and fixed-boundary families: each family
        # registers a `benchmark` and no `run`, holds every method on every problem it takes at three
        # resolutions, and each row builds its integrator with its own solver keywords, without a
        # step. A row is never `:smoke`, so the gate's two smoke testsets run none of them.
        problem_keys = Dict(:wave => keys(Wave.problem().params),
            :sine_gordon => keys(SineGordon.problem().params),
            :vector_wave => keys(VectorWave.problem().params))
        @test problem_keys[:wave] != problem_keys[:sine_gordon]
        fixed_boundary_methods = (:fixed_boundary_dead, :fixed_boundary_live_abg_joint,
            :fixed_boundary_live_tsr_joint, :fixed_boundary_live_abg_projected,
            :fixed_boundary_live_tsr_projected)
        families = (
            network_matrix = (count = 18, methods = (:network_all, :network_last_layer),
                problems = (:wave, :sine_gordon, :vector_wave),
                build = (network_matrix_problem, network_matrix_method)),
            symbolic_matrix = (count = 3, methods = (:symbolic,), problems = (:wave,),
                build = (symbolic_matrix_problem, symbolic_matrix_method)),
            elm_matrix = (count = 6, methods = (:elm,), problems = (:wave, :sine_gordon),
                build = (elm_matrix_problem, elm_matrix_method)),
            trialnn_matrix = (
                count = 6, methods = (:trialnn,), problems = (:wave, :sine_gordon),
                build = (trialnn_matrix_problem, trialnn_matrix_method)),
            fixed_boundary_matrix = (count = 30, methods = fixed_boundary_methods,
                problems = (:wave, :sine_gordon),
                build = (fixed_boundary_problem, fixed_boundary_method)))
        @test allunique(row.entry.stem for row in rows)
        for (name, family) in pairs(families)
            entries = family_entries(name)
            @test family_benchmark(name) isa Function
            @test family_run(name) === nothing
            @test length(entries) == family.count
            for method in family.methods, problem in family.problems

                group = [entry
                         for entry in entries
                         if entry.method === method && entry.problem === problem]
                @test length(group) == 3
                @test allunique(entry.resolution for entry in group)
            end
            makeproblem, makemethod = family.build
            for entry in entries
                @test entry.tag === :matrix
                @test entry.method in family.methods
                @test entry.problem in family.problems
                @test entry.label isa String && entry.problem_label isa String
                @test entry.resolution isa NamedTuple && haskey(entry.resolution, :timestep)
                @test entry.parameters isa NamedTuple
                keywords = get(entry, :solver, (;))
                @test keywords isa NamedTuple
                problem = makeproblem(entry)
                method = makemethod(entry)
                @test method isa SpaceTimeMethod
                @test problem.timestep == entry.resolution.timestep
                @test ntime(problem) >= 1
                # the problem the row names, told apart by its parameters
                @test keys(problem.params) == problem_keys[entry.problem]
                @test problem.D == (entry.problem === :vector_wave ? 2 : 1)
                integrator = GeometricIntegrator(problem, method; keywords...)
                @test integrator isa GeometricIntegrator
                # A network family's rows solve at `f_abstol = 0` and a measured relative tolerance:
                # the network family's `network_f_reltol(Float64)` for the network and the TrialNN
                # rows, the ELM family's `trained_f_reltol(Float64)` for the ELM rows.
                if name in (:network_matrix, :elm_matrix, :trialnn_matrix)
                    options = config(solver(integrator))
                    @test options.f_abstol == 0
                    @test options.f_reltol == (name === :elm_matrix ? 1e-2 : 1e-4)
                end
                # The built method is the one the row names.
                if name === :network_matrix
                    @test ansatz(method).mode === entry.parameters.mode
                    @test entry.parameters.mode ===
                          (entry.method === :network_last_layer ? :last_layer : :all)
                end
                if name === :fixed_boundary_matrix
                    @test MultisymplecticIntegrators.initial_guess(method) isa
                          (entry.parameters.solve === :newton ? Continuation : FitTo)
                    @test ansatz(method).Sx == entry.parameters.Sx
                    @test ansatz(method).St == entry.parameters.St
                    @test ansatz(method).bubble_form === entry.parameters.form
                end
            end
        end
        # The symbolic rows run at the standing wave's period `2l / c = 4` and at the tolerance the
        # symbolic family measured on its affine control.
        for entry in family_entries(:symbolic_matrix)
            @test entry.resolution.timestep == 4.0
            @test entry.solver == (f_abstol = 0.0, f_reltol = symbolic_f_reltol(Float64))
        end
        @test allunique(entry.resolution.ncells
        for entry in family_entries(:symbolic_matrix))
        # The fixed-boundary rows keep the base stems, reach `t = 1`, and name the solver method
        # their branch is solved by.
        solvers = Dict(:newton => SimpleSolvers.Newton(),
            :joint => MultisymplecticIntegrators.LevenbergMarquardt(),
            :projected => MultisymplecticIntegrators.LevenbergMarquardt(
                variable_projection = true))
        for entry in family_entries(:fixed_boundary_matrix)
            @test occursin("-fixed-boundary-", entry.stem)
            @test entry.parameters.steps * entry.resolution.timestep ≥ 1 - 1e-12
            @test entry.solver == (solver = solvers[entry.parameters.solve],)
            @test endswith(string(entry.method), string(entry.parameters.solve)) ||
                  entry.parameters.solve === :newton
        end
        @test count(entry -> entry.solver.solver == solvers[:projected],
            family_entries(:fixed_boundary_matrix)) == 12
        # A family of the figure archives contributes no row, and the run-only family is registered.
        elm = Set(entry.stem for entry in family_entries(:elm))
        @test !isempty(elm)
        @test !any(entry -> entry.stem in elm, [row.entry for row in rows])
    end
end
