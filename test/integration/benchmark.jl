using GeometricBase: ntime, timesteps
using JLD2
using LinearAlgebra: BLAS
using MultisymplecticIntegrators
using MultisymplecticIntegrators: space_quadrature_nodes, space_quadrature_weights
using Random: Xoshiro
using Test

# The scripts are no part of the package, so the whole harness is tested through its files. The
# reference records are included before the runner, which loads them itself only when its caller has
# not, and the driver then includes the archive layer, the registry, the matrix's runner and every
# model file in the order it includes them itself.
include(joinpath(@__DIR__, "..", "helpers", "reference_values.jl"))
include(joinpath(@__DIR__, "..", "..", "scripts", "run_benchmark.jl"))
include(joinpath(@__DIR__, "..", "helpers", "problem_generator.jl"))

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
function unsolved_problem(base, xstep)
    return LagrangianPDEProblem(base.density; D = base.D, ics_function = base.ics_function,
        bcs_function = base.bcs_function, timespan = base.timespan,
        timestep = base.timestep, xspan = base.xspan, xstep = xstep, params = base.params)
end

# The largest absolute difference of two archived runs' fields at each stored step, on the nodes
# their grids share — computed here, from the archives, and not by the code under test. An archive's
# `"error"` holds one entry per stored step, so the comparison is per step too.
function archived_difference(coarse, fine)
    shared = [searchsortedfirst(fine["x"], node) for node in coarse["x"]]
    return [maximum(abs(coarse["u"][d][n + 1, j] - fine["u"][d][n + 1, k])
            for d in eachindex(coarse["u"]), (j, k) in enumerate(shared))
            for n in 1:coarse["steps"]]
end

@testset "Benchmark matrix" begin
    @testset "a selection by method, problem or tag" begin
        entries = (
            row("sel-a", :method_one, :problem_one, :tag_one, (k = 1,)),
            row("sel-b", :method_one, :problem_two, :tag_two, (k = 2,)),
            row("sel-c", :method_two, :problem_one, :tag_one, (k = 3,)),
            row("sel-d", :method_two, :problem_two, :tag_two, (k = 4,)),
            # a tag that holds the first one as a prefix, which a substring match would take with it
            row("sel-e", :method_two, :problem_one, :tag_ten, (k = 5,)))
        register_family!(:selection, entries; benchmark = _ -> Dict{String, Any}())
        stems(selected) = [row.entry.stem for row in selected]
        @test stems(select_rows(method = :method_one)) == ["sel-a", "sel-b"]
        @test stems(select_rows(problem = :problem_one)) == ["sel-a", "sel-c", "sel-e"]
        @test stems(select_rows(tag = :tag_one)) == ["sel-a", "sel-c"]
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
        # One generated problem of the property suite, which passes no `solutions`, at three nested
        # grids of one time step: the coarse nodes are nodes of the fine ones.
        base = first(filter(entry -> entry.problem.D == 1,
            generate_problems(Float64; rng = Xoshiro(SEED)))).problem
        @test base.solutions === nothing
        span = base.xspan
        steps = (12, 24, 48)
        problems = [unsolved_problem(base, (span[2] - span[1]) / m) for m in steps]
        entries = [row("generated-coarse", :box, :generated, :matrix, (xstep = m,))
                   for m in steps]
        runs = [benchmark_run(problems[i], BoxScheme(), entries[i])
                for i in eachindex(steps)]
        @test all(archive -> archive["error_kind"] == "reference", runs)
        for (archive, entry) in zip(runs, entries)
            archive["stem"] = entry.stem
        end
        attach_references!(runs)
        finest = runs[end]
        @test finest["reference_stem"] == "generated-coarse"
        @test all(isnan, finest["error"])
        for (coarse, entry) in zip(runs[1:(end - 1)], entries[1:(end - 1)])
            @test coarse["reference_stem"] == finest["stem"]
            @test coarse["error"] ≈ archived_difference(coarse, finest)
        end
        @test runs[1]["error"][1] > 0
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
            xspan = (T(-1) / 2, zero(T)), xstep = T(1) / 20)
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
            @test archives[name]["charge"] === nothing
        end
        @test all(isfinite, archives[:vector_wave]["energy_drift"])
        @test first(archives[:vector_wave]["energy_drift"]) == 0

        # The box scheme's energy is the midpoint rule over its boxes and not the trapezoidal rule
        # at their nodes: `H` at the mean of each box's two nodes, `Σ Δx · H(½(zⱼ + zⱼ₊₁))`. The
        # run's first level is the problem's own initial data, so its energy is the closed form's
        # at `t₀` to round-off, while the two readings of the rule on that level differ by
        # `6.2e-3` relative: the assertion is of the rule and not of the scheme's order.
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
            if entry.problem === :linear_transport
                @test problem.D == 2
            end
        end
        # A family of the figure archives contributes no row, and the run-only family is registered.
        elm = Set(entry.stem for entry in family_entries(:elm))
        @test !isempty(elm)
        @test !any(entry -> entry.stem in elm, [row.entry for row in rows])
    end
end
