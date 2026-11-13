# The row runner of the matrix, the archive of one row, and the tables of a selection.
#
# A row is an entry of a family that registers a `benchmark`, and the family's `benchmark` builds
# its problem and its method and hands them here: `benchmark_run(problem, method, entry)` integrates
# the row, measures it and returns the record `store_run!` writes.
#
# The file assumes that `archives.jl` and `experiments.jl` are already included, as
# `run_benchmark.jl` and the tests include them, and it holds no work at top level.
#
# The per-step quantities — the solve's stats, the error, the energy drift, the cost — are recorded
# for every stored step and never for the last one alone, because a run that stops early, or that
# ends on a step which did not converge, is a result of the comparison and not a dropped row.

using MultisymplecticIntegrators
using Printf
using LinearAlgebra: BLAS
using GeometricBase: ntime, timesteps
using GeometricIntegratorsBase: GeometricIntegrator, solutionstep, cache, current, reset!,
                                integrate!
using GeometricSolutions: GeometricSolution

# The keys of the "benchmark" kind, as the table of scripts/README.md lists them.
const BENCHMARK_KEYS = ("kind", "stem", "label", "method", "problem", "tag", "resolution",
    "parameters", "status", "exception", "message", "failed_at_step", "steps", "t", "x", "u",
    "converged", "residuals", "initial_residuals", "iterations", "error", "error_kind",
    "reference_stem", "energy_drift", "charge", "cost", "wall_per_step", "wall_time", "pid",
    "load", "threads", "blas_threads", "julia", "check_bounds")

# The eight records of §A that the matrix keeps as historical data: five of the wave equation and
# three of the sine-Gordon equation, from `test/helpers/reference_values.jl`, which is their only
# machine-readable copy. They are named here because the set is a choice of names and not a
# property of a record: seven of the eight carry `status = :ok`, and none of them is an oracle.
#
# The helper is loaded here unless the caller has loaded it already, so that a test which includes
# the records itself and this file beside them reads one copy of each constant.
isdefined(@__MODULE__, :WAVE_NN_FULLY) ||
    include(joinpath(@__DIR__, "..", "test", "helpers", "reference_values.jl"))

const HISTORICAL = (
    ("Wave", :WAVE_NN_FULLY),
    ("Wave", :WAVE_NN_PARTIALLY),
    ("Wave", :WAVE_NN_SYMBOLIC_FULLY),
    ("Wave", :WAVE_ELM_LSGD),
    ("Wave", :WAVE_TRIALNN_OGA2D),
    ("SineGordon", :SINE_GORDON_NN_FULLY),
    ("SineGordon", :SINE_GORDON_ELM_LSGD),
    ("SineGordon", :SINE_GORDON_TRIALNN_OGA2D))

"""
    row_cost(method, problem) -> Int

The number of scalar unknowns one solve of `method` on `problem` handles, which is the matrix's
measure of a run's cost: a slab method's free coefficients, and the box scheme's unknowns of one
step.

It is a count and not a time, so it does not move with the machine, and it is the one work measure
every method of the package exposes through one interface.
"""
function row_cost(method::SpaceTimeGalerkin, problem::LagrangianPDEProblem)
    total_free_coefficients(method, problem)
end

row_cost(method::BoxScheme, problem::LagrangianPDEProblem) = box_grid(problem).unknowns

"""
    drive_steps(problem, method) -> NamedTuple

The steps of one run: the solution, the traces, the solver's record, the times of the steps after
the first, and how the run ended.

The steps are driven here rather than by [`integrate`](@ref), because the matrix records what a step
costs: every step after the first is timed and the first is not, since it carries the compilation of
the method's own code. A step that throws ends the run with `status = :failed`, the name of the
exception's type, the first line of its message and the number of the step, and the steps before it
stand.

The construction of the integrator is outside this function's `try`: a row whose method the problem
does not support is a failure of the row before a step runs, and the caller records it with
`failed_at_step = 0`.
"""
function drive_steps(problem::LagrangianPDEProblem, method::SpaceTimeMethod;
        solver_keywords = (;))
    steps = ntime(problem)
    integrator = GeometricIntegrator(problem, method; solver_keywords...)
    c = cache(integrator)
    reset!(c)
    solution = GeometricSolution(problem)
    sample = top_trace(c)
    traces = sizehint!(Vector{typeof(sample)}(), steps)
    solstep = solutionstep(integrator, solution[0])
    wall = Float64[]
    for n in 1:steps
        reset!(solstep, timesteps(solution)[n])
        started = time_ns()
        try
            integrate!(solstep, integrator)
        catch raised
            return (; solution, traces, stats = solve_stats(c), wall, status = :failed,
                exception = string(nameof(typeof(raised))), message = failure_message(raised),
                failed_at_step = n)
        end
        elapsed = (time_ns() - started) / 1.0e9
        n == 1 || push!(wall, elapsed)
        copy!(solution, current(solstep), n)
        push!(traces, deepcopy(top_trace(c)))
    end
    return (;
        solution, traces, stats = solve_stats(c), wall, status = :ok, exception = nothing,
        message = nothing, failed_at_step = 0)
end

# One recorded series, cut to the steps that were stored: the cache's vectors are the run's own, and
# an archive holds copies of the entries the run reached.
function recorded(series, kept::Integer)
    series === nothing && return nothing
    return collect(series[1:min(kept, length(series))])
end

"""
    benchmark_run(problem, method, entry) -> Dict{String, Any}

Integrate one row of the matrix and return its record: the `"benchmark"` archive of the schema in
`scripts/README.md`.

The solve's stats, the error against the problem's exact solution, the energy drift and the charge
are recorded per stored step, and the run's cost and the time of every step after the first with
them. A run that throws is archived rather than dropped: its `"status"` is `:failed`, `"exception"`
names the exception's type, `"message"` holds the first line of its text, `"failed_at_step"` the
number of the step that threw — `0` for the construction of the integrator — and the stats of the
steps before it. A run that completes with a step that did not converge leaves
`"status" = :unconverged`, which is a result too: the matrix compares methods that fail, and it
never runs under `on_failure = :error`.

A problem without an exact solution is measured against the finest run of the same method instead,
which [`attach_references!`](@ref) records once the selection's archives exist; this function then
stores no error and leaves the reference to it.

A row may name its solver's keywords in a `solver` field — `(max_iterations = 1, f_reltol = …)`, as
a family that compares a solver's settings needs — and they reach the integrator's constructor as
they are. A row that names none is solved at the method's own defaults.
"""
function benchmark_run(problem::LagrangianPDEProblem, method::SpaceTimeMethod, entry)
    load = collect(Sys.loadavg())
    pid = getpid()
    keywords = get(entry, :solver, (;))
    run = try
        drive_steps(problem, method; solver_keywords = keywords)
    catch raised
        (; solution = nothing, traces = Any[], stats = nothing, wall = Float64[],
            status = :failed, exception = string(nameof(typeof(raised))),
            message = failure_message(raised), failed_at_step = 0)
    end
    kept = length(run.traces)
    # A row whose integrator is not built has no cost: the count is read only once a step exists,
    # because a method the problem does not support can throw in the count as well.
    rounds = kept == 0 ? 0 : row_cost(method, problem)
    iterations = something(
        recorded(run.stats === nothing ? nothing : run.stats.iterations, kept),
        Int[])
    residuals = something(
        recorded(run.stats === nothing ? nothing : run.stats.residuals, kept),
        Float64[])
    initial_residuals = something(
        recorded(
            run.stats === nothing ? nothing : run.stats.initial_residuals, kept),
        Float64[])
    converged = something(
        recorded(run.stats === nothing ? nothing : run.stats.converged, kept),
        Bool[])
    status = run.status
    status === :ok && !all(converged) && (status = :unconverged)
    measured = (solution = run.solution, traces = run.traces)
    # The kind of error a row can have is the problem's, and the values are the run's: a problem
    # without an exact solution is measured against a finer run, which `attach_references!` fills in
    # once the selection's archives exist, and a run that stored no step records no value either way.
    exact = problem.solutions !== nothing
    errors = exact && run.solution !== nothing ? solution_error(problem, measured) : nothing
    drift = kept == 0 ? Float64[] : energy_error(problem, method, run.traces)
    charge = kept == 0 ? nothing : noether_charge(problem, method, run.traces)
    t, x, u = run.solution === nothing ? (Float64[], Float64[], [Float64[]]) :
              solution_data(problem, measured)
    data = row_identity(entry)
    merge!(data,
        Dict{String, Any}(
            "status" => status,
            "exception" => run.exception,
            "message" => run.message,
            "failed_at_step" => run.failed_at_step,
            "steps" => kept,
            "t" => t,
            "x" => x,
            "u" => u,
            "converged" => converged,
            "residuals" => residuals,
            "initial_residuals" => initial_residuals,
            "iterations" => iterations,
            "error" => errors === nothing ? fill(NaN, kept) : errors,
            "error_kind" => exact ? "exact" : "reference",
            "reference_stem" => nothing,
            "energy_drift" => drift,
            "charge" => charge,
            "cost" => fill(rounds, kept),
            "wall_per_step" => run.wall,
            "wall_time" => isempty(run.wall) ? 0.0 : sum(run.wall),
            "pid" => pid,
            "load" => load,
            "threads" => Threads.nthreads(),
            "blas_threads" => BLAS.get_num_threads(),
            "julia" => VERSION,
            "check_bounds" => Base.JLOptions().check_bounds))
    return data
end

"""
    reference_error(coarse, fine) -> Vector

The largest absolute difference of two archived runs' fields at the times the coarse run stored, on
the nodes their grids share.

`coarse` and `fine` are archives of one method on one problem, the second on a grid that holds the
first's nodes and on a time step that holds the first's times; a node of the coarse grid, or a time
of the coarse run, that the finer run does not hold is refused, because the comparison of a coarse
run with a finer one is a comparison at the nodes and the times the two share and not one by index.

The field of stored step `n` is row `n + 1` of an archive's `"u"`, because row `1` is the initial
data. The finer run's rows are cut to the coarse run's times, so that the finer run's own step index
names no row of the comparison.
"""
function reference_error(coarse, fine)
    xc = coarse["x"]
    xf = fine["x"]
    steps = coarse["steps"]
    iszero(steps) && return Float64[]
    shared = [searchsortedfirst(xf, node) for node in xc]
    for (j, node) in enumerate(xc)
        k = shared[j]
        k <= length(xf) && isapprox(xf[k], node; atol = 0, rtol = 1.0e-12) ||
            throw(ArgumentError("the node $node of the coarse run is not a node of the finer " *
                                "run's grid; a reference comparison is made at the nodes the two " *
                                "share"))
    end
    rows = fine_time_rows(coarse["t"], fine["t"], steps)
    fields = [field[rows, :] for field in fine["u"]]
    errors = zeros(Float64, steps)
    for n in 1:steps
        worst = 0.0
        for d in eachindex(coarse["u"])
            uc = coarse["u"][d]
            uf = fields[d]
            for (j, k) in enumerate(shared)
                worst = max(worst, abs(Float64(uc[n + 1, j]) - Float64(uf[n + 1, k])))
            end
        end
        errors[n] = worst
    end
    return errors
end

# The finer run's rows for the times the coarse run stored: row `n + 1` is the finer run's row of
# the coarse run's stored step `n`, and row `1` is the initial data of both.
#
# A time of the coarse run that the finer run does not hold is refused rather than matched to the
# nearest step: two resolutions of one method are compared at the times they share, and a run that
# reaches no time of the other is not comparable with it at all.
function fine_time_rows(tc, tf, steps)
    rows = Vector{Int}(undef, steps + 1)
    for n in 0:steps
        t = tc[n + 1]
        k = nearest_time(tf, t)
        k > 0 && isapprox(tf[k], t; atol = 0, rtol = 1.0e-12) ||
            throw(ArgumentError("the time $t of the coarse run is not a time of the finer run; " *
                                "a reference comparison is made at the times the two store"))
        rows[n + 1] = k
    end
    return rows
end

# The index of the time of `tf` nearest to `t`, or `0` for a run that stored no time. The two runs'
# times are the same values, but a time each computes from its own step count can sit a unit in the
# last place away, so the nearer of the sorted neighbours is the candidate and the caller checks it.
function nearest_time(tf, t)
    isempty(tf) && return 0
    k = searchsortedfirst(tf, t)
    k > length(tf) && return length(tf)
    k > 1 && abs(tf[k - 1] - t) < abs(tf[k] - t) && return k - 1
    return k
end

"""
    attach_references!(runs) -> Vector{Dict}

Fill in the reference comparison of the runs that have no exact solution, and return the archives
whose reference fields changed.

A group is the runs of one method on one problem, and the finest of a group is the run on the
largest grid, and of the runs on one grid the one with the smallest time step, so that the spline
rows, which refine the time step and the ansatz on one output grid, find theirs whatever the order
of `runs`. The finest run names itself as its reference and records no error, and every coarser run
names it and records the difference [`reference_error`](@ref) gives at the times and the nodes the
two share.

Nothing is written here, so that a test can ask for the comparison of archives it holds in memory;
the driver stores the changed archives again.
"""
function attach_references!(runs)
    changed = Dict{String, Any}[]
    groups = Dict{Tuple{String, String}, Vector{Dict{String, Any}}}()
    for run in runs
        get(run, "error_kind", "exact") == "reference" || continue
        key = (string(get(run, "method", "")), string(get(run, "problem", "")))
        push!(get!(groups, key, Dict{String, Any}[]), run)
    end
    for group in values(groups)
        finest = first(sort(group;
            by = run -> (-length(run["x"]), run["resolution"].timestep)))
        finest["reference_stem"] = finest["stem"]
        push!(changed, finest)
        for run in group
            run === finest && continue
            run["error"] = reference_error(run, finest)
            run["reference_stem"] = finest["stem"]
            push!(changed, run)
        end
    end
    return changed
end

"""
    benchmark_archive(stem, data)

The record of one row checked against the keys of the `"benchmark"` kind, ready for `store_run!`.

It is a check of the keys alone, and it names the ones the record lacks: what a run means is the
runner's, and a reader that globs `runs/` has only the keys to go by.
"""
function benchmark_archive(stem, data)
    # `"stem"` is `store_run!`'s, which adds it from its argument, so a record is checked for every
    # key but that one.
    missing_keys = [key for key in BENCHMARK_KEYS if key != "stem" && !haskey(data, key)]
    isempty(missing_keys) ||
        throw(ArgumentError("the record of `$(stem)` has no $(join(missing_keys, ", ")); the " *
                            "\"benchmark\" kind's keys are tabulated in scripts/README.md"))
    return data
end

# The keys of one row that do not depend on how it ran: the row's identity, which a reader needs
# even of a row that never started.
#
# `"problem"` is the problem's display name and not the row's selection symbol: a table is written
# per problem, and its heading and its file name are the name the problem module carries and the
# reference records of §A are filed under.
function row_identity(entry)
    return Dict{String, Any}(
        "kind" => "benchmark",
        "label" => string(entry.label),
        "method" => string(entry.method),
        "problem" => string(get(entry, :problem_label, entry.problem)),
        "tag" => string(entry.tag),
        "resolution" => entry.resolution,
        "parameters" => entry.parameters)
end

"""
    failed_archive(entry, reason) -> Dict{String, Any}

The archive of a row whose child process ended without writing one, and the reason it did.

A row that exits before archiving is a result of the comparison like a row that threw: the driver
writes this record with the child's exit code, or with the signal that ended it, continues with the
remaining rows, and still returns 0, because what it reports is whether every selected row has an
archive and not whether every row converged.
"""
function failed_archive(entry, reason)
    data = row_identity(entry)
    merge!(data,
        Dict{String, Any}(
            "status" => :failed,
            "exception" => "ProcessExit",
            "message" => string(reason),
            "failed_at_step" => 0,
            "steps" => 0,
            "t" => Float64[], "x" => Float64[], "u" => [Float64[]],
            "converged" => Bool[], "residuals" => Float64[], "initial_residuals" =>
                Float64[],
            "iterations" => Int[],
            "error" => Float64[], "error_kind" => "exact", "reference_stem" => nothing,
            "energy_drift" => Float64[], "charge" => nothing, "cost" => Float64[],
            "wall_per_step" => Float64[], "wall_time" => 0.0,
            "pid" => getpid(), "load" => collect(Sys.loadavg()),
            "threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
            "julia" => VERSION, "check_bounds" => Base.JLOptions().check_bounds))
    return data
end

"""
    final_time(run) -> Union{Float64, Nothing}

The time of an archive's last stored step, the last entry of its `"t"`, rounded to 12 significant
digits, and `nothing` for an archive that stored no step.

It is the time the archive's last error, energy drift and cost belong to, so two archives whose final
times differ are never compared on one panel or in one table. The rounding makes the times that two
resolutions reach by different numbers of steps compare equal.
"""
function final_time(run)
    iszero(get(run, "steps", 0)) && return nothing
    return round(Float64(last(run["t"])); sigdigits = 12)
end

# The final time of a historical record: its stored steps times the time step its problem names.
function record_final_time(record)
    timestep = parse(Float64, match(r"timestep = ([0-9.eE+-]+)", record.problem)[1])
    return round(record.steps * timestep; sigdigits = 12)
end

"""
    benchmark_group(problem, time) -> String

The name of the comparison of one problem's archives at one final time, as `benchmark_tables` keys
its tables and `figures.jl` names their files: `"Wave-t0.2"`, and `"Wave-no-step"` for the archives
that stored no step.
"""
function benchmark_group(problem, time)
    time === nothing ? "$(problem)-no-step" : "$(problem)-t$(time)"
end

"""
    benchmark_tables(runs; records = HISTORICAL) -> Dict{String, String}

The Markdown tables of a selection, one per problem and final time, as
[`benchmark_group`](@ref)` => text`.

The archives of one problem are grouped by their [`final_time`](@ref), and each group is one table,
so that a table never compares errors at two times. Each archive of a group is one row: its stem,
its method, its tag, the steps it stored, its status, and the error, the energy drift, the cost and
the wall time of its last stored step. An archive with no stored step still has its row, in the
problem's table of the archives that stored none, because such a run is a result and its status
says why it has no numbers.

The eight historical records of §A that a problem holds are appended to the table of their own final
time — their stored steps times their problem's time step — marked
`historical` and `unconverged`: they are the old stack's own numbers, from
`test/helpers/reference_values.jl`, and the matrix reproduces none of them. The set is a choice of
names and not a query of a record's status, so a record that converged is still marked — which is
what makes the comparison honest, and why the table reads the records by name. A record's series is
its energy error — the relative error of its slabs' quadrature energy, the quantity the matrix's
drift measures — so it is printed where a live row prints its energy drift and the record's error
cell is a dash.
"""
function benchmark_tables(runs; records = HISTORICAL)
    groups = Set{Tuple{String, Union{Float64, Nothing}}}()
    for run in runs
        get(run, "kind", "") == "benchmark" || continue
        push!(groups, (string(get(run, "problem", "")), final_time(run)))
    end
    for (problem, name) in records
        push!(groups, (problem, record_final_time(getfield(@__MODULE__, name))))
    end
    tables = Dict{String, String}()
    for (problem, time) in groups
        rows = sort(
            [run
             for run in runs
             if get(run, "kind", "") == "benchmark" &&
                    string(get(run, "problem", "")) == problem &&
                    isequal(final_time(run), time)];
            by = run -> run["stem"])
        text = IOBuffer()
        print(text, "# ", problem,
            time === nothing ? ", no stored step" : ", t = $(time)", "\n\n")
        print(
            text, "| stem | method | tag | steps | status | error | energy drift | cost | ",
            "wall time (s) |\n")
        print(text, "|:--|:--|:--|--:|:--|--:|--:|--:|--:|\n")
        for run in rows
            print(text, "| ", run["stem"], " | ", run["method"], " | ", run["tag"], " | ",
                run["steps"], " | ", run["status"], " | ",
                _table_number(run["error"]), " | ", _table_number(run["energy_drift"]), " | ",
                _table_number(run["cost"]), " | ", _table_number(run["wall_time"]), " |\n")
        end
        for (held, name) in records
            record = getfield(@__MODULE__, name)
            held == problem && isequal(record_final_time(record), time) || continue
            print(text, "| ", name, " | historical | — | ", record.steps,
                " | unconverged | — | ", _table_number(record.energy_error), " | — | — |\n")
        end
        tables[benchmark_group(problem, time)] = String(take!(text))
    end
    return tables
end

# The last entry of a recorded series as a table cell, and a dash where a run stored none.
function _table_number(series::AbstractVector)
    isempty(series) && return "—"
    return _table_number(last(series))
end

_table_number(value::Real) = isfinite(value) ? @sprintf("%.3e", value) : "—"
