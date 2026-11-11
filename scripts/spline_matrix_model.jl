# The spline family as the matrix's rows: the space-time Galerkin integrator on a tensor B-spline
# ansatz, under the three couplings the package ships, on the four example problems, at three
# resolutions each.
#
# The driver includes this file through its `*_model.jl` glob, and it is the only place the family's
# row constructors are written: a change to the ansatz or to a coupling is one edit. It builds
# nothing and runs nothing at top level, so `run_family.jl` can include it beside the figure
# families without paying for a solve.

using MultisymplecticIntegrators

"""
The three couplings of the family, as the `method` of a row: the coefficient restriction, the weak
initial condition, and the weak initial-and-boundary conditions.

The symbols are what `select_rows` selects on, so they are also what a report and a table show.
"""
const SPLINE_METHODS = (
    (method = :spline_restriction, coupling = :restriction),
    (method = :spline_initial, coupling = :initial),
    (method = :spline_initial_boundary, coupling = :initial_boundary))

"""The four problems of the family, as the `problem` of a row."""
const SPLINE_PROBLEMS = (:wave, :vector_wave, :sine_gordon, :linear_transport)

"""
The three resolutions of one method and problem, as the `resolution` of a row.

A resolution is the discretisation and not the method: the time step, the degree of the ansatz, its
cell counts and its quadrature order, which is what a coarsest-to-finest comparison of one method
reads without parsing a stem.
"""
const SPLINE_RESOLUTIONS = (
    (timestep = 0.1, degree = 2, ncells = (1, 10), n = 3),
    (timestep = 0.05, degree = 2, ncells = (2, 20), n = 3),
    (timestep = 0.025, degree = 2, ncells = (4, 40), n = 3))

# The space step of every row, and the span of each problem's field. `LinearTransport`'s default
# span is its own, because its initial profile is a Gaussian centred on it.
const SPLINE_XSTEP = 0.05
const SPLINE_XSPAN = (0.0, 1.0)
const SPLINE_TIMESPAN = (0.0, 0.2)
const TRANSPORT_XSPAN = (-0.5, 0.0)

"""
The stem of one row: `<problem>-<method>-h<time step>`, so that sorting a directory groups a
problem's methods together and a method's steps, as `scripts/README.md`'s naming rule says.
"""
function spline_matrix_stem(problem, method, resolution)
    "$(problem)-$(method)-h$(resolution.timestep)"
end

"""
    spline_problem_label(problem)

The display name of a problem of the family, which its rows archive and its table is named by.
"""
function spline_problem_label(problem)
    problem === :wave ? "Wave" :
    problem === :vector_wave ? "VectorWave" :
    problem === :sine_gordon ? "SineGordon" :
    problem === :linear_transport ? "LinearTransport" :
    string(problem)
end

"""
The spline matrix's rows: every coupling on every problem at every resolution, 36 in all, each with
its own stem, its method, its problem, its tag, its resolution and its method-defining parameters.
"""
const SPLINE_MATRIX = [(stem = spline_matrix_stem(problem, held.method, resolution),
                           label = "$(problem), $(held.method), h = $(resolution.timestep)",
                           method = held.method, problem, problem_label = spline_problem_label(problem),
                           tag = :matrix, resolution,
                           parameters = (degree = resolution.degree,
                               ncells = resolution.ncells, n = resolution.n,
                               coupling = held.coupling))
                       for held in SPLINE_METHODS, problem in SPLINE_PROBLEMS,
resolution in SPLINE_RESOLUTIONS]

"""
    spline_matrix_problem(entry)

The problem of a row: the example problem it names, on the row's time step and the family's space
step.

`LinearTransport` is built as the problem module builds it, which is the doubled field its density
needs: a row of it on a single field would have no transport equation to solve, and its density is
not `D = 1`.
"""
function spline_matrix_problem(entry)
    T = Float64
    timespan = SPLINE_TIMESPAN
    if entry.problem === :wave
        return Wave.problem(timespan = timespan, timestep = entry.resolution.timestep,
            xspan = SPLINE_XSPAN, xstep = SPLINE_XSTEP)
    elseif entry.problem === :vector_wave
        return VectorWave.problem(
            timespan = timespan, timestep = entry.resolution.timestep,
            xspan = SPLINE_XSPAN, xstep = SPLINE_XSTEP)
    elseif entry.problem === :sine_gordon
        return SineGordon.problem(
            timespan = timespan, timestep = entry.resolution.timestep,
            xspan = SPLINE_XSPAN, xstep = SPLINE_XSTEP)
    elseif entry.problem === :linear_transport
        return LinearTransport.problem(
            timespan = timespan, timestep = entry.resolution.timestep,
            xspan = TRANSPORT_XSPAN, xstep = SPLINE_XSTEP)
    end
    throw(ArgumentError("the spline matrix has no problem `$(entry.problem)`; it holds " *
                        join(string.(SPLINE_PROBLEMS), ", ")))
end

"""
    spline_matrix_method(entry)

The method of a row: the tensor B-spline ansatz at the row's resolution, under the coupling its
method names.

A coupling symbol the family does not hold raises an `ArgumentError` naming it: a mistyped coupling
that ran as another one would be a result about the wrong method.
"""
function spline_matrix_method(entry)
    xspan = entry.problem === :linear_transport ? TRANSPORT_XSPAN : SPLINE_XSPAN
    ansatz = TensorBSplineAnsatz(degree = entry.parameters.degree,
        ncells = entry.parameters.ncells, xspan = xspan, n = entry.parameters.n)
    coupling = entry.parameters.coupling === :restriction ? CoefficientRestriction() :
               entry.parameters.coupling === :initial ? WeakInitialCondition() :
               entry.parameters.coupling === :initial_boundary ?
               WeakInitialAndBoundaryConditions() :
               throw(ArgumentError("unknown spline coupling `$(entry.parameters.coupling)`; the " *
                                   "family takes :restriction, :initial or :initial_boundary"))
    return SpaceTimeGalerkin(ansatz, coupling)
end

"""
    spline_matrix_benchmark(entry) -> Dict{String, Any}

The registry's `benchmark` for the spline family: build the row's problem and method, run it, and
return its record for the archive of the `"benchmark"` kind.
"""
function spline_matrix_benchmark(entry)
    benchmark_run(spline_matrix_problem(entry), spline_matrix_method(entry), entry)
end

register_family!(:spline_matrix, SPLINE_MATRIX; benchmark = spline_matrix_benchmark)
