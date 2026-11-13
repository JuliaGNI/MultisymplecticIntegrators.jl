# The box scheme as the matrix's baseline rows: the Preissmann box scheme on the three example
# problems whose density is in its class, at two resolutions each, with a `smoke` tag on the
# coarsest row of the wave and the sine-Gordon equation.
#
# The driver includes this file through its `*_model.jl` glob. It builds nothing and runs nothing at
# top level, so `run_family.jl` includes it beside the figure families without paying for a solve.

using MultisymplecticIntegrators

"""
The problems of the box scheme: exactly those whose density is in its class, so `LinearTransport`,
whose density is not, has no row.
"""
const BOX_PROBLEMS = (:wave, :vector_wave, :sine_gordon)

"""
The two resolutions of the baseline: the time step and the space step, which are the box scheme's
whole discretisation, since it reads them from the problem.
"""
const BOX_RESOLUTIONS = ((timestep = 0.1, xstep = 0.1), (timestep = 0.05, xstep = 0.05))

"""
The stem of one row: `<problem>-box-h<time step>`, the family's name in the naming rule of
`scripts/README.md`.
"""
box_stem(problem, resolution) = "$(problem)-box-h$(resolution.timestep)"

"""
The box scheme's rows — the baseline of every comparison the matrix draws — with a `smoke` tag on
the coarsest row of each of two problems, which is the pair the gate runs.
"""
const BOX = [(stem = box_stem(problem, resolution),
                 label = "$(problem), box, h = $(resolution.timestep)",
                 method = :box, problem, problem_label = matrix_problem_label(problem),
                 tag = resolution.timestep == 0.1 && problem in (:wave, :sine_gordon) ?
                       :smoke : :matrix,
                 resolution,
                 parameters = (scheme = :preissmann,))
             for problem in BOX_PROBLEMS, resolution in BOX_RESOLUTIONS]

"""
    box_problem(entry)

The problem of a baseline row: the example problem it names, on the row's time step and space step
and the matrix's spans. The space step is the row's own, because the box scheme reads its whole
discretisation from the problem.
"""
function box_problem(entry)
    matrix_problem("box scheme", entry.problem, BOX_PROBLEMS;
        timestep = entry.resolution.timestep, xstep = entry.resolution.xstep)
end

"""
    box_method(entry)

The method of a baseline row, which is the box scheme itself: its resolution is the problem's, so a
row's discretisation reaches it through [`box_problem`](@ref).
"""
box_method(entry) = BoxScheme()

"""
    box_benchmark(entry) -> Dict{String, Any}

The registry's `benchmark` for the baseline: build the row's problem, run the box scheme on it, and
return its record for the archive of the `"benchmark"` kind.
"""
box_benchmark(entry) = benchmark_run(box_problem(entry), box_method(entry), entry)

register_family!(:box, BOX; benchmark = box_benchmark)
