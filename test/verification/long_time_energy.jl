using GeometricBase: parameters
using MultisymplecticIntegrators
using Test

const M = MultisymplecticIntegrators

include("../helpers/manufactured.jl")

# `Wave` at its defaults: `c = 0.5` on `[0, 1]`, the standing wave of period `2l/c = 4`. With `nx`
# space cells, `c·h/Δx = 0.5 · h · nx`.

# The `L²` errors of a run's top traces against the exact solution, one per step.
function trace_errors(problem, ansatz, result, h)
    nodes = space_quadrature_nodes(ansatz)
    weights = M.space_quadrature_weights(ansatz)
    return [l2_norm(weights,
                result.traces[n].u .-
                problem.solutions.u(n * h, nodes, parameters(problem)),
                1)
            for n in eachindex(result.traces)]
end

@testset "Long-time energy and the bound" begin
    @testset "ten periods of $name without drift" for (name, coupling, boundary) in COUPLINGS
        (; h, tend, nx, p) = LONG_TIME
        run = wave_run(coupling, boundary, p, h, nx, round(Int, tend / h))
        @test length(run.result.traces) == 800
        @test all_steps_converge(run.result, run.int)
        periods = energy_periods(slab_run_energies(run, h)..., h, tend)
        @test periods.last_period ≤ 2 * periods.first_period
        @info("long-time energy, $name, p = $p, nx = $nx, h = $h, t = $tend", periods...,
            run_metadata()...)
    end

    @testset "ten periods of the box scheme without drift" begin
        (; h, tend, box_xstep) = LONG_TIME
        run = box_wave_run(h, tend, box_xstep)
        @test length(run.result.traces) == 800
        @test all_steps_converge(run.result, run.int)
        periods = energy_periods(box_run_energies(run, h)..., h, tend)
        @test periods.last_period ≤ 2 * periods.first_period
        @info("long-time energy, box scheme, xstep = $box_xstep, h = $h, t = $tend",
            periods..., run_metadata()...)
    end

    @testset "below the bound, $name, p = $p" for (name, coupling, boundary) in COUPLINGS,
        p in (1, 2)
        # `c·h/Δx = 0.9`, 80 steps: the error stays at the size of a stable run's.
        run = wave_run(coupling, boundary, p, 0.09, 20, 80)
        @test length(run.result.traces) == 80
        @test all_steps_converge(run.result, run.int)
        errors = trace_errors(run.problem, run.ansatz, run.result, 0.09)
        largest = maximum(errors)
        @test largest ≤ 1e-2
        @info("below the bound, $name, p = $p, c·h/Δx = 0.9, 80 steps", largest,
            first=errors[1], ratio=largest/errors[1], run_metadata()...)
    end

    @testset "above the bound, $name, p = $p" for (name, coupling, boundary) in COUPLINGS,
        p in (1, 2)
        # `c·h/Δx = 4`, 40 steps: recorded, not bounded (KNOWN_ISSUES.md, the bound's entry).
        run = wave_run(coupling, boundary, p, 0.4, 20, 40)
        @test length(run.result.traces) == 40
        largest = maximum(trace_errors(run.problem, run.ansatz, run.result, 0.4))
        @info("above the bound, $name, p = $p, c·h/Δx = 4, 40 steps", largest,
            run_metadata()...)
    end
end
