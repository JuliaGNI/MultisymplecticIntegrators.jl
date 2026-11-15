using GeometricBase: parameters
using GeometricIntegratorsBase: GeometricIntegrator
using MultisymplecticIntegrators
using SimpleSplines: Dirichlet, Free
using Test

const M = MultisymplecticIntegrators

include("../helpers/manufactured.jl")

# `Wave` at its defaults: `c = 0.5` on `[0, 1]`, the standing wave of period `2l/c = 4`. With `nx`
# space cells, `c·h/Δx = 0.5 · h · nx`.
const COUPLINGS = (("CoefficientRestriction", CoefficientRestriction(), Dirichlet()),
    ("WeakInitialCondition", WeakInitialCondition(), Dirichlet()),
    ("WeakInitialAndBoundaryConditions", WeakInitialAndBoundaryConditions(), Free()))

const PERIOD = 4.0

# The energy of a trace on a rule: `Σ ωᵢ H(uᵢ, vᵢ, wᵢ)` over the rule's nodes.
function quadrature_energy(problem, weights, u, v, w)
    return sum(weights[i] *
               energy_density(problem.density, u[i], v[i], w[i], problem.params)
    for i in eachindex(weights))
end

# The energy errors `|E_h(t_n) − E_*(t_n)| / E_*(0)` of a series of run energies `Eh` against the
# exact energies `Ex`, whose first entry is the energy at `t = 0`: the largest error over the first
# period `(0, 4]`, the largest over the last `[tend − 4, tend]`, and their ratio.
function energy_periods(Eh, Ex, h, tend)
    errors = abs.(Eh .- Ex[2:end]) ./ Ex[1]
    times = h .* (1:length(errors))
    first_period = maximum(errors[times .≤ PERIOD + h / 2])
    last_period = maximum(errors[times .≥ tend - PERIOD - h / 2])
    return (; first_period, last_period, ratio = last_period / first_period)
end

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

function wave_run(coupling, boundary, p, h, nx, steps)
    problem = Wave.problem(timespan = (0.0, steps * h), timestep = h, xstep = 1 / nx)
    ansatz = TensorBSplineAnsatz(degree = p, ncells = (1, nx), xspan = (0.0, 1.0),
        x_boundary = boundary)
    int = GeometricIntegrator(problem, SpaceTimeGalerkin(ansatz, coupling))
    return (; problem, ansatz, int, result = integrate(int))
end

@testset "Long-time energy and the bound" begin
    @testset "ten periods of $name without drift" for (name, coupling, boundary) in COUPLINGS
        h, tend, nx = 0.05, 40.0, 20
        run = wave_run(coupling, boundary, 2, h, nx, 800)
        (; problem, ansatz, result) = run
        @test length(result.traces) == 800
        @test all_steps_converge(result, run.int)
        nodes = space_quadrature_nodes(ansatz)
        weights = M.space_quadrature_weights(ansatz)
        Eh = [quadrature_energy(problem, weights, trace.u, trace.v, trace.w)
              for trace in result.traces]
        params = parameters(problem)
        Ex = [quadrature_energy(
                  problem, weights, problem.solutions.u(n * h, nodes, params),
                  problem.solutions.v(n * h, nodes, params),
                  problem.solutions.w(n * h, nodes, params))
              for n in 0:800]
        periods = energy_periods(Eh, Ex, h, tend)
        @test periods.last_period ≤ 2 * periods.first_period
        @info("long-time energy, $name, p = 2, nx = $nx, h = $h, t = $tend", periods...,
            run_metadata()...)
    end

    @testset "ten periods of the box scheme without drift" begin
        h, tend = 0.05, 40.0
        problem = Wave.problem(timespan = (0.0, tend), timestep = h, xstep = 0.05)
        int = GeometricIntegrator(problem, BoxScheme())
        result = integrate(int)
        @test length(result.traces) == 800
        @test all_steps_converge(result, int)
        grid = collect(xgrid(problem))
        params = parameters(problem)
        exact = [(u = problem.solutions.u(n * h, grid, params),
                     v = problem.solutions.v(n * h, grid, params),
                     w = problem.solutions.w(n * h, grid, params)) for n in 0:800]
        periods = energy_periods(
            M.box_energies(problem, result.traces), M.box_energies(problem, exact), h, tend)
        @test periods.last_period ≤ 2 * periods.first_period
        @info("long-time energy, box scheme, xstep = 0.05, h = $h, t = $tend", periods...,
            run_metadata()...)
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
