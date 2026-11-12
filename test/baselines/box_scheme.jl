using GeometricBase: ntime
using MultisymplecticIntegrators
using MultisymplecticIntegrators: box_grid, xgrid
using Test

include("../helpers/testsetup.jl")

# The two constant matrices of the first-order form, written here as the form states them: the test
# builds the two-form of a run from the form, and never from the method's code.
const FIRST_ORDER_K = [0 -1 0; 1 0 0; 0 0 0]
const FIRST_ORDER_L = [0 0 -1; 0 0 0; 1 0 0]

function builder(name)
    name === "Wave" ? Wave.problem :
    name === "VectorWave" ? VectorWave.problem :
    name === "SineGordon" ? SineGordon.problem :
    throw(ArgumentError("the box scheme has no example problem `$(name)`"))
end

"""
    box_run(name; T, h, dx, span, keywords...)

One run of the box scheme on an example problem of its class, at the resolution the arguments give.
"""
function box_run(name; T = Float64, h = 0.05, dx = 0.05, span = 0.2, keywords...)
    problem = builder(name)(T = T, timespan = (zero(T), T(span)), timestep = T(h),
        xspan = (zero(T), one(T)), xstep = T(dx))
    return problem, integrate(problem, BoxScheme(); keywords...)
end

# The error of one run against the problem's exact solution at the last step it stored, which is the
# closed form's own value on the output grid.
function run_error(problem, result)
    nodes = collect(xgrid(problem))
    last_step = length(result.traces)
    closed = problem.solutions.u(result.solution.t[last_step], nodes, problem.params)
    return maximum(abs, result.solution.u[last_step] .- closed)
end

"""
    observed_order(errors, factor)

The observed orders of a ladder of errors whose resolutions differ by `factor`:
`log(errors[i] / errors[i + 1]) / log(factor)`.
"""
function observed_order(errors, factor)
    [log(errors[i] / errors[i + 1]) / log(factor) for i in 1:(length(errors) - 1)]
end

"""
    massive_problem(; T, A1, A2, h, dx)

The wave problem whose density carries a potential: the density of `Wave` with a mass term, on the
initial, boundary and exact data of `Wave`'s standing wave.

The equation stays linear, so that the difference of two of its runs is a solution of the scheme's
linearisation, and the potential is what makes the discrete conservation law see where `∇S` is
evaluated: a massless wave has `∇S = 0` in its `u` row and no such term at all.
"""
function massive_problem(; T = Float64, A1 = 0.8, A2 = 0.0, h = 0.05, dx = 0.05)
    base = Wave.problem(T = T,
        params = (c = T(1) / 2, A1 = T(A1), A2 = T(A2), B1 = T(8) / 10, B2 = zero(T),
            l = one(T)),
        timespan = (zero(T), 2 * T(h)), timestep = T(h), xspan = (zero(T), one(T)),
        xstep = T(dx))
    density = Wave.density(; params = base.params, m = T(1) / 2)
    return LagrangianPDEProblem(density; D = 1, ics_function = base.ics_function,
        bcs_function = base.bcs_function, solutions = base.solutions,
        timespan = base.timespan, timestep = base.timestep, xspan = base.xspan,
        xstep = base.xstep, params = base.params)
end

# The difference of two runs' traces at one stored step.
function trace_difference(a, b)
    return (u = a.u .- b.u, v = a.v .- b.v, w = a.w .- b.w)
end

# The three components of a trace at one node, the third as the first-order form's own: the state is
# `(u, v, w)` with `w = −c² ∂ₓu`, and the trace carries the physical slope `∂ₓu`.
corner(trace, j, c²) = (trace.u[j], trace.v[j], -c² * trace.w[j])

"""
    box_balance(U, V, problem, box, c²) -> (; residual, scale)

The residual of the discrete multisymplectic conservation law `δₜω + δₓκ = 0` in one box, with
`ω = UᵀKV` and `κ = UᵀLV`, and the scale it is measured against.

The two-forms are those of the perturbations averaged over the box: `ω` of the mean of a box's two
space nodes and `κ` of the mean of its two time levels. That pairing is what the box difference of a
product telescopes into exactly, so the statement holds to round-off and not to the scheme's order.
`U` and `V` are two solutions of the linearisation, each a named tuple `(old, new)` of the traces of
the two levels a step spans.
"""
function box_balance(U, V, problem, box, c²)
    h = problem.timestep
    dx = problem.xstep
    at(P, level, j) = corner(getfield(P, level), j, c²)
    mean3(a, b) = ntuple(k -> (a[k] + b[k]) / 2, 3)
    form(a, b, M) = sum(a[i] * M[i, j] * b[j] for i in 1:3, j in 1:3)
    ω(level) = form(mean3(at(U, level, box), at(U, level, box + 1)),
        mean3(at(V, level, box), at(V, level, box + 1)), FIRST_ORDER_K)
    κ(j) = form(mean3(at(U, :old, j), at(U, :new, j)),
        mean3(at(V, :old, j), at(V, :new, j)), FIRST_ORDER_L)
    time_term = (ω(:new) - ω(:old)) / h
    space_term = (κ(box + 1) - κ(box)) / dx
    return (; residual = time_term + space_term, scale = abs(time_term) + abs(space_term))
end

@testset "Box scheme" begin
    @testset "observed order 2 in h and Δx" begin
        # The ladders, and why they are the ones they are: the time step is refined on a grid whose
        # own error is two orders below the coarsest time step's, and the space step on a time step
        # whose error is below the finest space step's. Each run is measured against the problem's
        # exact solution at its final time.
        time_steps = (0.2, 0.1, 0.05)
        space_steps = (0.05, 0.025, 0.0125)
        for name in ("Wave", "VectorWave", "SineGordon")
            in_time = [run_error(box_run(name; h = h, dx = 0.00625, span = 0.2)...)
                       for h in time_steps]
            in_space = [run_error(box_run(name; h = 0.0025, dx = dx, span = 0.05)...)
                        for dx in space_steps]
            for order in observed_order(in_time, 2)
                @test 1.9 ≤ order ≤ 2.1
            end
            for order in observed_order(in_space, 2)
                @test 1.9 ≤ order ≤ 2.1
            end
        end
    end

    @testset "the discrete multisymplectic conservation law per box, $T" for T in (Float32, Float64)
        # Three runs of one linear problem whose potential is not zero. Their data differ, so two
        # differences of them are two solutions of the scheme's linearisation, and the law is
        # asserted on them and not on the method's own record of it.
        h = T(1) / 20
        dx = T(1) / 20
        base = integrate(massive_problem(; T = T, A1 = 0.8, h = h, dx = dx), BoxScheme())
        shifted = integrate(massive_problem(; T = T, A1 = 0.3, h = h, dx = dx), BoxScheme())
        second = integrate(massive_problem(; T = T, A2 = 0.5, h = h, dx = dx), BoxScheme())
        @test all(base.stats.converged)
        @test all(shifted.stats.converged)
        @test all(second.stats.converged)
        problem = massive_problem(; T = T, A1 = 0.8, h = h, dx = dx)
        U = (old = trace_difference(base.traces[end - 1], shifted.traces[end - 1]),
            new = trace_difference(base.traces[end], shifted.traces[end]))
        V = (old = trace_difference(base.traces[end - 1], second.traces[end - 1]),
            new = trace_difference(base.traces[end], second.traces[end]))
        # The density of that problem is `½v² − ½c²w² − ½m²u²` with `c = 1/2`, so the first-order
        # form's coefficient is `c² = 1/4`.
        for box in 1:box_grid(problem).boxes
            balance = box_balance(U, V, problem, box, T(1) / 4)
            @test balance.scale > zero(T)
            @test abs(balance.residual) ≤ 100 * eps(T) * balance.scale
        end
    end

    @testset "a density outside the class" begin
        transport = LinearTransport.problem(timespan = (0.0, 0.2), timestep = 0.1,
            xspan = (0.0, 1.0), xstep = 0.05)
        thrown = try
            integrate(transport, BoxScheme())
            nothing
        catch exception
            exception
        end
        @test thrown isa ArgumentError
        # The refusal names the partial derivative that disqualified the density: the doubled
        # field's `∂L/∂v` is not one `λ` for both components.
        @test occursin("∂L/∂v", sprint(showerror, thrown))
        for name in ("Wave", "VectorWave", "SineGordon")
            problem, result = box_run(name; h = 0.1, dx = 0.05, span = 0.2)
            @test length(result.traces) == ntime(problem)
        end
    end

    @testset "no silent upcast" begin
        # A `Float32` problem integrated for two steps: every array of the run is `Float32`, the
        # trace, the solution and the solve's record with it.
        T = Float32
        problem, result = box_run(
            "Wave"; T = T, h = T(1) / 20, dx = T(1) / 20, span = T(1) / 10)
        @test eltype(result.solution.u[1]) === T
        @test eltype(result.solution.v[1]) === T
        @test eltype(result.solution.w[1]) === T
        foreach(trace -> assert_no_upcast(trace, T), result.traces)
        @test eltype(result.stats.residuals) === T
        @test eltype(result.stats.initial_residuals) === T
        # The rule the energy is integrated by is an array of the run's too, and a rule of another
        # element type would promote every energy it integrates.
        @test eltype(box_quadrature_weights(problem)) === T
    end

    @testset "the box scheme's stats" begin
        # The solve's record is the run's own: an iteration cap of one leaves the sine-Gordon steps
        # unconverged, and the record says so rather than reporting a constant `true`.
        T = Float64
        _, capped = box_run("SineGordon"; h = T(1) / 20, dx = T(1) / 20, span = T(1) / 10,
            max_iterations = 1, f_reltol = 100 * eps(T))
        @test length(capped.stats.converged) == length(capped.traces)
        @test any(!, capped.stats.converged)
        @test all(isfinite, capped.stats.residuals)
        @test all(isfinite, capped.stats.initial_residuals)
        _, solved = box_run("SineGordon"; h = T(1) / 20, dx = T(1) / 20, span = T(1) / 10)
        @test all(solved.stats.converged)
        @test length(solved.stats.iterations) == length(solved.traces)
    end
end
