using MultisymplecticIntegrators
using MultisymplecticIntegrators: space_quadrature_nodes, space_quadrature_weights
using Test

include("helpers/testsetup.jl")

# Five steps of the wave equation on the ansatz of `test/ansatz/tensor_bspline.jl`.
function five_step_wave(::Type{T}) where {T}
    problem = Wave.problem(T = T, timespan = (zero(T), 5 * T(0.05)), timestep = T(0.05),
        xstep = T(0.05))
    ansatz = TensorBSplineAnsatz(degree = 2, ncells = (2, 20), xspan = (zero(T), one(T)),
        n = 3)
    method = SpaceTimeGalerkin(ansatz, CoefficientRestriction())
    return problem, method, integrate(problem, method)
end

# The problem whose density couples the two components (`VectorWave.vector_density`) over the data
# of `VectorWave.problem`: the same initial, boundary and exact data, the same spans and step, so
# that the two problems solve the same field through two densities.
function coupled_problem(::Type{T}; timespan = (zero(T), T(0.5)), timestep = T(0.1),
        xstep = T(0.05)) where {T}
    base = VectorWave.problem(T = T, timespan = timespan, timestep = timestep, xstep = xstep)
    return LagrangianPDEProblem(VectorWave.vector_density(; params = base.params, D = 2);
        D = 2, ics_function = base.ics_function, bcs_function = base.bcs_function,
        solutions = base.solutions,
        timespan = base.timespan, timestep = base.timestep, xspan = base.xspan,
        xstep = base.xstep, params = base.params)
end

# A density of the test's own type, of one component: a density that is not a `LagrangianDensity`
# is handed the entry of a node rather than a container, and its own `derivatives` is what closes
# the inner product of `energy_density`.
struct QuadraticDensity{T}
    k::T
end

(density::QuadraticDensity)(u, v, w, params) = density.k * v^2 / 2

function MultisymplecticIntegrators.derivatives(density::QuadraticDensity, u, v, w, params)
    (u = zero(u), v = density.k * v, w = zero(w))
end

@testset "Diagnostics" begin
    @testset "S2a the energy density is the density's Hamiltonian" begin
        problem = Wave.problem()
        density = problem.density
        c = problem.params.c
        for (u, v, w) in ((0.3, -0.7, 1.1), (-1.5, 0.25, -0.4), (zero(Float64), 2.0, 0.5))
            # `v ∂L/∂v − L` for `L = ½v² − ½c²w²`, and the density's own accessor
            @test energy_density(density, u, v, w, problem.params) ≈
                  v * derivative_v(density, u, v, w, problem.params) -
                  density(u, v, w, problem.params)
            @test energy_density(density, u, v, w, problem.params) ≈
                  (v^2 + c^2 * w^2) / 2
        end

        # A mass adds `½m²u²` to the energy, which is what the density's own value does not show:
        # the difference between the Hamiltonian and the density is the term the formula keeps.
        massive = Wave.density(; params = problem.params, m = 0.4)
        @test energy_density(massive, 0.3, -0.7, 1.1, problem.params) ≈
              (0.7^2 + (c * 1.1)^2 + (0.4 * 0.3)^2) / 2

        # A vector density sums the components' energies: `v` and `∂L/∂v` are vectors of the `D`
        # components at one point, and the product is their inner product.
        params = VectorWave.default_parameters()
        vector = VectorWave.vector_density(; params = params, D = 2)
        u, v, w = [0.3, -0.2], [-0.7, 0.4], [1.1, 0.5]
        scalar = Wave.density(; params = params)
        @test energy_density(vector, u, v, w, params) ≈
              sum(energy_density(scalar, u[i], v[i], w[i], params) for i in eachindex(u))
    end

    @testset "S2a the energy error of a five-step wave run" begin
        problem, method, result = five_step_wave(Float64)
        err = energy_error(problem, method, result.traces)

        # One value per stored step, of the problem's element type, and none of them negative.
        @test length(err) == length(result.traces) == 5
        @test eltype(err) === Float64
        @test all(isfinite, err)
        @test all(≥(0), err)

        # The deviation is relative to the first stored step, so it is zero there to round-off.
        @test err[1] == 0
        @test abs(err[1]) ≤ eps(Float64)

        # The energy is the conserved quantity of the wave equation, so the deviation stays at the
        # discretisation level of the run rather than drifting with it. This is the oracle: a
        # quantity the density defines, not a second run of the code under test.
        @test maximum(err) < 1e-3

        # The ansatz's quadrature is the rule the trace is stored on, and its own weights integrate
        # the space interval: a wrong measure would show here before it showed in the deviation.
        @test sum(MultisymplecticIntegrators.space_quadrature_weights(
            ansatz(method))) ≈ 1.0

        # An ansatz of another resolution is not the rule the trace was stored on, and the energy
        # would be integrated against the wrong nodes.
        coarse = TensorBSplineAnsatz(degree = 2, ncells = (2, 10), xspan = (0.0, 1.0), n = 3)
        @test_throws ArgumentError energy_error(problem, coarse, result.traces)

        # The same run through the ansatz, which is the signature the driver holds.
        @test energy_error(problem, ansatz(method), result.traces) == err
    end

    @testset "S2a a run without a trace, and an energy that vanishes" begin
        problem, method, result = five_step_wave(Float64)
        nq = length(result.traces[1].u)

        @test_throws ArgumentError energy_error(problem, method, [nothing])
        @test_throws ArgumentError energy_error(problem, method, Any[])
        # A trace that vanishes has no relative scale, and the deviation is the absolute one
        # rather than a `NaN` from `0 / 0`.
        zero_trace = (u = zeros(nq), v = zeros(nq), w = zeros(nq))
        @test energy_error(problem, method, [zero_trace, zero_trace]) == [0.0, 0.0]
    end

    @testset "S2a the quadrature the trace is stored on" begin
        # The energy of a step is the density integrated at the nodes of the ansatz's space
        # quadrature, so the rule there is part of the diagnostic: it has to integrate the exact
        # wave Lagrangian and its Hamiltonian to round-off, against the closed forms of the
        # standing wave, and not merely be a rule of the right total weight.
        problem, method, _ = five_step_wave(Float64)
        a = ansatz(method)
        nodes = collect(space_quadrature_nodes(a))
        weights = MultisymplecticIntegrators.space_quadrature_weights(a)
        p = problem.params
        c = p.c
        ω = π * c / p.l
        k = π / p.l

        @test sum(weights) ≈ problem.xspan[2] - problem.xspan[1]

        for t in (0.0, 0.3)
            u = Wave.solution_u.(t, nodes, Ref(p))
            v = Wave.solution_v.(t, nodes, Ref(p))
            w = Wave.solution_w.(t, nodes, Ref(p))
            # the two coefficients of the standing wave at this time, and the two closed forms
            A = -p.A1 * sin(ω * t) + p.B1 * cos(ω * t + π / 6)
            B = p.A1 * cos(ω * t) + p.B1 * sin(ω * t + π / 6)
            energy = (ω^2 * A^2 + c^2 * k^2 * B^2) * p.l / 4
            lagrangian = (ω^2 * A^2 - c^2 * k^2 * B^2) * p.l / 4

            @test sum(weights[j] * energy_density(problem.density, u[j], v[j], w[j], p)
            for j in eachindex(nodes)) ≈ energy rtol = 32 * eps(Float64)
            @test sum(weights[j] * problem.density(u[j], v[j], w[j], p)
            for j in eachindex(nodes)) ≈ lagrangian rtol = 32 * eps(Float64)
        end
    end

    @testset "S2a a run of two components" begin
        # `D = 2`: each component is its own scalar wave problem, and the trace of a step is the two
        # components at every node. The energy is the sum over the components, so the deviation is
        # that of the whole field and not of its first component.
        params = VectorWave.default_parameters()
        problem = VectorWave.problem(
            params = params, timespan = (0.0, 0.1), timestep = 0.05,
            xstep = 0.05)
        ansatz2 = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0), n = 3)
        method2 = SpaceTimeGalerkin(ansatz2, CoefficientRestriction())
        result2 = integrate(problem, method2)
        nq = length(space_quadrature_nodes(ansatz2))
        @test components(problem) == 2
        @test length(result2.traces) == 2
        @test length(result2.traces[1].u) == 2 * nq
        err = energy_error(problem, method2, result2.traces)
        @test length(err) == 2
        @test err[1] == 0
        @test all(isfinite, err)
        @test maximum(err) < 1e-2

        # The plotting layer's data: one matrix per component, of the step times against the output
        # grid, and component 2 is the second block of the state rather than the first again.
        t, x, u = solution_data(problem, result2)
        @test length(t) == 3
        @test length(x) == length(MultisymplecticIntegrators.xgrid(problem))
        @test length(u) == 2
        @test size(u[1]) == (3, length(x))
        exact2 = Wave.solution_u.(0.0, x, Ref(VectorWave.component_parameters(params, 2)))
        @test maximum(abs.(u[2][1, :] .- exact2)) ≤ 16 * eps(Float64) * maximum(abs, exact2)
    end

    @testset "S2a the data of a run that stopped early" begin
        # `integrate` ends a run at a NaN or a failed solve, and its traces then end before the
        # output grid does. The steps it never computed are not data: the times and the field stop
        # at the last stored trace, and the field there is the solution's own.
        problem, _, result = five_step_wave(Float64)
        stopped = (; solution = result.solution, traces = result.traces[1:2])
        t, x, u = solution_data(problem, stopped)
        @test t == [result.solution.t[n] for n in 0:2]
        @test size(only(u)) == (3, length(x))
        @test only(u)[3, :] == result.solution.u[2]
    end

    @testset "S2a the deviation is the whole field's, and it is unsigned" begin
        # A hand-built pair of traces, so the expected deviation is arithmetic and not a second run:
        # the wave density is `½v²` per component at `w = 0`, the weights of the space rule sum to
        # the interval, and the energy of a trace is therefore known.
        params = VectorWave.default_parameters()
        problem = VectorWave.problem(
            params = params, timespan = (0.0, 0.1), timestep = 0.05,
            xstep = 0.05)
        ansatz2 = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0), n = 3)
        nq = length(space_quadrature_nodes(ansatz2))
        moving(v1, v2) = (u = zeros(2 * nq), v = vcat(fill(v1, nq), fill(v2, nq)),
            w = zeros(2 * nq))
        # both components moving: energy ½ (1 + 1) = 1, then ½ (1 + 0) = ½
        err = energy_error(problem, ansatz2, [moving(1.0, 1.0), moving(1.0, 0.0)])
        @test err[2] ≈ 0.5
        # the fall is reported as a positive deviation: the error is unsigned
        @test err[2] > 0
    end

    @testset "S2a a trace of the wrong shape is refused" begin
        # A trace is the `D` components at the `nq` nodes of the space quadrature, and the energy of
        # a step reads all three of its parts at every node. A trace of another shape is refused
        # rather than read at the nodes it happens to hold.
        params = VectorWave.default_parameters()
        problem = VectorWave.problem(
            params = params, timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05)
        ansatz2 = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0), n = 3)
        nq = length(space_quadrature_nodes(ansatz2))
        entries = 2 * nq

        # One entry more than the nodes carry: the floored count of components per node agrees with
        # the quadrature's length, so without a remainder check the surplus entry goes unread.
        surplus = (u = zeros(entries + 1), v = ones(entries + 1), w = zeros(entries + 1))
        @test_throws ArgumentError energy_error(problem, ansatz2, [surplus])

        # The three parts of a trace are read at the same node, so a `v` shorter than `u` or a `w`
        # longer than it is a malformed trace as well.
        @test_throws ArgumentError energy_error(problem, ansatz2,
            [(u = zeros(entries), v = ones(entries - 1), w = zeros(entries))])
        @test_throws ArgumentError energy_error(problem, ansatz2,
            [(u = zeros(entries), v = ones(entries), w = zeros(entries + 1))])

        # The scalar case refuses the same disagreement.
        problem1, method1, result1 = five_step_wave(Float64)
        trace1 = result1.traces[1]
        @test_throws ArgumentError energy_error(problem1, method1,
            [(u = trace1.u, v = trace1.v[1:(end - 1)], w = trace1.w)])

        # A well-shaped trace is still accepted: the guard refuses a shape and not the value.
        @test energy_error(problem, ansatz2,
            [(u = zeros(entries), v = ones(entries), w = zeros(entries))]) == [0.0]
    end

    @testset "S2a a density that couples the components" begin
        # A problem may carry a density that takes the `D` components of one point at once
        # (`VectorWave.vector_density`). Uncoupled it is the sum of the scalar densities, so the
        # energy error on the same traces must equal the one the scalar density gives — an oracle
        # by construction, and the only path that reads a trace node as a vector of components.
        problem = VectorWave.problem(timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05)
        coupled = coupled_problem(Float64; timespan = (0.0, 0.1), timestep = 0.05, xstep = 0.05)
        ansatz2 = TensorBSplineAnsatz(degree = 2, ncells = (1, 20), xspan = (0.0, 1.0), n = 3)
        nq = length(space_quadrature_nodes(ansatz2))
        # The profile varies from node to node, so a node read at the wrong index is a different
        # value, and the two traces carry profiles of different shape rather than one profile at
        # two amplitudes, so that a wrong index does not rescale both energies alike and cancel in
        # their ratio. `v` is the only non-zero part of these traces.
        profile = [1.0 + 0.5 * (j - 1) / (nq - 1) for j in 1:nq]
        moving(v1, v2) = (u = zeros(2 * nq), v = vcat(v1, v2), w = zeros(2 * nq))
        traces = [moving(profile, 0.5 .* profile), moving(profile .^ 2, 0.25 .* profile)]
        err = energy_error(coupled, ansatz2, traces)
        @test err ≈ energy_error(problem, ansatz2, traces)
        # The first trace's energy is ½(1 + 0.25) ∫profile², the second's ½ ∫(profile⁴ +
        # 0.0625 profile²), each over the ansatz's own space quadrature.
        weights = space_quadrature_weights(ansatz2)
        first_energy = 0.625 * sum(weights .* profile .^ 2)
        second_energy = 0.5 * sum(weights .* (profile .^ 4 .+ 0.0625 .* profile .^ 2))
        @test err[2] ≈ abs(second_energy - first_energy) / abs(first_energy)
    end

    @testset "S2a the energy of a vector trace allocates no per-node container" begin
        # A density that couples the components of a point is handed them as a tuple whose length
        # is the problem's `D` at compile time; a container built per node escapes the call and
        # grows with the node count. The reading is taken after one warm-up call inside a function
        # of fixed arity, so that what is measured is the call and not its first dispatch.
        allocations(problem, method, traces) = (energy_error(problem, method, traces);
            @allocated energy_error(problem, method, traces))
        for T in (Float64, Float32)
            problem = coupled_problem(T)
            for ncells in ((4, 12), (4, 40))
                ansatz = TensorBSplineAnsatz(degree = 3, ncells = ncells,
                    xspan = (zero(T), one(T)), n = 5)
                method = SpaceTimeGalerkin(ansatz)
                nq = length(space_quadrature_nodes(ansatz))
                profile = [one(T) + T(0.5) * (j - 1) / (nq - 1) for j in 1:nq]
                traces = [(u = zeros(T, 2 * nq),
                              v = vcat(a * profile, (a / 2) * profile),
                              w = zeros(T, 2 * nq))
                          for a in (one(T), T(0.9), T(0.5), T(0.25), T(0.1))]
                # `--check-bounds=yes` puts its checks inside the measured call, so the reading is
                # skipped where a run forces `--check-bounds=yes` and taken under `auto`.
                Base.JLOptions().check_bounds == 1 && continue
                bytes = allocations(problem, method, traces)
                @test bytes <= 512
            end
        end
    end

    @testset "S2a a scalar density of another type reads one value per node" begin
        # `node_values` hands a density that is not a `LagrangianDensity` the entry itself at
        # `D = 1` and a tuple of all `D` components above it. A density of a type of the test's own
        # therefore sees scalars, and its energy is the quadrature of its own density: for
        # `L = ½kv²` the energy density is `½kv²`, so `v = 2` against `v = 1` deviates by `3`.
        for T in (Float64, Float32)
            problem = LagrangianPDEProblem(QuadraticDensity(T(2)); D = 1,
                ics_function = (x, p) -> (u = zeros(T, length(x), 1),
                    v = zeros(T, length(x), 1), w = zeros(T, length(x), 1)),
                bcs_function = (t, p) -> nothing,
                timespan = (zero(T), one(T)), timestep = T(0.1),
                xspan = (zero(T), one(T)), xstep = T(0.1))
            ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
                xspan = (zero(T), one(T)), n = 3)
            nq = length(space_quadrature_nodes(ansatz))
            traces = [(u = zeros(T, nq), v = fill(one(T), nq), w = zeros(T, nq)),
                (u = zeros(T, nq), v = fill(T(2), nq), w = zeros(T, nq))]
            err = energy_error(problem, SpaceTimeGalerkin(ansatz), traces)
            @test eltype(err) === T
            @test err ≈ T[0, 3]
        end
    end

    @testset "S2a the box's midpoint rule of a scalar density of another type" begin
        # The box scheme reads its density at the mean of a box's two nodes, and a density that is
        # not a `LagrangianDensity` gets that mean state as a scalar at `D = 1`, as the ansatz's node
        # rule hands it one value per node. For `L = ½kv²` the energy density is `½kv²`, so the
        # energy of a trace is the box sum of `½k v̄²`, which the test builds itself.
        for T in (Float64, Float32)
            problem = LagrangianPDEProblem(QuadraticDensity(T(2)); D = 1,
                ics_function = (x, p) -> (u = zeros(T, length(x), 1),
                    v = zeros(T, length(x), 1), w = zeros(T, length(x), 1)),
                bcs_function = (t, p) -> nothing,
                timespan = (zero(T), one(T)), timestep = T(0.1),
                xspan = (zero(T), one(T)), xstep = T(0.1))
            nodes = collect(xgrid(problem))
            weights = box_quadrature_weights(problem)
            @test length(weights) == length(nodes) - 1
            # A profile that varies from node to node, so that a box's mean is no node's value.
            profile = [one(T) + T(0.5) * (j - 1) / (length(nodes) - 1)
                       for j in eachindex(nodes)]
            traces = [(u = zeros(T, length(nodes)), v = copy(profile),
                w = zeros(T, length(nodes)))]
            boxed = MultisymplecticIntegrators.box_energies(problem, traces)
            closed = sum(weights[j] * T(1) * ((profile[j] + profile[j + 1]) / 2)^2
            for j in eachindex(weights))
            @test isapprox(first(boxed), closed; rtol = 8 * eps(T))
            # One stored trace has no deviation from itself, and what `energy_error` reports is the
            # deviation of the series and not the energies themselves.
            @test energy_error(problem, BoxScheme(), traces) == [zero(T)]
        end
    end

    @testset "S2a a per-component density sums the components of a node" begin
        # `VectorWave.problem` carries a `LagrangianDensity`, which `node_energy_density` applies to
        # each component of a node in turn. The energy of a `D = 2` trace is the sum of the two
        # components' own quadratures, read here through the density's own `energy_density`.
        for T in (Float64, Float32)
            problem = VectorWave.problem(T = T, timespan = (zero(T), T(0.1)),
                timestep = T(0.05), xstep = T(0.05))
            ansatz = TensorBSplineAnsatz(degree = 2, ncells = (1, 20),
                xspan = (zero(T), one(T)), n = 3)
            nq = length(space_quadrature_nodes(ansatz))
            weights = space_quadrature_weights(ansatz)
            profile = [one(T) + T(0.5) * (j - 1) / (nq - 1) for j in 1:nq]
            trace = (u = vcat(fill(one(T), nq), fill(T(2), nq)),
                v = vcat(profile, T(2) .* profile), w = vcat(T(0.5) .* profile, profile))
            density, params = problem.density, problem.params
            energy_of(t) = sum(sum(weights[j] *
                                   energy_density(density, t.u[(d - 1) * nq + j],
                                       t.v[(d - 1) * nq + j], t.w[(d - 1) * nq + j], params)
                               for j in 1:nq)
            for d in 1:2)
            traces = [trace, merge(trace, (v = T(2) .* trace.v,))]
            first_energy, second_energy = energy_of(traces[1]), energy_of(traces[2])
            err = energy_error(problem, SpaceTimeGalerkin(ansatz), traces)
            @test err ≈ [zero(T), abs(second_energy - first_energy) / abs(first_energy)]
        end
    end

    @testset "S2a a run in reduced precision" begin
        problem = Wave.problem(T = Float32, timespan = (0.0f0, 0.1f0), timestep = 0.05f0,
            xstep = 0.05f0)
        ansatz32 = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 20), xspan = (0.0f0, 1.0f0),
            n = 3)
        nodes = collect(space_quadrature_nodes(ansatz32))
        ics = problem.ics_function(nodes, problem.params)
        # Two traces of a Float32 run: the initial data, and the initial data with a perturbed
        # velocity, which is a non-zero energy error of that precision's own scale.
        traces = [(u = vec(ics.u), v = vec(ics.v), w = vec(ics.w)),
            (u = vec(ics.u), v = vec(ics.v) .+ 0.5f0, w = vec(ics.w))]
        err = energy_error(problem, ansatz32, traces)
        @test eltype(err) === Float32
        @test length(err) == 2
        @test err[1] == 0.0f0
        @test err[2] > 16 * eps(Float32)
    end

    @testset "the error of a run against the exact solution" begin
        # A run built by hand: the exact solution at every stored step, and `n ε sin(πx)` added to
        # the last one. The error of that step is arithmetic — `n ε` at the node where the sine is
        # one — and the steps before it are zero, so the value comes from the closed form and not
        # from a second run of the code.
        #
        # The two-component problem's perturbation sits on component 2 alone, so a value read from
        # component 1 reports no error at all.
        function exact_run(problem, ε; component = 1)
            solution = MultisymplecticIntegrators.GeometricSolution(problem)
            nodes = collect(MultisymplecticIntegrators.xgrid(problem))
            nx = length(nodes)
            steps = MultisymplecticIntegrators.ntime(problem)
            for n in 1:steps
                t = solution.t[n]
                solution[n] = (u = problem.solutions.u(t, nodes, problem.params),
                    v = problem.solutions.v(t, nodes, problem.params),
                    w = problem.solutions.w(t, nodes, problem.params))
            end
            last = solution.t[steps]
            u = problem.solutions.u(last, nodes, problem.params)
            for j in 1:nx
                u[(component - 1) * nx + j] += eltype(u)(steps) * ε * sinpi(nodes[j])
            end
            solution[steps] = (u = u,
                v = problem.solutions.v(last, nodes, problem.params),
                w = problem.solutions.w(last, nodes, problem.params))
            return (; solution, traces = fill(nothing, steps))
        end

        for T in (Float32, Float64)
            ε = T(1) / 20
            steps = 5
            scalar = Wave.problem(
                T = T, timespan = (zero(T), T(1) / 4), timestep = T(1) / 20,
                xstep = T(1) / 20)

            err = solution_error(scalar, exact_run(scalar, ε))
            @test eltype(err) === T
            @test length(err) == steps
            @test all(iszero, err[1:(end - 1)])
            @test isapprox(err[end], T(steps) * ε; rtol = 8 * eps(T))

            vector = VectorWave.problem(T = T, timespan = (zero(T), T(1) / 4),
                timestep = T(1) / 20, xstep = T(1) / 20)
            err2 = solution_error(vector, exact_run(vector, ε; component = 2))
            @test length(err2) == steps
            @test all(iszero, err2[1:(end - 1)])
            @test isapprox(err2[end], T(steps) * ε; rtol = 8 * eps(T))

            # A problem without an exact solution has no error to report, and says so rather than
            # returning a value of its own: the matrix measures such a run against a finer one.
            base = scalar
            unsolved = LagrangianPDEProblem(base.density; D = base.D,
                ics_function = base.ics_function, bcs_function = base.bcs_function,
                timespan = base.timespan, timestep = base.timestep, xspan = base.xspan,
                xstep = base.xstep, params = base.params)
            @test unsolved.solutions === nothing
            @test solution_error(unsolved, exact_run(scalar, ε)) === nothing
            # A call outside the method's signature is a `MethodError`, and not `Base.error` printing
            # its arguments.
            @test_throws MethodError solution_error(scalar, scalar, exact_run(scalar, ε))
        end
    end

    @testset "the rotation charge of a density with and without the symmetry" begin
        # `VectorWave`'s density is a function of the components' lengths, so a rotation of the two
        # components leaves it invariant and `∫ (u₁ v₂ − u₂ v₁)` is the run's conserved quantity.
        # `LinearTransport`'s doubled field has two components and no such symmetry: its density
        # reads one component against the other's velocity and slope. Whether the charge exists is
        # decided on the density's values, so the component count is not what tells the two apart,
        # and the answer is `nothing` for a density without the symmetry and a series for the other.
        T = Float64
        span = (zero(T), T(1) / 10)
        vector = VectorWave.problem(T = T, timespan = span, timestep = T(1) / 20,
            xstep = T(1) / 20)
        transport = LinearTransport.problem(T = T, timespan = span, timestep = T(1) / 20,
            xspan = (T(-1) / 2, zero(T)), xstep = T(1) / 20)
        ansatz = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 10), xspan = (zero(T), one(T)),
            n = 3)
        method = SpaceTimeGalerkin(ansatz, CoefficientRestriction())
        splined = integrate(vector, method)
        charge = noether_charge(vector, method, splined.traces)
        @test charge isa Vector{T}
        @test length(charge) == length(splined.traces)
        @test all(isfinite, charge)
        @test abs(first(charge)) > eps(T)
        # The box scheme keeps one convention for its energy and its charge: the charge of its run
        # is integrated by the midpoint rule over its boxes, so it too is a series.
        boxed = integrate(vector, BoxScheme())
        boxed_charge = noether_charge(vector, BoxScheme(), boxed.traces)
        @test boxed_charge isa Vector{T}
        @test length(boxed_charge) == length(boxed.traces)
        @test all(isfinite, boxed_charge)
        @test abs(first(boxed_charge)) > eps(T)
        # A density without the symmetry answers `nothing` under either rule, and the trace is never
        # read for it: the decision comes first.
        for problem in (
            Wave.problem(T = T, timespan = span, timestep = T(1) / 20,
            xstep = T(1) / 20),
            transport)
            @test noether_charge(problem, method, [nothing]) === nothing
            @test noether_charge(problem, BoxScheme(), [nothing]) === nothing
        end
    end

    @testset "the network ansatze's space weights" begin
        # A network stores its trace at the nodes of its trace basis' space rule, and a
        # fixed-boundary network at those of its own; each rule's weights sum to the span, and the
        # fixed-boundary rule, Gauss on each cell, integrates `x²` exactly.
        for T in (Float64, Float32), D in (1, 2)

            xspan = (T(-1) / 2, one(T))
            length_of_span = xspan[2] - xspan[1]
            network = NetworkAnsatz(
                S = 4, xspan = xspan, ncells = (2, 5), degree = 3, D = D)
            weights = space_quadrature_weights(network)
            @test eltype(weights) === T
            @test weights == space_quadrature_weights(trace_basis(network))
            @test isapprox(sum(weights), length_of_span; rtol = 8 * eps(T))
            for fixed in (FixedBoundaryNetwork(xspan = xspan, D = D),
                FixedBoundaryNetwork(
                Sx = 4, St = 4, xspan = xspan, ncells = 7, nq = (3, 4),
                D = D))
                fixed_weights = space_quadrature_weights(fixed)
                nodes = fixed.tabulation.xnode
                @test eltype(fixed_weights) === T
                @test length(fixed_weights) == fixed.ncells * fixed.nq[2] == length(nodes)
                @test all(>(zero(T)), fixed_weights)
                @test isapprox(sum(fixed_weights), length_of_span; rtol = 8 * eps(T))
                @test isapprox(sum(fixed_weights .* nodes .^ 2),
                    (xspan[2]^3 - xspan[1]^3) / 3; rtol = 64 * eps(T))
            end
        end
    end

    @testset "the network ansatze's energy error" begin
        # `Wave`'s density at `u = w = 0` is `½ v²` per component, so a trace of constant velocity
        # `c` on `[0, 1]` has the energy `½ D c²`, and two traces at `c₀` and `c₁` deviate by
        # `|c₁² − c₀²| / c₀²`; a first trace at rest has no scale, and the second's deviation is its
        # absolute energy.
        for T in (Float64, Float32), D in (1, 2)

            problem = (D == 1 ? Wave : VectorWave).problem(T = T,
                timespan = (zero(T), T(1) / 5), timestep = T(1) / 10,
                xspan = (zero(T), one(T)), xstep = T(1) / 20)
            @test problem.D == D
            c₀, c₁ = one(T), T(3) / 2
            unit = (zero(T), one(T))
            for ansatz in (NetworkAnsatz(
                S = 4, xspan = unit, ncells = (2, 5), degree = 3, D = D),
                FixedBoundaryNetwork(xspan = unit, D = D),
                FixedBoundaryNetwork(Sx = 4, St = 4, xspan = unit, ncells = 7, nq = (3, 4),
                D = D))
                n = D * length(space_quadrature_weights(ansatz))
                trace(c, m = n) = (u = zeros(T, m), v = fill(c, m), w = zeros(T, m))
                drift = energy_error(problem, ansatz, [trace(c₀), trace(c₁)])
                @test drift isa Vector{T}
                @test length(drift) == 2
                @test iszero(drift[1])
                @test isapprox(drift[2], abs(c₁^2 - c₀^2) / c₀^2; rtol = 64 * eps(T))
                at_rest = energy_error(problem, ansatz, [trace(zero(T)), trace(c₁)])
                @test isapprox(at_rest[2], D * c₁^2 / 2; rtol = 64 * eps(T))
                @test_throws ArgumentError energy_error(problem, ansatz, typeof(trace(c₀))[])
                @test_throws ArgumentError energy_error(problem, ansatz, [trace(c₀, n + D)])
                # the method's energy is its ansatz's
                method = SpaceTimeGalerkin(ansatz,
                    ansatz isa NetworkAnsatz ? WeakInitialAndBoundaryConditions() :
                    FixedBoundary())
                @test energy_error(problem, method, [trace(c₀), trace(c₁)]) == drift
            end
        end
    end

    @testset "the rotation charge of a network's two-component trace" begin
        # `u₁ = a`, `v₂ = b` and every other entry zero give `∫ (u₁ v₂ − u₂ v₁) dx = a b` on
        # `[0, 1]`, under each network's own space rule.
        for T in (Float64, Float32)
            problem = VectorWave.problem(T = T, timespan = (zero(T), T(1) / 5),
                timestep = T(1) / 10, xspan = (zero(T), one(T)), xstep = T(1) / 20)
            unit = (zero(T), one(T))
            a, b = T(3) / 4, T(-2)
            for method in (
                SpaceTimeGalerkin(
                NetworkAnsatz(S = 4, xspan = unit, ncells = (2, 5), degree = 3, D = 2),
                WeakInitialAndBoundaryConditions()),
                SpaceTimeGalerkin(FixedBoundaryNetwork(xspan = unit, D = 2), FixedBoundary()))
                nq = length(space_quadrature_weights(ansatz(method)))
                trace = (u = vcat(fill(a, nq), zeros(T, nq)),
                    v = vcat(zeros(T, nq), fill(b, nq)),
                    w = zeros(T, 2nq))
                charge = noether_charge(problem, method, [trace])
                @test charge isa Vector{T}
                @test isapprox(only(charge), a * b; rtol = 8 * eps(T))
            end
        end
    end
end
