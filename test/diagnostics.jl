using MultisymplecticIntegrators
using MultisymplecticIntegrators: space_quadrature_nodes
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
            u = Wave.exact_u.(t, nodes, Ref(p))
            v = Wave.exact_v.(t, nodes, Ref(p))
            w = Wave.exact_w.(t, nodes, Ref(p))
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
        traces = [(u = ics.u, v = ics.v, w = ics.w),
            (u = ics.u, v = ics.v .+ 0.5f0, w = ics.w)]
        err = energy_error(problem, ansatz32, traces)
        @test eltype(err) === Float32
        @test length(err) == 2
        @test err[1] == 0.0f0
        @test err[2] > 16 * eps(Float32)
    end
end
