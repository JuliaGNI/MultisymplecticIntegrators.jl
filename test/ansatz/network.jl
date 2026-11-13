using GeometricBase
using GeometricIntegratorsBase
using GeometricSolutions
using LinearAlgebra: BLAS, norm
using MultisymplecticIntegrators
using Random
using SimpleSolvers
using SimpleSplines
using Test

import GeometricIntegratorsBase: GeometricIntegrator, cache, current, nlsolution, reset!,
                                 solutionstep, solver, integrate!
import SimpleSolvers: jacobian!, jacobianmatrix

include("../helpers/testsetup.jl")
include("../helpers/reference_values.jl")
include("../helpers/action_oracle.jl")

const M = MultisymplecticIntegrators

# The deviation of a network run from its §A entry is a record, not an assertion: at `S = 10` the
# `:last_layer` runs have sixteen constraint rows that reach only ten output weights, and the Newton
# solve of the `:all` runs does not reach round-off (K38), so every step of these runs is an
# unconverged Newton iterate, and no bound on its distance from the entry holds across machines.
# Each test asserts that the deviation is finite and writes it with `@info`, beside the machine that
# produced it; K38–K40 and K42 record the figures of one machine.
#
# Neither entry reproduces bit for bit: the code that wrote them reads the bottom points in the
# normalized coordinate (`xspan` is `(0, 1)`, so there the two coincide) and keeps a duplicate atom
# in its dictionary, which costs the greedy a whole selection.

"""The steps of the §A case."""
const NETWORK_STEPS = 5

"""The most Newton iterations the affine control of a `:last_layer` step may take."""
const LAST_LAYER_ITERATIONS = 2

"""The hidden units of the affine control, which frees its output weights alone."""
const NETWORK_CONTROL_S = 20

"""
The relative tolerance of the network family's steps, measured on the affine control of `:last_layer`
at `S = 20`: the largest of its five step ratios `residual / initial residual` is `7.8e-6` in
`Float64` and `6.4e-3` in `Float32`, against the tolerance this function returns.

The control's steps are affine in its unknowns — the coupling's constraint rows reach the free
output weights alone and the hidden layer, which the fit fixed, enters the residual as data — so two
Newton iterations reach the root, and the ratio is the solver's own round-off. `f_abstol = 0`, so no
step converges on an absolute term.
"""
network_f_reltol(::Type{Float64}) = 1.0e-4
network_f_reltol(::Type{Float32}) = 1.0e-2

function network_ansatz(::Type{T}; S = 10, D = 1, mode = :all, degree = 3, ncells = (2, 5),
        xspan = (zero(T), one(T)), quadrature_order = nothing) where {T}
    return NetworkAnsatz(S = S, D = D, mode = mode, xspan = xspan, degree = degree,
        ncells = ncells, quadrature_order = quadrature_order)
end

function wave_problem(::Type{T}; timespan = (zero(T), T(0.25)), timestep = T(0.05),
        xstep = T(0.05), D = 1) where {T}
    return Wave.problem(
        T = T, D = D, timespan = timespan, timestep = timestep, xstep = xstep)
end

"""A whole run of one network mode on `Wave`: five steps of `h = 0.05`."""
function network_run(::Type{T}; mode = :all, D = 1, S = 10,
        guess = FitTo(InitialCondition(); method = OrthogonalGreedy()),
        problem = wave_problem(T; D = D), steps = NETWORK_STEPS) where {T}
    method = SpaceTimeGalerkin(network_ansatz(T; mode = mode, D = D, S = S),
        WeakInitialAndBoundaryConditions(); initial_guess = guess)
    int = GeometricIntegrator(problem, method)
    result = integrate(int)
    return problem, method, int, result
end

"""The sup-norm deviation of a run from the `u` of a §A entry, over the steps it took."""
function sup_deviation(result, entry)
    return maximum(maximum(abs.(result.solution.u[n] .- entry.u[n, :]))
    for n in 1:length(result.traces))
end

"""Whether every state of a solution series is finite."""
function finite_states(x)
    return all(n -> all(isfinite, x[n]), eachindex(x))
end

"""
    record_deviation(record, entry, T, deviation)

Write one §A deviation with the metadata that makes the figure readable on another machine: the
testset that measured it, the entry it is a distance from, the element type, the Julia version, the
machine, the CPU, the BLAS thread count and the `--check-bounds` setting of the process (`0` auto,
`1` yes, `2` no). No bound is asserted on the value.
"""
function record_deviation(record::Symbol, entry::Symbol, ::Type{T}, deviation) where {T}
    @info("the §A deviation of a network run", record, entry, T, deviation, VERSION,
        machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        blas_threads=BLAS.get_num_threads(),
        check_bounds=Base.JLOptions().check_bounds)
end

"""Whether two solution series are bit-identical on every state they hold."""
function same_states(x, y)
    return axes(x) == axes(y) && all(n -> x[n] == y[n], eachindex(x))
end

"""
    network_control(::Type{T})

The affine control of a `:last_layer` slab: `S = 20` hidden units, whose freed output weights are
the only unknowns the step's rows read, solved with `f_abstol = 0` at the family's relative
tolerance and at most two Newton iterations.
"""
function network_control(::Type{T}; steps = NETWORK_STEPS) where {T}
    problem = wave_problem(T; timespan = (zero(T), T(steps) * T(0.05)))
    method = SpaceTimeGalerkin(
        network_ansatz(T; S = NETWORK_CONTROL_S, mode = :last_layer),
        WeakInitialAndBoundaryConditions();
        initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
    int = GeometricIntegrator(problem, method; f_abstol = zero(T),
        f_reltol = network_f_reltol(T), max_iterations = LAST_LAYER_ITERATIONS)
    return integrate(int)
end

"""
    network_slab(::Type{T}; S, mode, D, xspan, problem)

The first slab of a network run, set up as the property suite does: its datum in the coupling's
block, and the solver's vector from the method's guess, before the solver runs.
"""
function network_slab(::Type{T}; S = 10, mode = :all, D = 1, xspan = (zero(T), one(T)),
        problem = wave_problem(T; D = D)) where {T}
    method = SpaceTimeGalerkin(network_ansatz(T; S = S, mode = mode, D = D, xspan = xspan),
        WeakInitialAndBoundaryConditions();
        initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
    int = GeometricIntegrator(problem, method)
    solution = GeometricSolution(problem)
    solstep = solutionstep(int, solution[0])
    reset!(solstep, timesteps(solution)[1])
    c = cache(int)
    M.slab_bottom!(c, c.coupling, problem, ansatz(method), current(solstep))
    M.initial_guess!(solstep, nothing, parameters(problem), int)
    return (; problem, method, int, solstep)
end

"""
    central_differences(int, θ, params)

The central differences of the slab's `residual!` at `θ`, one column per unknown, with the step
`δ_k = cbrt(eps(T)) max(1, |θ_k|)`. Each column divides by the step as it is represented.
"""
function central_differences(int, θ::AbstractVector{T}, params) where {T}
    N = length(θ)
    J = zeros(T, N, N)
    forward = zeros(T, N)
    backward = zeros(T, N)
    x = copy(θ)
    for k in 1:N
        δ = cbrt(eps(T)) * max(one(T), abs(θ[k]))
        x[k] = θ[k] + δ
        M.residual!(forward, x, nothing, params, int)
        up = x[k]
        x[k] = θ[k] - δ
        M.residual!(backward, x, nothing, params, int)
        J[:, k] .= (forward .- backward) ./ (up - x[k])
        x[k] = θ[k]
    end
    return J
end

# The physical Gauss nodes and weights of one direction of a slab, computed here so that a node set
# read in another coordinate shows up as a difference rather than as an error.
function gauss_rule(::Type{T}, a, b, ncells::Integer, n::Integer = 4) where {T}
    ξ = T[-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]
    ω = T[0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
    nodes = T[]
    weights = T[]
    width = (b - a) / ncells
    for c in 0:(ncells - 1), k in 1:n

        push!(nodes, a + width * (T(c) + (ξ[k] + one(T)) / 2))
        push!(weights, width / 2 * ω[k])
    end
    return nodes, weights
end

@testset "Network ansatz" begin
    @testset "S1b a projection on a network is refused" for T in (Float64, Float32)
        problem = wave_problem(T)
        ansatz = network_ansatz(T)
        make(fit) = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = fit)
        err = try
            GeometricIntegrator(problem, make(FitTo(InitialCondition(); method = Projection())))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Projection()", err.msg)
        @test occursin("NetworkAnsatz", err.msg)
        @test GeometricIntegrator(problem,
            make(FitTo(InitialCondition()))) isa GeometricIntegrator
        @test GeometricIntegrator(problem, make(Continuation())) isa GeometricIntegrator
    end

    @testset "S2b the size of one slab's system" for T in (Float64, Float32)
        problem = wave_problem(T)
        for mode in (:all, :last_layer), S in (4, 10)

            ansatz = network_ansatz(T; S = S, mode = mode)
            method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions())
            int = GeometricIntegrator(problem, method)
            @test length(nlsolution(int)) == total_free_coefficients(method, problem)
            @test length(nlsolution(int)) ==
                  M.nfree(ansatz) +
                  M.coupling_unknowns(coupling(method), ansatz,
                problem)
            free = mode === :last_layer ? S : 3S + S
            @test M.nfree(ansatz) == free
        end
        # the bottom multipliers span every space function that vanishes at both ends, as many as a
        # `Dirichlet()` basis of the same degree and cells has, and each side has one per time function
        dirichlet = TensorBSplineAnsatz(
            degree = 3, ncells = (2, 5), xspan = (zero(T), one(T)),
            x_boundary = Dirichlet())
        @test M.multiplier_counts(WeakInitialAndBoundaryConditions(), network_ansatz(T)) ==
              (M.nspace_functions(dirichlet), M.ntime_functions(dirichlet))
        # the tensor path keeps the base's count: free rows × space functions × components
        spline = TensorBSplineAnsatz(
            degree = 2, ncells = (2, 5), xspan = (zero(T), one(T)),
            x_boundary = Free())
        method = SpaceTimeGalerkin(spline, WeakInitialAndBoundaryConditions())
        int = GeometricIntegrator(problem, method)
        @test length(nlsolution(int)) == total_free_coefficients(method, problem)
        @test length(nlsolution(int)) ==
              M.free_rows(coupling(method), spline) *
              M.nspace_functions(spline) * 1 +
              M.coupling_unknowns(coupling(method), spline, problem)
    end

    @testset "N2 the parameters live in the cache, not in the method" for T in (Float64, Float32)
        problem = wave_problem(T)
        ansatz = network_ansatz(T; S = 4)
        before = (ansatz.mode, ansatz.S, ansatz.D, ansatz.np, ansatz.nfree,
            ansatz.first_free, ansatz.xspan, ansatz.chain)
        first_method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        second_method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        first_int = GeometricIntegrator(problem, first_method)
        second_int = GeometricIntegrator(problem, second_method)
        first = integrate(first_int)
        second = integrate(second_int)
        again = integrate(first_int)
        for run in (second, again)
            @test same_states(run.solution.u, first.solution.u)
            @test same_states(run.solution.v, first.solution.v)
            @test run.stats.iterations == first.stats.iterations
            @test run.stats.residuals == first.stats.residuals
        end
        after = (ansatz.mode, ansatz.S, ansatz.D, ansatz.np, ansatz.nfree,
            ansatz.first_free, ansatz.xspan, ansatz.chain)
        @test all(before[k] === after[k] for k in eachindex(before))
        # one ansatz object, two integrators: each keeps its parameters in its own block, so a write
        # into one is not seen by the other
        first_block = cache(first_int).ansatz
        second_block = cache(second_int).ansatz
        kept = copy(second_block.params)
        first_block.params .+= one(T)
        @test second_block.params == kept
    end

    @testset "N5 the element type is the problem's" for T in (Float64, Float32)
        _, _, int, result = network_run(T; steps = 2)
        c = cache(int)
        assert_no_upcast(result.solution.u, T)
        assert_no_upcast(result.solution.v, T)
        assert_no_upcast(c.trace_u, T)
        assert_no_upcast(c.x, T)
        assert_no_upcast(c.ansatz.params, T)
        assert_no_upcast(c.ansatz.u, T)
        assert_no_upcast(c.ansatz.Ju, T)
        assert_no_upcast(c.ansatz.gradient, T)
        foreach(t -> assert_no_upcast(t, T), result.traces)
    end

    # A point evaluation runs the network's in-place kernels from the block's own parameter set: it
    # agrees bit for bit with the allocating call of the generated functions, and allocates nothing.
    @testset "N6 a point evaluation allocates nothing" for T in (Float64, Float32)
        for mode in (:all, :last_layer)
            s = cache(network_slab(T; mode = mode).int).ansatz
            point = T[T(3 // 10), T(2 // 5)]
            M.evaluate_at!(s, point[1], point[2])
            @test s.values == s.ansatz.field(point, s.params)
            @test s.gradient == s.ansatz.gradient(point, s.params)
            evaluate(y) = M.evaluate_at!(s, y[1], y[2])
            allocations(f::F, y::Y) where {F, Y} = (f(y); @allocated f(y))
            # the helper sees an allocation, so a zero below is the evaluation's and not the helper's
            @test allocations(collect, point) > 0
            # asserted under `--check-bounds=auto` only; a run that forces `--check-bounds=yes`
            # skips the count
            if Base.JLOptions().check_bounds != 1
                @test allocations(evaluate, point) == 0
            end
        end
    end

    # The two `S = 10` arms are records: in `:last_layer` mode the sixteen constraint rows of the
    # coupling outnumber the ten output weights they reach, so the step's system is inconsistent,
    # and the `:all` arm's solve does not reach round-off (K38); neither arm meets the default
    # tolerances of the solver, and the run is its own expected value (U13, U24). The affine control
    # is the assertion: at `S = 20` the freed output weights carry every row, and two Newton
    # iterations reach the root of an affine system.
    @testset "N1 the network integrates Wave under both modes" for T in (Float64, Float32)
        for (mode, name, entry) in ((:all, :WAVE_NN_FULLY, WAVE_NN_FULLY),
            (:last_layer, :WAVE_NN_PARTIALLY, WAVE_NN_PARTIALLY))
            _, _, int, result = network_run(T; mode = mode)
            c = cache(int)
            @test length(result.traces) == NETWORK_STEPS
            @test length(result.stats.iterations) == NETWORK_STEPS
            @test length(result.stats.residuals) == NETWORK_STEPS
            @test length(result.stats.initial_residuals) == NETWORK_STEPS
            @test length(result.stats.converged) == NETWORK_STEPS
            @test minimum(result.stats.iterations) ≥ 1
            @test finite_states(result.solution.u)
            @test finite_states(result.solution.v)
            @test all(isfinite, c.trace_u)
            deviation = sup_deviation(result, entry)
            @test isfinite(deviation)
            record_deviation(:N1, name, T, deviation)
            assert_no_upcast(result.solution.u, T)
        end
        control = network_control(T)
        @test length(control.traces) == NETWORK_STEPS
        @test all(control.stats.converged)
        @test all(control.stats.iterations .<= LAST_LAYER_ITERATIONS)
        @test maximum(control.stats.residuals ./ control.stats.initial_residuals) ≤
              network_f_reltol(T)
        @test finite_states(control.solution.u)
        # the identity of the free rows: the control's per-step ratio is the solver's round-off,
        # where a step whose fixed parameters never reached the cache of the solver's element type
        # leaves the Jacobian at zero and the ratio at one
        @test maximum(control.stats.residuals ./ control.stats.initial_residuals) < one(T)
        assert_no_upcast(control.solution.u, T)
    end

    # The Jacobian the solver builds against central differences of the residual, on the first slab
    # of N1's case, in both modes: a `:last_layer` step whose fixed parameters never reached the
    # cache of the solver's element type has a Jacobian of zero, whatever its residual is.
    @testset "N1b the Jacobian is the derivative of the residual" for T in (Float64, Float32)
        for mode in (:all, :last_layer)
            slab = network_slab(T; S = 10, mode = mode)
            s = solver(slab.int)
            N = length(cache(slab.int).x)
            params = parameters(slab.problem)
            reference = network_slab(Float64; S = 10, mode = mode)
            # the reference slab carries the fixed parameters of the slab under test: its own fit
            # selects a hidden layer of another precision, and the two residuals would differ by
            # more than the solver's Jacobian differs from central differences
            cache(reference.int).ansatz.params .= Float64.(cache(slab.int).ansatz.params)
            rng = Xoshiro(1234)
            for _ in 1:16
                θ = T.(2 .* rand(rng, N) .- 1)
                jacobian!(s, θ, (current(slab.solstep), params, slab.int))
                J = copy(jacobianmatrix(s))
                scale = maximum(abs, J)
                @test scale ≥ T(1 // 100)
                @test maximum(abs,
                    J - central_differences(reference.int, Float64.(θ),
                        parameters(reference.problem))) ≤ sqrt(eps(T)) * scale
            end
        end
    end

    # The edges against the discrete action: a bottom or a top point read in the normalised
    # coordinate moves the residual away from the oracle by a quarter of its scale, where the
    # unmutated residual agrees with it to round-off.
    @testset "N3 the boundary points are physical" for T in (Float64, Float32)
        xspan = (T(-0.5), zero(T))
        problem = LinearTransport.problem(T = T, timespan = (zero(T), T(0.15)),
            timestep = T(0.05), xstep = T(0.05))
        ansatz = network_ansatz(T; degree = 3, ncells = (2, 5), D = 2, xspan = xspan)
        method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(ExactSolution(); method = OrthogonalGreedy()))
        int = GeometricIntegrator(problem, method)
        result = integrate(int)
        c = cache(int)
        @test length(result.traces) == 3
        @test length(result.stats.iterations) == 3
        @test length(result.stats.residuals) == 3
        # a fresh integrator's first slab, its datum fixed, at a vector away from its solution
        slab = network_slab(T; D = 2, xspan = xspan, problem = problem)
        oracle = ActionOracle(slab.problem, slab.method)
        params = parameters(slab.problem)
        N = length(cache(slab.int).x)
        @test N == unknown_count(oracle)
        θ = T.(2 .* rand(Xoshiro(1234), N) .- 1)
        b = zeros(T, N)
        M.residual!(b, θ, nothing, params, slab.int)
        ∇S = oracle_residual(oracle, first_datum(oracle), θ)
        @test all(isfinite, b)
        @test maximum(abs, ∇S) ≥ T(1 // 100)
        @test maximum(abs, b - ∇S) ≤ 100 * eps(T) * maximum(abs, ∇S)
        # the block's own nodes are the physical ones, and the initial condition is read there
        tnodes, xnodes = M.quadrature_points(c.ansatz)
        xt, _ = gauss_rule(T, xspan[1], xspan[2], 5,
            M.quadrature_order_of(M.trace_basis(ansatz))[2])
        @test collect(extrema(xnodes)) ≈ collect(extrema(xt))
        @test length(xnodes) == length(xt)
        sample = zeros(T, length(tnodes), length(xnodes), 2)
        M.evaluate!(sample, InitialCondition(), c, ansatz, problem,
            solutionstep(int,
                GeometricSolution(problem)[0]))
        values = problem.ics_function(collect(xt), parameters(problem))
        scale = maximum(abs, values.u)
        for r in eachindex(xt), d in 1:2

            @test abs(sample[1, r, d] - T(values.u[r, d])) ≤
                  16 * eps(T) * max(scale, one(T))
        end
        # a node set read in the normalized coordinate is not the physical one
        @test extrema(xnodes) != extrema(range(zero(T), one(T); length = length(xnodes)))
    end

    @testset "N4 a network with two outputs integrates VectorWave" for T in (Float64, Float32)
        params = VectorWave.default_parameters()
        problem = LagrangianPDEProblem(
            VectorWave.vector_density(; params = params, β = T(0.5));
            D = 2,
            ics_function = VectorWave.problem(T = T; params = params, timestep = T(0.05)).ics_function,
            bcs_function = VectorWave.problem(T = T; params = params, timestep = T(0.05)).bcs_function,
            solutions = VectorWave.problem(T = T; params = params, timestep = T(0.05)).solutions,
            timespan = (zero(T), T(0.5)), timestep = T(0.05), xspan = (zero(T), one(T)),
            xstep = T(0.05), params = params)
        ansatz = network_ansatz(T; S = 10, D = 2)
        method = SpaceTimeGalerkin(ansatz, WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        int = GeometricIntegrator(problem, method)
        result = integrate(int)
        c = cache(int)
        @test length(result.traces) == 10
        @test finite_states(result.solution.u)
        @test all(isfinite, c.trace_u)
        @test all(isfinite, c.trace_v)
        # each component holds its own output weights, one column of the fit's coefficients
        @test M.output_weights(c.ansatz.params, ansatz, 1) !=
              M.output_weights(c.ansatz.params, ansatz, 2)
        # the step's residual is recorded: a network step is a record, and it may fail (U13, U24)
        @test result.stats.residuals[1] ≤ result.stats.initial_residuals[1]
        # the output dimension is the problem's, and the input layer's is two in either case: a
        # network of two outputs on a scalar problem is refused when its cache is built
        @test M.noutputs(ansatz) == 2
        scalar = SpaceTimeGalerkin(network_ansatz(T; D = 2),
            WeakInitialAndBoundaryConditions())
        @test_throws ArgumentError GeometricIntegrator(wave_problem(T), scalar)
        @test_throws ArgumentError NetworkAnsatz(
            S = 4, xspan = (zero(T),), ncells = (2, 5),
            degree = 3)
        @test_throws ArgumentError NetworkAnsatz(S = 4, D = 0, xspan = (zero(T), one(T)),
            ncells = (2, 5), degree = 3)
        @test_throws ArgumentError NetworkAnsatz(S = 0, xspan = (zero(T), one(T)),
            ncells = (2, 5), degree = 3)
        @test_throws ArgumentError NetworkAnsatz(S = 4, xspan = (zero(T), one(T)),
            ncells = (2, 5), degree = 3, mode = :full)
    end

    @testset "N7 networks of one architecture share their kernels" for T in (Float64, Float32)
        a = network_ansatz(T; S = 6, D = 2)
        b = network_ansatz(T; S = 6, D = 2, mode = :last_layer)
        # the mode and the element type do not enter the symbolic build
        c = network_ansatz(T === Float64 ? Float32 : Float64; S = 6, D = 2)
        @test a.field === b.field
        @test a.gradient === b.gradient
        @test a.field === c.field
        @test a.gradient === c.gradient
        # another width, another output count or another activation is another architecture
        @test network_ansatz(T; S = 7, D = 2).field !== a.field
        @test network_ansatz(T; S = 6, D = 1).field !== a.field
        other = NetworkAnsatz(S = 6, D = 2, activation = sin, xspan = (zero(T), one(T)),
            ncells = (2, 5), degree = 3)
        @test other.field !== a.field
        @test other.gradient !== a.gradient
    end

    @testset "A1 the §A case in both modes" for T in (Float64, Float32)
        for (mode, name, entry) in ((:all, :WAVE_NN_FULLY, WAVE_NN_FULLY),
            (:last_layer, :WAVE_NN_PARTIALLY, WAVE_NN_PARTIALLY))
            _, _, _, result = network_run(T; mode = mode,
                guess = FitTo(ExactSolution(); method = OrthogonalGreedy()))
            @test length(result.traces) == NETWORK_STEPS
            @test length(result.stats.iterations) == NETWORK_STEPS
            @test finite_states(result.solution.u)
            deviation = sup_deviation(result, entry)
            @test isfinite(deviation)
            record_deviation(:A1, name, T, deviation)
            if mode === :all
                symbolic = sup_deviation(result, WAVE_NN_SYMBOLIC_FULLY)
                @test isfinite(symbolic)
                record_deviation(:A1, :WAVE_NN_SYMBOLIC_FULLY, T, symbolic)
            end
        end
        # the three-step SineGordon case of the entry
        problem = SineGordon.problem(
            T = T, timespan = (zero(T), T(0.15)), timestep = T(0.05),
            xstep = T(0.05))
        method = SpaceTimeGalerkin(network_ansatz(T), WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(ExactSolution(); method = OrthogonalGreedy()))
        result = integrate(GeometricIntegrator(problem, method))
        @test length(result.traces) == 3
        deviation = sup_deviation(result, SINE_GORDON_NN_FULLY)
        @test isfinite(deviation)
        record_deviation(:A1, :SINE_GORDON_NN_FULLY, T, deviation)
    end
end

# ---- the fixed-boundary ansatz -------------------------------------------------------
#
# The spatial-and-bubble network of `FixedBoundaryNetwork`: its two charts and its field map, read
# here from their own formulas. Its slab is built with the coupling, which only carries the datum.

"""A fixed-boundary ansatz on `Wave`'s span at the family's coarse rule."""
function fixed_boundary_ansatz(::Type{T}; Sx = 8, St = 4, form = :abg, mode = :all,
        ncells = 3, nq = (4, 3)) where {T}
    return FixedBoundaryNetwork(Sx = Sx, St = St, bubble_form = form, mode = mode,
        xspan = (zero(T), one(T)), ncells = ncells, nq = nq, D = 1)
end

"""
The field of a fixed-boundary slab at one node, from its formula alone: the two levels' networks,
the lift of the side data, and the bubble.
"""
function fixed_boundary_field(problem, a, φ0, φ1, Θ, t̂, x, h)
    xa, xb = xspan(problem)
    L, span = xb - xa, xb - xa
    ŝ = (x - xa) / span
    bump = (x - xa) * (xb - x)
    slope = xa + xb - 2x
    p = parameters(problem)
    at0 = problem.bcs_function(0.0, p)
    at1 = problem.bcs_function(h, p)
    bcs = problem.bcs_function(h * t̂, p)
    ga0, gb0 = at0.left.u[1], at0.right.u[1]
    ga1, gb1 = at1.left.u[1], at1.right.u[1]
    ga, gb = bcs.left.u[1], bcs.right.u[1]
    dga = ForwardDiff.derivative(τ -> problem.bcs_function(τ, p).left.u[1], h * t̂)
    dgb = ForwardDiff.derivative(τ -> problem.bcs_function(τ, p).right.u[1], h * t̂)
    level(φ, ga_, gb_) = begin
        m, mŝ = M.spatial_values(φ, a.Sx, a.D, 1, ŝ)
        ((1 - ŝ) * ga_ + ŝ * gb_ + bump * m,
            (gb_ - ga_) / span + slope * m + bump * mŝ / span)
    end
    n0, n0x = level(φ0, ga0, gb0)
    n1, n1x = level(φ1, ga1, gb1)
    ψ, ψt, ψŝ = zero(eltype(Θ)), zero(eltype(Θ)), zero(eltype(Θ))
    for k in 1:a.St
        α, β, γ, _ = M.bubble_unit(a, Θ, k)
        ak = Θ[3a.St + k]
        s = tanh(α * t̂ + β * ŝ + γ)
        ψ += ak * s
        ψt += ak * α * (1 - s^2)
        ψŝ += ak * β * (1 - s^2)
    end
    ra = ga - (1 - t̂) * ga0 - t̂ * ga1
    rb = gb - (1 - t̂) * gb0 - t̂ * gb1
    rat = h * dga + ga0 - ga1
    rbt = h * dgb + gb0 - gb1
    b = t̂ * (1 - t̂)
    u = (1 - t̂) * n0 + t̂ * n1 + (1 - ŝ) * ra + ŝ * rb + b * bump * ψ
    ut = n1 - n0 + (1 - ŝ) * rat + ŝ * rbt + bump * ((1 - 2t̂) * ψ + b * ψt)
    ux = (1 - t̂) * n0x + t̂ * n1x + (rb - ra) / L + b * (slope * ψ + bump * ψŝ / L)
    return u, ut / h, ux
end

@testset "the fixed-boundary ansatz's Gauss rule" begin
    # The rule of the slab's own quadrature, at the two counts the family uses and at three element
    # types: its nodes are the rule's own, sorted on `[0, 1]`, and it integrates `xᵏ` exactly for
    # every `k` below `2n`. `BigFloat` is the leg that shows the Newton iteration's own
    # convergence — a step that is only slowed still reaches the same root at `Float64` in
    # `100` iterations, and does not at a 256-bit mantissa.
    for T in (Float32, Float64, BigFloat), n in (4, 6)

        nodes, weights = M.gauss_legendre(n, T)
        @test eltype(nodes) === T
        @test eltype(weights) === T
        @test length(nodes) == n && length(weights) == n
        @test issorted(nodes)
        @test all(zero(T) .< nodes .< one(T))
        @test all(>(zero(T)), weights)
        for k in 0:(2n - 1)
            @test abs(sum(weights .* nodes .^ k) - one(T) / (k + 1)) ≤ 8 * eps(T)
        end
    end
end

@testset "the fixed-boundary ansatz's two charts" begin
    for form in (:abg, :tsr)
        a = fixed_boundary_ansatz(Float64; form = form)
        Θ = M.bubble_start(a, Float64)
        # one unit, read in its own coordinates, has the radius those coordinates carry
        for k in 1:a.St
            α, β, γ, r = M.bubble_unit(a, Θ, k)
            @test r == sqrt(α^2 + β^2)
            @test isfinite(α) && isfinite(β) && isfinite(γ)
        end
        # the chart of an `:abg` atom is `:tsr`'s and back
        grid = M.bubble_start(fixed_boundary_ansatz(Float64; form = :abg), Float64)
        for k in 1:a.St
            θ, s, ρ = M.bubble_chart(grid[k], grid[a.St + k], grid[2a.St + k], k)
            @test exp(ρ) * cos(θ) ≈ grid[k]
            @test exp(ρ) * sin(θ) ≈ grid[a.St + k]
            @test -exp(ρ) * s ≈ grid[2a.St + k]
        end
        # no `(θ, s, ρ)` writes a unit at `r = 0`, and the refusal names it
        @test_throws r"^ArgumentError: the bubble's unit 3 has α = β = 0" M.bubble_chart(
            0.0, 0.0, 0.5, 3)
        # the dictionary is one dictionary, less its atoms at `α = β = 0`
        atoms = bubble_atoms(BubbleGreedy(), Float64)
        @test size(atoms, 1) == 2184
        @test !any(iszero(atoms[i, 1]) && iszero(atoms[i, 2]) for i in axes(atoms, 1))
    end
end

@testset "the fixed-boundary ansatz's field and its top" begin
    problem = wave_problem(Float64)
    for form in (:abg, :tsr)
        a = fixed_boundary_ansatz(Float64; form = form)
        method = SpaceTimeGalerkin(a, FixedBoundary())
        int = GeometricIntegrator(problem, method)
        c = cache(int)
        solution = GeometricSolution(problem)
        solstep = solutionstep(int, solution[0])
        reset!(solstep, timesteps(solution)[1])
        M.slab_bottom!(c, c.coupling, problem, a, current(solstep))
        M.initial_guess!(current(solstep), nothing, parameters(problem), int)
        M.residual!(zeros(length(c.x)), c.x, nothing, parameters(problem), int)
        h = timestep(problem)
        φ0 = copy(c.coupling.phi)
        φ1 = copy(c.ansatz.params[1:M.nspatial(a)])
        Θ = copy(M.bubble_parameters(c.ansatz))
        tb = a.tabulation
        worst = (u = 0.0, v = 0.0, w = 0.0)
        for q in eachindex(tb.tnode), r in eachindex(tb.xnode)

            u, v, w = fixed_boundary_field(problem, a, φ0, φ1, Θ, tb.tnode[q],
                tb.xnode[r], h)
            worst = (u = max(worst.u, abs(u - c.u[q, r, 1])),
                v = max(worst.v, abs(v - c.v[q, r, 1])),
                w = max(worst.w, abs(w - c.w[q, r, 1])))
        end
        @test worst.u ≤ 1e3 * eps(Float64) * max(1, maximum(abs, c.u))
        @test worst.v ≤ 1e3 * eps(Float64) * max(1, maximum(abs, c.v))
        @test worst.w ≤ 1e3 * eps(Float64) * max(1, maximum(abs, c.w))
        # the top of the slab is the solved level's network: the bubble's weights vanish there
        ev = M.output_evaluation(c.ansatz)
        u = zeros(length(ev.points), 1)
        v = similar(u)
        w = similar(u)
        M.top_values!(u, v, w, c.ansatz, ev, h)
        at1 = problem.bcs_function(h, parameters(problem))
        for (i, x) in enumerate(ev.points)
            ŝ = (x - xspan(problem)[1]) / (xspan(problem)[2] - xspan(problem)[1])
            m, _ = M.spatial_values(φ1, a.Sx, a.D, 1, ŝ)
            expect = (1 - ŝ) * at1.left.u[1] + ŝ * at1.right.u[1] +
                     (x - xspan(problem)[1]) * (xspan(problem)[2] - x) * m
            @test abs(u[i, 1] - expect) ≤ 1e3 * eps(Float64) * max(1, abs(expect))
        end
    end
end
