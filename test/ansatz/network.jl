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
# The helpers this file shares with `test/ansatz/network_cases.jl`, which holds N3, N4 and A1.
include("../helpers/network_setup.jl")

@testset "Network ansatz" begin
    # The `:all` run of N1's case, once per element type: N1 and N5 read it, and neither changes it.
    all_runs = Dict(T => network_run(T; mode = :all) for T in (Float64, Float32))

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
        _, _, int, result = all_runs[T]
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
            _, _, int, result = mode === :all ? all_runs[T] : network_run(T; mode = mode)
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

    # The solver's `regularization_factor` reaches the step: one Newton iteration of N1's first slab
    # at `λ = 0` and at `λ = 1e-2` solves two different linear systems, so the two steps' states
    # differ by far more than round-off, and each integrator's options hold the `λ` it was built
    # with. No convergence is asserted: one iteration of the `:all` mode is an unconverged iterate.
    @testset "N8 the regularization factor reaches the step" for T in (Float64, Float32)
        problem = wave_problem(T; timespan = (zero(T), T(0.05)))
        method = SpaceTimeGalerkin(network_ansatz(T), WeakInitialAndBoundaryConditions();
            initial_guess = FitTo(InitialCondition(); method = OrthogonalGreedy()))
        states = map((zero(T), T(1e-2))) do λ
            int = GeometricIntegrator(problem, method; max_iterations = 1,
                regularization_factor = λ)
            @test SimpleSolvers.config(solver(int)).regularization_factor == λ
            result = integrate(int)
            @test length(result.traces) == 1
            @test result.stats.iterations == [1]
            u = result.solution.u[1]
            @test eltype(u) === T
            @test all(isfinite, u)
            u
        end
        plain, shifted = states
        @test norm(shifted - plain) > 100 * eps(T) * norm(plain)
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
        # two separate builds of equal bodies are `===` too, so the cache itself is the check
        @test lock(kernels -> kernels[(6, 2, tanh)], M.NETWORK_KERNELS) ===
              (a.field, a.gradient)
        entries = lock(length, M.NETWORK_KERNELS)
        network_ansatz(T; S = 6, D = 2)
        @test lock(length, M.NETWORK_KERNELS) == entries
        # another width, another output count or another activation is another architecture
        @test network_ansatz(T; S = 7, D = 2).field !== a.field
        @test network_ansatz(T; S = 6, D = 1).field !== a.field
        other = NetworkAnsatz(S = 6, D = 2, activation = sin, xspan = (zero(T), one(T)),
            ncells = (2, 5), degree = 3)
        @test other.field !== a.field
        @test other.gradient !== a.gradient
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
